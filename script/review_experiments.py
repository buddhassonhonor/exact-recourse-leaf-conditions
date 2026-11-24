"""Independent solver verification and held-out follow-up experiments."""

import argparse
import itertools
import json
import time
import traceback
import warnings
from copy import copy
from datetime import datetime
from pathlib import Path

from run_recourse import (
    CONFIG, ROOT, Action, LeafIndex, create_model, distance, encode, historical_baseline,
    load_openml, save_provenance, take_queries, train_test_split, transform, write_jsonl,
)

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

FRACTIONS = (.125, .25, .5, .75)
ORACLE_DATASETS = (29, 31, 1461, 1590)
FOLLOWUP_DATASETS = (31, 44, 151, 1461, 1486, 1590, 42477, 45026)


def compile_flow(model, levels):
    """Read tree comparisons without using the projection index."""
    fields = [(j, c) for j, count in enumerate(levels) for c in range(max(1, count))]
    tree = model.tree_
    stack = [(0, ())]
    target_paths = []
    cuts = [set() for _ in levels]
    while stack:
        node, path = stack.pop()
        j = tree.feature[node]
        if j < 0:
            if model.classes_[tree.value[node, 0].argmax()] == 1:
                target_paths.append(path)
            continue
        field, category = fields[j]
        t = float(tree.threshold[node])
        if levels[field] == 0:
            v = np.float32(t)
            lo = v if float(v) <= t else np.nextafter(v, np.float32(-np.inf))
            hi = v if float(v) > t else np.nextafter(v, np.float32(np.inf))
            cuts[field].update([float(lo), float(hi)])
        stack.extend([
            (tree.children_left[node], (*path, (field, category, t, True))),
            (tree.children_right[node], (*path, (field, category, t, False))),
        ])
    return cuts, target_paths


def solve_milp(model, compiled, x, levels, scales, action):
    cuts, paths = compiled
    if not paths:
        return None, 2, 0., None, None, None
    choices = []
    for j, level_count in enumerate(levels):
        v = x[j]
        if j in action.frozen:
            allowed = np.array([v])
        elif level_count:
            allowed = np.arange(level_count)
        else:
            allowed = np.unique([v, *cuts[j]])
            if j in action.increasing:
                allowed = allowed[allowed >= v]
            if j in action.decreasing:
                allowed = allowed[allowed <= v]
        choices.append(allowed)
    starts = np.r_[0, np.cumsum([len(c) for c in choices])]
    objective = np.concatenate([np.zeros(len(paths)), *[
        action.categorical_cost * (v != x[j]) if levels[j] else abs(v - x[j]) / scales[j]
        for j, v in enumerate(choices)
    ]])
    # Use O(1) binary coefficients for the branch logic even at adjacent float32 values.
    row_index, column_index, rhs, lhs = [], [], [], []

    def constraint(columns, lower, upper):
        row_index.extend([len(rhs)] * len(columns))
        column_index.extend(columns)
        lhs.append(lower)
        rhs.append(upper)

    constraint(range(len(paths)), 1., 1.)
    for j in range(len(choices)):
        constraint(range(len(paths) + starts[j], len(paths) + starts[j + 1]), 1., 1.)
    for leaf, path in enumerate(paths):
        for j, cat, threshold, left in path:
            encoded = choices[j] == cat if levels[j] else choices[j]
            violates = np.flatnonzero((encoded <= threshold) != left)
            if len(violates):
                constraint([leaf, *(len(paths) + starts[j] + violates)], 0., 1.)
    matrix = coo_matrix((np.ones(len(row_index)), (row_index, column_index)),
                        shape=(len(rhs), len(objective))).tocsc()
    start = time.perf_counter_ns()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected")
        out = milp(objective * 1e6, integrality=np.ones(len(objective)),
                   bounds=Bounds(0., 1.), constraints=LinearConstraint(matrix, lhs, rhs),
                   options={"time_limit": 5., "mip_rel_gap": 0., "threads": 1})
    elapsed = (time.perf_counter_ns() - start) / 1000.
    if out.status != 0:
        return None, int(out.status), elapsed, None, None, None
    z = np.asarray([v[np.argmax(out.x[len(paths) + starts[j]:len(paths) + starts[j + 1]])]
                    for j, v in enumerate(choices)])
    assert int(model.predict(encode(z[None], levels))[0]) == 1
    assert all(z[j] == x[j] for j in action.frozen)
    assert all(z[j] >= x[j] for j in action.increasing)
    assert all(z[j] <= x[j] for j in action.decreasing)
    cost = float(distance(x, z, levels, scales, action.categorical_cost))
    return cost, int(out.status), elapsed, int(out.mip_node_count), float(out.mip_gap), z.tolist()


def oracle_fold(data_root, did, seed):
    frame, y, meta = load_openml(data_root, did)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, random_state=seed, stratify=y)
    x, levels, scales, details = transform(frame, train)
    rng = np.random.default_rng(seed)
    encoded = encode(x, levels)
    model, _ = create_model(encoded[train], y[train], seed, "CART", 6)
    ids = take_queries(model.predict(encoded), test, rng, 8)
    index = LeafIndex(model, levels, scales)
    compiled = compile_flow(model, levels)
    numeric = np.flatnonzero(levels == 0)
    selected = rng.permutation(len(levels))[:max(1, len(levels) * 3 // 4)]
    rows = []
    for mode, action in (
        ("free", Action()),
        ("restricted", Action(tuple(selected), tuple(numeric[:1]), tuple(numeric[1:2]), 1.)),
    ):
        _, expected, _ = index.solve(x[ids], action, "interned")
        for record_id, optimum in zip(ids, expected):
            value, status, us, nodes, gap, witness = solve_milp(
                model, compiled, x[record_id], levels, scales, action)
            reference = None if np.isinf(optimum) else float(optimum)
            error = abs(value - reference) if value is not None and reference is not None else None
            agreement = (status == 2 and reference is None) or (
                status == 0 and error is not None and error <= 1e-7)
            rows.append({"experiment": "oracle", "dataset": did, "seed": seed,
                         "query": int(record_id), "mode": mode, "oracle_cost": value,
                         "reference_cost": reference, "status": status, "agreement": agreement,
                         "solve_us": us, "nodes": nodes, "mip_gap": gap, "error": error,
                         "factual": x[record_id].tolist() if mode == "restricted" else None,
                         "oracle_answer": witness if mode == "restricted" else None,
                         "frozen": [int(j) for j in action.frozen],
                         "increasing": [int(j) for j in action.increasing],
                         "decreasing": [int(j) for j in action.decreasing]})
    return meta, {"dataset": did, "seed": seed, "train_indices": train.tolist(),
                  "test_indices": test.tolist(), "query_indices": ids.tolist(),
                  "preprocessing": details, "target_leaves": len(compiled[1])}, rows


def timed_call(fn, times):
    samples = []
    for _ in range(times):
        start = time.perf_counter_ns()
        value = fn()
        samples.append((time.perf_counter_ns() - start) / 1000.)
    return value, float(np.median(samples))


def sparse_without_sharing(index):
    """Use the same sparse incidence pattern while evaluating each occurrence."""
    result = copy(index)
    instances = index.matrix.indices
    duplicated = np.sort(instances, kind="stable")
    new_indices = np.empty(len(instances), dtype=int)
    new_indices[np.argsort(instances, kind="stable")] = np.arange(len(instances))
    from scipy.sparse import csr_matrix
    result.matrix = csr_matrix((np.ones(len(instances)), new_indices, index.matrix.indptr.copy()),
                               shape=(len(index.nodes), len(instances)))
    result.features = index.features[duplicated]
    result.condition_low = index.condition_low[duplicated]
    result.condition_high = index.condition_high[duplicated]
    result.condition_allowed = index.condition_allowed[duplicated]
    result.n_numeric_conditions = int((duplicated < index.n_numeric_conditions).sum())
    return result


def followup_fold(data_root, did, seed, batch_only=False):
    frame, y, meta = load_openml(data_root, did)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, random_state=seed, stratify=y)
    fit, calibration = train_test_split(train, test_size=.2, random_state=seed + 1000, stratify=y[train])
    x, levels, scales, details = transform(frame, fit)
    enc = encode(x, levels)
    rng = np.random.default_rng(seed + 500)
    model, _ = create_model(enc[fit], y[fit], seed, "CART", 9)
    model_preds = model.predict(enc)
    calibrate = take_queries(model_preds, calibration, rng, 48)
    heldout = take_queries(model_preds, test, rng, 256)
    if not len(calibrate) or not len(heldout):
        return meta, {"dataset": did, "seed": seed, "calibration": len(calibrate), "queries": len(heldout)}, []
    index, build_us = timed_call(lambda: LeafIndex(model, levels, scales), 3)
    unshared = sparse_without_sharing(index)
    optimized = {}
    rows = []
    pool = x[fit][model_preds[fit] == 1]
    timings = [("dense", .25), ("interned", .25), *[("cascade", f) for f in FRACTIONS]]
    for method, fraction in timings:
        if method == "cascade":
            index.stages(fraction)

    def search(method, fraction, ids):
        if method == "sparse_unshared":
            return unshared.solve(x[ids], Action(), "interned", fraction, seed)
        return index.solve(x[ids], Action(), method, fraction, seed)

    # Choose each batch policy from calibration queries only. A full test fold is never used to choose it.
    for requested in (1, 8, 32, 128, 256):
        chunk = heldout[:requested]
        calibration_chunks = [calibrate[j:j + requested] for j in range(0, len(calibrate), requested)]
        if requested > len(calibrate):
            calibration_chunks = [np.resize(calibrate, requested)]
        by_method = {}
        for pick in rng.permutation(len(timings)):
            method, fraction = timings[pick]
            times = [timed_call(lambda: search(method, fraction, block), 3)[1] / len(block)
                     for block in calibration_chunks]
            by_method[method, fraction] = float(np.median(times))
        tuned = min(FRACTIONS, key=lambda value: by_method["cascade", value])
        dispatch = min(by_method, key=by_method.get)
        optimized[str(requested)] = {"prefix": tuned, "dispatch": list(dispatch),
                                      "calibration_us": {f"{m}:{f}": t for (m, f), t in by_method.items()}}
        comparisons = [("dense", "dense", .25), ("interned", "interned", .25),
                       ("sparse_unshared", "sparse_unshared", .25),
                       ("fixed_screen", "cascade", .25), ("tuned_screen", "cascade", tuned),
                       ("calibrated_dispatch", *dispatch)]
        results = {}
        for i in rng.permutation(len(comparisons)):
            label, method, fraction = comparisons[i]
            _, elapsed = timed_call(lambda: search(method, fraction, chunk), 1)
            result, elapsed = timed_call(lambda: search(method, fraction, chunk), 5)
            results[label] = (result, elapsed)
        for label, method, fraction in comparisons:
            result, elapsed = results[label]
            assert np.allclose(result[1], results["dense"][0][1], rtol=0, atol=1e-9)
            rows.append({"experiment": "batch", "dataset": did, "seed": seed,
                         "size": requested, "n_queries": len(chunk), "method": label,
                         "selected_method": method, "fraction": fraction, "latency_us": elapsed / len(chunk),
                         "build_us": build_us, "target_leaves": len(index.nodes)})

    split = {"dataset": did, "seed": seed, "fit_indices": fit.tolist(), "test_indices": test.tolist(),
             "calibration_indices": calibration.tolist(), "calibration_queries": calibrate.tolist(),
             "queries": heldout.tolist(), "preprocessing": details, "tuning": optimized}
    if batch_only:
        return meta, split, rows

    # Use the original held-out factuals and a single set of fixed-model explanations.
    ids = heldout[:32]
    baseline = historical_baseline(x[ids], pool, index, Action(), "nearest")[0]
    exact = search("interned", .25, ids)[0]
    for noise in (0., .025, .05, .10, .20):
        for repeat in range(4):
            resample = rng.choice(fit, len(fit), replace=True)
            labels = y[resample].copy()
            flipped = rng.choice(len(labels), round(noise * len(labels)), replace=False)
            labels[flipped] = 1 - labels[flipped]
            seed_refit = seed * 100 + repeat
            refit, _ = create_model(enc[resample], labels, seed_refit, "CART", 9)
            still_negative = refit.predict(enc[ids]) == 0
            for label, answers in (("interned", exact), ("nearest", baseline)):
                eligible = np.isfinite(answers).all(axis=1)
                keep = np.zeros(len(ids), dtype=bool)
                if eligible.any():
                    keep[eligible] = refit.predict(encode(answers[eligible], levels)) == 1
                for local, factual in enumerate(ids):
                    rows.append({"experiment": "refit", "dataset": did, "seed": seed,
                                 "noise": noise, "repeat": repeat, "method": label,
                                 "query": int(factual), "eligible": bool(eligible[local]),
                                 "kept_target": bool(keep[local]),
                                 "factual_still_negative": bool(still_negative[local])})
    return meta, split, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["oracle", "followup", "batch_ablation"])
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    run = args.out or ROOT / "result" / "runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + args.stage)
    save_provenance(run, args, ("script/review_experiments.py",))
    sets = ORACLE_DATASETS if args.stage == "oracle" else FOLLOWUP_DATASETS
    function = oracle_fold if args.stage == "oracle" else followup_fold
    manifest = {}
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        try:
            for did, seed in itertools.product(sets, CONFIG["primary_seeds"]):
                extra = {"batch_only": True} if args.stage == "batch_ablation" else {}
                meta, split, rows = function(args.data_root, did, seed, **extra)
                manifest[str(did)] = meta
                write_jsonl(run / "artifacts" / "splits.jsonl", [split])
                write_jsonl(run / "metrics.jsonl", rows)
                message = f"{did} seed={seed} {len(rows)} measurement rows"
                print(message, flush=True)
                print(message, file=log, flush=True)
        except Exception:
            traceback.print_exc(file=log)
            raise
    if args.stage == "oracle":
        from scipy.optimize._highspy._core import HIGHS_VERSION_MAJOR, HIGHS_VERSION_MINOR, HIGHS_VERSION_PATCH
        details = {"solver": "HiGHS", "version": f"{HIGHS_VERSION_MAJOR}.{HIGHS_VERSION_MINOR}.{HIGHS_VERSION_PATCH}",
                   "time_limit_s": 5., "relative_gap": 0., "threads": 1, "warm_start": False,
                   "objective_multiplier": 1e6, "solver_timer_excludes_model_build": True}
        (run / "artifacts" / "solver.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (run / "artifacts" / "datasets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
