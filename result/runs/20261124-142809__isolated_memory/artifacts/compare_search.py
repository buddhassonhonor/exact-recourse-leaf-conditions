"""Paired comparison with geometric and released counterfactual-search implementations."""

import argparse
import contextlib
import io
import itertools
import json
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from importlib.metadata import version

from run_recourse import CONFIG, ROOT, Action, LeafIndex, create_model, encode, distance
from run_recourse import load_openml, train_test_split, transform, take_queries, save_provenance, write_jsonl
from region_search import RegionSearch

import numpy as np
import pandas as pd
import psutil

GEOMETRY_DATA = (3, 24, 29, 31, 37, 44, 151, 1461, 1486, 1590, 42477, 45026)
PUBLISHED_DATA = (29, 31, 1461, 1590)


def warm_timing(fn, repeats=3):
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = fn()
        elapsed.append((time.perf_counter_ns() - start) / 1000.)
    return value, float(np.median(elapsed))


def tracked_setup(build):
    current = psutil.Process()
    before = current.memory_info().rss
    value, us = warm_timing(build, 1)
    return value, {"setup_us": us, "rss_delta": current.memory_info().rss - before}


def bound_scale(x, train, levels):
    numeric = levels == 0
    lower = x[train][:, numeric].min(axis=0)
    span = x[train][:, numeric].max(axis=0) - lower
    span[span == 0] = 1.
    x[:, numeric] = np.clip((x[:, numeric] - lower) / span, 0., 1.).astype(np.float32)
    return lower.tolist(), span.tolist()


def action_set(seed, levels):
    rng = np.random.default_rng(seed)
    frozen = tuple(int(v) for v in rng.permutation(len(levels))[: len(levels) // 2])
    remaining = [int(j) for j in np.flatnonzero(levels == 0) if j not in frozen]
    return Action(frozen, tuple(remaining[:1]), tuple(remaining[1:2]))


def validate(model, q, z, levels, scales, action):
    if z is None or not np.isfinite(z).all():
        return False, None
    if np.any((z[levels == 0] < 0.) | (z[levels == 0] > 1.)):
        return False, None
    categories = z[levels > 0]
    if np.any((categories < 0) | (categories >= levels[levels > 0]) | (categories != np.floor(categories))):
        return False, None
    if (np.any(z[list(action.frozen)] != q[list(action.frozen)])
            or np.any(z[list(action.increasing)] < q[list(action.increasing)])
            or np.any(z[list(action.decreasing)] > q[list(action.decreasing)])):
        return False, None
    valid = model.predict(encode(z.reshape(1, -1), levels))[0] == 1
    return bool(valid), float(distance(q, z, levels, scales)) if valid else None


def make_ocean_mapper(levels):
    from ocean.abc import Mapper
    from ocean.feature import Feature

    columns, features = [], {}
    for j, level in enumerate(levels):
        name = f"f{j}"
        if level:
            codes = [f"{k:07d}" for k in range(level)]
            features[name] = Feature(Feature.Type.ONE_HOT_ENCODED, codes=codes)
            columns.extend((name, k) for k in codes)
        else:
            features[name] = Feature(Feature.Type.CONTINUOUS, levels=[0., 1.])
            columns.append((name, ""))
    return Mapper(features, columns=pd.MultiIndex.from_tuples(columns))


def decode_ocean(point, levels):
    start = np.r_[0, np.cumsum(np.maximum(1, levels))]
    decoded = np.empty(len(levels))
    for j, n in enumerate(levels):
        if n:
            block = point[start[j]:start[j + 1]]
            if not np.allclose(block, np.eye(1, int(n), int(block.argmax()))[0], rtol=0, atol=1e-6):
                return None
            decoded[j] = block.argmax()
        else:
            # The classifier operates on this numerical grid.
            decoded[j] = np.float32(point[start[j]])
    return decoded


def ocean_query(backend, q, action, levels, seed, max_time=3):
    constraints = {f"f{j}": rule
                   for fields, rule in [(action.frozen, "fixed"), (action.increasing, "increase"),
                                        (action.decreasing, "decrease")]
                   for j in fields}
    x = encode(q[None], levels)[0].astype(float)
    try:
        response = backend.explain(x, y=1, norm=1, num_workers=1, max_time=max_time, random_seed=seed,
                                   actionability=constraints)
        status = backend.get_solving_status()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    if response is not None:
        return decode_ocean(response.x, levels), status
    return None, status


class DiceAdapter:
    def __init__(self, model, levels):
        self.model = model
        self.levels = levels
        self.columns = [f"f{j}" for j in range(len(levels))]
        self.numeric = [f"f{j}" for j in np.flatnonzero(levels == 0)]

    def frame(self, points):
        return pd.DataFrame({
            tag: points[:, j] if not self.levels[j] else points[:, j].astype(int).astype(str)
            for j, tag in enumerate(self.columns)
        })

    def read(self, frame):
        return frame[self.columns].to_numpy(dtype=float)

    def predict_proba(self, points):
        return self.model.predict_proba(encode(self.read(points), self.levels))

    def predict(self, points):
        return self.model.predict(encode(self.read(points), self.levels))


def make_dice(model, train, y, levels):
    import dice_ml

    adapter = DiceAdapter(model, levels)
    frame = adapter.frame(train)
    frame["target"] = y
    data = dice_ml.Data(dataframe=frame, continuous_features=adapter.numeric, outcome_name="target",
                        continuous_features_precision={k: 7 for k in adapter.numeric})
    api = dice_ml.Model(model=adapter, backend="sklearn")
    return adapter, dice_ml.Dice(data, api, method="random")


def dice_query(prepared, q, action, levels, seed):
    adapter, explainer = prepared
    frame = adapter.frame(q[None])
    change = [f"f{j}" for j in range(len(levels)) if j not in action.frozen]
    ranges = {
        f"f{j}": [float(q[j]), 1.] if j in action.increasing else
        [0., float(q[j])] if j in action.decreasing else [0., 1.]
        for j in np.flatnonzero(levels == 0) if j not in action.frozen
    }
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            explanations = explainer.generate_counterfactuals(
                frame, total_CFs=10, desired_class=1, stopping_threshold=0.5,
                features_to_vary=change, permitted_range=ranges, sample_size=512,
                posthoc_sparsity_param=0., random_seed=seed, verbose=False)
        candidates = explanations.cf_examples_list[0].final_cfs_df
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    if candidates is None or not len(candidates):
        return None, "no candidate"
    points = adapter.read(candidates)
    points[:, levels == 0] = points[:, levels == 0].astype(np.float32)
    # DiCE displays rounded output; enforce frozen dimensions before independent validation.
    points[:, list(action.frozen)] = q[list(action.frozen)]
    clean = []
    for z in points:
        valid, cost = validate(adapter.model, q, z, levels, np.ones(len(levels)), action)
        if valid:
            clean.append((cost, z))
    return (min(clean, key=lambda entry: entry[0])[1], "returned") if clean else (None, "invalid")


def packages_for(args):
    if args.stage != "published":
        return None, None
    if args.ocean_path:
        sys.path.insert(0, str(Path(args.ocean_path).resolve()))
    import gurobipy as gp
    import ocean

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return ocean, env


def run_fold(data_root, stage, dataset, seed, packages):
    frame, y, meta = load_openml(data_root, dataset)
    train, test = train_test_split(np.arange(len(y)), test_size=.30, stratify=y, random_state=seed)
    x, levels, _, transform_info = transform(frame, train)
    norm_min, norm_span = bound_scale(x, train, levels)
    scales = np.ones(x.shape[1])
    enc = encode(x, levels)
    model, _ = create_model(enc[train], y[train], seed, "CART", 9 if stage == "geometry" else 6)
    rng = np.random.default_rng(seed)
    query_ids = take_queries(model.predict(enc), test, rng, 24 if stage == "geometry" else 8)
    idx, setup = tracked_setup(lambda: LeafIndex(model, levels, scales))
    region, region_setup = tracked_setup(lambda: RegionSearch(idx, bucket=8))
    methods = ["dense", "interned", "volumetric"]
    setup_rows = dict(dense=setup, interned=setup, volumetric={
        "setup_us": setup["setup_us"] + region_setup["setup_us"],
        "rss_delta": setup["rss_delta"] + region_setup["rss_delta"],
        "index_increment_bytes": region.structure_bytes(),
    })
    external = {}
    if stage == "published":
        ocean, env = packages
        for name, maker, opts in [
            ("ocean_mip", ocean.MixedIntegerProgramExplainer, {"env": env}),
            ("ocean_cp", ocean.ConstraintProgrammingExplainer, {}),
        ]:
            try:
                external[name], setup_rows[name] = tracked_setup(
                    lambda: maker([model], mapper=make_ocean_mapper(levels), **opts))
            except Exception as exc:
                external[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                setup_rows[name] = {"setup_us": None, "rss_delta": None}
        external["dice_random"], setup_rows["dice_random"] = tracked_setup(
            lambda: make_dice(model, x[train], y[train], levels))
        methods.extend(external)
    rows = []
    for mode, action in [("free", Action()), ("restricted", action_set(seed, levels))]:
        for row in query_ids:
            q = x[row]
            _, optimum, _ = idx.solve(q[None], action, "interned")
            for name in rng.permutation(methods):
                seen = visited = None
                if name == "volumetric":
                    (z, _, seen, visited), elapsed = warm_timing(lambda: region.solve(q, action))
                    status = "returned" if z is not None else "infeasible"
                elif name in {"dense", "interned"}:
                    (z, obj, _), elapsed = warm_timing(lambda: idx.solve(q[None], action, name))
                    z = z[0] if np.isfinite(obj[0]) else None
                    status = "returned" if z is not None else "infeasible"
                else:
                    backend = external[name]
                    if isinstance(backend, str):
                        z, status, elapsed = None, backend, None
                    elif name == "dice_random":
                        (z, status), elapsed = warm_timing(lambda: dice_query(backend, q, action, levels, seed), 1)
                    else:
                        (z, status), elapsed = warm_timing(lambda: ocean_query(backend, q, action, levels, seed), 1)
                ok, cost = validate(model, q, z, levels, scales, action)
                opt = float(optimum[0]) if np.isfinite(optimum[0]) else None
                gap = cost - opt if ok and opt is not None else None
                if name in {"dense", "interned", "volumetric"}:
                    assert ok == (opt is not None) and (not ok or abs(gap) <= 1e-9), (dataset, seed, row, name)
                rows.append({"dataset": dataset, "seed": seed, "query": int(row), "mode": mode,
                             "method": str(name), "valid": ok, "cost": cost, "gap": gap, "status": status,
                             "returned": z is not None, "reference": opt, "latency_us": elapsed,
                             "target_leaves": len(idx.nodes), "regions_scored": seen, "nodes_visited": visited,
                             "answer": z.tolist() if z is not None and row == query_ids[0] else None})
    for name in ("ocean_mip",):
        item = external.get(name)
        if item is not None and not isinstance(item, str):
            item.dispose()
    split = {"dataset": dataset, "seed": seed, "source": meta, "train": train.tolist(), "test": test.tolist(),
             "query_ids": query_ids.tolist(), "train_min": norm_min, "train_span": norm_span,
             "preprocessing": transform_info, "target_leaves": len(idx.nodes), "setup": setup_rows}
    return split, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("geometry", "published"))
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--ocean-path")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    run = args.out or ROOT / "result" / "runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + args.stage + ("_pilot" if args.pilot else ""))
    save_provenance(run, args, ("script/compare_search.py", "script/region_search.py",
                               "docs/experiment_plan_round2.md"))
    packages = packages_for(args)
    versions = ["psutil"]
    if args.stage == "published":
        versions.extend(["oceanpy", "gurobipy", "ortools", "protobuf", "dice-ml", "anytree", "python-sat"])
    (run / "artifacts" / "baseline_env.json").write_text(
        json.dumps({name: version(name) for name in versions}, indent=2), encoding="utf-8")
    data = GEOMETRY_DATA if args.stage == "geometry" else PUBLISHED_DATA
    seeds = CONFIG["primary_seeds"]
    if args.pilot:
        data, seeds = (31,), [7]
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        try:
            for dataset, seed in itertools.product(data, seeds):
                fold, rows = run_fold(args.data_root, args.stage, dataset, seed, packages)
                write_jsonl(run / "metrics.jsonl", rows)
                write_jsonl(run / "artifacts" / "splits.jsonl", [fold])
                progress = f"{dataset} seed={seed}, {len(rows)} measured rows"
                print(progress, flush=True)
                print(progress, file=log, flush=True)
        except Exception:
            traceback.print_exc(file=log)
            raise
    if packages[1] is not None:
        packages[1].dispose()


if __name__ == "__main__":
    main()
