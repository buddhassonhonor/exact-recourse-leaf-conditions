"""Reproducible experiments for mixed-domain decision-tree counterfactuals."""

import argparse
import hashlib
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from recourse import Action, LeafIndex, distance, encode

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "script" / "benchmarks.json").read_text())
MODELS = {"CART": DecisionTreeClassifier, "ExtraTree": ExtraTreeClassifier}
METHODS = ("dense", "interned", "cascade", "random_prefix", "heap", "nearest", "medoid",
           "no_change")
PRIMARY = ("dense", "interned", "cascade")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata(path):
    return {x.tag.split("}")[-1]: x.text for x in ET.parse(path).getroot()}


def load_openml(data_root, did):
    filename = f"dataset_{did}.pq"
    for name in ("openml", "openml_cache"):
        base = data_root / name / "org" / "openml" / "www" / "datasets" / str(did)
        path = base / filename
        if path.is_file():
            break
    else:
        raise FileNotFoundError(f"OpenML {did} is missing from {data_root}")
    meta = metadata(base / "description.xml")
    frame = pd.read_parquet(path)
    target = meta["default_target_attribute"]
    if target not in frame:
        raise KeyError(f"Missing target {target} on OpenML {did}")
    exclude = {target, *str(meta.get("ignore_attribute") or "").split(","),
               *str(meta.get("row_id_attribute") or "").split(",")}
    x = frame.drop(columns=list(exclude), errors="ignore")
    y = frame[target]
    keep = y.notna()
    codes, labels = pd.factorize(y[keep].astype(str), sort=True)
    if len(labels) != 2:
        raise ValueError(f"OpenML {did} has {len(labels)} target values in the source file")
    return x.loc[keep].reset_index(drop=True), codes, {
        "dataset": did, "name": meta["name"], "target": target, "version": meta["version"],
        "labels": labels.tolist(), "count_0": int((codes == 0).sum()),
        "count_1": int((codes == 1).sum()), "source": f"https://www.openml.org/d/{did}",
        "cached_file": str(path.relative_to(data_root)), "sha256": digest(path),
        "licence": meta.get("licence"), "missing_cells": int(x.loc[keep].isna().sum().sum()),
        "rows": len(codes), "features": x.shape[1], "target_missing": int((~keep).sum()),
        "columns": x.columns.tolist(), "excluded_present": sorted(exclude.intersection(frame) - {target}),
    }


def transform(frame, train):
    data, levels, scales, descriptions = [], [], [], []
    for name in frame:
        field = frame[name]
        fit = field.iloc[train]
        numeric = pd.api.types.is_numeric_dtype(field) and not pd.api.types.is_bool_dtype(field)
        if numeric:
            values = field.to_numpy(dtype=float, na_value=np.nan)
            observed = values[train][np.isfinite(values[train])]
            median = float(np.median(observed)) if len(observed) else 0.
            values[~np.isfinite(values)] = median
            q1, q3 = np.percentile(values[train], [25, 75])
            spread = q3 - q1 or np.ptp(values[train]) or 1.
            data.append(values.astype(np.float32))
            levels.append(0)
            scales.append(float(spread))
            descriptions.append({"name": name, "median": median, "scale": float(spread)})
        else:
            # Prefix observed categories so sentinels cannot collide with actual text.
            values = ("v:" + field.astype("string")).fillna("missing")
            observed = sorted(values.iloc[train].unique())
            categories = observed + ["unseen"]
            codes = pd.Categorical(values.where(values.isin(observed), "unseen"),
                                   categories=categories).codes
            data.append(codes)
            levels.append(len(categories))
            scales.append(1.)
            descriptions.append({"name": name, "levels": categories})
    return np.column_stack(data).astype(float), np.asarray(levels), np.asarray(scales), descriptions


def allowed_candidates(q, pos, action):
    if not len(pos):
        return np.zeros(0, dtype=bool)
    return ((pos[:, action.frozen] == q[list(action.frozen)]).all(axis=1)
            & (pos[:, action.increasing] >= q[list(action.increasing)]).all(axis=1)
            & (pos[:, action.decreasing] <= q[list(action.decreasing)]).all(axis=1))


def historical_baseline(x, pos, index, action, method):
    z = np.full_like(x, np.nan)
    costs = np.full(len(x), np.inf)
    for i, q in enumerate(x):
        pool = pos[allowed_candidates(q, pos, action)]
        if len(pool) and method != "no_change":
            ds = distance(q, pool, index.levels, index.scales, action.categorical_cost)
            best = ds.argmin()
            z[i], costs[i] = pool[best], ds[best]
    return z, costs, np.full(len(x), len(pos))


def take_queries(pred, test, rng, count):
    eligible = test[pred[test] == 0]
    return np.sort(rng.choice(eligible, min(count, len(eligible)), replace=False))


def evaluate(index, queries, ref, reference_ids, seed, action, methods, fractions, repeat=2):
    pred = index.model.predict(encode(ref, index.levels))
    positives = ref[pred == 1]
    ids = index.model.apply(encode(positives, index.levels)) if len(positives) else np.array([])
    rep = []
    for leaf in np.unique(ids):
        part = positives[ids == leaf]
        median = np.median(part, axis=0)
        for j in index.categorical:
            median[j] = np.bincount(part[:, j].astype(int)).argmax()
        rep.append(part[distance(median, part, index.levels, index.scales).argmin()])
    medoids = np.array(rep).reshape(-1, index.n_features)
    actions = [(m, CONFIG["prefix_fraction"]) for m in methods]
    actions.extend(("cascade", value) for value in fractions if value != CONFIG["prefix_fraction"])
    schedule = np.random.default_rng(seed).permutation(len(actions))
    outputs = {}
    for m, f in actions:
        if m in {"cascade", "random_prefix"}:
            index.stages(f, seed if m == "random_prefix" else None)
    for i in schedule:
        method, fraction = actions[i]
        if method in {"nearest", "medoid", "no_change"}:
            pool = medoids if method == "medoid" else positives
            solve = lambda q: historical_baseline(q, pool, index, action, method)
        else:
            solve = lambda q: index.solve(q, action, method, fraction, seed)
        solve(queries[:1])
        if method in PRIMARY or method == "random_prefix":
            batch_times = []
            for _ in range(repeat):
                start = time.perf_counter_ns()
                solve(queries)
                batch_times.append((time.perf_counter_ns() - start) / len(queries) / 1000)
            batch_us = float(np.median(batch_times))
        else:
            batch_us = None
        candidate_rows = []
        for x in queries:
            times = []
            for _ in range(repeat):
                start = time.perf_counter_ns()
                z, c, work = solve(x[None])
                times.append((time.perf_counter_ns() - start) / 1000.)
            candidate_rows.append((z[0], c[0], work[0], np.median(times)))
        outputs[method, fraction] = (candidate_rows, batch_us)

    reference = outputs["dense", CONFIG["prefix_fraction"]][0]
    rows = []
    for (method, fraction), (results, batch_us) in outputs.items():
        valid_indices = [i for i, r in enumerate(results) if np.isfinite(r[1])]
        support = np.full(len(queries), np.nan)
        if valid_indices:
            answers = np.array([results[i][0] for i in valid_indices])
            assert np.all(index.model.predict(encode(answers, index.levels)) == 1)
            factual = queries[valid_indices]
            assert np.all(answers[:, action.frozen] == factual[:, action.frozen])
            assert np.all(answers[:, action.increasing] >= factual[:, action.increasing])
            assert np.all(answers[:, action.decreasing] <= factual[:, action.decreasing])
            recomputed = distance(factual, answers, index.levels, index.scales, action.categorical_cost)
            assert np.allclose(recomputed, [results[i][1] for i in valid_indices], rtol=0, atol=1e-9)
            if method != "no_change":
                for i, answer in zip(valid_indices, answers):
                    support[i] = float(distance(answer, ref[reference_ids], index.levels, index.scales).min())
        for i, ((z, cost, work, elapsed), (_, optimum, _, _)) in enumerate(zip(results, reference)):
            hit = np.isfinite(cost)
            gap = float(cost - optimum) if hit else None
            if method in {"dense", "interned", "cascade", "heap", "random_prefix"}:
                assert hit == np.isfinite(optimum), (method, i, cost, optimum)
                assert not hit or abs(gap) <= 1e-9, (method, i, cost, optimum)
            rows.append({
                "query_pos": i, "method": method, "fraction": fraction, "valid": bool(hit),
                "cost": float(cost) if hit else None, "gap": gap,
                "changed": int(np.count_nonzero(z != queries[i])) if hit else None,
                "support_distance": float(support[i]) if hit else None,
                "work": int(work), "latency_us": float(elapsed), "batch_us": batch_us,
                "answer": z.tolist() if hit and i < 3 else None,
            })
    return rows


def create_model(x, y, seed, model_name, depth):
    start = time.perf_counter()
    model = MODELS[model_name](max_depth=depth, min_samples_leaf=2, random_state=seed).fit(x, y)
    return model, (time.perf_counter() - start) * 1000


def run_task(job):
    task, data_root, did, seed = job
    if task == "scale":
        return scale(seed)
    frame, y, meta = load_openml(data_root, did)
    train, test = train_test_split(np.arange(len(y)), stratify=y, random_state=seed,
                                  test_size=CONFIG["test_fraction"])
    x, levels, scales, transform_info = transform(frame, train)
    meta.update(n_numeric=int((levels == 0).sum()), n_categorical=int((levels > 0).sum()))
    x_encoded = encode(x, levels)
    rng = np.random.default_rng(seed)
    support_ids = rng.choice(len(train), min(len(train), 256), replace=False)
    rows, splits = [], []
    for model_name in MODELS if task == "main" else ("CART",):
        model, train_ms = create_model(x_encoded[train], y[train], seed, model_name,
                                       CONFIG["primary_depth"])
        predictions = model.predict(x_encoded)
        query_ids = take_queries(predictions, test, rng, CONFIG["query_limit"])
        fold_acc = float((predictions[test] == y[test]).mean())
        fold = {"seed": seed, "dataset": did, "model": model_name, "accuracy": fold_acc,
                "training_ms": train_ms, "nodes": model.tree_.node_count, "depth": model.get_depth(),
                "encoded_features": x_encoded.shape[1], "features": x.shape[1],
                "train_size": len(train), "test_size": len(test), "n_queries": len(query_ids)}
        if not len(query_ids):
            splits.append({**fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                           "query_indices": [], "preprocessing": transform_info})
            continue
        start = time.perf_counter()
        index = LeafIndex(model, levels, scales)
        build_ms = (time.perf_counter() - start) * 1000
        fold.update(target_leaves=len(index.nodes), condition_terms=index.matrix.nnz,
                    unique_conditions=len(index.features), index_ms=build_ms)
        splits.append({**fold, "train_indices": train.tolist(), "test_indices": test.tolist(),
                       "query_indices": query_ids.tolist(), "preprocessing": transform_info,
                       "factual": x[query_ids[:3]].tolist()})
        settings = [(0., Action())]
        fractions = []
        eval_methods = METHODS
        if task == "stress":
            directions = np.flatnonzero(levels == 0)[:2]
            ranked = rng.permutation(x.shape[1])
            settings = [(f, Action(tuple(ranked[:int(f * x.shape[1])]), tuple(directions[:1]),
                                   tuple(directions[1:2]), 1.))
                        for f in (0., .25, .5, .75)]
            settings.extend([(0., Action(categorical_cost=c)) for c in (.25, .5, 2., 4.)])
            fractions = [.125, .5, .75]
            eval_methods = PRIMARY + ("random_prefix", "nearest")
        for freeze_fraction, action in settings:
            measures = evaluate(index, x[query_ids], x[train], support_ids, seed, action,
                                eval_methods, fractions)
            for row in measures:
                rows.append({**fold, **row, "query": int(query_ids[row["query_pos"]]),
                             "freeze": freeze_fraction, "categorical_cost": action.categorical_cost,
                             "directional": bool(action.increasing or action.decreasing)})
    return meta, splits, rows


def scale(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    data = rng.uniform(-1, 1, (n, 256)).astype(np.float32).astype(float)
    y = (np.sin(7 * data[:, 0]) + np.sin(9 * data[:, 1]) + .5 * data[:, 2]
         + rng.normal(0., .65, n) > 0).astype(int)
    train, test = np.arange(3000), np.arange(3000, 4000)
    rows = []
    for width, depth in itertools.product((8, 32, 128, 256), (3, 6, 9, 12)):
        x = data[:, :width]
        levels, scales = np.zeros(width, int), np.ones(width)
        model, train_ms = create_model(x[train], y[train], seed, "CART", depth)
        queries = take_queries(model.predict(x), test, rng, 24)
        start = time.perf_counter()
        index = LeafIndex(model, levels, scales)
        build_ms = (time.perf_counter() - start) * 1000
        measures = evaluate(index, x[queries], x[train], train[:256], seed, Action(),
                            PRIMARY + ("random_prefix",), [.125, .5, .75], repeat=3)
        for row in measures:
            rows.append({**row, "dataset": "synthetic", "features": width, "depth": depth,
                         "seed": seed, "model": "CART", "target_leaves": len(index.nodes),
                         "condition_terms": index.matrix.nnz, "unique_conditions": len(index.features),
                         "training_ms": train_ms, "index_ms": build_ms,
                         "query": int(queries[row["query_pos"]])})
    return {"dataset": "synthetic"}, [], rows


def exhaustive():
    rng = np.random.default_rng(2209)
    cube = np.array(list(itertools.product(range(3), range(3), range(2))), dtype=float)
    # The exhaustive oracle treats every field as discrete, including binary coordinates.
    levels = np.array([3, 3, 2])
    enc = encode(cube, levels)
    checked = []
    for seed in range(32):
        y = rng.integers(0, 2, len(cube))
        model, _ = create_model(enc, y, seed, "CART", 5)
        index = LeafIndex(model, levels, np.ones(3))
        target = cube[model.predict(enc) == 1]
        factual = cube[model.predict(enc) == 0]
        if not len(factual):
            continue
        for frozen in [(), (0,), (0, 1), (0, 1, 2)]:
            a = Action(frozen)
            oracle = np.asarray([distance(q, target[allowed_candidates(q, target, a)], levels,
                                          np.ones(3)).min(initial=np.inf) for q in factual])
            for method in ("dense", "interned", "cascade", "heap", "random_prefix"):
                z, got, _ = index.solve(factual, a, method)
                assert np.array_equal(got, oracle), (seed, method, frozen, got, oracle)
                success = np.isfinite(got)
                if success.any():
                    assert np.all(model.predict(encode(z[success], levels)) == 1)
                    assert np.all(z[success][:, frozen] == factual[success][:, frozen])
                checked.append({"seed": seed, "frozen": list(frozen), "method": method,
                                "cases": len(factual), "feasible": int(success.sum())})
    for value in (0., .1, 1., -1., 1e15, -1e-15):
        lower, upper = threshold_on_grid_for_test(value)
        assert lower <= value < upper
        assert np.nextafter(np.float32(lower), np.float32(np.inf)) == upper
    return {"dataset": "exhaustive_grid"}, [], checked


def threshold_on_grid_for_test(value):
    from recourse import threshold_on_grid
    return threshold_on_grid(value, False), threshold_on_grid(value, True)


def command_or_none(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_provenance(run, args):
    run.mkdir(parents=True, exist_ok=False)
    (run / "artifacts").mkdir()
    (run / "plots").mkdir()
    config = {**CONFIG, **vars(args), "data_root": str(args.data_root), "out": str(args.out)}
    (run / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    shell_args = ["python", *sys.argv]
    actual_args = [sys.executable, *sys.argv]
    (run / "commands.sh").write_text("#!/usr/bin/env bash\n" + shlex.join(shell_args) + "\n")
    (run / "commands.ps1").write_text(
        "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in actual_args) + "\n",
        encoding="utf-8")
    code = ["script/recourse.py", "script/run_recourse.py", "script/benchmarks.json"]
    (run / "diff.patch").write_text("\n".join(
        subprocess.run(["git", "diff", "--no-index", "--", "/dev/null", path],
                       capture_output=True, text=True).stdout for path in code),
        encoding="utf-8")
    import shutil
    for path in code:
        shutil.copyfile(path, run / "artifacts" / Path(path).name)
    env = {"python": sys.version, "command": actual_args, "platform": platform.platform(),
           "processor": platform.processor(), "workers": args.workers,
           "CPU": command_or_none(["powershell", "-NoProfile", "-Command",
                                   "(Get-CimInstance Win32_Processor).Name"]),
           "RAM_bytes": command_or_none(["powershell", "-NoProfile", "-Command",
                                         "(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum"]),
           "GPU": command_or_none(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
           "gpu_used": False, "BLAS_threads": 1, "git": command_or_none(["git", "rev-parse", "HEAD"]),
           "libraries": {m: version(m) for m in ["numpy", "scipy", "scikit-learn", "pandas", "pyarrow"]},
           "sha256": {p: digest(Path(p)) for p in code}}
    (run / "artifacts" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["exhaustive", "pilot", "main", "stress", "scale"])
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    task = args.stage
    run = args.out or ROOT / "result" / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + task)
    save_provenance(run, args)
    metrics = run / "metrics.jsonl"
    manifest = {}
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        def record(meta, splits, rows):
            manifest[str(meta["dataset"])] = meta
            write_jsonl(metrics, rows)
            write_jsonl(run / "artifacts" / "splits.jsonl", splits)
            message = f"{meta['dataset']} seed={rows[0]['seed'] if rows else '-'} rows={len(rows)}"
            print(message, flush=True)
            print(message, file=log, flush=True)

        try:
            if task == "exhaustive":
                record(*exhaustive())
            else:
                datasets = CONFIG["openml_ids"]
                seeds = CONFIG["primary_seeds"]
                if task == "pilot":
                    task, datasets, seeds = "main", [31, 44, 151, 1486], [7, 13, 19]
                elif task == "stress":
                    datasets = [31, 44, 151, 1461, 1486, 1590, 42477, 45026]
                elif task == "scale":
                    datasets = [None]
                jobs = list(itertools.product([task], [args.data_root], datasets, seeds))
                if args.workers == 1:
                    for result in map(run_task, jobs):
                        record(*result)
                else:
                    with ProcessPoolExecutor(max_workers=args.workers) as executor:
                        for result in executor.map(run_task, jobs):
                            record(*result)
        except Exception:
            import traceback
            traceback.print_exc(file=log)
            raise
    (run / "artifacts" / "datasets.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
