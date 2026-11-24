"""Bounded mixed-feature benchmark and isolated-process memory measurement."""

import argparse
import contextlib
import io
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from compare_search import (
    bound_scale, dice_query, make_dice, make_ocean_mapper, ocean_query, validate, warm_timing,
)
from region_search import RegionSearch
from run_recourse import (
    CONFIG, ROOT, Action, LeafIndex, create_model, digest, encode, load_openml,
    save_provenance, train_test_split, transform, write_jsonl,
)

import numpy as np
import psutil

DATASETS = (29, 31, 38, 49, 51, 151, 1461, 1486, 1498, 1506, 1590, 40701)
DEPTHS = (6, 10, 14)
BENCHMARK = ("shared", "region", "highs_milp", "ocean_cp", "ocean_mip", "dice", "nice_sparse", "nice_proximity")
MEMORY = ("dense", "shared", "compiled_shared", "compiled_region", "ocean_cp", "ocean_mip")
MEMORY_DATA = (31, 1461, 1486, 1590)


def load_fold(root, dataset, seed):
    raw, y, source = load_openml(root, dataset)
    train, test = train_test_split(np.arange(len(y)), test_size=.30, stratify=y, random_state=seed)
    x, levels, _, preprocessing = transform(raw, train)
    minimum, span = bound_scale(x, train, levels)
    metadata = dict(dataset=dataset, seed=seed, train=train.tolist(), test=test.tolist(),
                    transformations=preprocessing, minimum=minimum, span=span, source=source)
    return x, y, levels, train, test, metadata


def actions_for(levels, seed):
    order = np.random.default_rng(seed).permutation(len(levels)).tolist()
    available = []
    for label, fraction in (("half", .50), ("three_quarters", .75)):
        fixed = tuple(order[:int(fraction * len(order))])
        numeric = [int(j) for j in np.flatnonzero(levels == 0) if j not in fixed]
        available.append((label, Action(fixed, tuple(numeric[:1]), tuple(numeric[1:2]))))
    return [("free", Action()), *available]


def nice_solver(model, train, levels, mode):
    from nice import NICE

    tags = np.flatnonzero(levels > 0).tolist()
    app = NICE(lambda values: model.predict_proba(encode(values, levels)),
               train.copy(), tags, justified_cf=False,
               optimization="proximity" if mode == "nice_proximity" else "sparsity")
    positives = model.predict(encode(train, levels)) == 1

    def query(q, action):
        fixed, up, down = map(list, (action.frozen, action.increasing, action.decreasing))
        admissible = ((train[:, fixed] == q[fixed]).all(axis=1)
                      & (train[:, up] >= q[up]).all(axis=1) & (train[:, down] <= q[down]).all(axis=1))
        if not (positives & admissible).any():
            return None, "no_admissible_prototype"
        app.data.candidates_mask = admissible
        z = np.asarray(app.explain(q[None].copy(), target_class=[1]), dtype=float).reshape(-1)
        return z, "returned"

    return query


def build_solver(name, model, x, y, levels, train, seed, environment=None):
    scales = np.ones(len(levels))
    if name in {"shared", "dense", "compiled_shared", "compiled_region", "region"}:
        idx = LeafIndex(model, levels, scales)
        if name == "region":
            region = RegionSearch(idx)

            def region_query(q, action):
                z, _, _, _ = region.solve(q, action)
                return z, "returned" if z is not None else "infeasible"

            return region_query, None
        if name == "compiled_shared":
            from native_kernel import kernel
            idx.stages(.25)
            solve = lambda values, action: kernel(idx, values, action, False)
        elif name == "compiled_region":
            from compiled_geometry import CompiledRegion
            index = CompiledRegion(idx, x[train])
            solve = index.solve
        else:
            solve = lambda values, action: idx.solve(values, action, "dense" if name == "dense" else "interned")

        def shared_query(q, action):
            z, cost, _ = solve(q[None], action)
            return (z[0], "returned") if np.isfinite(cost[0]) else (None, "infeasible")

        return shared_query, solve
    if name in ("ocean_cp", "ocean_mip"):
        import ocean
        if name == "ocean_mip":
            maker, opts = ocean.MixedIntegerProgramExplainer, {"env": environment}
        else:
            maker, opts = ocean.ConstraintProgrammingExplainer, {}
        model_ocean = maker([model], mapper=make_ocean_mapper(levels), **opts)
        return lambda q, a: ocean_query(model_ocean, q, a, levels, seed), None
    if name == "highs_milp":
        from flow_milp import NodeFlow
        graph = NodeFlow.from_tree(model, levels)
        return lambda q, action: graph.solve(q, action, scales), None
    if name == "dice":
        backend = make_dice(model, x[train], y[train], levels)
        return lambda q, a: dice_query(backend, q, a, levels, seed), None
    return nice_solver(model, x[train], levels, name), None


def gurobi_env():
    import gurobipy as gp
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def result_kind(status, valid):
    if valid:
        return "valid"
    folded = status.casefold()
    if "time_limit" in folded or "unknown" == folded:
        return "time_limit_or_unresolved"
    if "size-limited" in folded or "model too large" in folded or "license" in folded:
        return "license_limit"
    if folded == "infeasible":
        return "reported_infeasible"
    if "no counterfactuals" in folded or folded.startswith("no_"):
        return "no_candidate"
    if status in ("OPTIMAL", "returned", "invalid"):
        return "unvalidated_return"
    return "other"


def exact_reference(idx, q, action):
    _, costs, _ = idx.solve(q[None], action, "interned")
    return float(costs[0]) if np.isfinite(costs[0]) else None


def evaluate_split(root, dataset, seed, depths, query_limit, environment):
    x, y, levels, train, test, metadata = load_fold(root, dataset, seed)
    scales = np.ones(len(levels))
    enc = encode(x, levels)
    models = {depth: create_model(enc[train], y[train], seed, "CART", depth)[0] for depth in depths}
    rng = np.random.default_rng(seed)
    eligible = test[np.logical_and.reduce([m.predict(enc[test]) == 0 for m in models.values()])]
    queries = np.sort(rng.choice(eligible, min(query_limit, len(eligible)), replace=False))
    policies = actions_for(levels, seed)
    metadata["queries"] = queries.tolist()
    metadata["eligible"] = int(len(eligible))
    metadata["policies"] = {k: dict(frozen=v.frozen, increasing=v.increasing, decreasing=v.decreasing)
                           for k, v in policies}
    rows, setups = [], []
    for depth, model in models.items():
        reference = LeafIndex(model, levels, scales)
        setup = {}
        for method in rng.permutation(BENCHMARK):
            start = time.perf_counter_ns()
            try:
                query, _ = build_solver(method, model, x, y, levels, train, seed, environment)
                failure = None
            except Exception as exc:
                query, failure = None, f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            setup[str(method)] = ((time.perf_counter_ns() - start) / 1000, query, failure)
        for mode, action in policies:
            for i in queries:
                q = x[i]
                target = exact_reference(reference, q, action)
                for method in rng.permutation(BENCHMARK):
                    setup_us, query, failure = setup[method]
                    start = time.perf_counter_ns()
                    z, status = None, failure
                    if query is not None:
                        try:
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                z, status = query(q, action)
                        except Exception as exc:
                            status = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    latency = (time.perf_counter_ns() - start) / 1000 if query is not None else None
                    valid, cost = validate(model, q, z, levels, scales, action)
                    gap = float(cost - target) if valid and target is not None else None
                    if method in ("shared", "region"):
                        assert valid == (target is not None)
                        assert gap is None or abs(gap) <= 1e-9
                    elif valid:
                        assert target is not None and gap >= -1e-9
                    rows.append({
                        "dataset": dataset, "seed": seed, "depth": depth, "actual_depth": model.get_depth(),
                        "query": int(i), "mode": mode, "method": str(method), "valid": valid,
                        "status": status, "kind": result_kind(status, valid), "latency_us": latency,
                        "setup_us": setup_us, "cost": cost, "reference": target, "gap": gap,
                        "leaves": int(len(reference.nodes)), "width": int(enc.shape[1]),
                    })
        setups.append({"depth": depth, "nodes": int(model.tree_.node_count), "target_leaves": len(reference.nodes),
                       "methods": {k: dict(setup_us=v[0], failure=v[2]) for k, v in setup.items()}})
    metadata["setups"] = setups
    return metadata, rows


def memory_snapshot():
    memory = psutil.Process().memory_info()
    if not hasattr(memory, "peak_wset"):
        raise RuntimeError("This benchmark reads Windows working-set high-water counters.")
    return {"rss": int(memory.rss), "peak": int(memory.peak_wset)}


def memory_worker(args):
    x, y, levels, train, test, _ = load_fold(args.data_root, args.dataset, args.seed)
    enc = encode(x, levels)
    model, _ = create_model(enc[train], y[train], args.seed, "CART", args.depth)
    rng = np.random.default_rng(args.seed)
    eligible = test[model.predict(enc[test]) == 0]
    count = min(128, len(eligible))
    if not count:
        raise ValueError("No negative test predictions for the memory workload")
    selected = np.sort(rng.choice(eligible, count, replace=False))
    sizes = (1, 32, 128)
    before = memory_snapshot()
    start = time.perf_counter_ns()
    env = gurobi_env() if args.method == "ocean_mip" else None
    query, batch = build_solver(args.method, model, x, y, levels, train, args.seed, env)
    setup_us = (time.perf_counter_ns() - start) / 1000
    setup = memory_snapshot()
    supplied = actions_for(levels, args.seed)[1][1]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        (_, cold_us) = warm_timing(lambda: query(x[selected[0]], supplied), 1)
    cold = memory_snapshot()
    rows = []
    for nominal in sizes:
        ids = selected[:nominal]
        start = time.perf_counter_ns()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if batch is None:
                outcomes = [query(x[j], supplied) for j in ids]
                returned = sum(z is not None for z, _ in outcomes)
                states = [state for _, state in outcomes]
            else:
                _, cost, _ = batch(x[ids], supplied)
                returned = int(np.isfinite(cost).sum())
                states = ["returned"] * returned + ["infeasible"] * (len(ids) - returned)
        online_us = (time.perf_counter_ns() - start) / 1000
        now = memory_snapshot()
        rows.append(dict(dataset=args.dataset, seed=args.seed, depth=args.depth, actual_depth=model.get_depth(),
                         method=args.method, nominal=nominal, queries=len(ids), returned=returned,
                         pre_rss=before["rss"], pre_peak=before["peak"], setup_rss=setup["rss"],
                         setup_peak=setup["peak"], cold_peak=cold["peak"], end_rss=now["rss"],
                         end_peak=now["peak"], setup_us=setup_us, cold_us=cold_us, batch_us=online_us,
                         statuses={state: states.count(state) for state in sorted(set(states))}))
    print("RESULT " + json.dumps(rows, allow_nan=False))


def independent_memory(args, run, log):
    command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()), "memory-worker",
               "--data-root", str(args.data_root)]
    if args.package_path:
        command.extend(("--package-path", args.package_path))
    jobs = itertools.product(MEMORY_DATA, (7, 29), (6, 14), MEMORY)
    if args.pilot:
        jobs = itertools.product((31,), (7,), (6,), ("shared", "ocean_mip"))
    for did, seed, depth, method in jobs:
        command_job = [*command, "--dataset", str(did), "--seed", str(seed),
                       "--depth", str(depth), "--method", method]
        start = time.perf_counter()
        fields = dict(dataset=did, seed=seed, depth=depth, method=method)
        try:
            output = subprocess.run(command_job, capture_output=True, encoding="utf-8", errors="replace",
                                    check=False, timeout=180)
            result = next((line[7:] for line in output.stdout.splitlines() if line.startswith("RESULT ")), None)
            if output.returncode or result is None:
                entry = {**fields, "status": "error", "detail": output.stderr[-1500:]}
            else:
                entry = {**fields, "status": "ok"}
                write_jsonl(run / "metrics.jsonl", json.loads(result))
        except subprocess.TimeoutExpired:
            entry = {**fields, "status": "timeout"}
        elapsed = time.perf_counter() - start
        write_jsonl(run / "artifacts" / "worker_commands.jsonl",
                    [{**entry, "command": command_job, "worker_seconds": elapsed}])
        message = f"{did} seed={seed} depth={depth} {method}: {entry['status']} ({elapsed:.1f}s)"
        print(message, flush=True)
        print(message, file=log, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["bench", "memory", "memory-worker"])
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--package-path", help="Path to independently unpacked release wheels.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--dataset", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--method")
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    if args.package_path:
        args.package_path = str(Path(args.package_path).resolve())
        sys.path.insert(0, args.package_path)
    if args.stage == "memory-worker":
        memory_worker(args)
        return
    slug = {"bench": "published_deep", "memory": "isolated_memory"}[args.stage]
    if args.pilot:
        slug += "_pilot"
    run = args.out or ROOT / "result" / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + slug)
    files = ("script/deep_benchmark.py", "script/compare_search.py", "script/region_search.py",
             "script/flow_milp.py",
             "script/compiled_geometry.py", "script/native_kernel.py", "docs/experiment_plan_round3.md")
    save_provenance(run, args, files)
    package_versions = ("oceanpy", "gurobipy", "ortools", "protobuf", "NICEx", "dice-ml", "numba", "psutil")
    (run / "artifacts" / "software.json").write_text(json.dumps(
        {p: version(p) for p in package_versions}, indent=2), encoding="utf-8")
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        if args.stage == "memory":
            independent_memory(args, run, log)
            return
        env = gurobi_env()
        datasets = (args.dataset,) if args.dataset is not None else (31, 1486) if args.pilot else DATASETS
        seeds = (args.seed,) if args.seed is not None else (7,) if args.pilot else CONFIG["primary_seeds"]
        jobs = itertools.product(datasets, seeds)
        for did, seed in jobs:
            fold, rows = evaluate_split(args.data_root, did, seed, DEPTHS, 3 if args.pilot else 8, env)
            write_jsonl(run / "artifacts" / "splits.jsonl", [fold])
            write_jsonl(run / "metrics.jsonl", rows)
            message = f"{did} seed={seed}, {len(rows)} method observations"
            print(message, flush=True)
            print(message, file=log, flush=True)
        env.dispose()


if __name__ == "__main__":
    main()
