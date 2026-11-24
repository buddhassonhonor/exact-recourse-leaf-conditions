"""Hold the numerical backend constant while testing the screening mechanism."""

import argparse
import itertools
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from compare_search import action_set, warm_timing
from run_recourse import CONFIG, ROOT, Action, LeafIndex, create_model, encode, take_queries
from run_recourse import load_openml, transform, train_test_split, save_provenance, write_jsonl
from native_kernel import kernel
from compiled_geometry import CompiledRegion

import numpy as np

TASKS = (3, 31, 44, 151, 1461, 1486, 1590, 42477)


def blocks(data_root, did, seed):
    if did == "synthetic":
        rng = np.random.default_rng(seed)
        matrix = rng.uniform(-1, 1, (6000, 64)).astype(np.float32).astype(float)
        y = (np.sin(9 * matrix[:, 0]) + np.sin(13 * matrix[:, 1]) + .4 * matrix[:, 2]
             + rng.normal(0., .65, len(matrix)) > 0).astype(int)
        train, test = np.arange(4000), np.arange(4000, 6000)
        for depth in (6, 9, 12, 16):
            yield matrix, y, np.zeros(64, dtype=int), np.ones(64), train, test, depth, None
    else:
        frame, y, meta = load_openml(data_root, did)
        train, test = train_test_split(np.arange(len(y)), test_size=.30, stratify=y, random_state=seed)
        x, levels, scales, _ = transform(frame, train)
        yield x, y, levels, scales, train, test, 9, meta


def measure(fit, did, seed, with_geometry):
    x, y, levels, scales, train, test, depth, meta = fit
    rng = np.random.default_rng(seed)
    enc = encode(x, levels)
    tree, _ = create_model(enc[train], y[train], seed, "CART", depth)
    idx = LeafIndex(tree, levels, scales)
    idx.stages(.25)
    geometry = CompiledRegion(idx, x[train]) if with_geometry else None
    queries = take_queries(tree.predict(enc), test, rng, 32)
    records = []
    for mode, action in [("free", Action()), ("restricted", action_set(seed, levels))]:
        subset = x[queries]
        methods = {
            "scipy": lambda q: idx.solve(q, action, "interned"),
            "compiled_enum": lambda q: kernel(idx, q, action, False),
            "compiled_screen": lambda q: kernel(idx, q, action, True),
        }
        if geometry is not None:
            methods["compiled_region"] = lambda q: geometry.solve(q, action)
        _, reference, _ = methods["scipy"](subset)
        for name in rng.permutation(list(methods)):
            solve = methods[name]
            z, obj, counts = solve(subset)
            assert np.allclose(obj, reference, rtol=0, atol=1e-9)
            good = np.isfinite(obj)
            if good.any():
                assert np.all(tree.predict(encode(z[good], levels)) == 1)
                assert np.all(z[good][:, action.frozen] == subset[good][:, action.frozen])
                assert np.all(z[good][:, action.increasing] >= subset[good][:, action.increasing])
                assert np.all(z[good][:, action.decreasing] <= subset[good][:, action.decreasing])
            _, batch_us = warm_timing(lambda: solve(subset), 12)
            for i in rng.permutation(len(subset)):
                _, us = warm_timing(lambda: solve(subset[i:i + 1]), 9)
                records.append({"dataset": did, "seed": seed, "depth": depth, "method": str(name),
                                "mode": mode, "query": int(queries[i]), "latency_us": us,
                                "batch_us": float(batch_us / len(subset)),
                                "leaf_terms": int(idx.matrix.nnz),
                                "summations": int(counts[i]) if name.startswith("compiled_") and name != "compiled_region"
                                else int(idx.matrix.nnz) if name == "scipy" else None,
                                "scored_regions": int(counts[i]) if name == "compiled_region" else None,
                                "target_leaves": len(idx.nodes), "feasible": bool(good[i])})
    return records, {"dataset": did, "seed": seed, "depth": depth, "dataset_source": meta,
                     "train": train.tolist(), "test": test.tolist(), "queries": queries.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--geometry", action="store_true", help="Include the compiled volumetric control.")
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    stage = "native_geometry" if args.geometry else "native_kernel"
    run = args.out or ROOT / "result" / "runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + stage)
    save_provenance(run, args, ("script/native_experiment.py", "script/native_kernel.py",
                               "script/compare_search.py", "script/region_search.py",
                               "script/compiled_geometry.py", "docs/experiment_plan_round2.md"))
    sample, _ = create_model(np.array([[0.], [0.], [1.], [1.]]), [0, 0, 1, 1], 7, "CART", 2)
    start = LeafIndex(sample, np.array([0]), np.ones(1))
    _, compilation_us = warm_timing(lambda: kernel(start, np.zeros((1, 1)), Action(), False), 1)
    metadata = {"numba": version("numba"), "cold_us": compilation_us}
    if args.geometry:
        region = CompiledRegion(start, np.array([[0.], [1.]]))
        _, region_cold = warm_timing(lambda: region.solve(np.zeros((1, 1)), Action()), 1)
        metadata["region_cold_us"] = region_cold
    (run / "artifacts" / "compilation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        for did, seed in itertools.product([*TASKS, "synthetic"], CONFIG["primary_seeds"]):
            for fit in blocks(args.data_root, did, seed):
                rows, split = measure(fit, did, seed, args.geometry)
                write_jsonl(run / "metrics.jsonl", rows)
                write_jsonl(run / "artifacts" / "splits.jsonl", [split])
                msg = f"{did} seed={seed} depth={fit[6]} rows={len(rows)}"
                print(msg, flush=True)
                print(msg, file=log, flush=True)


if __name__ == "__main__":
    main()
