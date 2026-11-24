"""Replay material solver discrepancies against explicitly feasible witnesses."""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from deep_benchmark import actions_for, load_fold
from flow_milp import NodeFlow
from run_recourse import ROOT, LeafIndex, create_model, distance, encode, save_provenance, write_jsonl

import numpy as np
import pandas as pd


def vector_in_model(program, tree, explanation, levels):
    witness = np.zeros(len(program.objective))
    witness[tree.decision_path(encode(explanation[None], levels)).indices] = 1
    for k, items in enumerate(program.options):
        hit = np.flatnonzero(items == explanation[k])
        assert len(hit) == 1
        witness[program.offsets[k] + hit[0]] = 1
    evaluated = program.constraints.A @ witness
    row_ok = np.all(evaluated <= program.constraints.ub) and np.all(evaluated >= program.constraints.lb)
    bounds_ok = np.all(witness <= program.bounds.ub) and np.all(witness >= program.bounds.lb)
    assert row_ok and bounds_ok
    return float(program.objective @ witness)


def replay(args, row):
    x, y, levels, train, _, _ = load_fold(args.data_root, int(row.dataset), int(row.seed))
    tree, _ = create_model(encode(x[train], levels), y[train], int(row.seed), "CART", int(row.depth))
    q = x[int(row.query)]
    a = dict(actions_for(levels, int(row.seed)))[row.mode]
    scales = np.ones(len(levels))
    index = LeafIndex(tree, levels, scales)
    z, value, _ = index.solve(q[None], a, "interned")
    assert np.isclose(value[0], row.reference, rtol=0, atol=1e-9)
    graph = NodeFlow.from_tree(tree, levels)
    program = graph.program(q, a, scales)
    objective = vector_in_model(program, tree, z[0], levels)
    assert np.isclose(objective, value[0], rtol=0, atol=1e-9)
    for presolve in (True, False):
        start = time.perf_counter_ns()
        candidate, status = graph.solve(q, a, scales, time_limit=5., presolve=presolve)
        us = (time.perf_counter_ns() - start) / 1000
        got = float(distance(q, candidate, levels, scales)) if candidate is not None else None
        yield dict(dataset=int(row.dataset), depth=int(row.depth), seed=int(row.seed), query=int(row.query),
                   mode=row.mode, presolve=presolve, status=status, runtime_us=us, simulated_cost=got,
                   matrix_feasible=True, verified_minimum=objective,
                   origin_run=row.origin_run, variables=len(program.objective), constraints=program.constraints.A.shape[0],
                   factual=q.tolist(), verified_explanation=z[0].tolist(),
                   solver_explanation=candidate.tolist() if candidate is not None else None,
                   gap=float(got - objective) if got is not None else None,
                   candidate_prediction=int(tree.predict(encode(candidate[None], levels))[0])
                   if candidate is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="A deep benchmark run whose outputs contain discrepancies.")
    parser.add_argument("--pilot-run", help="An optional earlier pilot containing a known discrepancy.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    output = args.out or ROOT / "result" / "runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "__flow_replay")
    offsets = ("script/replay_flow.py", "script/flow_milp.py", "script/deep_benchmark.py",
               "script/compare_search.py", "docs/experiment_plan_round3.md")
    save_provenance(output, args, offsets)
    data = pd.read_json(Path(args.run) / "metrics.jsonl", lines=True)
    data["origin_run"] = Path(args.run).name
    bad = data[data.method.eq("highs_milp") & data.reference.notna()
               & (data.gap.gt(1e-5) | data.kind.eq("reported_infeasible"))]
    bad = bad.sort_values(["gap", "dataset", "seed", "depth", "query", "mode"],
                          ascending=[False, True, True, True, True, True]).head(16)
    if args.pilot_run:
        pilot = pd.read_json(Path(args.pilot_run) / "metrics.jsonl", lines=True)
        extra = pilot[pilot.method.eq("highs_milp") & pilot.gap.gt(1e-5)].head(3).copy()
        extra["origin_run"] = Path(args.pilot_run).name
        bad = pd.concat([bad, extra], ignore_index=True).drop_duplicates(
            subset=["dataset", "seed", "depth", "query", "mode"])
    metrics = []
    with (output / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        for record in bad.itertuples(index=False):
            results = list(replay(args, record))
            metrics.extend(results)
            msg = f"dataset={record.dataset} depth={record.depth} query={record.query} {record.mode}: {[x['status'] for x in results]}"
            print(msg, flush=True)
            print(msg, file=log, flush=True)
    write_jsonl(output / "metrics.jsonl", metrics)
    (output / "artifacts" / "selection.json").write_text(
        json.dumps({"run": args.run, "pilot_run": args.pilot_run, "cases": len(bad),
                    "maximum_final_cases": 16, "maximum_pilot_cases": 3, "time_limit_s": 5,
                    "known_method": "highs_milp", "absolute_gap_threshold": 1e-5}), encoding="utf-8")


if __name__ == "__main__":
    main()
