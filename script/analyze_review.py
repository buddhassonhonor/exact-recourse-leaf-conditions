"""Summarize the independent verification and follow-up experiments."""

import argparse
import json
from pathlib import Path

from run_recourse import Action, LeafIndex, create_model, encode, load_openml, transform
from analyze_recourse import save_fig, percentile_ci, read

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def case_from_oracle(oracle, splits, data_root):
    primary = oracle[oracle.dataset.eq(31) & oracle.seed.eq(7)]
    free = primary[primary["mode"].eq("free")].set_index("query")
    restricted = primary[primary["mode"].eq("restricted") & primary.status.eq(2)]
    fold = splits[splits.dataset.eq(31) & splits.seed.eq(7)].iloc[0]
    frame, y, _ = load_openml(data_root, 31)
    train = np.array(fold.train_indices)
    x, levels, scales, _ = transform(frame, train)
    model, _ = create_model(encode(x[train], levels), y[train], 7, "CART", 6)
    idx = LeafIndex(model, levels, scales)
    for row in restricted.itertuples(index=False):
        factual = x[row.query:row.query + 1]
        for field in row.frozen:
            action = Action(tuple(j for j in row.frozen if j != field), tuple(row.increasing),
                            tuple(row.decreasing))
            z, costs, _ = idx.solve(factual, action, "interned")
            if np.isfinite(costs[0]):
                changes = np.flatnonzero(z[0] != factual[0])
                return {"dataset": 31, "seed": 7, "query": int(row.query),
                        "free_cost": float(free.loc[row.query, "oracle_cost"]),
                        "frozen_fields": [str(frame.columns[j]) for j in row.frozen],
                        "released_field": str(frame.columns[field]),
                        "increasing": [str(frame.columns[j]) for j in row.increasing],
                        "decreasing": [str(frame.columns[j]) for j in row.decreasing],
                        "release_cost": float(costs[0]),
                        "changed_fields": [str(frame.columns[j]) for j in changes]}
    raise RuntimeError("No single-field release exposed a finite certificate")


def chart(batch, refit, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    first, second = axes
    for tag, name, colour in [
        ("dense", "Dense", "#262626"), ("interned", "Shared", "#00778c"),
        ("tuned_screen", "Tuned screen", "#bf683e"),
    ]:
        d = batch[batch.method.eq(tag)].groupby(["dataset", "size"]).latency_us.median().unstack()
        p50 = d.median()
        first.plot(p50.index, p50, marker="o", label=name, color=colour)
        first.fill_between(d.columns, d.quantile(.25), d.quantile(.75), color=colour, alpha=.16)
    first.set_xscale("log", base=2)
    first.set_yscale("log")
    first.set_xticks([1, 8, 32, 128, 256], ["1", "8", "32", "128", "256"])
    plt.setp(first.get_xticklabels(), rotation=27, ha="right", rotation_mode="anchor")
    first.set_ylabel("Latency per query (µs)")
    first.set_xlabel("Requested batch size")
    first.legend(fontsize=16)
    for tag, name, colour in [("interned", "Exact", "#00778c"), ("nearest", "Nearest", "#bf683e")]:
        r = refit[refit.method.eq(tag)].groupby(["dataset", "noise"]).kept_target.mean().unstack()
        second.plot(r.columns * 100, 100 * r.median(), marker="o", label=name, color=colour)
        second.fill_between(r.columns * 100, 100 * r.quantile(.25), 100 * r.quantile(.75),
                            color=colour, alpha=.16)
    second.set_xticks([0, 5, 10, 20])
    second.set_ylim(0, 100)
    second.set_ylabel("Target-label retention (%)")
    second.set_xlabel("Training label flips (%)")
    second.legend(fontsize=16)
    save_fig(fig, out, "workload-and-refit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--followup", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    args = parser.parse_args()
    oracle, followup = read(args.oracle), read(args.followup)
    splits = pd.read_json(args.oracle / "artifacts" / "splits.jsonl", lines=True)
    batch = read(args.batch)
    refit = followup[followup.experiment.eq("refit") & followup.eligible.eq(True)].copy()
    exact_budget = oracle[oracle.status.eq(0)]
    assert oracle.agreement.all()
    assert len(batch) == 6 * 5 * 8 * 5
    audit = {"cases": len(oracle), "feasible": len(exact_budget), "infeasible": int(oracle.status.eq(2).sum()),
             "largest_error": float(exact_budget.error.max()), "median_solver_ms": float(oracle.solve_us.median()) / 1000,
             "p95_solver_ms": float(oracle.solve_us.quantile(.95)) / 1000,
             "case": case_from_oracle(oracle, splits, args.data_root),
             "fold_median_index_ms": float(batch.build_us.median() / 1000)}
    b = batch.groupby(["size", "dataset", "seed", "method"]).latency_us.median().unstack()
    b["speedup"] = b.dense / b.interned
    b["tuned_speedup"] = b.fixed_screen / b.tuned_screen
    b["dispatch_speedup"] = b.interned / b.calibrated_dispatch
    b["sharing_speedup"] = b.sparse_unshared / b.interned
    ds = b.groupby(["size", "dataset"]).median()
    ratios = {int(i): {m: list(percentile_ci(v[m])) for m in
                      ("speedup", "tuned_speedup", "dispatch_speedup", "sharing_speedup")}
              for i, v in ds.groupby(level="size")}
    audit["batch_paired_ratios"] = ratios
    audit["batch_per_query_us"] = ds.groupby(level="size").median().T.to_dict()
    audit["batch_request_realized"] = batch.groupby("size").n_queries.agg(["min", "median", "max"]).T.to_dict()
    selection = batch[batch.method.eq("calibrated_dispatch")]
    audit["dispatch_count"] = selection.groupby("selected_method").size().to_dict()
    audit["prefix_counts"] = batch[batch.method.eq("tuned_screen")].fraction.value_counts().to_dict()
    all_refit = refit.groupby(["dataset", "noise", "method"]).kept_target.mean()
    audit["refit_rate"] = all_refit.groupby(level=["noise", "method"]).median().unstack().T.to_dict()
    negative = refit[refit.factual_still_negative.eq(True)]
    audit["refit_rate_factual_negative"] = negative.groupby(
        ["dataset", "noise", "method"]).kept_target.mean().groupby(
            level=["noise", "method"]).median().unstack().T.to_dict()
    audit["n_retraining_comparisons"] = len(refit)
    audit["n_retraining_folds"] = len(refit[["dataset", "seed", "noise", "repeat"]].drop_duplicates())
    (args.followup / "artifacts" / "analysis.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    ds.to_csv(args.followup / "artifacts" / "batch_dataset_summary.csv")
    all_refit.to_csv(args.followup / "artifacts" / "refit_dataset_summary.csv")
    (args.oracle / "artifacts" / "case.json").write_text(json.dumps(audit["case"], indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
    chart(batch, refit, Path("figures"))

    sources = json.loads((args.main / "artifacts" / "datasets.json").read_text())
    dataset_table = pd.DataFrame(sources.values())
    dataset_table[["dataset", "name", "version", "target", "count_0", "count_1", "labels", "rows",
                   "n_numeric", "n_categorical", "missing_cells", "target_missing", "licence",
                   "source", "cached_file", "sha256"]].to_csv("docs/datasets.csv", index=False)
    diagnostics = pd.read_json(args.main / "artifacts" / "splits.jsonl", lines=True)
    diagnostics[["dataset", "model", "seed", "depth", "nodes", "target_leaves", "condition_terms",
                 "unique_conditions", "accuracy", "index_ms", "n_queries"]].to_csv(
                     args.main / "artifacts" / "fold_diagnostics.csv", index=False)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
