"""Summarize deep solver, isolated memory, and delayed-update observations."""

import argparse
import json
from pathlib import Path

from analyze_recourse import read, save_fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {
    "shared": "Shared enumeration", "region": "Volumetric index",
    "highs_milp": "HiGHS MILP", "ocean_cp": "OCEAN CP", "ocean_mip": "OCEAN MIP",
    "dice": "DiCE Random", "nice_sparse": "NICE sparsity", "nice_proximity": "NICE proximity",
    "dense": "Dense projection", "compiled_shared": "Compiled shared",
    "compiled_region": "Compiled volumetric",
}
TASKS = {"bank": "Bank Marketing", "electricity": "Electricity",
         "bike": "Capital Bikeshare", "seoul": "Seoul Bike Sharing"}


def table(path, rows):
    text = "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
    path.write_text(text + "\n" + r"\bottomrule" + "\n", encoding="utf-8")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def stats_for(block):
    reference = block.reference.notna()
    valid = block.valid
    expenses = block.loc[valid, "gap"]
    return dict(
        queries=len(block), feasible=int(reference.sum()), valid=int(valid.sum()),
        matched=int((block.gap.abs() <= 1e-5).sum()), above_tolerance=int((block.gap > 1e-5).sum()),
        valid_p50_us=float(block.loc[valid, "latency_us"].median()) if valid.any() else None,
        valid_p95_us=float(block.loc[valid, "latency_us"].quantile(.95)) if valid.any() else None,
        gap_median=float(expenses.median()) if len(expenses) else None,
        max_gap=float(expenses.abs().max()) if len(expenses) else None,
        license_limits=int(block.kind.eq("license_limit").sum()),
        timeout_or_unresolved=int(block.kind.eq("time_limit_or_unresolved").sum()),
        no_candidate=int(block.kind.eq("no_candidate").sum()),
        unvalidated_return=int(block.kind.eq("unvalidated_return").sum()),
        other=int(block.kind.eq("other").sum()),
        true_infeasible=int((block.kind.eq("reported_infeasible") & ~reference).sum()),
        alleged_infeasible=int((block.kind.eq("reported_infeasible") & reference).sum()),
        validated_status={str(k): int(v) for k, v in block.status.value_counts().items()},
    )


def deep(run, figures):
    raw = read(run)
    names = ["shared", "region", "highs_milp", "ocean_cp", "ocean_mip", "dice", "nice_sparse", "nice_proximity"]
    assert set(raw.method) == set(names)
    assert raw.dataset.nunique() == 12 and raw.seed.nunique() == 5
    expected = raw[raw.method.eq("shared")]
    assert len(expected) == len(raw) // len(names)
    assert expected.kind.isin(["valid", "reported_infeasible"]).all()
    records = [{"method": method, "depth": int(depth), "policy": policy, **stats_for(part)}
               for (method, depth, policy), part in raw.groupby(["method", "depth", "mode"])]
    pd.DataFrame(records).to_csv(run / "artifacts" / "by_depth_policy.csv", index=False)
    methods = {key: stats_for(block) for key, block in raw.groupby("method")}
    by_depth = {str(depth): {key: stats_for(block) for key, block in subset.groupby("method")}
                for depth, subset in raw.groupby("depth")}
    references = {str(d): int(frame.reference.notna().sum()) for d, frame in expected.groupby("depth")}
    examples = raw[raw.method.isin(["ocean_cp", "ocean_mip", "highs_milp"]) & (
                   (raw.gap.abs() > 1e-5) | (raw.reference.notna() & ~raw.valid))]
    examples.to_csv(run / "artifacts" / "discrepancies.csv", index=False)
    info = dict(methods=methods, by_depth=by_depth, queries=len(expected), feasible_by_depth=references,
                deepest_actual=int(raw.actual_depth.max()), maximum_target_leaves=int(raw.leaves.max()),
                maximum_encoded_width=int(raw.width.max()))
    save(run / "artifacts" / "analysis.json", info)
    output = []
    for method in names:
        row = methods[method]
        v = [by_depth[str(d)][method]["valid"] for d in (6, 10, 14)]
        output.append([METHODS[method], *v, row["matched"], row["license_limits"],
                       row["timeout_or_unresolved"], f"{row['valid_p50_us'] / 1000:.3f}"])
    table(figures / "deep_external_table.tex", output)
    return info


def memory(run, figures):
    raw = read(run)
    workers = pd.read_json(run / "artifacts" / "worker_commands.jsonl", lines=True)
    assert len(workers) == 4 * 2 * 2 * 6
    mib = 2 ** 20
    phases = raw.assign(
        peak_mib=raw.end_peak / mib, extra_peak_mib=(raw.end_peak - raw.pre_peak) / mib,
        high_after_setup=(raw.setup_peak - raw.pre_peak) / mib, after_warm_mib=raw.cold_peak / mib,
        online_growth_mib=(raw.end_peak - raw.cold_peak) / mib,
    )
    status_frame = phases.statuses.apply(pd.Series).fillna(0)
    phases["license_errors"] = status_frame.filter(regex=r"size-limited|Model too large").sum(axis=1)
    summaries = []
    for (method, nominal), part in phases.groupby(["method", "nominal"]):
        secs = part[["peak_mib", "extra_peak_mib", "high_after_setup", "after_warm_mib",
                     "online_growth_mib", "setup_us", "cold_us", "batch_us"]].median().to_dict()
        summaries.append({"method": method, "nominal": int(nominal), "processes": len(part),
                          "n_min": int(part.queries.min()), "n_max": int(part.queries.max()),
                          "license_errors": int(part.license_errors.sum()),
                          "peak_p95_mib": float(part.peak_mib.quantile(.95)), **secs})
    status_records = {str(status): int(total) for status, total in workers.status.value_counts().items()}
    final = {r["method"]: r for r in summaries if r["nominal"] == 128}
    table(figures / "peak_memory_table.tex", [
        [METHODS[key], row["processes"], f"{row['setup_us']/1000:.2f}", f"{row['cold_us']/1000:.2f}",
         f"{row['extra_peak_mib']:.1f}", f"{row['peak_mib']:.1f}", f"{row['peak_p95_mib']:.1f}"]
        for key in ("dense", "shared", "compiled_shared", "compiled_region", "ocean_cp", "ocean_mip")
        if (row := final.get(key))
    ])
    pd.DataFrame(summaries).to_csv(run / "artifacts" / "by_method_batch.csv", index=False)
    info = dict(workers=status_records, rows=len(raw), summary=summaries)
    save(run / "artifacts" / "analysis.json", info)
    return info


def aggregate_time(part):
    return dict(n=len(part), admissible=int(part.admissible.sum()), found=int(part.found.sum()),
                retained=int(part.retained.sum()), usable=int(part.usable.sum()),
                usable_per_100=100 * float(part.usable.mean()),
                conditional_retention=float(part.retained.sum() / part.found.sum()) if part.found.any() else None)


def temporal(run, figures):
    raw = read(run)
    configs = pd.read_json(run / "artifacts" / "splits.jsonl", lines=True)
    assert len(configs) == 32 and raw.policy.nunique() == 4
    info = {"total": int(len(raw) // 4), "sources": {}, "paired": {}, "windows": [], "policy": {}}
    for task, group in configs.groupby("task"):
        descriptors = group[["base_accuracy", "updated_accuracy"]].median().to_dict()
        thresholds = sorted(group[["origin", "vote_cutoff"]].drop_duplicates().itertuples(index=False, name=None))
        outcomes = {str(k): aggregate_time(part) for k, part in raw[raw.task.eq(task)].groupby("policy")}
        info["sources"][task] = {**descriptors, "cutoffs": thresholds, "outcomes": outcomes}
        subset = raw[raw.task.eq(task) & raw.policy.isin(["constrained", "fixed_four", "calibrated"])]
        p = subset.pivot(index=["origin", "lead", "query"], columns="policy", values=["found", "retained", "usable"])
        cases = {}
        for method in ("fixed_four", "calibrated"):
            common = p[p.found.constrained.astype(bool) & p.found[method].astype(bool)]
            cases[method] = {
                "eligible": len(common), "constrained_retained": int(common.retained.constrained.sum()),
                "filtered_retained": int(common.retained[method].sum()),
                "constrained_usable": int(common.usable.constrained.sum()),
                "filtered_usable": int(common.usable[method].sum()),
            }
        info["paired"][task] = cases
    info["windows"] = [{"task": task, "lead": lead, "policy": policy, **aggregate_time(part)}
                       for (task, lead, policy), part in raw.groupby(["task", "lead", "policy"])]
    info["policy"] = {str(k): aggregate_time(data) for k, data in raw.groupby("policy")}
    frame = pd.DataFrame(info["windows"])
    frame.to_csv(run / "artifacts" / "by_task_horizon.csv", index=False)
    table(figures / "horizon_table.tex", [
        [TASKS[task], data["outcomes"]["constrained"]["n"],
         data["outcomes"]["constrained"]["admissible"],
         *[data["outcomes"][key]["usable"] for key in ("free", "constrained", "fixed_four", "calibrated")]]
        for task, data in info["sources"].items()
    ])
    time_figure(frame, figures)
    save(run / "artifacts" / "analysis.json", info)
    return info


def time_figure(frame, figures):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    values = frame[frame.policy.eq("constrained")]
    colors = {"bank": "#8d4d20", "electricity": "#714b8b", "bike": "#1c7d76", "seoul": "#2d61a6"}
    for task, rows in values.groupby("task"):
        axes[0].plot(rows.lead * 100, rows.usable_per_100, color=colors[task], marker="o", label=TASKS[task])
    axes[0].legend(fontsize=16)
    seo = frame[frame.task.eq("seoul")]
    for i, (label, name) in enumerate([("constrained", "Constrained"), ("fixed_four", "Four votes"),
                                      ("calibrated", "Calibrated votes")]):
        data = seo[seo.policy.eq(label)]
        axes[1].plot(data.lead * 100, data.usable_per_100, marker=("o", "s", "^")[i],
                     ls=("-", "--", "-.")[i], color=("#2d61a6", "#a95c25", "#4f746d")[i], label=name)
    axes[1].legend(fontsize=16)
    for ax in axes:
        ax.set_xlabel("Update delay (% of timeline)")
        ax.set_xticks([5, 15, 25, 35])
        ax.set_ylabel("Usable per 100 queries")
        ax.set_ylim(bottom=-.8)
    save_fig(fig, figures, "delayed-explanation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("deep", "memory", "time"):
        parser.add_argument("--" + stage, type=Path)
    args = parser.parse_args()
    figures = Path("figures")
    plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
    outputs = {}
    for name, function in (("deep", deep), ("memory", memory), ("time", temporal)):
        path = getattr(args, name)
        if path:
            outputs[name] = function(path, figures)
    print(json.dumps(outputs, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
