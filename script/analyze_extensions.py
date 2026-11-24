"""Produce matched comparisons and chronological results from archived query logs."""

import argparse
import json
from pathlib import Path

from analyze_recourse import percentile_ci, read, save_fig

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"dense": "Dense projection", "interned": "Shared enumeration",
         "volumetric": "Volumetric index", "ocean_mip": "OCEAN MIP",
         "ocean_cp": "OCEAN CP", "dice_random": "DiCE Random"}
TASKS = {"bank": "Bank Marketing", "bike": "Bike Sharing", "electricity": "Electricity"}
RULES = {"free": "Unrestricted", "semantic_exact": "Constrained",
         "semantic_vote": "Constrained + vote"}


def baselines(geometry, published, geometry_run, published_run):
    g = geometry.groupby(["dataset", "seed", "method"]).latency_us.median().unstack()
    ratio = (g.volumetric / g.interned).groupby("dataset").median()
    shared_dense = (g.dense / g.interned).groupby("dataset").median()
    regions = (geometry[geometry.method.eq("volumetric")]
               .assign(share=lambda a: a.regions_scored / a.target_leaves)
               .groupby("dataset").share.median())
    data = {}
    for method, rows in published.groupby("method"):
        feasible = rows.reference.notna()
        returned = rows[rows.valid]
        data[method] = {
            "valid_free": int(rows.loc[rows["mode"].eq("free"), "valid"].sum()),
            "valid_restricted": int(rows.loc[rows["mode"].eq("restricted"), "valid"].sum()),
            "valid_given_feasible": int((rows.valid & feasible).sum()),
            "invalid_returns": int((rows.returned & ~rows.valid).sum()),
            "feasible_cases": int(feasible.sum()), "cases": len(rows),
            "both_feasible_count": int((rows.valid & feasible).sum()),
            "matched_cost_tol": int((rows.gap.abs() <= 1e-5).sum()),
            "median_positive_gap": float(returned.gap.clip(lower=0).median()),
            "largest_abs_gap": float(returned.gap.abs().max()),
            "median_latency_us": float(rows.latency_us.median()),
            "p95_latency_us": float(rows.latency_us.quantile(.95)),
            "status": {str(k): int(v) for k, v in rows.status.value_counts().items()},
        }
    splits = pd.read_json(published_run / "artifacts" / "splits.jsonl", lines=True)
    gsplits = pd.read_json(geometry_run / "artifacts" / "splits.jsonl", lines=True)
    for method in data:
        prep = pd.DataFrame([setting[method] for setting in splits.setup])
        data[method]["setup_us"] = float(prep.setup_us.median())
        data[method]["rss_delta_bytes"] = float(prep.rss_delta.median())
    stats = {
        "n_geometry": int(len(geometry) / 3), "n_published": int(len(published) / len(data)),
        "geo_datasets": len(ratio), "geo_max_gap": float(geometry.gap.abs().max()),
        "geometry_ratio": list(percentile_ci(ratio)), "geometry_wins_shared": int((ratio > 1).sum()),
        "geometry_shared_dense": list(percentile_ci(shared_dense)),
        "geometry_retained_share": float(regions.median()),
        "geometry_added_array_bytes": float(pd.Series(
            [i["volumetric"]["index_increment_bytes"] for i in gsplits.setup]).median()),
        "geometry_setup_median_us": {
            k: float(pd.Series([i[k]["setup_us"] for i in gsplits.setup]).median())
            for k in ("interned", "volumetric")},
        "published": data,
    }
    ratios = pd.concat([ratio.rename("volumetric_over_shared"), shared_dense.rename("dense_over_shared"),
                        regions.rename("share_scored")], axis=1)
    ratios.to_csv(geometry_run / "artifacts" / "paired_geometry.csv")
    pd.DataFrame(data).T.to_csv(published_run / "artifacts" / "summary.csv")
    return stats


def temporal(shift, out):
    shift["usable"] = shift.semantic & shift.retained & shift.factual_still_negative
    shift["prediction_persists"] = shift.found & shift.retained
    records = []
    for (task, method), data in shift.groupby(["task", "method"]):
        feasible = data[data.found]
        records.append({
            "task": task, "method": method, "queries": len(data),
            "found": int(data.found.sum()), "semantic": int(data.semantic.sum()),
            "retained": int(data.prediction_persists.sum()), "usable": int(data.usable.sum()),
            "relative_retention": float(feasible.retained.mean()) if len(feasible) else None,
            "found_but_immutable_violation": int((data.found & ~data.semantic).sum()),
            "median_cost": float(feasible.cost.median()) if len(feasible) else None,
        })
    paired = shift[shift.method.isin(["semantic_exact", "semantic_vote"])].pivot(
        index=["task", "origin", "query"], columns="method", values=["found", "retained", "usable", "cost"])
    common = paired[paired.found.semantic_exact.astype(bool) & paired.found.semantic_vote.astype(bool)]
    common_counts = {}
    for task, same in common.groupby(level="task"):
        common_counts[task] = {
            "both_found": len(same),
            "exact_retained": int(same.retained.semantic_exact.sum()),
            "vote_retained": int(same.retained.semantic_vote.sum()),
            "exact_usable": int(same.usable.semantic_exact.sum()),
            "vote_usable": int(same.usable.semantic_vote.sum()),
            "median_cost_increase": float(np.median(same.cost.semantic_vote - same.cost.semantic_exact)),
        }
    pd.DataFrame(records).to_csv(out / "summary.csv", index=False)
    temporal_plot(shift, Path("figures"))
    return {"rows": records, "paired": common_counts}


def temporal_plot(raw, out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.8))
    opt = raw[raw.method.eq("semantic_exact")].groupby(["task", "origin"]).semantic.mean()
    votes = raw[raw.method.eq("semantic_vote")].groupby(["task", "origin"]).semantic.mean()
    joint = raw.groupby(["task", "origin", "method"]).usable.mean()
    colors = {"bank": "#aa6030", "bike": "#007b8d", "electricity": "#514480"}
    for task, name in TASKS.items():
        color = colors[task]
        a = opt.loc[task]
        idx = a.index * 100
        left.plot(idx, a * 100, "-", color=color, marker="o", label=name)
        left.plot(idx, votes.loc[task] * 100, "--", color=color, marker="o")
        right.plot(idx, joint.loc[task].unstack().semantic_exact * 100, "-", color=color, marker="o")
        right.plot(idx, joint.loc[task].unstack().semantic_vote * 100, "--", color=color, marker="o")
    for ax in (left, right):
        ax.set_xlabel("Early training cutoff (%)")
        ax.set_xticks([35, 43, 51, 59, 67])
        ax.set_ylim(-2, 45)
    left.legend(fontsize=16, loc="upper left")
    left.set_ylabel("Admissible answers (%)")
    right.set_ylabel("Usable after update (%)")
    save_fig(fig, out, "chronological-recourse")


def latex_for(stats, timeline, output):
    rows = []
    for method, label in NAMES.items():
        d = stats["published"][method]
        line = [label, d["valid_free"], d["valid_restricted"], d["matched_cost_tol"],
                f"{d['median_positive_gap']:.3g}", f"{d['setup_us'] / 1000:.2f}",
                f"{d['median_latency_us'] / 1000:.3f}", f"{d['p95_latency_us'] / 1000:.3f}"]
        rows.append(" & ".join(map(str, line)) + r" \\")
    (output / "external_table.tex").write_text("\n".join(rows) + "\n" + r"\bottomrule" + "\n")
    rows = []
    for d in timeline["rows"]:
        line = [TASKS[d["task"]], RULES[d["method"]], d["found"], d["semantic"], d["retained"], d["usable"]]
        rows.append(" & ".join(map(str, line)) + r" \\")
    (output / "time_table.tex").write_text("\n".join(rows) + "\n" + r"\bottomrule" + "\n")


def compiled_ablation(run):
    raw = read(run)
    assert len(raw) == 60 * 64 * 3
    groups = raw.groupby(["dataset", "seed", "depth", "method"])
    timings = groups[["latency_us", "batch_us"]].median().unstack()
    timings.columns = ["_".join(pair) for pair in timings.columns]
    term_counts = raw[raw.method.eq("compiled_screen")].copy()
    term_share = (term_counts.summations / term_counts.leaf_terms.replace(0, np.nan))
    timings["term_share"] = term_share.groupby(
        [term_counts.dataset, term_counts.seed, term_counts.depth]).median()
    timings["screen_over_enum"] = timings.latency_us_compiled_screen / timings.latency_us_compiled_enum
    timings["screen_over_enum_batch"] = timings.batch_us_compiled_screen / timings.batch_us_compiled_enum
    timings["scipy_over_enum"] = timings.latency_us_scipy / timings.latency_us_compiled_enum
    timings.to_csv(run / "artifacts" / "native_pairings.csv")
    synthetic = timings.loc["synthetic", :].groupby(level="depth").median()
    real = timings.drop(index="synthetic", level="dataset").groupby(level="dataset").median()
    stats = json.loads((run / "artifacts" / "compilation.json").read_text())
    stats.update({
        "query_count": int(len(raw) / 3), "tree_count": len(timings),
        "public_screen_ratio": list(percentile_ci(real.screen_over_enum)),
        "public_screen_batch_ratio": list(percentile_ci(real.screen_over_enum_batch)),
        "public_scipy_ratio": list(percentile_ci(real.scipy_over_enum)),
        "public_screen_term_share": float(real.term_share.median()),
        "synthetic": synthetic.T.to_dict(),
    })
    (run / "artifacts" / "analysis.json").write_text(json.dumps(stats, indent=2, allow_nan=False))
    return stats


def compiled_region(run, tuned=False):
    raw = read(run)
    assert len(raw) == 60 * 64 * (5 if tuned else 4)
    folds = raw.groupby(["dataset", "seed", "depth", "method"])[["latency_us", "batch_us"]].median().unstack()
    method = "tuned_region" if tuned else "compiled_region"
    effects = pd.DataFrame({
        "single_region_over_enum": folds.latency_us[method] / folds.latency_us.compiled_enum,
        "batch_region_over_enum": folds.batch_us[method] / folds.batch_us.compiled_enum,
    })
    if tuned:
        effects["fixed_over_tuned_batch"] = folds.batch_us.compiled_region / folds.batch_us.tuned_region
    ratio = raw[raw.method.eq(method)]
    work = ratio.scored_regions / ratio.target_leaves.replace(0, np.nan)
    effects["scored_region_fraction"] = work.groupby(
        [ratio.dataset, ratio.seed, ratio.depth]).median()
    ds = effects.drop(index="synthetic", level="dataset").groupby(level="dataset").median()
    synthetic = effects.loc["synthetic"].groupby(level="depth").median()
    summary = {
        "queries": int(len(ratio)),
        "single": list(percentile_ci(ds.single_region_over_enum)),
        "batch": list(percentile_ci(ds.batch_region_over_enum)),
        "wins_shared_single": int((ds.single_region_over_enum > 1).sum()),
        "wins_shared_batch": int((ds.batch_region_over_enum > 1).sum()),
        "scored_region_fraction": float(ds.scored_region_fraction.median()),
        "public_datasets": ds.to_dict("index"),
        "synthetic": synthetic.to_dict("index"),
    }
    if tuned:
        split = pd.read_json(run / "artifacts" / "splits.jsonl", lines=True)
        summary["bucket_counts"] = {int(k): int(v) for k, v in split.region_bucket.value_counts().items()}
        summary["fixed_over_tuned_batch"] = list(percentile_ci(ds.fixed_over_tuned_batch))
    summary.update(json.loads((run / "artifacts" / "compilation.json").read_text()))
    effects.to_csv(run / "artifacts" / "compiled_geometry_pairings.csv")
    (run / "artifacts" / "analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def matched_bank(original, run):
    weighted = read(run)
    common = weighted[["seed", "query"]].drop_duplicates()
    reference = original[original.task.eq("bank")].merge(common, on=["seed", "query"], validate="many_to_one")
    outputs = []
    for label, data in (("original", reference), ("class_weight_balanced", weighted)):
        data["usable"] = data.semantic & data.retained & data.factual_still_negative
        for method, part in data.groupby("method"):
            outputs.append({"model": label, "method": method, "queries": len(part),
                            "admissible": int(part.semantic.sum()), "retained": int(part.retained.sum()),
                            "usable": int(part.usable.sum())})
    pd.DataFrame(outputs).to_csv(run / "artifacts" / "matched_bank.csv", index=False)
    splits = pd.read_json(run / "artifacts" / "splits.jsonl", lines=True)
    return {"rows": outputs, "new_base_accuracy_median": float(splits.base_accuracy.median()),
            "new_updated_accuracy_median": float(splits.updated_accuracy.median())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("geometry", "published", "time", "native", "weighted"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--native-geometry", type=Path)
    parser.add_argument("--tuned-geometry", type=Path)
    args = parser.parse_args()
    geometry, published, shift = [read(path) for path in (args.geometry, args.published, args.time)]
    assert len(geometry) == 12 * 5 * 24 * 2 * 3
    assert len(published) == 4 * 5 * 8 * 2 * 6
    assert len(shift) == 3 * 5 * 160 * 3
    plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
    results = baselines(geometry, published, args.geometry, args.published)
    timeline = temporal(shift, args.time / "artifacts")
    timeline["bank_weight_sensitivity"] = matched_bank(shift, args.weighted)
    native = compiled_ablation(args.native)
    latex_for(results, timeline, Path("figures"))
    (args.published / "artifacts" / "analysis.json").write_text(
        json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (args.time / "artifacts" / "analysis.json").write_text(
        json.dumps(timeline, indent=2, allow_nan=False), encoding="utf-8")
    record = {"baselines": results, "temporal": timeline, "native": native}
    if args.native_geometry:
        record["native_geometry"] = compiled_region(args.native_geometry)
    if args.tuned_geometry:
        record["tuned_geometry"] = compiled_region(args.tuned_geometry, tuned=True)
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
