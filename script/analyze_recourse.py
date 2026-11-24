"""Read experiment records and produce statistical summaries, figures, and LaTeX data."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


METHODS = {
    "dense": "Dense projection", "interned": "Shared-condition enumeration",
    "cascade": "Screened search", "random_prefix": "Random screen",
    "heap": "Tree best-first", "nearest": "Nearest training point",
    "medoid": "One training exemplar per leaf", "no_change": "No change",
}


def read(run):
    frame = pd.read_json(run / "metrics.jsonl", lines=True)
    return frame


def percentile_ci(values, seed=20261124):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    simulations = np.median(rng.choice(v, size=(10000, len(v))), axis=1)
    return float(np.median(v)), *map(float, np.quantile(simulations, [.025, .975]))


def paired(work, m, baseline, measure="latency_us"):
    med = work.groupby(["dataset", "model", "seed", "method"])[measure].median().unstack()
    ratio = (med[baseline] / med[m]).dropna()
    by_dataset = ratio.groupby(level="dataset").median()
    positive = int((by_dataset > 1).sum())
    negative = int((by_dataset < 1).sum())
    return {"effect": percentile_ci(by_dataset), "wins": positive, "losses": negative,
            "n": len(by_dataset), "sign_p": binomtest(positive, positive + negative).pvalue
            if positive + negative else 1.}, by_dataset


def main_table(main):
    records = []
    for method, part in main.groupby("method"):
        f = part.groupby(["dataset", "model", "seed"])
        fold = f.agg(coverage=("valid", "mean"), cost=("cost", "median"),
                     changed=("changed", "median"), support=("support_distance", "median"),
                     p50=("latency_us", "median"),
                     p95=("latency_us", lambda x: x.quantile(.95)))
        ds = fold.groupby(level="dataset").median()
        record = {"method": method, **ds.median().to_dict()}
        record["coverage"] = f.valid.mean().groupby(level="dataset").mean().mean()
        batch = part.dropna(subset=["batch_us"])
        record["batch"] = batch.groupby("dataset").batch_us.median().median() if len(batch) else np.nan
        records.append(record)
    return pd.DataFrame(records).set_index("method").loc[list(METHODS)]


def save_fig(fig, output, name):
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=100)
    plt.close(fig)


def latency_plot(ratios, out):
    fig, ax = plt.subplots(figsize=(10, 6))
    names = ["Shared vs dense\nSingle query", "Screened vs dense\nSingle query",
             "Screened vs shared\nSingle query", "Shared vs dense\nBatched"]
    ax.boxplot(list(ratios), tick_labels=names, showfliers=True)
    ax.axhline(1, color="#666666", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("Baseline / method latency")
    ax.tick_params(axis="x", labelsize=16)
    save_fig(fig, out, "paired-latency")


def scale_plot(scale, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    main = scale[scale.fraction.eq(.25) & scale.method.isin(["dense", "interned", "cascade"])]
    for ax, width in zip(axes, [8, 256]):
        subset = main[main.features.eq(width)]
        for method, color in zip(["dense", "interned", "cascade"], ["#222222", "#00778c", "#bf683e"]):
            d = subset[subset.method.eq(method)].groupby(["seed", "depth"]).latency_us.median().unstack()
            m = d.median()
            ax.plot(m.index, m, marker="o", label=METHODS[method].replace(" enumeration", ""), color=color)
            ax.fill_between(m.index, d.min(), d.max(), alpha=.17, color=color)
        ax.set_yscale("log")
        ax.set_xticks([3, 6, 9, 12])
        ax.set_title(f"{width} input fields")
        ax.set_xlabel("Maximum depth")
    axes[0].set_ylabel("Single-query latency (µs)")
    axes[1].legend(fontsize=16, loc="best")
    save_fig(fig, out, "synthetic-scale")


def pruning_plot(stress, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    core = stress[stress.directional & stress.freeze.eq(0)]
    variants = core[core.method.eq("cascade")].copy()
    variants["retained"] = variants.work / variants.target_leaves
    all_values = variants.groupby(["dataset", "fraction"]).retained.median().unstack()
    axes[0].boxplot([all_values[col] for col in all_values], tick_labels=["1/8", "1/4", "1/2", "3/4"])
    axes[0].set_xlabel("Prefix fraction")
    axes[0].set_ylabel("Share of candidate leaves retained")
    dense = stress[stress.directional & stress.method.eq("dense")].copy()
    matrix = dense.groupby(["dataset", "freeze"]).valid.mean().unstack()
    for _, row in matrix.iterrows():
        axes[1].plot(row.index * 100, row, marker="o", alpha=.38, color="#00778c")
    axes[1].plot(matrix.columns * 100, matrix.median(), color="#222222", lw=2.8, label="Dataset median")
    axes[1].set_xticks([0, 25, 50, 75])
    axes[1].set_xlabel("Frozen fields (%)")
    axes[1].set_ylabel("Fraction of feasible queries")
    axes[1].legend(fontsize=16)
    save_fig(fig, out, "screen-and-constraints")


def quality_plot(main, out):
    methods = ["interned", "nearest", "medoid"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.8))
    labels = ["Exact", "Nearest", "Exemplar"]
    for ax, metric, title in zip(axes, ["cost", "support_distance"],
                                 ["Cost to change the prediction", "Distance to training support"]):
        ds = main.groupby(["method", "dataset"])[metric].median()
        ax.boxplot([ds.loc[m] for m in methods], tick_labels=labels)
        ax.set_title(title)
        ax.set_ylabel("Mixed distance")
        ax.set_yscale("symlog", linthresh=.05)
    save_fig(fig, out, "quality-tradeoff")


def format_num(v, precision=2):
    return f"{v:.{precision}f}" if pd.notna(v) else "--"


def latex_table(table, out):
    labels = {"interned": "Shared enumeration", "nearest": "Nearest training point",
              "medoid": "Leaf exemplar"}
    rows = []
    for method, row in table.iterrows():
        values = [row["coverage"] * 100, row["cost"], row["changed"], row["support"], row["p50"],
                  row["p95"], row["batch"]]
        if method == "no_change":
            values[-3:] = [np.nan] * 3
        rows.append(" & ".join([labels.get(method, METHODS[method]),
                               *[format_num(v) for v in values]]) + r" \\")
    out.write_text("\n".join(rows) + "\n" + r"\bottomrule" + "\n", encoding="utf-8")


def main():
    arg = argparse.ArgumentParser()
    for stage in ("main", "stress", "scale", "exhaustive"):
        arg.add_argument("--" + stage, required=True, type=Path)
    args = arg.parse_args()
    out = Path("figures")
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 18, "axes.spines.top": False, "axes.spines.right": False})
    raw, stress, scale, checks = (read(getattr(args, name)) for name in
                                  ("main", "stress", "scale", "exhaustive"))
    dataset_info = json.loads((args.main / "artifacts" / "datasets.json").read_text())
    splits = pd.read_json(args.main / "artifacts" / "splits.jsonl", lines=True)
    active_folds = splits[splits.n_queries.gt(0)]
    dense = raw[raw.method.eq("dense")]
    exact_methods = ["interned", "cascade", "random_prefix", "heap"]
    summary = {
        "source_runs": {k: str(v) for k, v in vars(args).items()},
        "datasets": len(dataset_info), "folds": len(splits), "active_folds": len(active_folds),
        "queries": len(dense), "queries_by_model": dense.groupby("model").size().to_dict(),
        "models_accuracy": splits.groupby("model").accuracy.agg(["median", "min", "max"]).T.to_dict(),
        "gap_max": float(raw[raw.method.isin(exact_methods)].gap.abs().max()),
        "stress_gap_max": float(stress[stress.method.isin(exact_methods)].gap.abs().max()),
        "scale_gap_max": float(scale[scale.method.isin(exact_methods)].gap.abs().max()),
        "exhaustive_cases_per_method": int(checks[checks.method.eq("dense")].cases.sum()),
        "exhaustive_feasible": int(checks[checks.method.eq("dense")].feasible.sum()),
        "width_range": [int(splits.features.min()), int(splits.features.max())],
        "size_range": [min(m["rows"] for m in dataset_info.values()),
                       max(m["rows"] for m in dataset_info.values())],
        "target_leaf_range": [int(active_folds.target_leaves.min()), int(active_folds.target_leaves.max())],
        "median_leaf_count": float(active_folds.target_leaves.median()),
        "index_median_ms": float(active_folds.index_ms.median()),
        "index_p95_ms": float(active_folds.index_ms.quantile(.95)),
        "condition_share_median": float((active_folds.unique_conditions / active_folds.condition_terms).dropna().median()),
    }
    table = main_table(raw)
    table.to_csv(args.main / "artifacts" / "summary.csv")
    latex_table(table, out / "main_table.tex")
    ratios = []
    for name, method, baseline, col in [
        ("interned_vs_dense", "interned", "dense", "latency_us"),
        ("cascade_vs_dense", "cascade", "dense", "latency_us"),
        ("cascade_vs_interned", "cascade", "interned", "latency_us"),
        ("batch_interned_vs_dense", "interned", "dense", "batch_us"),
        ("batch_cascade_vs_interned", "cascade", "interned", "batch_us"),
    ]:
        summary[name], latency = paired(raw, method, baseline, col)
        ratios.append(latency)
        latency.rename(name).to_csv(args.main / "artifacts" / f"{name}.csv")
    for method in ["cascade", "random_prefix"]:
        part = raw[raw.method.eq(method)].copy()
        part["retained"] = part.work / part.target_leaves
        summary[method + "_retained"] = part.groupby("dataset").retained.median().median()
    dense_fold = dense.groupby(["dataset", "seed", "model"]).latency_us.median()
    sparse_fold = raw[raw.method.eq("interned")].groupby(["dataset", "seed", "model"]).latency_us.median()
    ratio = (dense_fold / sparse_fold).rename("ratio").reset_index()
    diagnostics = ratio.merge(active_folds.drop(columns=["accuracy"]), on=["dataset", "seed", "model"])
    diagnostics["duplication"] = diagnostics.condition_terms / diagnostics.unique_conditions.replace(0, np.nan)
    for feature in ("duplication", "target_leaves", "features", "encoded_features"):
        subset = diagnostics[["ratio", feature]].dropna().groupby(diagnostics.dataset).median()
        summary["spearman_" + feature] = float(spearmanr(subset.ratio, subset[feature]).statistic)
    cases = ratio.groupby("dataset").ratio.median().sort_values()
    summary["slowest_datasets"] = {dataset_info[str(k)]["name"]: v for k, v in cases.head(4).items()}
    summary["fastest_datasets"] = {dataset_info[str(k)]["name"]: v for k, v in cases.tail(4).items()}
    directional = stress[stress.directional & stress.method.eq("dense")]
    summary["freeze_rates"] = directional.groupby(["dataset", "freeze"]).valid.mean().unstack().median().to_dict()
    summary["freeze_case"] = {dataset_info[str(k)]["name"]: v for k, v in
                              directional[directional.freeze.eq(.75)].groupby("dataset").valid.mean().items()}
    summary["stress_queries"] = len(stress[stress.method.eq("dense")])
    summary["scale_queries"] = len(scale[scale.method.eq("dense")])
    core = stress[stress.directional & stress.freeze.eq(0) & stress.method.eq("cascade")].copy()
    summary["prefix_retention"] = (core.work / core.target_leaves).groupby(
        [core.dataset, core.fraction]).median().unstack().median().to_dict()
    cw = stress[~stress.directional & stress.method.eq("dense")].copy()
    summary["category_cost"] = cw.groupby(["dataset", "categorical_cost"]).cost.median().unstack().median().to_dict()
    s = scale[scale.fraction.eq(.25)].groupby(["features", "depth", "method"]).latency_us.median().unstack()
    s["ratio"] = s.dense / s.interned
    s.to_csv(args.scale / "artifacts" / "scale_summary.csv")
    summary["scale_corner"] = s.loc[[(8, 3), (8, 12), (256, 3), (256, 12)]].reset_index().to_dict("records")
    (args.main / "artifacts" / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latency_plot(ratios[:4], out)
    scale_plot(scale, out)
    pruning_plot(stress, out)
    quality_plot(raw, out)
    with (out / "evidence.tex").open("w", encoding="utf-8") as stream:
        def write(name, value):
            stream.write(f"\\newcommand{{\\{name}}}{{{value}}}\n")

        write("DatasetCount", summary["datasets"])
        write("FoldCount", summary["folds"])
        write("ActiveFoldCount", summary["active_folds"])
        write("MainQueryCount", f"{summary['queries']:,}")
        write("StressQueryCount", f"{summary['stress_queries']:,}")
        write("ScaleQueryCount", f"{summary['scale_queries']:,}")
        write("GridCaseCount", f"{summary['exhaustive_cases_per_method']:,}")
        for key, name in [("interned_vs_dense", "Shared"), ("cascade_vs_dense", "Screen"),
                          ("cascade_vs_interned", "ScreenShared"), ("batch_interned_vs_dense", "BatchShared")]:
            lo, lb, ub = summary[key]["effect"]
            write(name + "Ratio", f"{lo:.2f}")
            write(name + "CI", f"{lb:.2f}--{ub:.2f}")
            write(name + "Wins", summary[key]["wins"])
        write("ScreenRetention", f"{100 * summary['cascade_retained']:.1f}")
        write("RandomRetention", f"{100 * summary['random_prefix_retained']:.1f}")
        write("IndexMedian", f"{summary['index_median_ms']:.2f}")
        write("IndexPninetyfive", f"{summary['index_p95_ms']:.2f}")
    print(table.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
