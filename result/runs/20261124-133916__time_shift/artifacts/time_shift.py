"""Rolling time-split evaluation with semantically bounded counterfactual inputs."""

import argparse
import itertools
import json
import traceback
from datetime import datetime
from pathlib import Path

from run_recourse import ROOT, Action, LeafIndex, create_model, digest, distance, encode, load_openml
from run_recourse import save_provenance, take_queries, transform, write_jsonl
from recourse import threshold_on_grid
from sklearn.metrics import balanced_accuracy_score

import numpy as np
import pandas as pd

DEPTHS = (5, 7, 9, 12)
ORIGINS = (.35, .43, .51, .59, .67)
N_QUERIES = 160


def source(data_root, task):
    if task == "electricity":
        original, y, meta = load_openml(data_root, 151)
        frame = original.drop(columns=["date", "nswprice", "vicprice"])
        dates = original["date"].to_numpy()
        order = np.lexsort((original["period"], dates))
        return frame.iloc[order].reset_index(drop=True), y[order], dates[order], meta
    if task == "bank":
        file = data_root / "bank" / "bank-full.csv"
        raw = pd.read_csv(file, sep=";")
        frame = raw.drop(columns=["y", "duration"])
        y = raw.y.eq("yes").to_numpy(dtype=int)
        dates = np.arange(len(y))
        doi = "10.24432/C5K306"
        origin = "provider chronology in bank-full.csv"
    else:
        file = data_root / "bike_sharing" / "hour.csv"
        raw = pd.read_csv(file).sort_values(["dteday", "hr"]).reset_index(drop=True)
        dates = raw["dteday"].to_numpy()
        initial_dates = np.unique(dates)
        earliest = dates < initial_dates[int(len(initial_dates) * ORIGINS[0])]
        threshold = float(raw.loc[earliest, "cnt"].median())
        y = raw.cnt.gt(threshold).to_numpy(dtype=int)
        frame = raw.drop(columns=["dteday", "instant", "yr", "casual", "registered", "cnt"])
        for j in ("season", "mnth", "holiday", "weekday", "workingday", "weathersit"):
            frame[j] = frame[j].astype("category")
        doi = "10.24432/C5W894"
        origin = "hour.csv, UCI dataset via checksum-recorded public mirror"
    meta = {"dataset": task, "origin": origin, "doi": doi, "sha256": digest(file),
            "path": str(file.relative_to(data_root)), "rows": len(y), "features": frame.columns.tolist()}
    if task == "bike":
        meta["target_rule"] = f"hourly rental count > {threshold}"
    return frame, y, dates, meta


def temporal_indices(dates, origin):
    days, group = np.unique(dates, return_inverse=True)
    positions = np.ceil(len(days) * np.array([origin, origin + .06, origin + .13, origin + .20])).astype(int)
    a, b, c, d = np.searchsorted(group, positions)
    assert 0 < a < b < c < d <= len(group)
    return np.arange(a), np.arange(a, b), np.arange(b, c), np.arange(c, d)


def allowed_spec(task, names, info, x, fit, models, levels):
    if task == "bike":
        field = names.index("hr")
        return [field], lambda q: [np.arange(max(0, int(q[field]) - 2), min(23, int(q[field]) + 2) + 1)], None
    if task == "bank":
        channel, calls = names.index("contact"), names.index("campaign")
        values = [i for i, v in enumerate(info[channel]["levels"]) if v in {"v:telephone", "v:cellular"}]
        cap = int(x[fit, calls].max())

        def allowed(q):
            end = max(int(q[calls]), min(cap, int(q[calls]) + 3))
            return [np.unique([q[channel], *values]), np.arange(int(q[calls]), end + 1)]

        return [channel, calls], allowed, None
    field = names.index("transfer")
    encoded_pos = int(np.maximum(1, levels[:field]).sum())
    bounds = [float(x[fit, field].min()), float(x[fit, field].max())]
    cuts = [
        threshold_on_grid(float(cut), right)
        for model in models
        for cut in model.tree_.threshold[model.tree_.feature == encoded_pos]
        for right in (False, True)
    ]
    representatives = np.unique(np.asarray([*bounds, *cuts], dtype=float))

    def allowed(q):
        current = q[field]
        low, high = max(bounds[0], current - .10), min(bounds[1], current + .10)
        within = representatives[(representatives >= low) & (representatives <= high)]
        return [np.unique(np.r_[current, within])]

    return [field], allowed, bounds


def record_fold(task, data_root, origin, seed):
    frame, y, dates, metadata = source(data_root, task)
    train, cal, updated, test = temporal_indices(dates, origin)
    x, levels, scales, info = transform(frame, train)
    enc = encode(x, levels)
    scores = {}
    for depth in DEPTHS:
        candidate, _ = create_model(enc[train], y[train], seed, "CART", depth)
        scores[depth] = balanced_accuracy_score(y[cal], candidate.predict(enc[cal]))
    depth = max(scores, key=scores.get)
    fit = np.r_[train, cal]
    final_fit = np.r_[fit, updated]
    base, _ = create_model(enc[fit], y[fit], seed, "CART", depth)
    future, _ = create_model(enc[final_fit], y[final_fit], seed, "CART", depth)
    rng = np.random.default_rng(seed)
    replicas = []
    for j in range(5):
        sample = rng.choice(fit, len(fit), replace=True)
        model, _ = create_model(enc[sample], y[sample], seed * 10 + j, "CART", depth)
        replicas.append(model)
    fields, allowed, bounds = allowed_spec(task, list(frame.columns), info, x, fit, [base, *replicas], levels)
    frozen = sorted(set(range(x.shape[1])) - set(fields))
    queries = take_queries(base.predict(enc), test, rng, N_QUERIES)
    tree = LeafIndex(base, levels, scales)
    unconstrained = tree.solve(x[queries], Action(), "interned")
    update_negative = future.predict(enc[queries]) == 0
    rows = []
    for i, row in enumerate(queries):
        q = x[row]
        levels_allowed = allowed(q)
        options = list(itertools.product(*levels_allowed))
        choices = np.tile(q, (len(options), 1))
        choices[:, fields] = options
        p = encode(choices, levels)
        valid = base.predict(p) == 1
        votes = sum(model.predict(p) == 1 for model in replicas)
        costs = distance(q, choices, levels, scales)

        def best(mask):
            return choices[np.where(mask, costs, np.inf).argmin()] if mask.any() else None

        def semantic(z):
            if z is None or not np.array_equal(q[frozen], z[frozen]):
                return False
            if bounds is None:
                return all(v in choices for v, choices in zip(z[fields], levels_allowed))
            j = fields[0]
            return z[j] == q[j] or (bounds[0] <= z[j] <= bounds[1] and abs(z[j] - q[j]) <= .10)

        outputs = {
            "free": unconstrained[0][i] if np.isfinite(unconstrained[1][i]) else None,
            "semantic_exact": best(valid),
            "semantic_vote": best(valid & (votes >= 4)),
        }
        for label, z in outputs.items():
            cost = float(distance(q, z, levels, scales)) if z is not None else None
            permitted = bool(semantic(z))
            retained = False
            if z is not None:
                decoded = encode(z[None], levels)
                assert base.predict(decoded)[0] == 1
                if label != "free":
                    assert permitted and cost >= unconstrained[1][i] - 1e-9
                retained = bool(future.predict(decoded)[0] == 1)
            rows.append({
                "task": task, "origin": origin, "seed": seed, "query": int(row), "method": label,
                "found": z is not None, "semantic": permitted, "retained": retained,
                "factual_still_negative": bool(update_negative[i]), "cost": cost,
                "candidates": len(choices),
                "changed_fields": [str(f) for f in frame.columns[z != q]] if z is not None else [],
                "factual_action": q[fields].tolist(),
                "answer_action": z[fields].tolist() if z is not None else None,
            })
    split = {"task": task, "origin": origin, "seed": seed, "train": train.tolist(),
             "calibration": cal.tolist(), "update": updated.tolist(), "test": test.tolist(),
             "queries": queries.tolist(), "validation_score": scores, "selected_depth": depth,
             "last_train": str(dates[train[-1]]), "last_calibration": str(dates[cal[-1]]),
             "last_update": str(dates[updated[-1]]), "first_test": str(dates[test[0]]),
             "last_test": str(dates[test[-1]]), "proportion_target": float(y[test].mean()),
             "model_fields": list(frame.columns), "base_accuracy": balanced_accuracy_score(y[test], base.predict(enc[test])),
             "updated_accuracy": balanced_accuracy_score(y[test], future.predict(enc[test])),
             "transform": info, "action_fields": [str(frame.columns[j]) for j in fields]}
    return metadata, split, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    run = args.out or ROOT / "result" / "runs" / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "__time_shift")
    save_provenance(run, args, ("script/time_shift.py", "docs/experiment_plan_round2.md"))
    meta = {}
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        try:
            for task in ("electricity", "bank", "bike"):
                for origin, seed in zip(ORIGINS, (7, 13, 19, 29, 37)):
                    info, split, rows = record_fold(task, args.data_root, origin, seed)
                    meta[task] = info
                    write_jsonl(run / "metrics.jsonl", rows)
                    write_jsonl(run / "artifacts" / "splits.jsonl", [split])
                    progress = f"{task} origin {origin:.2f} {len(rows) // 3} queries"
                    print(progress, flush=True)
                    print(progress, file=log, flush=True)
        except Exception:
            traceback.print_exc(file=log)
            raise
    (run / "artifacts" / "datasets.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
