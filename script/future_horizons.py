"""Frozen early models, delayed updates, and chronological external validation."""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

from run_recourse import ROOT, Action, LeafIndex, create_model, digest, distance, encode
from run_recourse import transform, save_provenance, take_queries, write_jsonl
from time_shift import DEPTHS, allowed_spec, source
from sklearn.metrics import balanced_accuracy_score

import numpy as np
import pandas as pd

TASKS = ("electricity", "bank", "bike", "seoul")
ORIGINS = (.35, .43)
LEADS = (.05, .15, .25, .35)
THRESHOLDS = (0, 1, 2, 3, 4, 5)


def external_source(data_root, task):
    if task != "seoul":
        return source(data_root, task)
    file = data_root / "seoul_bike_sharing" / "SeoulBikeData.csv"
    raw = pd.read_csv(file, encoding="latin-1")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%d/%m/%Y")
    df = raw.loc[raw["Functioning Day"].eq("Yes")].sort_values(["Date", "Hour"]).reset_index(drop=True)
    dates = df.Date.to_numpy()
    day_order = np.unique(dates)
    early = df.Date.lt(day_order[int(.35 * len(day_order))])
    threshold = float(df.loc[early, "Rented Bike Count"].median())
    y = df["Rented Bike Count"].gt(threshold).to_numpy(np.int64)
    x = df.drop(columns=["Date", "Rented Bike Count", "Functioning Day"]).rename(columns={"Hour": "hr"})
    info = json.loads((file.parent / "source.json").read_text(encoding="utf-8"))
    assert digest(file) == info["sha256"]
    info.update(dataset="seoul", raw_rows=len(raw), rows=len(df), features=list(x),
                target_rule=f"operating-hour rental count greater than {threshold}",
                path=str(file.relative_to(data_root)))
    return x, y, dates, info


def fit(encoded, y, indices, seed, depth):
    return create_model(encoded[indices], y[indices], seed, "CART", depth)[0]


def replicas(encoded, y, indices, seed, depth):
    rng = np.random.default_rng(seed + 542)
    return [fit(encoded, y, rng.choice(indices, len(indices), replace=True), seed + j, depth)
            for j in range(5)]


def action_domain(task, frame, transform_info, x, index, levels, base, ensemble):
    key = "bike" if task == "seoul" else task
    fields, allowed, bounds = allowed_spec(key, list(frame), transform_info, x, index, [base, *ensemble], levels)
    frozen = [j for j in range(x.shape[1]) if j not in fields]
    return fields, frozen, allowed, bounds


def use_schedule(restrictions, day, schedule):
    if schedule is None:
        return restrictions
    fields, frozen, allowed, bounds = restrictions

    def operating_hours(q):
        return [np.intersect1d(allowed(q)[0], schedule[day], assume_unique=True)]

    return fields, frozen, operating_hours, bounds


def semantic_answer(q, z, restrictions):
    fields, frozen, allowed, bounds = restrictions
    if z is None or not np.array_equal(q[frozen], z[frozen]):
        return False
    if bounds is not None:
        j = fields[0]
        return z[j] == q[j] or (bounds[0] <= z[j] <= bounds[1] and abs(z[j] - q[j]) <= .10)
    return all(v in pool for v, pool in zip(z[fields], allowed(q)))


def costs_by_vote(q, restrictions, base, ensemble, levels, scales):
    fields, _, allowed, _ = restrictions
    edits = list(itertools.product(*allowed(q)))
    values = np.tile(q, (len(edits), 1))
    values[:, fields] = edits
    encoded = encode(values, levels)
    positive = base.predict(encoded) == 1
    vote = sum(tree.predict(encoded) == 1 for tree in ensemble)
    cost = distance(q, values, levels, scales)
    result = {}
    for lim in THRESHOLDS:
        ok = positive & (vote >= lim)
        if ok.any():
            best = np.where(ok, cost, np.inf).argmin()
            result[lim] = (values[best], float(cost[best]), int(vote[best]))
        else:
            result[lim] = (None, None, None)
    return result


def choose_votes(x, encoded, y, levels, scales, frame, early, calibration, seed, depth, task, info,
                 dates, schedule):
    cal_first, cal_last = np.array_split(calibration, 2)
    model = fit(encoded, y, early, seed, depth)
    proxy = fit(encoded, y, np.r_[early, cal_first], seed, depth)
    ensemble = replicas(encoded, y, early, seed, depth)
    restrictions = action_domain(task, frame, info, x, early, levels, model, ensemble)
    queries = take_queries(model.predict(encoded), cal_last, np.random.default_rng(seed + 311), 128)
    desired = dict.fromkeys(THRESHOLDS, 0)
    for i in queries:
        if proxy.predict(encoded[i:i + 1])[0] != 0:
            continue
        current = use_schedule(restrictions, dates[i], schedule)
        for threshold, (edited, _, _) in costs_by_vote(x[i], current, model, ensemble, levels, scales).items():
            if edited is not None and proxy.predict(encode(edited[None], levels))[0] == 1:
                desired[threshold] += 1
    limit = max(desired, key=lambda threshold: (desired[threshold], -threshold))
    return limit, desired, queries


def by_day(dates, fractions):
    days, group = np.unique(dates, return_inverse=True)
    assert np.all(group[:-1] <= group[1:])
    boundaries = np.ceil(len(days) * np.asarray(fractions)).astype(int)
    return np.searchsorted(group, boundaries)


def compare_origin(frame, y, dates, task, origin, seed):
    a, b = by_day(dates, [origin, origin + .06])
    early, calibration, base_indices = np.arange(a), np.arange(a, b), np.arange(b)
    x, levels, scales, info = transform(frame, early)
    encoded = encode(x, levels)
    schedule = None
    if task == "seoul":
        schedule = pd.DataFrame({"day": dates, "hour": frame.hr}).groupby("day").hour.unique().to_dict()
        schedule = {np.datetime64(key, "ns"): value for key, value in schedule.items()}
    accuracy = {}
    for depth in DEPTHS:
        model = fit(encoded, y, early, seed, depth)
        accuracy[depth] = float(balanced_accuracy_score(y[calibration], model.predict(encoded[calibration])))
    depth = max(accuracy, key=accuracy.get)
    limit, early_use, training_queries = choose_votes(
        x, encoded, y, levels, scales, frame, early, calibration, seed, depth, task, info, dates, schedule)
    base = fit(encoded, y, base_indices, seed, depth)
    ensemble = replicas(encoded, y, base_indices, seed, depth)
    restrictions = action_domain(task, frame, info, x, base_indices, levels, base, ensemble)
    region = LeafIndex(base, levels, scales)
    base_predict = base.predict(encoded)
    for lead in LEADS:
        c, d = by_day(dates, [origin + .06 + lead, origin + .06 + lead + .07])
        assert b < c < d
        update, test = np.arange(c), np.arange(c, d)
        next_model = fit(encoded, y, update, seed, depth)
        next_predict = next_model.predict(encoded[test])
        query_ids = take_queries(base_predict, test, np.random.default_rng(seed), 192)
        early_cf, early_cost, _ = region.solve(x[query_ids], Action(), "interned")
        rows = []
        for i, query in enumerate(query_ids):
            q = x[query]
            current = use_schedule(restrictions, dates[query], schedule)
            candidates = costs_by_vote(q, current, base, ensemble, levels, scales)
            free = ((early_cf[i], float(early_cost[i]), None)
                    if np.isfinite(early_cost[i]) else (None, None, None))
            factual_negative = next_predict[query - c] == 0
            for policy, item in [("free", free), ("constrained", candidates[0]),
                                 ("fixed_four", candidates[4]), ("calibrated", candidates[limit])]:
                z, cost, votes = item
                admissible = semantic_answer(q, z, current)
                retained = False
                if z is not None:
                    enc = encode(z[None], levels)
                    assert base.predict(enc)[0] == 1
                    if policy != "free":
                        assert admissible and cost >= early_cost[i] - 1e-9
                    retained = bool(next_model.predict(enc)[0] == 1)
                rows.append(dict(task=task, origin=origin, lead=lead, query=int(query), policy=policy,
                                 found=z is not None, admissible=bool(admissible), retained=retained,
                                 originally_negative=bool(factual_negative), cost=cost, votes=votes,
                                 usable=bool(admissible and retained and factual_negative)))
        configuration = dict(
            task=task, origin=origin, seed=seed, lead=lead, depth=depth, vote_cutoff=limit,
            calibration_effect=early_use, calibration_queries=training_queries.tolist(),
            train=[0, int(a)], calibration=[int(a), int(b)], updated=[0, int(c)],
            test=[int(c), int(d)], test_dates=[str(dates[c]), str(dates[d - 1])],
            calibration_accuracy=accuracy, base_accuracy=float(balanced_accuracy_score(y[test], base_predict[test])),
            updated_accuracy=float(balanced_accuracy_score(y[test], next_predict)),
            target_prevalence=float(y[test].mean()), queries=query_ids.tolist(),
        )
        yield configuration, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/data"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    parser.set_defaults(workers=1)
    args = parser.parse_args()
    name = "long_horizons_pilot" if args.pilot else "long_horizons"
    run = args.out or ROOT / "result" / "runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "__" + name)
    save_provenance(run, args, ("script/future_horizons.py", "script/time_shift.py",
                               "docs/experiment_plan_round3.md"))
    data = {}
    with (run / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        for task in ("seoul",) if args.pilot else TASKS:
            frame, y, dates, meta = external_source(args.data_root, task)
            data[task] = meta
            for origin, seed in zip(ORIGINS[:1], (7,)) if args.pilot else zip(ORIGINS, (7, 13)):
                for config, rows in compare_origin(frame, y, dates, task, origin, seed):
                    write_jsonl(run / "metrics.jsonl", rows)
                    write_jsonl(run / "artifacts" / "splits.jsonl", [config])
                    msg = f"{task} early {origin:.2f}, lead {config['lead']:.2f}, {len(rows)//4} queries"
                    print(msg, flush=True)
                    print(msg, file=log, flush=True)
    (run / "artifacts" / "datasets.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
