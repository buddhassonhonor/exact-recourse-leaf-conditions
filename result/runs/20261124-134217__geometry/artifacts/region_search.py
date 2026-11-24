"""Exact best-first search of bounding boxes over mixed-feature target leaves."""

import heapq

import numpy as np


class RegionSearch:
    """Single-tree adaptation of the volumetric index in Counterfactual Maps."""

    def __init__(self, index, bucket=8):
        self.index = index
        self.children, self.rows, self.lo, self.hi = [], [], [], []
        self.allowed = {j: [] for j in index.categorical}
        low = np.maximum(index.low, 0.)
        high = np.minimum(index.high, 1.)
        centers = [(low + high) / 2]
        centers.extend(index.allowed[j].astype(float) for j in index.categorical)
        self.build(np.arange(len(index.nodes)), np.concatenate(centers, axis=1), bucket)
        self.lo = np.asarray(self.lo)
        self.hi = np.asarray(self.hi)
        self.allowed = {j: np.asarray(values) for j, values in self.allowed.items()}
        self.numeric_scale = index.scales[index.numeric]

    def build(self, rows, centers, bucket):
        if not len(rows):
            return -1
        i = len(self.rows)
        index = self.index
        self.rows.append(rows if len(rows) <= bucket else None)
        self.children.append(None)
        self.lo.append(index.low[rows].min(axis=0))
        self.hi.append(index.high[rows].max(axis=0))
        for j in self.allowed:
            self.allowed[j].append(index.allowed[j][rows].any(axis=0))
        if len(rows) > bucket:
            split_field = np.ptp(centers[rows], axis=0).argmax()
            sorted_rows = rows[np.argsort(centers[rows, split_field], kind="stable")]
            middle = len(rows) // 2
            self.children[i] = (self.build(sorted_rows[:middle], centers, bucket),
                                self.build(sorted_rows[middle:], centers, bucket))
        return i

    def structure_bytes(self):
        return (self.lo.nbytes + self.hi.nbytes + sum(x.nbytes for x in self.allowed.values())
                + sum(x.nbytes for x in self.rows if x is not None))

    def project(self, q, values, costs, action, rows, node):
        idx = self.index
        lo, hi = (self.lo[rows], self.hi[rows]) if node else (idx.low[rows], idx.high[rows])
        delta = np.clip(values, lo, hi) - values
        numeric = (abs(delta) / self.numeric_scale).sum(axis=-1)
        fixed, up, down = costs
        impossible = ((lo > hi) | (fixed & (delta != 0)) | (up & (delta < 0)) | (down & (delta > 0)))
        numeric[impossible.any(axis=1)] = np.inf
        for j, masks in (self.allowed if node else idx.allowed).items():
            allowed = masks[rows]
            legal = allowed[:, int(q[j])]
            numeric += np.where(legal, 0., np.where(
                allowed.any(axis=1) & (j not in action.frozen), action.categorical_cost, np.inf))
        return numeric

    def solve(self, q, action):
        if not self.rows:
            return None, np.inf, 0, 0
        idx = self.index
        values = q[idx.numeric]
        flags = tuple(np.isin(idx.numeric, indices) for indices in
                      (action.frozen, action.increasing, action.decreasing))
        best, selected, seen, visited = np.inf, None, 0, 0
        queue = [(0., 0)]
        while queue:
            bound, node = heapq.heappop(queue)
            if bound > best + 1e-12 * (1. + best):
                break
            visited += 1
            rows = self.rows[node]
            if rows is None:
                children = list(self.children[node])
                for value, child in zip(self.project(q, values, flags, action, children, True), children):
                    if np.isfinite(value) and value <= best + 1e-12 * (1. + best):
                        heapq.heappush(queue, (value, child))
                continue
            scores = self.project(q, values, flags, action, rows, False)
            seen += len(rows)
            winner = scores.argmin()
            if scores[winner] < best:
                best, selected = float(scores[winner]), int(rows[winner])
        return (idx.materialize(q, selected) if selected is not None else None, best, seen, visited)
