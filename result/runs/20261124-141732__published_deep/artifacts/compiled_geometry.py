"""Volumetric heap search using the same compiled execution backend as enumeration."""

import heapq

import numba as nb
import numpy as np

from region_search import RegionSearch


@nb.njit
def box_distance(point, lo, hi, eligible, numeric, categorical, offsets, scales, rules, category_cost):
    distance = 0.
    for i in range(len(numeric)):
        j = numeric[i]
        x = point[j]
        z = min(max(x, lo[i]), hi[i])
        delta = z - x
        if (lo[i] > hi[i] or (rules[j] & 1 and delta != 0.)
                or (rules[j] & 2 and delta < 0.) or (rules[j] & 4 and delta > 0.)):
            return np.inf
        distance += abs(delta) / scales[j]
    for i in range(len(categorical)):
        j = categorical[i]
        if eligible[offsets[i] + int(point[j])]:
            continue
        if rules[j] & 1 or not eligible[offsets[i]:offsets[i + 1]].any():
            return np.inf
        distance += category_cost
    return distance


@nb.njit
def search(points, left, right, starts, rows, box_lo, box_hi, box_eligible,
           leaf_lo, leaf_hi, leaf_eligible, numeric, categorical, offsets, scales, rules, category_cost):
    chosen = np.full(len(points), -1, dtype=np.int64)
    costs = np.full(len(points), np.inf)
    work = np.zeros(len(points), np.int64)
    for i in range(len(points)):
        if len(left) == 0:
            break
        q = points[i]
        queue = [(0., np.int64(0))]
        while len(queue):
            bound, node = heapq.heappop(queue)
            best = costs[i]
            if bound > best + 1e-12 * (1. + best):
                break
            if left[node] < 0:
                for slot in range(starts[node], starts[node + 1]):
                    k = rows[slot]
                    d = box_distance(q, leaf_lo[k], leaf_hi[k], leaf_eligible[k], numeric,
                                     categorical, offsets, scales, rules, category_cost)
                    work[i] += 1
                    if d < costs[i]:
                        costs[i], chosen[i] = d, k
            else:
                for child in (left[node], right[node]):
                    d = box_distance(q, box_lo[child], box_hi[child], box_eligible[child],
                                     numeric, categorical, offsets, scales, rules, category_cost)
                    if np.isfinite(d) and d <= best + 1e-12 * (1. + best):
                        heapq.heappush(queue, (d, child))
    return chosen, costs, work


class CompiledRegion:
    def __init__(self, index, train, bucket=8):
        numeric_fit = train[:, index.numeric]
        limits = (numeric_fit.min(axis=0), numeric_fit.max(axis=0))
        region = RegionSearch(index, bucket, limits)
        children = [(-1, -1) if value is None else value for value in region.children]
        self.left, self.right = np.asarray(children, dtype=np.int64).reshape(-1, 2).T.copy()
        self.starts = np.zeros(len(region.rows) + 1, dtype=np.int64)
        self.starts[1:] = np.cumsum([0 if ids is None else len(ids) for ids in region.rows])
        self.rows = np.asarray([int(j) for ids in region.rows if ids is not None for j in ids], dtype=np.int64)
        leaf_shape = (len(index.nodes), len(index.numeric))
        box_shape = (len(region.rows), len(index.numeric))
        self.lo, self.hi = region.lo.reshape(box_shape), region.hi.reshape(box_shape)
        self.offsets = np.r_[0, np.cumsum(index.levels[index.categorical])]
        self.box_categories = self.join_categories(region.allowed, len(region.rows))
        self.leaf_categories = self.join_categories(index.allowed, leaf_shape[0])
        self.index = index

    def join_categories(self, allowed, n_rows):
        if not allowed:
            return np.empty((n_rows, 0), dtype=bool)
        if not n_rows:
            return np.empty((0, self.offsets[-1]), dtype=bool)
        return np.concatenate(list(allowed.values()), axis=1)

    def solve(self, x, action):
        index = self.index
        rules = np.zeros(index.n_features, dtype=np.uint8)
        rules[list(action.frozen)] |= 1
        rules[list(action.increasing)] |= 2
        rules[list(action.decreasing)] |= 4
        leaf, costs, work = search(x, self.left, self.right, self.starts, self.rows,
                                   self.lo, self.hi, self.box_categories, index.low, index.high,
                                   self.leaf_categories, index.numeric, index.categorical,
                                   self.offsets, index.scales, rules, action.categorical_cost)
        z = np.full_like(x, np.nan)
        for i in np.flatnonzero(np.isfinite(costs)):
            z[i] = index.materialize(x[i], leaf[i])
        return z, costs, work
