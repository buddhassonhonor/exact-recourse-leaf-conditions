"""Exact mixed-feature counterfactual search for a single binary tree."""

import heapq
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Action:
    frozen: tuple[int, ...] = ()
    increasing: tuple[int, ...] = ()
    decreasing: tuple[int, ...] = ()
    categorical_cost: float = 1.0


def encode(x, levels):
    """Encode one original categorical field with one indicator per level."""
    columns = [
        x[:, j:j + 1] if k == 0 else x[:, j, None] == np.arange(k)
        for j, k in enumerate(levels)
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


def threshold_on_grid(value, right):
    rounded = np.float32(value)
    stored = float(rounded)
    if (right and stored <= value) or (not right and stored > value):
        direction = np.float32(np.inf if right else -np.inf)
        return float(np.nextafter(rounded, direction))
    return float(rounded)


class LeafIndex:
    def __init__(self, model, levels, scales):
        self.model = model
        self.levels = np.asarray(levels)
        self.scales = np.asarray(scales, dtype=float)
        self.n_features = len(levels)
        self.starts = np.r_[0, np.cumsum(np.maximum(1, levels))]
        self.numeric = np.flatnonzero(self.levels == 0)
        self.categorical = np.flatnonzero(self.levels > 0)
        tree = model.tree_
        self.nodes, self.left, self.right = [], tree.children_left, tree.children_right
        width = self.starts[-1]
        all_lower, all_upper = {}, {}
        stack = [(0, np.full(width, -np.inf), np.full(width, np.inf))]
        while stack:
            node, lower, upper = stack.pop()
            if self.left[node] == self.right[node]:
                if model.classes_[tree.value[node, 0].argmax()] == 1:
                    self.nodes.append(node)
            else:
                j, cut = tree.feature[node], tree.threshold[node]
                lower_right, upper_left = lower.copy(), upper.copy()
                lower_right[j] = max(lower[j], threshold_on_grid(cut, True))
                upper_left[j] = min(upper[j], threshold_on_grid(cut, False))
                stack.append((self.right[node], lower_right, upper))
                stack.append((self.left[node], lower, upper_left))
            all_lower[node], all_upper[node] = lower, upper

        self.nodes = np.asarray(self.nodes, dtype=int)
        self._boxes(all_lower, all_upper)
        self._compile_conditions()

    def _boxes(self, lower, upper):
        nodes = sorted(lower)
        low = np.stack([lower[i] for i in nodes])
        high = np.stack([upper[i] for i in nodes])
        columns = self.starts[self.numeric]
        self.node_low, self.node_high = low[:, columns], high[:, columns]
        self.low, self.high = self.node_low[self.nodes], self.node_high[self.nodes]
        self.node_allowed, self.allowed = {}, {}
        for j in self.categorical:
            block = slice(self.starts[j], self.starts[j + 1])
            zero = (low[:, block] <= 0) & (high[:, block] >= 0)
            one = (low[:, block] <= 1) & (high[:, block] >= 1)
            other = np.count_nonzero(~zero, axis=1, keepdims=True) - (~zero)
            self.node_allowed[j] = one & (other == 0)
            self.allowed[j] = self.node_allowed[j][self.nodes]

    def _compile_conditions(self):
        n_leaves = len(self.nodes)
        self.codes = np.full((n_leaves, self.n_features), -1, dtype=int)
        feature_ids, lower, upper, admissible = [], [], [], []
        max_levels = int(max(self.levels.max(), 1))
        for column, j in enumerate(self.numeric):
            pairs = np.column_stack([self.low[:, column], self.high[:, column]])
            unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
            restricted = np.any(np.isfinite(unique), axis=1)
            ids = np.full(len(unique), -1, dtype=int)
            ids[restricted] = np.arange(len(feature_ids), len(feature_ids) + restricted.sum())
            self.codes[:, j] = ids[inverse]
            for lo, hi in unique[restricted]:
                feature_ids.append(j)
                lower.append(lo)
                upper.append(hi)
                admissible.append(np.ones(max_levels, dtype=bool))
        self.n_numeric_conditions = len(feature_ids)
        for j in self.categorical:
            unique, inverse = np.unique(self.allowed[j], axis=0, return_inverse=True)
            restricted = ~unique.all(axis=1)
            ids = np.full(len(unique), -1, dtype=int)
            ids[restricted] = np.arange(len(feature_ids), len(feature_ids) + restricted.sum())
            self.codes[:, j] = ids[inverse]
            for row in unique[restricted]:
                feature_ids.append(j)
                lower.append(-np.inf)
                upper.append(np.inf)
                admissible.append(np.pad(row, (0, max_levels - len(row))))
        self.features = np.asarray(feature_ids, dtype=int)
        self.condition_low, self.condition_high = np.asarray(lower), np.asarray(upper)
        self.condition_allowed = np.asarray(admissible, dtype=bool).reshape(-1, max_levels)
        self.feature_order = np.argsort(-np.count_nonzero(self.codes >= 0, axis=0), kind="stable")
        self.feature_order = self.feature_order[np.any(self.codes[:, self.feature_order] >= 0, axis=0)]
        self.matrix = self._matrix(self.codes)
        self.cache = {}

    def _matrix(self, codes):
        row, col = np.where(codes >= 0)
        return csr_matrix((np.ones(len(row)), (row, codes[row, col])),
                          shape=(len(codes), len(self.features)))

    def stages(self, fraction, random_order=None):
        key = (fraction, random_order)
        if key not in self.cache:
            order = self.feature_order
            if random_order is not None:
                order = np.random.default_rng(random_order).permutation(order)
            k = max(1, int(np.ceil(fraction * len(order))))
            prefix, suffix = order[:k], order[k:]
            self.cache[key] = (self._matrix(self.codes[:, prefix]),
                               self._matrix(self.codes[:, suffix]))
        return self.cache[key]

    def constraint_costs(self, x, action):
        points = x[:, self.features]
        minimum, maximum = self.condition_low, self.condition_high
        frozen = np.isin(self.features, action.frozen)
        target = np.clip(points, minimum, maximum)
        illegal = ((frozen & (target != points))
                   | (np.isin(self.features, action.increasing) & (target < points))
                   | (np.isin(self.features, action.decreasing) & (target > points))
                   | (minimum > maximum))
        cost = np.where(illegal, np.inf, abs(target - points) / self.scales[self.features])
        cat = slice(self.n_numeric_conditions, None)
        cat_points = points[:, cat].astype(int)
        cats = self.condition_allowed[cat]
        keep = cats[np.arange(len(cats)), cat_points]
        cost[:, cat] = np.where(keep, 0., np.where(
            cats.any(axis=1) & ~frozen[cat], action.categorical_cost, np.inf))
        return cost

    def dense_costs(self, x, action, rows, internal=False):
        lower = (self.node_low if internal else self.low)[rows]
        upper = (self.node_high if internal else self.high)[rows]
        points = x[:, None, self.numeric]
        targets = np.clip(points, lower, upper)
        blocked = ((np.isin(self.numeric, action.frozen) & (targets != points))
                   | (np.isin(self.numeric, action.increasing) & (targets < points))
                   | (np.isin(self.numeric, action.decreasing) & (targets > points))
                   | (lower > upper))
        with np.errstate(invalid="ignore"):
            cost = np.where(blocked, np.inf, abs(targets - points) / self.scales[self.numeric]).sum(axis=-1)
        allowed = self.node_allowed if internal else self.allowed
        for j in self.categorical:
            options = allowed[j][rows]
            keep = options[:, x[:, j].astype(int)].T
            replacement = np.where(options.any(axis=1) & (j not in action.frozen),
                                   action.categorical_cost, np.inf)
            cost += np.where(keep, 0., replacement)
        return cost

    def materialize(self, x, row):
        z = x.copy()
        z[self.numeric] = np.clip(x[self.numeric], self.low[row], self.high[row])
        for j, allowed in self.allowed.items():
            if not allowed[row, int(x[j])]:
                z[j] = np.argmax(allowed[row])
        return z

    def solve(self, x, action=Action(), method="cascade", fraction=.25, seed=7):
        count = len(self.nodes)
        if not count:
            return np.full_like(x, np.nan), np.full(len(x), np.inf), np.zeros(len(x), int)
        if method == "dense":
            values = self.dense_costs(x, action, np.arange(count))
            picked = values.argmin(axis=1)
            costs = values[np.arange(len(x)), picked]
            survivors = np.full(len(x), count)
        elif method == "heap":
            picked, costs, survivors = map(np.asarray, zip(*(self._heap(q, action) for q in x)))
        else:
            conditions = self.constraint_costs(x, action)
            if method == "interned":
                values = (self.matrix @ conditions.T).T
                picked = values.argmin(axis=1)
                costs = values[np.arange(len(x)), picked]
                survivors = np.full(len(x), count)
            else:
                first, rest = self.stages(fraction, seed if method == "random_prefix" else None)
                partial = (first @ conditions.T).T
                initial = partial.argmin(axis=1)
                incumbent = partial[np.arange(len(x)), initial] + np.array([
                    (rest[row] @ v)[0] for row, v in zip(initial, conditions)
                ])
                eligible = ((partial <= incumbent[:, None] + 1e-12 * (1 + incumbent[:, None]))
                            & np.isfinite(partial))
                survivors = np.count_nonzero(eligible, axis=1)
                picked = np.zeros(len(x), int)
                costs = np.full(len(x), np.inf)
                for i, ids in enumerate(eligible):
                    sub = np.flatnonzero(ids)
                    if len(sub):
                        values = partial[i, sub] + rest[sub] @ conditions[i]
                        best = values.argmin()
                        picked[i], costs[i] = sub[best], values[best]
        z = np.full_like(x, np.nan)
        for i in np.flatnonzero(np.isfinite(costs)):
            z[i] = self.materialize(x[i], picked[i])
        return z, costs, survivors

    def _heap(self, x, action):
        queue = [(0., 0)]
        target = {node: row for row, node in enumerate(self.nodes)}
        visited = 0
        while queue:
            cost, node = heapq.heappop(queue)
            visited += 1
            if node in target:
                return target[node], cost, visited
            if self.left[node] == self.right[node]:
                continue
            children = [self.left[node], self.right[node]]
            bounds = self.dense_costs(x[None], action, children, internal=True)[0]
            for value, child in zip(bounds, children):
                if np.isfinite(value):
                    heapq.heappush(queue, (value, child))
        return 0, np.inf, visited


def distance(x, z, levels, scales, categorical_cost=1.):
    columns = np.asarray(levels) > 0
    return ((abs(x[..., ~columns] - z[..., ~columns]) / scales[~columns]).sum(axis=-1)
            + categorical_cost * (x[..., columns] != z[..., columns]).sum(axis=-1))
