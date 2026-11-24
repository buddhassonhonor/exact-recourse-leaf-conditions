"""Independent finite-grid MILP using unit flow through the fitted binary tree."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from recourse import threshold_on_grid


@dataclass
class NodeFlow:
    levels: np.ndarray
    splits: list
    edges: list
    numeric_cells: list
    node_limits: np.ndarray

    @classmethod
    def from_tree(cls, model, levels):
        pointers = [(j, code) for j, size in enumerate(levels) for code in range(max(1, size))]
        tree = model.tree_
        levels_by_field = [set() for _ in levels]
        upper = np.ones(tree.node_count)
        splits = []
        for node, encoded_id in enumerate(tree.feature):
            if encoded_id < 0:
                if model.classes_[tree.value[node, 0].argmax()] != 1:
                    upper[node] = 0.
                continue
            field, category = pointers[encoded_id]
            cut = float(tree.threshold[node])
            if levels[field] == 0:
                levels_by_field[field].update([threshold_on_grid(cut, False), threshold_on_grid(cut, True)])
            splits.append((node, int(tree.children_left[node]), int(tree.children_right[node]), field, category, cut))
        cells = [np.array(sorted(cuts)) for cuts in levels_by_field]
        edges = [(node, left, right) for node, left, right, *_ in splits]
        return cls(levels, splits, edges, cells, upper)

    def choices(self, q, a):
        options = []
        for field, size in enumerate(self.levels):
            x = q[field]
            if field in a.frozen:
                values = np.array([x])
            elif size:
                values = np.arange(size)
            else:
                values = np.unique(np.r_[self.numeric_cells[field], x])
                if field in a.increasing:
                    values = values[values >= x]
                if field in a.decreasing:
                    values = values[values <= x]
                values = values[(values >= 0.) & (values <= 1.)]
            options.append(values)
        return options

    def solve(self, q, a, scales, time_limit=2.):
        options = self.choices(q, a)
        n_nodes = len(self.node_limits)
        offsets = np.r_[n_nodes, n_nodes + np.cumsum([len(items) for items in options])]
        n_vars = int(offsets[-1])
        row_ids, columns, entries, low, high = [], [], [], [], []

        def row(ids, weights, lower, upper):
            row_ids.extend([len(low)] * len(ids))
            columns.extend(ids)
            entries.extend(weights)
            low.append(lower)
            high.append(upper)

        row([0], [1.], 1., 1.)
        for node, left, right in self.edges:
            row([node, left, right], [-1., 1., 1.], 0., 0.)
        objective = np.zeros(n_vars)
        for j, values in enumerate(options):
            left, right = offsets[j:j + 2]
            row(range(left, right), [1.] * len(values), 1., 1.)
            objective[left:right] = (a.categorical_cost * (values != q[j]) if self.levels[j]
                                     else abs(values - q[j]) / scales[j])
        for _, left, right, j, category, cut in self.splits:
            values = options[j]
            test = values == category if self.levels[j] else values
            route_left = test <= cut
            for child, eligible in ((left, route_left), (right, ~route_left)):
                allowed = offsets[j] + np.flatnonzero(eligible)
                row([child, *allowed], [1., *([-1.] * len(allowed))], -np.inf, 0.)
        matrix = coo_matrix((entries, (row_ids, columns)), shape=(len(low), n_vars)).tocsc()
        integrality = np.r_[np.zeros(n_nodes), np.ones(n_vars - n_nodes)]
        upper = np.r_[self.node_limits, np.ones(n_vars - n_nodes)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected")
            answer = milp(objective * 1e6, integrality=integrality, bounds=Bounds(0., upper),
                          constraints=LinearConstraint(matrix, low, high),
                          options={"time_limit": time_limit, "mip_rel_gap": 0., "threads": 1})
        states = {0: "OPTIMAL", 1: "TIME_LIMIT", 2: "INFEASIBLE"}
        status = states.get(answer.status, f"HiGHS status {answer.status}")
        if answer.status != 0:
            return None, status
        selected = [values[answer.x[offsets[j]:offsets[j + 1]].argmax()] for j, values in enumerate(options)]
        return np.asarray(selected, dtype=float), status
