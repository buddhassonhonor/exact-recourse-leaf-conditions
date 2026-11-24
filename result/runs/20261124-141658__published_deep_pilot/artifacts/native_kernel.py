"""Compiled shared enumeration and allocation-free suffix screening."""

import numba as nb
import numpy as np


@nb.njit
def condition_cost(query, feature, lower, upper, allowed, n_numeric, scales, rules, category_cost):
    score = np.empty(len(feature), np.float64)
    for j in range(len(feature)):
        field = feature[j]
        factual = query[field]
        if j < n_numeric:
            projected = min(max(factual, lower[j]), upper[j])
            move = projected - factual
            forbidden = (lower[j] > upper[j] or
                         (rules[field] & 1 and move != 0.) or
                         (rules[field] & 2 and move < 0.) or
                         (rules[field] & 4 and move > 0.))
            score[j] = np.inf if forbidden else abs(move) / scales[field]
        elif allowed[j, int(factual)]:
            score[j] = 0.
        else:
            score[j] = category_cost if not (rules[field] & 1) and allowed[j].any() else np.inf
    return score


@nb.njit
def sparse_sums(score, indices, rows):
    v = np.zeros(len(rows) - 1)
    for leaf in range(len(v)):
        for i in range(rows[leaf], rows[leaf + 1]):
            v[leaf] += score[indices[i]]
    return v


@nb.njit
def screened_leaf(score, prefix_indices, prefix_rows, suffix_indices, suffix_rows):
    bounds = sparse_sums(score, prefix_indices, prefix_rows)
    initial = bounds.argmin()
    incumbent = initial
    best = bounds[initial]
    attempts = len(prefix_indices)
    for i in range(suffix_rows[incumbent], suffix_rows[incumbent + 1]):
        best += score[suffix_indices[i]]
        attempts += 1
    for leaf in range(len(bounds)):
        v = bounds[leaf]
        if leaf == initial or not np.isfinite(v) or v > best + 1e-12 * (1. + best):
            continue
        for i in range(suffix_rows[leaf], suffix_rows[leaf + 1]):
            v += score[suffix_indices[i]]
            attempts += 1
            if not np.isfinite(v) or v > best + 1e-12 * (1. + best):
                break
        if v < best:
            incumbent, best = leaf, v
    return incumbent, best, attempts


@nb.njit
def solve_batch(x, features, lower, upper, allowed, n_numeric, scales, rules, category_cost,
                indices, rows, prefix_indices, prefix_rows, suffix_indices, suffix_rows, screen):
    picked = np.empty(len(x), dtype=np.int64)
    values = np.full(len(x), np.inf)
    operations = np.zeros(len(x), dtype=np.int64)
    if len(rows) <= 1:
        return picked, values, operations
    for i in range(len(x)):
        score = condition_cost(x[i], features, lower, upper, allowed, n_numeric, scales, rules, category_cost)
        if screen:
            picked[i], values[i], operations[i] = screened_leaf(
                score, prefix_indices, prefix_rows, suffix_indices, suffix_rows)
        else:
            costs = sparse_sums(score, indices, rows)
            picked[i] = costs.argmin()
            values[i] = costs[picked[i]]
            operations[i] = len(indices)
    return picked, values, operations


def kernel(index, x, action, screen):
    """Keep return materialization identical to the SciPy control."""
    first, rest = index.stages(.25)
    rules = np.zeros(index.n_features, dtype=np.uint8)
    rules[list(action.frozen)] |= 1
    rules[list(action.increasing)] |= 2
    rules[list(action.decreasing)] |= 4
    rows, costs, terms = solve_batch(x, index.features, index.condition_low,
                                    index.condition_high, index.condition_allowed,
                                    index.n_numeric_conditions, index.scales, rules, action.categorical_cost,
                                    index.matrix.indices, index.matrix.indptr,
                                    first.indices, first.indptr, rest.indices, rest.indptr, screen)
    z = np.full_like(x, np.nan)
    for i in np.flatnonzero(np.isfinite(costs)):
        z[i] = index.materialize(x[i], rows[i])
    return z, costs, terms
