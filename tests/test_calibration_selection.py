"""Check selection order independently of the dense model or face data."""

import numpy as np
import pytest

from occprune.calibration_selection import cosine_kernel, select_calibration


def exhaustive_facility(kernel, candidates, budget, anchor=None):
    """Recompute all remaining marginal gains after each choice."""
    order = [] if anchor is None else [anchor]
    candidates = set(candidates)
    while len(order) < budget:
        coverage = kernel[order].max(axis=0) if order else np.zeros(len(kernel))
        def priority(row):
            gain = np.maximum(kernel[row] - coverage, 0).sum()
            return -gain, row
        order.append(min(candidates.difference(order), key=priority))
    return order


def test_random_centralized_and_distributed_match_independent_greedy():
    rng = np.random.default_rng(10)
    kernel = cosine_kernel(rng.normal(size=(45, 8)))
    rows = np.arange(len(kernel))
    random_order = rng.permutation(rows)[:12]
    first = random_order[0]
    budgets = (4, 8, 12)
    choices = select_calibration(kernel, rows * 10, random_order, seed=43, budgets=budgets)

    shards = np.array_split(np.random.default_rng(43).permutation(rows), 2)
    if first not in shards[0]:
        shards.reverse()
    local_a = exhaustive_facility(kernel, shards[0], max(budgets), first)
    local_b = exhaustive_facility(kernel, shards[1], max(budgets))
    merged = []
    for i in range(0, max(budgets) // 2, 2):
        merged.extend([local_a[i], local_b[i], local_b[i + 1], local_a[i + 1]])

    for budget in budgets:
        expected = {
            "random": random_order[:budget],
            "coreset": exhaustive_facility(kernel, rows, budget, first),
            "greedy_merge": merged[:budget],
            "greedi": exhaustive_facility(kernel, local_a[:budget] + local_b[:budget], budget, first),
        }
        for strategy, order in expected.items():
            selected, diagnostic = choices[strategy, budget]
            np.testing.assert_array_equal(selected, order)
            assert diagnostic["facility_objective"] == pytest.approx(kernel[order].max(axis=0).mean())
        for shard in shards:
            assert np.isin(merged[:budget:2], shard).sum() == np.isin(merged[1:budget:2], shard).sum()


def test_kcenter_uses_largest_nearest_euclidean_distance():
    x = np.random.default_rng(98).normal(size=(24, 6))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    rows = np.arange(len(x))
    draw = np.roll(rows, 6)
    budget = 12
    order, diagnostic = select_calibration(
        cosine_kernel(x), rows, draw, seed=42, budgets=(budget,), strategies=("kcenter",)
    )["kcenter", budget]
    expected = [draw[0]]
    for _ in range(budget - 1):
        nearest = np.linalg.norm(x[:, None] - x[expected][None], axis=2).min(axis=1)
        expected.append(np.argmax(nearest))
    np.testing.assert_array_equal(order, expected)
    nearest = np.linalg.norm(x[:, None] - x[expected][None], axis=2).min(axis=1)
    assert diagnostic["cover_radius"] == pytest.approx(nearest.max())


def test_duplicate_features_and_stream_completion():
    # There is zero residual gain after the shared anchor; every sieve underfills.
    k = np.ones((10, 10))
    draw = np.array([4, 2, 6, 5])
    choices = select_calibration(k, np.arange(10), draw, seed=44, budgets=(4,))
    for method in ("kcenter", "coreset"):
        np.testing.assert_array_equal(choices[method, 4][0], [4, 0, 1, 2])
    stream = np.random.default_rng(44).permutation(10)
    order, diag = choices["sieve", 4]
    assert diag["raw_selected_count"] == 1 and diag["completion_count"] == 3
    np.testing.assert_array_equal(order, [4, *stream[stream != 4][:3]])


def test_reject_unmatched_clean_occluded_banks_or_infeasible_shards():
    kernel = np.eye(8)
    rows = np.arange(8)
    for draw, budget in ((rows, 3), ([0, 1, 1, 2], 4), (rows, 6)):
        with pytest.raises(ValueError):
            select_calibration(kernel, rows, draw, seed=42, budgets=(budget,))
    # Only GreeDi needs B images in each shard; Greedy-Merge needs B/2.
    choices = select_calibration(kernel, rows, rows, seed=42, budgets=(6,), strategies=("greedy_merge",))
    assert len(np.unique(choices["greedy_merge", 6][0])) == 6
