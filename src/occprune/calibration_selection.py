"""Training-photo selectors for the matched CASIA/LFW calibration-budget study.

All algorithms return positions in the same pool of original photos. Alternating
positions receive clean and square-occluded views. Greedy algorithms use frozen
clean embeddings; gradients, identities and LFW pairs are not selection inputs.
"""
import heapq
from time import perf_counter

import numpy as np

from ._sieve_selection import sieve_order

STRATEGIES = ("random", "coreset", "kcenter", "greedi", "greedy_merge", "sieve")


def cosine_kernel(features):
    """Shift unit-vector cosine into [0, 1]; squared Euclidean distance is 4*(1-K)."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty matrix of finite clean embeddings")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("Expected finite, nonzero embedding norms")
    unit = x / norms
    kernel = unit @ unit.T
    kernel += 1.
    kernel *= .5
    np.clip(kernel, 0., 1., out=kernel)
    np.fill_diagonal(kernel, 1.)
    return kernel


def _lazy_greedy(kernel, candidates, stops, first=None):
    """Exact marginal maximization over eligible rows with the full reference pool.

    Pool rows are sorted by original photo ID. The row key in each heap tuple
    settles exact ties. Submodularity makes a stale marginal an upper bound.
    """
    started = perf_counter()
    order = np.empty(max(stops), dtype=np.int64)
    best = np.zeros(len(kernel))
    count = 0
    times = {0: 0.}
    if first is not None:
        order[0] = first
        best[:] = kernel[first]
        count = 1
        times[1] = perf_counter() - started
    if count == len(order):
        return order, times
    heap = []
    for i in range(0, len(candidates), 64):
        block = candidates[i:i + 64]
        gains = np.maximum(kernel[block] - best, 0.).sum(axis=1, dtype=np.float64)
        heap.extend((-float(gain), int(pos), count) for pos, gain in zip(block, gains) if pos != first)
    heapq.heapify(heap)
    while count < len(order):
        _, pos, stamp = heapq.heappop(heap)
        if stamp != count:
            gain = np.maximum(kernel[pos] - best, 0.).sum(dtype=np.float64)
            heapq.heappush(heap, (-float(gain), pos, count))
            continue
        order[count] = pos
        np.maximum(best, kernel[pos], out=best)
        count += 1
        if count in stops:
            times[count] = perf_counter() - started
    return order, times


def _kcenter(kernel, first, stops):
    """Farthest-first traversal. Smaller best similarity means greater distance."""
    started = perf_counter()
    order = np.empty(max(stops), dtype=np.int64)
    best = np.zeros(len(kernel))
    chosen = np.zeros(len(kernel), dtype=bool)
    times = {}
    center = first
    for count in range(1, len(order) + 1):
        order[count - 1] = center
        chosen[center] = True
        np.maximum(best, kernel[center], out=best)
        if count in stops:
            times[count] = perf_counter() - started
        if count < len(order):
            center = np.argmin(np.where(chosen, np.inf, best))
    return order, times


def _distributed(kernel, perm, first, budgets, strategies):
    """Two shards restricted to local candidates, both scoring all reference rows."""
    started = perf_counter()
    shards = np.array_split(perm, 2)
    if max(budgets) > min(map(len, shards)) and "greedi" in strategies:
        raise ValueError("GreeDi must be able to choose B candidates in each of the two shards")
    # Name the shard containing the common first photo A; the other starts at
    # its best singleton. Prefix lengths for each worker depend on the merger.
    if first not in shards[0]:
        shards.reverse()
    stops = set(budgets) if "greedi" in strategies else set()
    if "greedy_merge" in strategies:
        stops.update(b // 2 for b in budgets)
    setup_time = perf_counter() - started
    a, a_time = _lazy_greedy(kernel, np.sort(shards[0]), stops, first)
    b, b_time = _lazy_greedy(kernel, np.sort(shards[1]), stops)
    selections = {}
    for budget in budgets:
        if "greedy_merge" in strategies:
            merge_start = perf_counter()
            quota = budget // 2
            # A0 B0 B1 A1 A2 B2 ... retains each worker's order, and splits
            # its photos evenly across clean/occluded views at the study budgets.
            pair = np.column_stack((a[:quota], b[:quota]))
            pair[1::2] = pair[1::2, ::-1]
            selections["greedy_merge", budget] = (pair.ravel(), {
                "selection_seconds": setup_time + a_time[quota] + b_time[quota] + perf_counter() - merge_start,
                "local_quota": quota,
            })
        if "greedi" in strategies:
            merge_start = perf_counter()
            candidates = np.concatenate((a[:budget], b[:budget]))
            order, _ = _lazy_greedy(kernel, candidates, {budget}, first)
            selections["greedi", budget] = (order, {
                "selection_seconds": setup_time + a_time[budget] + b_time[budget] + perf_counter() - merge_start,
                "local_quota": budget,
            })
    return selections


def select_calibration(kernel, original_ids, random_order, *, seed, budgets=(256, 512, 1024),
                       strategies=STRATEGIES):
    """Return {(strategy, B): (ordered pool positions, diagnostics)} for one seed.

    `original_ids` are sorted, unique CASIA training-row IDs. `random_order`
    interleaves the baseline seed's clean/occluded draws, already mapped to pool
    positions. All selectors start at random_order[0]. Even/odd output positions
    correspond to the clean/occluded banks, respectively. Use the same per-seed
    square boxes for every strategy. Selection time excludes kernel construction
    and the final diagnostic calculation.
    """
    ids = np.asarray(original_ids)
    k = np.asarray(kernel, dtype=np.float64)
    rows = np.asarray(random_order)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or np.any(ids[1:] <= ids[:-1])):
        raise ValueError("Supply strictly increasing integer original IDs")
    if k.shape != (len(ids), len(ids)) or not np.isfinite(k).all() or k.min() < 0 or k.max() > 1:
        raise ValueError("Expected a finite square similarity matrix in [0,1]")
    budgets = tuple(sorted(set(budgets)))
    if not budgets or any(not isinstance(b, (int, np.integer)) or b < 2 or b % 2 for b in budgets):
        raise ValueError("Budgets must be positive even integers for equal clean and occluded banks")
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or len(rows) < max(budgets)
            or len(np.unique(rows)) != len(rows) or np.any((rows < 0) | (rows >= len(ids)))):
        raise ValueError("Supply distinct, valid baseline pool positions covering all budgets")
    strategies = set(strategies)
    if not strategies or not strategies.issubset(STRATEGIES):
        raise ValueError(f"Choose among {STRATEGIES}")
    first = int(rows[0])
    selections = {}
    if "random" in strategies:
        for budget in budgets:
            started = perf_counter()
            replay = rows[:budget].copy()
            selections["random", budget] = (replay, {"selection_seconds": perf_counter() - started})
    for strategy in ("coreset", "kcenter"):
        if strategy not in strategies:
            continue
        if strategy == "coreset":
            order, times = _lazy_greedy(k, np.arange(len(k)), set(budgets), first)
        else:
            order, times = _kcenter(k, first, set(budgets))
        for budget in budgets:
            selections[strategy, budget] = (order[:budget].copy(), {"selection_seconds": times[budget]})
    permutation = np.random.default_rng(seed).permutation(len(k))
    if strategies.intersection(("greedi", "greedy_merge")):
        selections.update(_distributed(k, permutation, first, budgets, strategies))
    if "sieve" in strategies:
        for budget in budgets:
            started = perf_counter()
            order, raw_count = sieve_order(k, permutation, first, budget)
            selections["sieve", budget] = (order, {
                "selection_seconds": perf_counter() - started, "epsilon": .2,
                "raw_selected_count": raw_count, "completion_count": budget - raw_count,
            })
    for order, info in selections.values():
        best = k[order].max(axis=0)
        info["facility_objective"] = float(best.mean())
        info["cover_radius"] = float(np.sqrt(4 * (1 - best.min())))
    return selections
