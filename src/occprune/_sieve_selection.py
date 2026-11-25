"""Sieve-Streaming on the residual facility-location objective of a fixed pool."""
from dataclasses import dataclass
import math

import numpy as np


def _guesses(m, k, epsilon):
    """Powers of 1 + epsilon in the inclusive interval [m, 2*k*m]."""
    base = 1.0 + epsilon
    lower = math.floor(math.log(m) / math.log(base)) - 1
    upper = math.ceil(math.log(2 * k * m) / math.log(base)) + 1
    # Filter on the actual values; logs of exact powers may round to either side.
    values = ((i, base**i) for i in range(lower, upper + 1))
    return {i: v for i, v in values if m <= v <= 2 * k * m}


@dataclass
class _Sieve:
    target: float
    best: np.ndarray
    picked: list[int]
    residual: float = 0.0


def sieve_order(kernel, stream, first, budget, epsilon=0.2):
    """Return ordered positions and the size before exact-budget completion.

    Each sieve maximizes coverage gain beyond the shared first photo with capacity
    k = budget - 1. A newly opened sieve processes the current stream item first.
    The published experiment uses epsilon=0.2; all nine runs filled naturally.
    """
    initial = kernel[first]
    k = budget - 1
    active = {}
    m = 0.0
    scratch = np.empty(len(initial), dtype=np.float64)
    if k:
        for pos in stream:
            if pos == first:
                continue
            candidate = kernel[pos]
            np.subtract(candidate, initial, out=scratch)
            np.maximum(scratch, 0., out=scratch)
            singleton = scratch.mean()
            if singleton > m:
                m = singleton
                guesses = _guesses(m, k, epsilon)
                active = {i: state for i, state in active.items() if i in guesses}
                for i, target in guesses.items():
                    if i not in active:
                        active[i] = _Sieve(target, initial.copy(), [])
            for state in active.values():
                remaining = k - len(state.picked)
                if not remaining:
                    continue
                np.subtract(candidate, state.best, out=scratch)
                np.maximum(scratch, 0., out=scratch)
                gain = scratch.mean()
                if gain >= (state.target / 2 - state.residual) / remaining:
                    state.picked.append(pos)
                    np.maximum(state.best, candidate, out=state.best)
                    state.residual += gain

    useful = [s for s in active.values() if s.picked]
    # Equal coverage favors the smaller guess, keeping stream order within its set.
    winner = min(useful, key=lambda s: (-s.best.mean(), s.target), default=None)
    chosen = [first] + (winner.picked if winner else [])
    raw_count = len(chosen)
    if raw_count < budget:
        taken = set(chosen)
        for pos in stream:
            if pos not in taken:
                chosen.append(pos)
                if len(chosen) == budget:
                    break
    return np.asarray(chosen, dtype=np.int64), raw_count
