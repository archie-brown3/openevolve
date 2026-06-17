"""
UCB1 bandit operator selector (numerical AOS, Condition 3).

Each operator is an arm. ``select`` plays every arm once, then maximises the
UCB1 index ``mean_i + c * sqrt(2 * ln t / N_i)``. ``update`` keeps an
incremental sample mean of the reward (fitness gain). All state lives on the
parent controller process, so no locking is required.
"""

import math
import random
from typing import Dict, List, Optional

from openevolve.operators.selector import OperatorSelector


class UCBSelector(OperatorSelector):
    """UCB1 over a fixed set of operator arms."""

    # Magic numbers extracted as class constants
    DEFAULT_C = 1.414  # Approximately sqrt(2)
    UCB1_LOG_FACTOR = 2.0  # Factor in UCB1 formula: c * sqrt(2 * ln(t) / N_i)
    MIN_T = 1  # Minimum value for t to avoid log(0)

    def __init__(self, operator_ids: List[str], c: float = None, seed: Optional[int] = None):
        if not operator_ids:
            raise ValueError("UCBSelector requires at least one operator id")
        self._ids = list(operator_ids)
        self.c = c if c is not None else self.DEFAULT_C
        self._rng = random.Random(seed)
        self._counts: Dict[str, int] = {oid: 0 for oid in self._ids}
        self._means: Dict[str, float] = {oid: 0.0 for oid in self._ids}
        self._t = 0  # total number of updates observed

    def select(self) -> str:
        # Play each arm at least once before applying the UCB index.
        unplayed = [oid for oid in self._ids if self._counts[oid] == 0]
        if unplayed:
            return self._rng.choice(unplayed)

        t = max(self._t, self.MIN_T)
        best_id = None
        best_index = -math.inf
        for oid in self._ids:
            bonus = self.c * math.sqrt(self.UCB1_LOG_FACTOR * math.log(t) / self._counts[oid])
            index = self._means[oid] + bonus
            if index > best_index:
                best_index = index
                best_id = oid
        return best_id

    def update(self, operator_id: str, reward: float) -> None:
        if operator_id not in self._counts:
            raise KeyError(f"Operator id not in selector: {operator_id!r}")
        self._t += 1
        self._counts[operator_id] += 1
        n = self._counts[operator_id]
        # Incremental sample mean.
        self._means[operator_id] += (reward - self._means[operator_id]) / n

    def stats(self) -> Dict[str, Dict[str, float]]:
        """Per-arm pull counts and mean rewards (for logging/analysis)."""
        return {
            oid: {"count": self._counts[oid], "mean_reward": self._means[oid]} for oid in self._ids
        }
