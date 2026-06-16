"""
Operator selector interface and a uniform-random baseline.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Optional


class OperatorSelector(ABC):
    """Chooses an operator per iteration and learns from the resulting reward."""

    @abstractmethod
    def select(self) -> str:
        """Return the operator id to apply this iteration."""

    @abstractmethod
    def update(self, operator_id: str, reward: float) -> None:
        """Record the reward (fitness gain) obtained from ``operator_id``."""


class RandomSelector(OperatorSelector):
    """Uniform-random choice over the pool. Control condition; ignores reward."""

    def __init__(self, operator_ids: List[str], seed: Optional[int] = None):
        if not operator_ids:
            raise ValueError("RandomSelector requires at least one operator id")
        self._ids = list(operator_ids)
        self._rng = random.Random(seed)

    def select(self) -> str:
        return self._rng.choice(self._ids)

    def update(self, operator_id: str, reward: float) -> None:
        # Random scheduling does not adapt.
        return None
