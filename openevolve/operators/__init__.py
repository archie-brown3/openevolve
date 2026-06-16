"""
Operator-selection layer for OpenEvolve.

An *operator* is a named prompt strategy resolved to a user-message template. A
*selector* chooses which operator to apply each iteration and receives a reward
(fitness gain) afterwards, so it can adapt over time. State lives on the parent
controller process; workers only receive the resolved ``template_key``.
"""

from openevolve.operators.bandit import UCBSelector
from openevolve.operators.cost_meter import CostMeter
from openevolve.operators.pool import OperatorPool
from openevolve.operators.selector import OperatorSelector, RandomSelector

__all__ = [
    "OperatorPool",
    "OperatorSelector",
    "RandomSelector",
    "UCBSelector",
    "CostMeter",
    "build_selector",
]


def build_selector(name: str, operator_ids, ucb_c: float = 1.414, seed=None) -> OperatorSelector:
    """Construct a selector by config name (``"random"`` or ``"ucb"``)."""
    name = (name or "ucb").lower()
    if name == "random":
        return RandomSelector(operator_ids, seed=seed)
    if name == "ucb":
        return UCBSelector(operator_ids, c=ucb_c, seed=seed)
    raise ValueError(f"Unknown operator selector: {name!r} (expected 'random' or 'ucb')")
