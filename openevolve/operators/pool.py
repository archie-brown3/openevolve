"""
Operator pool: maps operator ids to prompt template keys.

An operator id of ``"baseline"`` resolves to ``None``, meaning the worker falls
through to OpenEvolve's default ``diff_user``/``full_rewrite_user`` behaviour
(driven by ``config.diff_based_evolution``). Any other operator resolves to a
named template loaded by :class:`~openevolve.prompt.templates.TemplateManager`.
"""

from typing import Dict, List, Optional

# Built-in operators available in this experiment slice.
# operator_id -> template_key (None means "use OpenEvolve's default template")
DEFAULT_OPERATORS: Dict[str, Optional[str]] = {
    "baseline": None,
    "reflect_rewrite": "op_reflect_rewrite",
}


class OperatorPool:
    """Registry mapping operator ids to prompt template keys."""

    def __init__(self, operator_ids: Optional[List[str]] = None):
        if operator_ids is None:
            self._mapping = dict(DEFAULT_OPERATORS)
        else:
            unknown = [oid for oid in operator_ids if oid not in DEFAULT_OPERATORS]
            if unknown:
                raise ValueError(
                    f"Unknown operator id(s): {unknown}. "
                    f"Known operators: {sorted(DEFAULT_OPERATORS)}"
                )
            self._mapping = {oid: DEFAULT_OPERATORS[oid] for oid in operator_ids}

        if not self._mapping:
            raise ValueError("OperatorPool requires at least one operator")

    def ids(self) -> List[str]:
        """Operator ids in registration order."""
        return list(self._mapping.keys())

    def resolve(self, operator_id: str) -> Optional[str]:
        """Return the template key for an operator id (``None`` for the default)."""
        if operator_id not in self._mapping:
            raise KeyError(f"Operator id not in pool: {operator_id!r}")
        return self._mapping[operator_id]
