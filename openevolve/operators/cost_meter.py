"""
Minimal per-operator token accounting.

Honest cost accounting for the experiment: every metered LLM call's token usage
is attributed to the operator that produced it. Lives on the parent controller.
"""

from typing import Dict, Optional


class CostMeter:
    """Accumulates prompt/completion tokens and call counts per operator."""

    def __init__(self):
        self._by_operator: Dict[str, Dict[str, int]] = {}

    def add(self, operator_id: Optional[str], usage: Optional[Dict[str, int]]) -> None:
        """Attribute a single call's token usage to ``operator_id``.

        ``usage`` is ``None`` when the provider omits it (e.g. some OpenAI-compatible
        servers), in which case the call count is still recorded.
        """
        key = operator_id if operator_id is not None else "unknown"
        bucket = self._by_operator.setdefault(
            key, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        )
        bucket["calls"] += 1
        if usage:
            bucket["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            bucket["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Return a copy of the per-operator token totals."""
        return {k: dict(v) for k, v in self._by_operator.items()}

    def grand_total(self) -> Dict[str, int]:
        """Aggregate token totals across all operators."""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        for bucket in self._by_operator.values():
            for k in total:
                total[k] += bucket[k]
        return total
