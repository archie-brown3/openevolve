"""
Reflexion layer: stagnation-triggered prompt-strategy adaptation.

When an island's search stalls, the orchestrator asks a frontier model to
diagnose the failed trajectory and propose a revised system message for that
island. Reflections are kept in a small bounded memory (optionally deduped by
embedding cosine). See .claude/plans/REFLEXION_SPRINT_PLAN.md.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

# Imported at module level so tests can patch them and the orchestrator can build them.
from openevolve.embedding import EmbeddingClient
from openevolve.llm.ensemble import LLMEnsemble


## todo: change this from a generic message to a specific prompt based on the problem 
REFLEXION_SYSTEM = (
    "You are a senior researcher diagnosing why an automated code-evolution search has "
    "stalled on one island. Read the recent attempts and the current instructions given to "
    "the coding model. In the first person, identify the specific repeated pattern that is "
    "keeping the search stuck, then propose a REVISED system message that pushes the model "
    "toward a different strategy. Respond ONLY as JSON: "
    '{"reflection": "...", "new_system_message": "..."}'
)


def update_stagnation(
    counters: Dict[int, int],
    bests: Dict[int, float],
    island_id: int,
    score: float,
    threshold: float,
) -> None:
    """Per-island mirror of the global early-stopping bump/reset (process_parallel.py:711).
    Resets on improvement >= threshold, else increments. Mutates the dicts in place."""
    if score >= bests.get(island_id, float("-inf")) + threshold:
        bests[island_id] = score
        counters[island_id] = 0
    else:
        counters[island_id] = counters.get(island_id, 0) + 1


def apply_reflexion_override(
    prompt: Dict[str, str], override: Optional[Dict[str, Any]]
) -> Dict[str, str]:
    """Return prompt with 'system' replaced iff override carries a non-empty system_message.
    Returns a new dict; never mutates the caller's prompt (workers are pooled across islands)."""
    if override and override.get("system_message"):
        return {**prompt, "system": override["system_message"]}
    return prompt


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _extract_json(text: Optional[str]) -> Dict[str, Any]:
    """Lenient parse copied from evaluator._llm_evaluate (evaluator.py:587-610):
    try a ```json fence, else first '{' to last '}'. Any failure -> {}."""
    if not text:
        return {}
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if m:
        s = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}") + 1
        if i < 0 or j <= i:
            return {}
        s = text[i:j]
    try:
        return json.loads(s)
    except Exception:
        return {}


class ReflexionMemory:
    """Bounded list of reflections, persisted as JSON; optional embedding dedup."""

    def __init__(
        self,
        path: Optional[str],
        memory_size: int = 3,
        embedding_client=None,
        similarity_threshold: float = 0.95,
    ):
        self.path = path
        self.memory_size = memory_size
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self._entries: List[Dict[str, Any]] = []
        if path and os.path.exists(path):
            with open(path) as f:
                self._entries = json.load(f)

    def add(self, reflection: Dict[str, Any]) -> bool:
        emb = None
        if self.embedding_client is not None:
            emb = self.embedding_client.get_embedding(reflection["reflection"])
            for e in self._entries:
                if e.get("embedding") and _cosine(emb, e["embedding"]) >= self.similarity_threshold:
                    return False  # near-duplicate
        self._entries.append({**reflection, "embedding": emb})
        self._entries = self._entries[-self.memory_size :]
        self._save()
        return True

    def recent(self, n: int) -> List[Dict[str, Any]]:
        return self._entries[-n:]

    def _save(self) -> None:
        # ponytail: non-atomic; reflexion writes are rare. Atomic write if a crash
        # mid-write ever corrupts the memory file.
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump(self._entries, f)


class ReflexionOrchestrator:
    """Diagnoses a stalled island and proposes a revised system message."""

    def __init__(self, config, output_dir: Optional[str]):
        cfg = config.reflexion
        ec = EmbeddingClient(cfg.embedding_model) if cfg.embedding_model else None
        path = os.path.join(output_dir, "reflexion_memory.json") if output_dir else None
        self.memory = ReflexionMemory(path, cfg.memory_size, ec, cfg.similarity_threshold)
        self.ensemble = LLMEnsemble(config.llm.reflexion_models)

    async def reflect(
        self, island_id: int, trajectory: List[Dict[str, Any]], current_system_message: str
    ) -> Dict[str, str]:
        lines = ["Recent attempts on the stalled island:"]
        for i, t in enumerate(trajectory, 1):
            lines.append(f"{i}. {t.get('changes_description', '')} -> {t.get('metrics', {})}")
        past = self.memory.recent(self.memory.memory_size)
        if past:
            lines.append("\nPast reflections:")
            lines += [f"- {p.get('reflection', '')}" for p in past]
        lines.append(f"\nCurrent system message:\n{current_system_message}")
        lines.append('\nRespond ONLY as JSON: {"reflection": "...", "new_system_message": "..."}')

        resp = await self.ensemble.generate_with_context(
            REFLEXION_SYSTEM, [{"role": "user", "content": "\n".join(lines)}]
        )
        parsed = _extract_json(resp)
        new_sys = parsed.get("new_system_message")
        if not new_sys:
            return {}
        self.memory.add(
            {
                "island_id": island_id,
                "reflection": parsed.get("reflection", ""),
                "new_system_message": new_sys,
            }
        )
        return {"system_message": new_sys}
