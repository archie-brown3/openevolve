"""
Tests for the Reflexion layer (openevolve/reflexion.py) and its config.

Written test-first (TDD RED phase) against the interface contract in
.claude/plans/REFLEXION_SPRINT_PLAN.md. Production code does NOT exist yet,
so this whole module is expected to fail at import (ModuleNotFoundError /
ImportError) until the corresponding sprints land. Each test maps 1:1 to a
sprint task (T1.x ↔ S-task) noted in its docstring.

No network, no API keys: the LLM ensemble and embedding client are stubbed at
their boundaries.
"""

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from openevolve.config import Config, ReflexionConfig
from openevolve.reflexion import (
    ReflexionMemory,
    ReflexionOrchestrator,
    apply_reflexion_override,
    update_stagnation,
)


# --------------------------------------------------------------------------- #
# Test doubles (boundaries only)
# --------------------------------------------------------------------------- #
class StubEnsemble:
    """Stands in for LLMEnsemble at the network boundary."""

    def __init__(self, response: str):
        self._response = response
        self.calls = []

    async def generate_with_context(self, system_message, messages, **kwargs):
        self.calls.append((system_message, messages))
        return self._response


class StubEmbeddingClient:
    """Deterministic embeddings: identical/similar text -> near-identical vectors."""

    def get_embedding(self, text):
        if "stagnant-A'" in text:  # near-duplicate of stagnant-A
            return [0.999, 0.001, 0.0]
        if "stagnant-A" in text:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]  # orthogonal -> cosine 0


def _reflection(text, sys="NEW SYS"):
    return {"island_id": 0, "reflection": text, "new_system_message": sys}


# --------------------------------------------------------------------------- #
# Sprint 3.1 — ReflexionMemory  (T1.2, T1.3, T1.4)
# --------------------------------------------------------------------------- #
class TestReflexionMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "reflexion_memory.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_memory_bounded(self):
        """T1.2: keeps only the last `memory_size` entries."""
        mem = ReflexionMemory(self.path, memory_size=3)
        for i in range(5):
            mem.add(_reflection(f"r{i}"))
        recent = mem.recent(10)
        self.assertEqual(len(recent), 3)
        self.assertEqual([e["reflection"] for e in recent], ["r2", "r3", "r4"])

    def test_recent_fewer_than_requested(self):
        """T1.2 edge: recent(n) with fewer entries returns all of them."""
        mem = ReflexionMemory(self.path, memory_size=5)
        mem.add(_reflection("only"))
        self.assertEqual(len(mem.recent(10)), 1)

    def test_memory_roundtrip(self):
        """T1.3: entries persist across re-instantiation from the same path."""
        mem = ReflexionMemory(self.path, memory_size=5)
        mem.add(_reflection("alpha", sys="SYS-A"))
        mem.add(_reflection("beta", sys="SYS-B"))

        reloaded = ReflexionMemory(self.path, memory_size=5)
        entries = reloaded.recent(10)
        self.assertEqual([e["reflection"] for e in entries], ["alpha", "beta"])
        self.assertEqual(entries[-1]["new_system_message"], "SYS-B")

    def test_path_none_is_in_memory(self):
        """T1.3 edge: path=None works in-memory and writes no file."""
        mem = ReflexionMemory(None, memory_size=3)
        self.assertTrue(mem.add(_reflection("x")))
        self.assertEqual(len(mem.recent(10)), 1)

    def test_memory_dedup_skips_near_duplicate(self):
        """T1.4: with an embedding client, a near-duplicate add is skipped."""
        mem = ReflexionMemory(
            self.path,
            memory_size=5,
            embedding_client=StubEmbeddingClient(),
            similarity_threshold=0.95,
        )
        self.assertTrue(mem.add(_reflection("stagnant-A insight")))
        self.assertFalse(mem.add(_reflection("stagnant-A' insight")))  # deduped
        self.assertTrue(mem.add(_reflection("totally different")))  # orthogonal kept

        texts = [e["reflection"] for e in mem.recent(10)]
        self.assertEqual(len(texts), 2)
        self.assertNotIn("stagnant-A' insight", texts)

    def test_dedup_off_without_embedding_client(self):
        """T1.4 edge: no embedding client => dedup is a no-op (plain bounded list)."""
        mem = ReflexionMemory(self.path, memory_size=5, embedding_client=None)
        self.assertTrue(mem.add(_reflection("stagnant-A insight")))
        self.assertTrue(mem.add(_reflection("stagnant-A' insight")))
        self.assertEqual(len(mem.recent(10)), 2)


# --------------------------------------------------------------------------- #
# Sprint 3.2 — ReflexionOrchestrator.reflect  (T1.5, T1.6)
# --------------------------------------------------------------------------- #
class TestReflexionOrchestrator(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.config = Config()  # default reflexion: enabled=False, embedding_model=None
        self.trajectory = [
            {"changes_description": "tried X", "metrics": {"combined_score": 0.40}},
            {"changes_description": "tried Y", "metrics": {"combined_score": 0.41}},
        ]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _make_orchestrator(self, response: str):
        # Patch the ensemble class so __init__ never builds a real client
        # (LLMEnsemble([]) divides by zero on an empty model list).
        with patch("openevolve.reflexion.LLMEnsemble"):
            orch = ReflexionOrchestrator(self.config, output_dir=self.dir)
        orch.ensemble = StubEnsemble(response)
        return orch

    def test_reflect_parses_strategy(self):
        """T1.5: a clean JSON reply yields {'system_message': ...} and grows memory."""
        resp = json.dumps({"reflection": "I over-fit X", "new_system_message": "NEW SYS"})
        orch = self._make_orchestrator(resp)

        result = asyncio.run(orch.reflect(0, self.trajectory, "OLD SYS"))

        self.assertEqual(result, {"system_message": "NEW SYS"})
        self.assertEqual(len(orch.memory.recent(10)), 1)

    def test_reflect_lenient_json_fenced(self):
        """T1.6: JSON wrapped in a ```json fence parses."""
        resp = '```json\n{"reflection": "r", "new_system_message": "FENCED"}\n```'
        orch = self._make_orchestrator(resp)
        result = asyncio.run(orch.reflect(0, self.trajectory, "OLD"))
        self.assertEqual(result, {"system_message": "FENCED"})

    def test_reflect_lenient_json_with_prose(self):
        """T1.6: a bare {...} embedded in prose parses (first { to last })."""
        resp = (
            "Sure, here is my analysis.\n"
            '{"reflection": "r", "new_system_message": "PROSE"}\n'
            "Hope that helps!"
        )
        orch = self._make_orchestrator(resp)
        result = asyncio.run(orch.reflect(0, self.trajectory, "OLD"))
        self.assertEqual(result, {"system_message": "PROSE"})

    def test_reflect_malformed_returns_empty_and_no_memory(self):
        """T1.6 edge: unparseable reply => {} override and NOTHING written to memory."""
        orch = self._make_orchestrator("not json at all, sorry")
        result = asyncio.run(orch.reflect(0, self.trajectory, "OLD"))
        self.assertEqual(result, {})
        self.assertEqual(len(orch.memory.recent(10)), 0)

    def test_reflect_missing_key_returns_empty(self):
        """T1.6 edge: valid JSON lacking new_system_message => {} and no memory write."""
        resp = json.dumps({"reflection": "diagnosed but proposed nothing"})
        orch = self._make_orchestrator(resp)
        result = asyncio.run(orch.reflect(0, self.trajectory, "OLD"))
        self.assertEqual(result, {})
        self.assertEqual(len(orch.memory.recent(10)), 0)


# --------------------------------------------------------------------------- #
# Sprint 2 — config  (T1.7)
# --------------------------------------------------------------------------- #
class TestReflexionConfig(unittest.TestCase):
    def test_config_defaults(self):
        """T1.7: ReflexionConfig defaults match the contract."""
        cfg = ReflexionConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.stagnation_patience, 10)
        self.assertEqual(cfg.memory_size, 3)
        self.assertEqual(cfg.improvement_metric, "combined_score")
        self.assertIsNone(cfg.embedding_model)

    def test_omitted_block_yields_defaults(self):
        """T1.7 edge: a config without a reflexion block is disabled by default."""
        cfg = Config.from_dict({})
        self.assertFalse(cfg.reflexion.enabled)

    def test_reflexion_block_parses(self):
        """T1.7: nested reflexion + reflexion_models parse via dacite."""
        cfg = Config.from_dict(
            {
                "reflexion": {"enabled": True, "stagnation_patience": 3, "memory_size": 2},
                "llm": {"reflexion_models": [{"name": "deepseek-v4-pro"}]},
            }
        )
        self.assertTrue(cfg.reflexion.enabled)
        self.assertEqual(cfg.reflexion.stagnation_patience, 3)
        self.assertEqual(cfg.reflexion.memory_size, 2)
        self.assertEqual(len(cfg.llm.reflexion_models), 1)
        self.assertEqual(cfg.llm.reflexion_models[0].name, "deepseek-v4-pro")


# --------------------------------------------------------------------------- #
# Sprint 5.1 — update_stagnation  (T1.8)
# --------------------------------------------------------------------------- #
class TestStagnationHelper(unittest.TestCase):
    def test_first_observation_resets(self):
        """T1.8: first score for an island establishes the baseline, counter=0."""
        counters, bests = {}, {}
        update_stagnation(counters, bests, 0, 0.5, threshold=0.001)
        self.assertEqual(counters[0], 0)
        self.assertEqual(bests[0], 0.5)

    def test_no_improvement_increments(self):
        """T1.8: a non-improving score increments the island counter."""
        counters, bests = {0: 0}, {0: 0.5}
        update_stagnation(counters, bests, 0, 0.5, threshold=0.001)  # no change
        self.assertEqual(counters[0], 1)
        update_stagnation(counters, bests, 0, 0.4, threshold=0.001)  # worse
        self.assertEqual(counters[0], 2)

    def test_improvement_at_threshold_resets(self):
        """T1.8 edge: improvement of exactly `threshold` counts as progress (>=),
        matching the existing early_stopping convention (process_parallel.py:711)."""
        counters, bests = {0: 3}, {0: 0.5}
        update_stagnation(counters, bests, 0, 0.501, threshold=0.001)
        self.assertEqual(counters[0], 0)
        self.assertEqual(bests[0], 0.501)

    def test_islands_tracked_independently(self):
        """T1.8 edge: counters/bests are per-island."""
        counters, bests = {}, {}
        update_stagnation(counters, bests, 0, 0.5, threshold=0.001)
        update_stagnation(counters, bests, 1, 0.9, threshold=0.001)
        update_stagnation(counters, bests, 0, 0.5, threshold=0.001)  # island 0 stalls
        self.assertEqual(counters[0], 1)
        self.assertEqual(counters[1], 0)


# --------------------------------------------------------------------------- #
# Sprint 5.4/5.5 — apply_reflexion_override (the worker interrupt)  (T1.9)
# --------------------------------------------------------------------------- #
class TestApplyOverride(unittest.TestCase):
    def _prompt(self):
        # Shape returned by PromptSampler.build_prompt (process_parallel.py:181-194)
        return {"system": "BASE SYS", "user": "USER CONTENT"}

    def test_override_replaces_system_only(self):
        """T1.9: a present override swaps system, leaves user untouched."""
        out = apply_reflexion_override(self._prompt(), {"system_message": "NEW"})
        self.assertEqual(out["system"], "NEW")
        self.assertEqual(out["user"], "USER CONTENT")

    def test_none_override_unchanged(self):
        """T1.9: no override => prompt unchanged."""
        out = apply_reflexion_override(self._prompt(), None)
        self.assertEqual(out, {"system": "BASE SYS", "user": "USER CONTENT"})

    def test_empty_override_unchanged(self):
        """T1.9 edge: an override dict without system_message changes nothing."""
        out = apply_reflexion_override(self._prompt(), {})
        self.assertEqual(out["system"], "BASE SYS")

    def test_empty_string_system_message_ignored(self):
        """T1.9 edge: an empty new system message must NOT blank the prompt."""
        out = apply_reflexion_override(self._prompt(), {"system_message": ""})
        self.assertEqual(out["system"], "BASE SYS")

    def test_does_not_mutate_input(self):
        """T1.9 edge: the override is transient — the caller's prompt dict is not
        mutated in place (guards against cross-island leakage in pooled workers)."""
        original = self._prompt()
        apply_reflexion_override(original, {"system_message": "NEW"})
        self.assertEqual(original["system"], "BASE SYS")


if __name__ == "__main__":
    unittest.main()
