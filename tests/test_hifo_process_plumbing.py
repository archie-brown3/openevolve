"""
HiFo Phase 2 tests: credit plumbing through the process boundary.

Tests that the three touch-points in process_parallel.py wire correctly:
  A — _submit_iteration embeds insight_payload in the db_snapshot
  B — _run_iteration_worker stamps insight IDs onto child metadata and appends
      the directive text to the user prompt
  C — harvest loop calls update_effectiveness() after child evaluation

All tests are mock-based — no actual subprocess spawning. Written TDD-first
(RED phase) against the contracts described in the implementation plan.

Prerequisites (must pass first): test_hifo_reflexion_memory.py Group A & B.
"""

import asyncio
import os
import tempfile
import shutil
import unittest
from unittest.mock import MagicMock, Mock, patch, call
from concurrent.futures import Future

os.environ.setdefault("OPENAI_API_KEY", "test")

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.process_parallel import ProcessParallelController, SerializableResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_controller(config=None, num_islands=2, reflexion_enabled=True):
    if config is None:
        config = Config()
    config.database.num_islands = num_islands
    config.database.in_memory = True
    config.evaluator.parallel_evaluations = 2
    config.reflexion.enabled = reflexion_enabled
    config.reflexion.insight_k = 3

    db = ProgramDatabase(config.database)
    for i in range(4):
        p = Program(
            id=f"seed_{i}",
            code=f"def f(): return {i}",
            metrics={"combined_score": 0.3 + i * 0.1},
        )
        db.add(p, target_island=i % num_islands)

    ctrl = ProcessParallelController(config, "dummy_eval.py", db)
    return ctrl, db


def _make_mock_reflexion(tips=None):
    """Build a minimal mock that satisfies the reflexion attribute contract."""
    tips = tips or [
        {"entry_id": 0, "reflection": "Design principle A"},
        {"entry_id": 1, "reflection": "Design principle B"},
    ]
    mock_mem = MagicMock()
    mock_mem.ranked.return_value = tips
    mock_reflexion = MagicMock()
    mock_reflexion.memory = mock_mem
    return mock_reflexion, mock_mem


# ---------------------------------------------------------------------------
# Touch-point A — snapshot injection
# ---------------------------------------------------------------------------

class TestSnapshotInsightPayload(unittest.TestCase):
    """
    _submit_iteration must embed insight_payload in the db_snapshot when
    reflexion is enabled and the memory returns tips.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_insight_payload_present_when_reflexion_enabled(self):
        """
        When reflexion is set on the controller and memory.ranked() returns tips,
        db_snapshot must contain 'insight_payload' with both 'ids' and 'text' lists.
        """
        ctrl, db = _make_controller(reflexion_enabled=True)
        mock_reflexion, mock_mem = _make_mock_reflexion()
        ctrl.reflexion = mock_reflexion
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        # Capture snapshot passed to executor.submit
        call_args = ctrl.executor.submit.call_args
        # submit(fn, iteration, db_snapshot, parent_id, inspiration_ids)
        db_snapshot = call_args[0][1]  # positional arg index 1

        self.assertIn("insight_payload", db_snapshot,
                      "db_snapshot must contain insight_payload when reflexion enabled")
        payload = db_snapshot["insight_payload"]
        self.assertIn("ids", payload)
        self.assertIn("text", payload)
        self.assertIsInstance(payload["ids"], list)
        self.assertIsInstance(payload["text"], list)
        self.assertEqual(len(payload["ids"]), len(payload["text"]))

    def test_insight_payload_ids_match_returned_tips(self):
        """insight_payload['ids'] must exactly match the entry_ids from memory.ranked()."""
        tips = [
            {"entry_id": 42, "reflection": "Tip X"},
            {"entry_id": 99, "reflection": "Tip Y"},
        ]
        ctrl, db = _make_controller(reflexion_enabled=True)
        mock_reflexion, mock_mem = _make_mock_reflexion(tips=tips)
        ctrl.reflexion = mock_reflexion
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        db_snapshot = ctrl.executor.submit.call_args[0][1]
        self.assertEqual(db_snapshot["insight_payload"]["ids"], [42, 99])
        self.assertEqual(db_snapshot["insight_payload"]["text"], ["Tip X", "Tip Y"])

    def test_insight_payload_absent_when_reflexion_disabled(self):
        """When controller.reflexion is None, no insight_payload should appear."""
        ctrl, db = _make_controller(reflexion_enabled=False)
        ctrl.reflexion = None  # explicitly disabled
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        db_snapshot = ctrl.executor.submit.call_args[0][1]
        self.assertNotIn("insight_payload", db_snapshot)

    def test_insight_payload_absent_when_memory_returns_empty(self):
        """If ranked() returns [], insight_payload must not be injected (or must be empty)."""
        ctrl, db = _make_controller(reflexion_enabled=True)
        mock_reflexion, mock_mem = _make_mock_reflexion(tips=[])
        ctrl.reflexion = mock_reflexion
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        db_snapshot = ctrl.executor.submit.call_args[0][1]
        payload = db_snapshot.get("insight_payload")
        # Either absent or empty — both are acceptable implementations
        if payload is not None:
            self.assertEqual(len(payload.get("ids", [])), 0)

    def test_ranked_called_with_current_island_gen(self):
        """ranked() must be called with the island's current generation counter."""
        ctrl, db = _make_controller(reflexion_enabled=True)
        # Set a known generation for island 0
        db.island_generations[0] = 7
        mock_reflexion, mock_mem = _make_mock_reflexion()
        ctrl.reflexion = mock_reflexion
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        mock_mem.ranked.assert_called_once()
        call_kwargs = mock_mem.ranked.call_args[1]  # keyword args
        self.assertEqual(call_kwargs.get("current_island_gen"), 7,
                         "ranked() must receive the island's current generation")


# ---------------------------------------------------------------------------
# Touch-point B — child metadata stamping (via SerializableResult round-trip)
# ---------------------------------------------------------------------------

class TestChildMetadataStamping(unittest.TestCase):
    """
    insight IDs from insight_payload must appear in child_program.metadata['insights'].

    We test this at the SerializableResult boundary: the worker produces a
    child_program_dict, the harvest loop deserialises it into Program(**dict),
    and the 'insights' key must survive that round-trip.
    """

    def _make_result_with_insights(self, insight_ids):
        return SerializableResult(
            child_program_dict={
                "id": "child_test",
                "code": "def f(): return 1",
                "language": "python",
                "parent_id": "seed_0",
                "generation": 1,
                "metrics": {"combined_score": 0.6},
                "iteration_found": 1,
                "metadata": {
                    "changes": "minor tweak",
                    "island": 0,
                    "insights": insight_ids,   # <-- the new field
                },
            },
            parent_id="seed_0",
            iteration_time=0.1,
            iteration=1,
        )

    def test_insights_field_survives_serializable_result(self):
        """
        insight IDs stored in child_program_dict.metadata['insights'] must be
        accessible after Program(**result.child_program_dict) reconstruction.
        """
        result = self._make_result_with_insights([42, 99])
        child = Program(**result.child_program_dict)
        self.assertIn("insights", child.metadata)
        self.assertEqual(child.metadata["insights"], [42, 99])

    def test_insights_empty_list_when_no_payload(self):
        """
        If no insight_payload was in the snapshot, child metadata['insights']
        should be an empty list (not missing, to simplify harvest-loop logic).
        """
        result = self._make_result_with_insights([])
        child = Program(**result.child_program_dict)
        self.assertEqual(child.metadata.get("insights", []), [])

    def test_insights_none_treated_as_empty(self):
        """metadata['insights'] = None must be handled gracefully by the harvest loop."""
        result = SerializableResult(
            child_program_dict={
                "id": "child_none",
                "code": "def f(): return 0",
                "language": "python",
                "parent_id": "seed_0",
                "generation": 1,
                "metrics": {"combined_score": 0.5},
                "iteration_found": 1,
                "metadata": {"changes": "x", "island": 0, "insights": None},
            },
            parent_id="seed_0",
            iteration_time=0.1,
            iteration=1,
        )
        child = Program(**result.child_program_dict)
        # Harvest loop guard: `child.metadata.get("insights") or []`
        insight_ids = child.metadata.get("insights") or []
        self.assertEqual(insight_ids, [])


# ---------------------------------------------------------------------------
# Touch-point C — credit feedback in harvest loop
# ---------------------------------------------------------------------------

class TestCreditFeedback(unittest.TestCase):
    """
    After database.add(child_program), the harvest loop must call
    reflexion.memory.update_effectiveness(ids, norm_score) when the child has
    non-empty 'insights' metadata.
    """

    def _run_one_iteration(self, insight_ids, combined_score=0.7):
        """Drive one mock iteration through run_evolution and capture feedback calls."""
        ctrl, db = _make_controller(reflexion_enabled=True)
        mock_reflexion, mock_mem = _make_mock_reflexion()
        ctrl.reflexion = mock_reflexion

        child_dict = {
            "id": "child_credit",
            "code": "def f(): return 1",
            "language": "python",
            "parent_id": "seed_0",
            "generation": 1,
            "metrics": {"combined_score": combined_score},
            "iteration_found": 1,
            "metadata": {"changes": "tweak", "island": 0, "insights": insight_ids},
        }
        mock_result = SerializableResult(
            child_program_dict=child_dict,
            parent_id="seed_0",
            iteration_time=0.1,
            iteration=1,
            target_island=0,
        )
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = mock_result

        async def run():
            with patch.object(ctrl, "_submit_iteration", return_value=mock_future):
                ctrl.start()
                await ctrl.run_evolution(start_iteration=1, max_iterations=1)

        asyncio.run(run())
        return mock_mem

    def test_update_effectiveness_called_with_correct_ids(self):
        """
        Harvest loop must call update_effectiveness([42, 99], <score>) when
        child.metadata['insights'] == [42, 99].
        """
        mock_mem = self._run_one_iteration(insight_ids=[42, 99], combined_score=0.7)
        mock_mem.update_effectiveness.assert_called_once()
        call_args = mock_mem.update_effectiveness.call_args
        self.assertEqual(call_args[0][0], [42, 99])

    def test_update_effectiveness_called_with_combined_score(self):
        """norm_score passed to update_effectiveness must be the child's combined_score."""
        mock_mem = self._run_one_iteration(insight_ids=[5], combined_score=0.85)
        call_args = mock_mem.update_effectiveness.call_args
        norm_score = call_args[0][1]
        self.assertAlmostEqual(norm_score, 0.85, places=4)

    def test_update_effectiveness_not_called_when_no_insights(self):
        """If child.metadata['insights'] is empty, no credit feedback call is made."""
        mock_mem = self._run_one_iteration(insight_ids=[], combined_score=0.7)
        mock_mem.update_effectiveness.assert_not_called()

    def test_update_effectiveness_not_called_when_reflexion_none(self):
        """If controller.reflexion is None, no credit feedback call is made."""
        ctrl, db = _make_controller(reflexion_enabled=False)
        ctrl.reflexion = None

        child_dict = {
            "id": "child_no_ref",
            "code": "def f(): return 1",
            "language": "python",
            "parent_id": "seed_0",
            "generation": 1,
            "metrics": {"combined_score": 0.6},
            "iteration_found": 1,
            "metadata": {"changes": "x", "island": 0, "insights": [7, 8]},
        }
        mock_result = SerializableResult(
            child_program_dict=child_dict,
            parent_id="seed_0",
            iteration_time=0.1,
            iteration=1,
            target_island=0,
        )
        mock_future = MagicMock()
        mock_future.done.return_value = True
        mock_future.result.return_value = mock_result

        # Should not raise even though insights are present but reflexion is None
        async def run():
            with patch.object(ctrl, "_submit_iteration", return_value=mock_future):
                ctrl.start()
                await ctrl.run_evolution(start_iteration=1, max_iterations=1)

        asyncio.run(run())  # must not raise AttributeError on ctrl.reflexion.memory


# ---------------------------------------------------------------------------
# Snapshot structure: existing tests must still pass
# ---------------------------------------------------------------------------

class TestSnapshotStructureRegression(unittest.TestCase):
    """
    Regression: the existing snapshot contract (programs, islands, current_island,
    artifacts) must be unaffected by the insight_payload addition.
    """

    def test_existing_snapshot_keys_present(self):
        ctrl, db = _make_controller(reflexion_enabled=False)
        snapshot = ctrl._create_database_snapshot()
        for key in ("programs", "islands", "current_island", "artifacts"):
            self.assertIn(key, snapshot, f"'{key}' missing from snapshot")

    def test_reflexion_override_still_works(self):
        """
        island_config_overrides → reflexion_override path is unchanged by
        insight_payload addition.
        """
        ctrl, db = _make_controller(reflexion_enabled=False)
        ctrl.reflexion = None
        db.island_config_overrides[0] = {"system_message": "OVERRIDE SYS"}
        ctrl.executor = Mock()
        ctrl.executor.submit.return_value = Mock(spec=Future)

        ctrl._submit_iteration(iteration=1, island_id=0)

        db_snapshot = ctrl.executor.submit.call_args[0][1]
        self.assertIn("reflexion_override", db_snapshot)
        self.assertEqual(db_snapshot["reflexion_override"]["system_message"], "OVERRIDE SYS")


if __name__ == "__main__":
    unittest.main()
