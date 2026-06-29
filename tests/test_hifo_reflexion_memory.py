"""
HiFo Phase 1 tests: credit-assigned InsightPool behaviour on ReflexionMemory.

Written TDD-first (RED phase). Every test documents the exact contract that the
production implementation must satisfy. Tests are grouped by the new capability
they gate:

  Group A — Entry metadata fields (entry_id, effectiveness, used_count, last_used_island_gen)
  Group B — EWMA effectiveness update + clamp
  Group C — Probation eviction: young entries immune, oldest-low evicted when pool full
  Group D — ranked() retrieval ordering
  Group E — Backward-compatibility: existing recent() and add() callers unchanged

No network, no LLM, no filesystem side effects (tmp dirs cleaned in tearDown).
"""

import json
import math
import os
import shutil
import tempfile
import unittest

from openevolve.reflexion import ReflexionMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(text: str, sys: str = "SYS") -> dict:
    """Minimal reflection dict matching the shape expected by ReflexionMemory.add()."""
    return {"island_id": 0, "reflection": text, "new_system_message": sys}


def _make_mem(size: int = 30, path: str | None = None) -> ReflexionMemory:
    return ReflexionMemory(path, memory_size=size, embedding_client=None)


# ---------------------------------------------------------------------------
# Group A — Entry metadata fields
# ---------------------------------------------------------------------------

class TestEntryMetadataFields(unittest.TestCase):
    """Every stored entry must carry the four new HiFo credit-assignment fields."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "mem.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_stores_entry_id(self):
        """Each entry gets a unique integer entry_id starting from 0."""
        mem = _make_mem(path=self.path)
        mem.add(_entry("first"))
        mem.add(_entry("second"))
        entries = mem.recent(10)
        ids = [e["entry_id"] for e in entries]
        self.assertEqual(len(ids), 2)
        self.assertIsInstance(ids[0], int)
        self.assertNotEqual(ids[0], ids[1], "entry_ids must be unique")

    def test_add_stores_effectiveness_zero(self):
        """Fresh entries start with effectiveness=0.0 (no prior feedback)."""
        mem = _make_mem()
        mem.add(_entry("tip"))
        e = mem.recent(1)[0]
        self.assertEqual(e["effectiveness"], 0.0)

    def test_add_stores_used_count_zero(self):
        """Fresh entries start with used_count=0."""
        mem = _make_mem()
        mem.add(_entry("tip"))
        e = mem.recent(1)[0]
        self.assertEqual(e["used_count"], 0)

    def test_add_stores_last_used_island_gen_zero(self):
        """Fresh entries start with last_used_island_gen=0."""
        mem = _make_mem()
        mem.add(_entry("tip"))
        e = mem.recent(1)[0]
        self.assertEqual(e["last_used_island_gen"], 0)

    def test_entry_ids_monotonically_increase(self):
        """entry_id counter never resets across multiple add() calls."""
        mem = _make_mem(size=5)
        for i in range(5):
            mem.add(_entry(f"tip-{i}"))
        ids = [e["entry_id"] for e in mem.recent(10)]
        self.assertEqual(ids, sorted(ids), "entry_ids must be monotonically increasing")

    def test_metadata_persists_to_disk(self):
        """Credit fields survive a save/reload cycle."""
        mem = _make_mem(path=self.path)
        mem.add(_entry("persisted"))
        # Reload from disk
        mem2 = ReflexionMemory(self.path, memory_size=30)
        e = mem2.recent(1)[0]
        self.assertIn("entry_id", e)
        self.assertIn("effectiveness", e)
        self.assertIn("used_count", e)
        self.assertIn("last_used_island_gen", e)


# ---------------------------------------------------------------------------
# Group B — EWMA effectiveness update
# ---------------------------------------------------------------------------

class TestEffectivenessUpdate(unittest.TestCase):
    """update_effectiveness() applies EWMA with α=0.3, clamped to [-1, 1]."""

    def _get_entry_id(self, mem: ReflexionMemory) -> int:
        return mem.recent(1)[0]["entry_id"]

    def test_single_update_ewma(self):
        """First update: (1-0.3)*0.0 + 0.3*norm_score = 0.3*norm_score."""
        mem = _make_mem()
        mem.add(_entry("alpha"))
        eid = self._get_entry_id(mem)
        mem.update_effectiveness([eid], norm_score=1.0)
        eff = mem.recent(1)[0]["effectiveness"]
        self.assertAlmostEqual(eff, 0.3, places=6)

    def test_repeated_update_converges(self):
        """After many identical updates, effectiveness converges to norm_score."""
        mem = _make_mem()
        mem.add(_entry("beta"))
        eid = self._get_entry_id(mem)
        for _ in range(50):
            mem.update_effectiveness([eid], norm_score=0.8)
        eff = mem.recent(1)[0]["effectiveness"]
        # Converged value ≈ 0.8; allow 1% tolerance
        self.assertAlmostEqual(eff, 0.8, delta=0.01)

    def test_negative_score_drives_effectiveness_down(self):
        """A norm_score of 0.0 applied to a positive effectiveness drives it down."""
        mem = _make_mem()
        mem.add(_entry("gamma"))
        eid = self._get_entry_id(mem)
        # Warm it up to ~0.8
        for _ in range(30):
            mem.update_effectiveness([eid], norm_score=1.0)
        before = mem.recent(1)[0]["effectiveness"]
        # Now apply bad feedback
        for _ in range(30):
            mem.update_effectiveness([eid], norm_score=0.0)
        after = mem.recent(1)[0]["effectiveness"]
        self.assertLess(after, before)

    def test_effectiveness_clamped_at_plus_one(self):
        """Repeated perfect feedback must not push effectiveness above 1.0."""
        mem = _make_mem()
        mem.add(_entry("upper"))
        eid = self._get_entry_id(mem)
        for _ in range(100):
            mem.update_effectiveness([eid], norm_score=1.0)
        eff = mem.recent(1)[0]["effectiveness"]
        self.assertLessEqual(eff, 1.0)

    def test_effectiveness_clamped_at_minus_one(self):
        """Repeated zero feedback on a negative-leaning tip must not go below -1.0."""
        mem = _make_mem()
        mem.add(_entry("lower"))
        eid = self._get_entry_id(mem)
        # Start from 0, push down with 0 scores (0 < current expected value when negative)
        # First drive it negative with sub-zero signals — here norm_score=-0 (0.0 mapped from -1)
        # We test the clamp by manually setting effectiveness low then updating
        entry = mem.recent(1)[0]
        entry["effectiveness"] = -0.95
        for _ in range(20):
            mem.update_effectiveness([eid], norm_score=0.0)
        eff = mem.recent(1)[0]["effectiveness"]
        self.assertGreaterEqual(eff, -1.0)

    def test_update_unknown_id_is_noop(self):
        """Passing a non-existent entry_id must not raise and must not corrupt state."""
        mem = _make_mem()
        mem.add(_entry("real"))
        before = mem.recent(1)[0]["effectiveness"]
        mem.update_effectiveness([9999], norm_score=1.0)  # unknown id
        after = mem.recent(1)[0]["effectiveness"]
        self.assertEqual(before, after, "Unknown IDs must be silently skipped")

    def test_multiple_ids_all_updated(self):
        """update_effectiveness() with a list of IDs updates all of them."""
        mem = _make_mem()
        mem.add(_entry("tip-A"))
        mem.add(_entry("tip-B"))
        entries = mem.recent(10)
        ids = [e["entry_id"] for e in entries]
        mem.update_effectiveness(ids, norm_score=1.0)
        for e in mem.recent(10):
            self.assertGreater(e["effectiveness"], 0.0)


# ---------------------------------------------------------------------------
# Group C — Probation eviction
# ---------------------------------------------------------------------------

class TestProbationEviction(unittest.TestCase):
    """
    Pool-full behaviour:
      - Entries with used_count < 3 (probation) are immune from eviction.
      - When all entries are on probation, the oldest (by last_used_island_gen) is evicted.
      - Among mature entries (used_count >= 3), the one with the lowest eviction score
        (effectiveness - idle_gens * 0.01) is evicted.
    """

    def _fill_pool(self, mem: ReflexionMemory, n: int) -> list:
        """Add n entries and return their entry_ids in order."""
        ids = []
        for i in range(n):
            mem.add(_entry(f"unique-tip-number-{i}"))
            ids.append(mem.recent(1)[0]["entry_id"])
        return ids

    def test_young_entry_not_evicted_when_pool_full(self):
        """A just-added entry (used_count=0) is never evicted to make room."""
        pool_size = 5
        mem = _make_mem(size=pool_size)
        # Fill pool with 5 entries
        self._fill_pool(mem, pool_size)
        # Manually mature the first entry so it can be evicted
        first_entry = mem.recent(10)[0]
        first_entry["used_count"] = 5
        first_entry["effectiveness"] = -0.9  # very poor, should be evicted

        # Add one more: pool is full, eviction should remove the poor mature one
        mem.add(_entry("brand-new-tip-zzzz"))
        reflections = [e["reflection"] for e in mem.recent(10)]
        self.assertIn("brand-new-tip-zzzz", reflections,
                      "The newly added entry must survive eviction")
        self.assertEqual(len(mem.recent(10)), pool_size,
                         "Pool size must not exceed capacity")

    def test_poor_mature_entry_evicted_over_good_young(self):
        """When pool is full: evict a mature, low-effectiveness entry not a young one."""
        pool_size = 4
        mem = _make_mem(size=pool_size)
        self._fill_pool(mem, pool_size)

        entries = mem.recent(10)
        # Make entry[0] mature with very low effectiveness
        entries[0]["used_count"] = 10
        entries[0]["effectiveness"] = -0.9
        target_reflection = entries[0]["reflection"]

        # Make entry[1] young (used_count=1 < 3)
        entries[1]["used_count"] = 1

        # Add a new entry: triggers eviction
        mem.add(_entry("newcomer-unique-string-abc"))

        reflections = [e["reflection"] for e in mem.recent(10)]
        self.assertNotIn(target_reflection, reflections,
                         "Mature low-effectiveness entry should be evicted")
        self.assertIn("newcomer-unique-string-abc", reflections)

    def test_all_probation_evicts_oldest_by_gen(self):
        """If every entry is on probation, evict the one with the lowest last_used_island_gen."""
        pool_size = 3
        mem = _make_mem(size=pool_size)
        self._fill_pool(mem, pool_size)

        entries = mem.recent(10)
        # All have used_count=0 (on probation). Set different gen stamps.
        entries[0]["last_used_island_gen"] = 1   # oldest → should be evicted
        entries[1]["last_used_island_gen"] = 5
        entries[2]["last_used_island_gen"] = 10
        oldest_reflection = entries[0]["reflection"]

        mem.add(_entry("evict-oldest-probation-unique"))

        reflections = [e["reflection"] for e in mem.recent(10)]
        self.assertNotIn(oldest_reflection, reflections,
                         "Oldest probation entry (lowest gen) should be evicted")

    def test_pool_size_never_exceeded(self):
        """Pool size is strictly bounded even after many adds."""
        pool_size = 5
        mem = _make_mem(size=pool_size)
        for i in range(20):
            mem.add(_entry(f"tip-overflow-test-{i}"))
        self.assertLessEqual(len(mem.recent(30)), pool_size)


# ---------------------------------------------------------------------------
# Group D — ranked() retrieval
# ---------------------------------------------------------------------------

class TestRankedRetrieval(unittest.TestCase):
    """ranked(k) returns top-k entries sorted by adaptive score."""

    def _setup_two_entries(self) -> ReflexionMemory:
        mem = _make_mem()
        mem.add(_entry("low-eff-tip"))
        mem.add(_entry("high-eff-tip"))
        entries = mem.recent(10)
        low_id  = entries[0]["entry_id"]
        high_id = entries[1]["entry_id"]
        # Give high-eff-tip a high effectiveness
        for _ in range(20):
            mem.update_effectiveness([high_id], norm_score=1.0)
        # Drive low-eff-tip down
        entries[0]["effectiveness"] = -0.5
        return mem, low_id, high_id

    def test_ranked_returns_k_entries(self):
        mem = _make_mem()
        for i in range(10):
            mem.add(_entry(f"t{i}"))
        result = mem.ranked(k=3)
        self.assertEqual(len(result), 3)

    def test_ranked_prefers_high_effectiveness(self):
        """ranked() must return high-effectiveness entries before low-effectiveness ones."""
        mem, low_id, high_id = self._setup_two_entries()
        result = mem.ranked(k=1)
        self.assertEqual(result[0]["entry_id"], high_id,
                         "High-effectiveness entry must rank first")

    def test_ranked_usage_penalty(self):
        """
        A heavily used high-effectiveness entry can be beaten by a less-used one
        with equal effectiveness, due to the −0.1·log(used+1) penalty.
        """
        mem = _make_mem()
        mem.add(_entry("heavy-user"))
        mem.add(_entry("fresh-entry"))
        entries = mem.recent(10)
        heavy_id = entries[0]["entry_id"]
        fresh_id  = entries[1]["entry_id"]
        # Same effectiveness
        for _ in range(10):
            mem.update_effectiveness([heavy_id], norm_score=0.7)
            mem.update_effectiveness([fresh_id], norm_score=0.7)
        # Make heavy-user heavily used
        entries[0]["used_count"] = 50
        entries[1]["used_count"] = 1
        result = mem.ranked(k=1)
        # fresh-entry should win due to lower usage penalty
        self.assertEqual(result[0]["entry_id"], fresh_id,
                         "Less-used entry wins when effectiveness is equal")

    def test_ranked_recency_bonus(self):
        """
        An entry used within 2 island-generations of current gets +0.2 recency bonus,
        enough to beat a slightly higher-effectiveness but stale entry.
        """
        mem = _make_mem()
        mem.add(_entry("stale-entry"))
        mem.add(_entry("recent-entry"))
        entries = mem.recent(10)
        stale_id  = entries[0]["entry_id"]
        recent_id = entries[1]["entry_id"]
        # Give stale a slightly higher base effectiveness
        entries[0]["effectiveness"] = 0.5
        entries[0]["last_used_island_gen"] = 0  # stale
        entries[1]["effectiveness"] = 0.4
        entries[1]["last_used_island_gen"] = 98  # recent (current_gen=100, delta=2)
        result = mem.ranked(k=1, current_island_gen=100)
        self.assertEqual(result[0]["entry_id"], recent_id,
                         "Recent entry wins over stale with recency bonus")

    def test_ranked_fewer_than_k_returns_all(self):
        """ranked(k) with fewer entries than k returns all available entries."""
        mem = _make_mem()
        mem.add(_entry("only-one"))
        result = mem.ranked(k=5)
        self.assertEqual(len(result), 1)

    def test_ranked_increments_used_count(self):
        """Calling ranked() increments used_count on the returned entries."""
        mem = _make_mem()
        mem.add(_entry("traceable"))
        eid = mem.recent(1)[0]["entry_id"]
        mem.ranked(k=1)
        updated = mem.recent(1)[0]
        self.assertEqual(updated["used_count"], 1)

    def test_ranked_updates_last_used_island_gen(self):
        """ranked() stamps last_used_island_gen with the provided current_island_gen."""
        mem = _make_mem()
        mem.add(_entry("gen-tracked"))
        mem.ranked(k=1, current_island_gen=42)
        e = mem.recent(1)[0]
        self.assertEqual(e["last_used_island_gen"], 42)


# ---------------------------------------------------------------------------
# Group E — Backward-compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    """
    All existing callers of add(), recent(), and ReflexionMemory(path, size, ...)
    must continue to work without modification.
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "mem.json")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_returns_bool_true_on_success(self):
        mem = _make_mem(path=self.path)
        result = mem.add({"island_id": 0, "reflection": "r", "new_system_message": "s"})
        self.assertTrue(result)

    def test_recent_still_works(self):
        """recent(n) still returns the n most recent entries (unchanged behaviour)."""
        mem = _make_mem(path=self.path)
        for i in range(5):
            mem.add({"island_id": 0, "reflection": f"r{i}", "new_system_message": "s"})
        recent = mem.recent(3)
        self.assertEqual(len(recent), 3)
        self.assertEqual([e["reflection"] for e in recent], ["r2", "r3", "r4"])

    def test_memory_size_still_bounds_recent(self):
        """memory_size cap still applies when using recent()."""
        mem = ReflexionMemory(self.path, memory_size=3)
        for i in range(6):
            mem.add({"island_id": 0, "reflection": f"r{i}", "new_system_message": "s"})
        self.assertEqual(len(mem.recent(10)), 3)

    def test_roundtrip_without_embedding(self):
        """Existing save/load path works with new fields present."""
        mem = ReflexionMemory(self.path, memory_size=5)
        mem.add({"island_id": 1, "reflection": "persisted", "new_system_message": "X"})
        reloaded = ReflexionMemory(self.path, memory_size=5)
        entries = reloaded.recent(10)
        self.assertEqual(entries[0]["reflection"], "persisted")

    def test_path_none_no_disk_write(self):
        """path=None still works in-memory; no file is created."""
        mem = ReflexionMemory(None, memory_size=5)
        mem.add({"island_id": 0, "reflection": "x", "new_system_message": "y"})
        self.assertEqual(len(mem.recent(10)), 1)


if __name__ == "__main__":
    unittest.main()
