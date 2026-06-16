"""
Tests for the operator-selection layer (openevolve.operators) and its wiring.
"""

import random
import unittest

from openevolve.config import Config, OperatorConfig
from openevolve.operators import (
    CostMeter,
    OperatorPool,
    RandomSelector,
    UCBSelector,
    build_selector,
)
from openevolve.operators.pool import DEFAULT_OPERATORS
from openevolve.prompt.templates import TemplateManager
from openevolve.utils.metrics_utils import get_fitness_score


class TestOperatorPool(unittest.TestCase):
    def test_default_pool_resolution(self):
        pool = OperatorPool()
        self.assertEqual(pool.ids(), ["baseline", "reflect_rewrite"])
        # Baseline resolves to None -> worker uses OpenEvolve's default template.
        self.assertIsNone(pool.resolve("baseline"))
        self.assertEqual(pool.resolve("reflect_rewrite"), "op_reflect_rewrite")

    def test_subset_and_order_preserved(self):
        pool = OperatorPool(["reflect_rewrite", "baseline"])
        self.assertEqual(pool.ids(), ["reflect_rewrite", "baseline"])

    def test_unknown_operator_rejected(self):
        with self.assertRaises(ValueError):
            OperatorPool(["nope"])
        with self.assertRaises(KeyError):
            OperatorPool().resolve("nope")

    def test_empty_pool_rejected(self):
        with self.assertRaises(ValueError):
            OperatorPool([])


class TestRandomSelector(unittest.TestCase):
    def test_returns_only_pool_ids_and_is_uniform(self):
        ids = ["a", "b", "c"]
        sel = RandomSelector(ids, seed=123)
        counts = {i: 0 for i in ids}
        for _ in range(3000):
            choice = sel.select()
            self.assertIn(choice, ids)
            counts[choice] += 1
        # Roughly uniform: each arm within a generous band around 1000.
        for i in ids:
            self.assertGreater(counts[i], 800)
            self.assertLess(counts[i], 1200)

    def test_update_is_noop(self):
        sel = RandomSelector(["a", "b"], seed=1)
        sel.update("a", 5.0)  # must not raise or change behaviour


class TestUCBSelector(unittest.TestCase):
    def test_plays_each_arm_before_repeating(self):
        sel = UCBSelector(["a", "b", "c"], seed=7)
        first_three = set()
        for _ in range(3):
            choice = sel.select()
            first_three.add(choice)
            sel.update(choice, 0.0)
        self.assertEqual(first_three, {"a", "b", "c"})

    def test_converges_to_better_arm(self):
        rng = random.Random(42)
        sel = UCBSelector(["good", "bad"], c=1.0, seed=42)
        picks = []
        for _ in range(2000):
            arm = sel.select()
            # Stationary rewards: good arm centred at 1.0, bad arm at 0.0.
            reward = rng.gauss(1.0, 0.1) if arm == "good" else rng.gauss(0.0, 0.1)
            sel.update(arm, reward)
            picks.append(arm)
        # The bandit should overwhelmingly prefer the better arm by the end.
        tail = picks[-500:]
        self.assertGreater(tail.count("good"), 480)
        self.assertGreater(sel.stats()["good"]["mean_reward"], sel.stats()["bad"]["mean_reward"])

    def test_unknown_arm_update_raises(self):
        sel = UCBSelector(["a", "b"])
        with self.assertRaises(KeyError):
            sel.update("c", 1.0)


class TestBuildSelector(unittest.TestCase):
    def test_build_by_name(self):
        self.assertIsInstance(build_selector("random", ["a", "b"]), RandomSelector)
        self.assertIsInstance(build_selector("ucb", ["a", "b"]), UCBSelector)
        with self.assertRaises(ValueError):
            build_selector("bogus", ["a", "b"])


class TestCostMeter(unittest.TestCase):
    def test_accumulates_per_operator(self):
        meter = CostMeter()
        meter.add("baseline", {"prompt_tokens": 10, "completion_tokens": 5})
        meter.add("baseline", {"prompt_tokens": 3, "completion_tokens": 2})
        meter.add("reflect_rewrite", {"prompt_tokens": 7, "completion_tokens": 1})
        totals = meter.totals()
        self.assertEqual(
            totals["baseline"], {"prompt_tokens": 13, "completion_tokens": 7, "calls": 2}
        )
        self.assertEqual(totals["reflect_rewrite"]["calls"], 1)
        grand = meter.grand_total()
        self.assertEqual(grand["prompt_tokens"], 20)
        self.assertEqual(grand["completion_tokens"], 8)
        self.assertEqual(grand["calls"], 3)

    def test_missing_usage_counts_call_only(self):
        meter = CostMeter()
        meter.add("baseline", None)
        totals = meter.totals()
        self.assertEqual(
            totals["baseline"], {"prompt_tokens": 0, "completion_tokens": 0, "calls": 1}
        )

    def test_none_operator_bucketed_as_unknown(self):
        meter = CostMeter()
        meter.add(None, {"prompt_tokens": 4, "completion_tokens": 0})
        self.assertIn("unknown", meter.totals())


class TestOperatorTemplate(unittest.TestCase):
    def test_template_loaded_and_valid(self):
        tm = TemplateManager()
        self.assertIn("op_reflect_rewrite", tm.templates)
        text = tm.templates["op_reflect_rewrite"]
        self.assertTrue(text.strip())
        # Keeps OpenEvolve's diff output contract and core placeholder.
        self.assertIn("<<<<<<< SEARCH", text)
        self.assertIn(">>>>>>> REPLACE", text)
        self.assertIn("{current_program}", text)


class TestRewardMath(unittest.TestCase):
    def test_fitness_gain_uses_combined_score(self):
        # Mirrors the reward computed in the result loop: fit(child) - fit(parent).
        parent_metrics = {"combined_score": 0.4}
        child_metrics = {"combined_score": 0.7}
        feature_dims = []
        captured = {}

        class _Recorder(UCBSelector):
            def update(self, operator_id, reward):
                captured["reward"] = reward
                super().update(operator_id, reward)

        sel = _Recorder(["baseline"], seed=0)
        sel.select()
        reward = get_fitness_score(child_metrics, feature_dims) - get_fitness_score(
            parent_metrics, feature_dims
        )
        sel.update("baseline", reward)
        self.assertAlmostEqual(captured["reward"], 0.3)


class TestOperatorConfig(unittest.TestCase):
    def test_defaults_disabled(self):
        cfg = Config()
        self.assertIsInstance(cfg.operators, OperatorConfig)
        self.assertFalse(cfg.operators.enabled)
        self.assertEqual(cfg.operators.operator_ids, list(DEFAULT_OPERATORS.keys()))

    def test_from_dict_parses_operators(self):
        cfg = Config.from_dict(
            {"operators": {"enabled": True, "selector": "ucb", "operator_ids": ["baseline"]}}
        )
        self.assertTrue(cfg.operators.enabled)
        self.assertEqual(cfg.operators.operator_ids, ["baseline"])


if __name__ == "__main__":
    unittest.main()
