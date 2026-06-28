"""
Evaluator for the cache-replacement-policy evolution experiment.

Thin adapter: loads the evolved score() function, calls benchmark.evaluate_policy(),
maps BenchmarkResult -> EvaluationResult for OpenEvolve.

The evolved program must define only:
    score(age, frequency, recency, size, cache_size, now) -> float

benchmark.py provides all simulation, baselines, and fitness computation.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from benchmark import evaluate_policy, format_table  # noqa: E402


def _load_score_fn(program_path: str):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "score"):
        raise ValueError("Program must define score(age, frequency, recency, size, cache_size, now)")
    return module.score


def _evaluate(program_path: str, workloads=None):
    score_fn = _load_score_fn(program_path)
    result = evaluate_policy(score_fn, split="train", workloads=workloads)
    metrics = {
        "combined_score":          result.combined_score,
        "stable_frequency_hitrate": result.per_workload["stable_frequency"].hr_cand,
        "churn_hitrate":           result.per_workload["churn"].hr_cand,
    }
    artifacts = {
        "hit_rate_table": format_table(result),
        "n_workloads_retained": result.n_retained,
    }
    return metrics, artifacts


def evaluate(program_path: str) -> EvaluationResult:
    """Full evaluation over all workloads (OpenEvolve entry point)."""
    try:
        metrics, artifacts = _evaluate(program_path)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={
                "error_type":    type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Program must define score() — no other functions needed.",
            },
        )


def evaluate_stage1(program_path: str) -> EvaluationResult:
    """Fast gate: verify score() loads and returns a finite float. Stage 2 does real scoring."""
    import math
    try:
        score_fn = _load_score_fn(program_path)
        val = float(score_fn(1, 1, 1, 1, 64, 10))
        if not math.isfinite(val):
            raise ValueError(f"score() returned non-finite value: {val}")
        # Return a passing stub — stage2 will compute real metrics including feature dims
        return EvaluationResult(
            metrics={"combined_score": 0.001},
            artifacts={"stage1": "score() callable and returns finite float"},
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={"error_type": type(e).__name__, "error_message": str(e)},
        )


def evaluate_stage2(program_path: str) -> EvaluationResult:
    """Full evaluation (cascade stage 2)."""
    return evaluate(program_path)


if __name__ == "__main__":
    import json
    path = sys.argv[1] if len(sys.argv) > 1 else str(_THIS_DIR / "initial_program.py")
    t0 = time.time()
    result = evaluate(path)
    print(json.dumps(result.metrics, indent=2))
    print(result.artifacts.get("hit_rate_table", ""))
    print(f"\nevaluate() took {time.time() - t0:.2f}s")
