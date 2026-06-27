"""
Evaluator for the cache-replacement-policy evolution experiment.

Loads a candidate program (which exposes the evolved ``score()`` plus the fixed
``simulate()`` — see ``initial_program.py`` and ``IMPLEMENTATION.md`` §A), runs it
over the synthetic workload suite, and scores it by how much of the **LRU → Belady
(OPT)** hit-rate gap it closes, averaged across workloads.

Design (see ``IMPLEMENTATION.md`` §A.4/§A.5):
  * The candidate provides ``simulate(trace, capacity) -> hit_rate``; the evaluator
    never calls ``score`` directly.
  * Reference policies (LRU, LFU, ARC, Belady) are implemented HERE as fixed, trusted
    code — never part of the evolvable surface. Fitness uses only LRU + Belady; LFU
    and ARC are reported for context.
  * Traces are seeded, so evaluation is deterministic: identical code → identical
    score (required for clean preference labels downstream).
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import math
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult

# --- make resources/trace_generators.py importable -------------------------
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "resources"))
import trace_generators  # noqa: E402

# --- evaluation knobs (tuned for ~1-3 s per evaluation; see plan) ----------
SEED = 42                      # RNG seed for trace generation (reproducibility)
EVAL_LENGTH = 8000             # truncate generated traces for speed
CAPACITIES = [64, 256]         # eval at two cache sizes; eviction is O(C) per miss
WORKLOADS = ["temporal_locality", "stable_frequency", "scan", "churn"]
PER_CALL_TIMEOUT = 15          # seconds; guards a pathological candidate

# Metric names exposed per workload (these double as MAP-Elites feature dims).
HITRATE_METRIC = {w: f"{w}_hitrate" for w in WORKLOADS}


# ===========================================================================
# Fixed reference policies (trusted; NOT evolvable). Each returns a hit rate.
# ===========================================================================
def _lru(trace, cap):
    cache = OrderedDict()
    hits = 0
    for k in trace:
        if k in cache:
            hits += 1
            cache.move_to_end(k)
        else:
            if len(cache) >= cap:
                cache.popitem(last=False)
            cache[k] = True
    return hits / len(trace)


def _lfu(trace, cap):
    freq, tie = {}, {}
    hits = 0
    for t, k in enumerate(trace):
        if k in freq:
            hits += 1
            freq[k] += 1
            tie[k] = t
        else:
            if len(freq) >= cap:
                victim = min(freq, key=lambda x: (freq[x], tie[x]))
                del freq[victim]
                del tie[victim]
            freq[k] = 1
            tie[k] = t
    return hits / len(trace)


def _arc(trace, cap):
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p = 0
    hits = 0

    def replace(k):
        nonlocal p
        if t1 and (len(t1) > p or (k in b2 and len(t1) == p)):
            old, _ = t1.popitem(last=False)
            b1[old] = True
        else:
            old, _ = t2.popitem(last=False)
            b2[old] = True

    for k in trace:
        if k in t2:
            hits += 1
            t2.move_to_end(k)
        elif k in t1:
            hits += 1
            del t1[k]
            t2[k] = True
        elif k in b1:
            p = min(cap, p + max(1, len(b2) // max(1, len(b1))))
            replace(k)
            del b1[k]
            t2[k] = True
        elif k in b2:
            p = max(0, p - max(1, len(b1) // max(1, len(b2))))
            replace(k)
            del b2[k]
            t2[k] = True
        else:
            if len(t1) + len(b1) == cap:
                if len(t1) < cap:
                    b1.popitem(last=False)
                    replace(k)
                else:
                    t1.popitem(last=False)
            elif len(t1) + len(t2) + len(b1) + len(b2) >= cap:
                if len(t1) + len(t2) + len(b1) + len(b2) == 2 * cap:
                    b2.popitem(last=False)
                replace(k)
            t1[k] = True
    return hits / len(trace)


def _belady(trace, cap):
    """Oracle upper bound: evict the resident whose next use is furthest ahead."""
    nxt = [0] * len(trace)
    last = {}
    for i in range(len(trace) - 1, -1, -1):
        k = trace[i]
        nxt[i] = last.get(k, math.inf)
        last[k] = i
    cache = {}  # key -> next-use index
    hits = 0
    for i, k in enumerate(trace):
        if k in cache:
            hits += 1
        else:
            if len(cache) >= cap:
                victim = max(cache, key=lambda x: cache[x])
                del cache[victim]
        cache[k] = nxt[i]
    return hits / len(trace)


# ===========================================================================
# Workload suite + reference hit rates (computed once, cached at module level)
# ===========================================================================
_SUITE = None     # OrderedDict[name -> trace]
_REFS = None      # {name: {cap: {"lru","lfu","arc","opt"}}}


def _get_suite_and_refs():
    """Lazily build the trace suite and precompute reference hit rates. These are
    fixed inputs, so computing them once per worker process is enough."""
    global _SUITE, _REFS
    if _SUITE is not None:
        return _SUITE, _REFS
    suite = trace_generators.generate_all(seed=SEED, length=EVAL_LENGTH)
    refs = {}
    for name in WORKLOADS:
        trace = suite[name]
        refs[name] = {}
        for cap in CAPACITIES:
            refs[name][cap] = {
                "lru": _lru(trace, cap),
                "lfu": _lfu(trace, cap),
                "arc": _arc(trace, cap),
                "opt": _belady(trace, cap),
            }
    _SUITE, _REFS = suite, refs
    return _SUITE, _REFS


def _clamp01(x):
    return max(0.0, min(1.0, x))


def _run_with_timeout(func, args, timeout):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(func, *args).result(timeout=timeout)


def _load_candidate(program_path):
    spec = importlib.util.spec_from_file_location("candidate", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# Core scoring
# ===========================================================================
def _evaluate(program_path, workloads):
    """Run the candidate over `workloads`, return (metrics, artifacts).

    Per workload: gap_closed = mean over capacities of
        clamp((hr_cand - hr_lru) / (hr_opt - hr_lru), 0, 1)
    (a saturated workload where hr_opt == hr_lru contributes 1.0).
    combined_score = mean(gap_closed) over workloads.
    """
    suite, refs = _get_suite_and_refs()

    module = _load_candidate(program_path)
    if not hasattr(module, "simulate") or not hasattr(module, "score"):
        return (
            {"combined_score": 0.0, "error": "missing simulate/score"},
            {
                "error_type": "MissingFunction",
                "error_message": "Program must define both score() and simulate().",
                "suggestion": "Keep the fixed simulate() harness and a score() in the EVOLVE-BLOCK.",
            },
        )

    metrics = {}
    gaps = []
    rows = ["workload            cap    cand    LRU    LFU    ARC    OPT     gap"]
    errors = []

    for name in workloads:
        trace = suite[name]
        cand_rates, wl_gaps = [], []
        for cap in CAPACITIES:
            ref = refs[name][cap]
            try:
                hr = _run_with_timeout(module.simulate, (trace, cap), PER_CALL_TIMEOUT)
                hr = float(hr)
                if math.isnan(hr) or math.isinf(hr):
                    raise ValueError(f"non-finite hit rate: {hr}")
            except Exception as e:  # broken candidate → worst score on this cell
                hr = 0.0
                errors.append(f"{name}@{cap}: {type(e).__name__}: {e}")
            cand_rates.append(hr)
            denom = ref["opt"] - ref["lru"]
            if denom <= 1e-9:
                # Saturated cell: LRU already matches OPT, so there is no LRU→OPT
                # headroom and thus no signal. Exclude it from fitness entirely
                # (rewarding it would give spurious credit for merely tying LRU).
                gap = None
            else:
                gap = _clamp01((hr - ref["lru"]) / denom)
                wl_gaps.append(gap)
            rows.append(
                f"{name:<18}{cap:>5}{hr:7.3f}{ref['lru']:7.3f}{ref['lfu']:7.3f}"
                f"{ref['arc']:7.3f}{ref['opt']:7.3f}"
                + ("     n/a" if gap is None else f"{gap:8.3f}")
            )
        metrics[HITRATE_METRIC[name]] = sum(cand_rates) / len(cand_rates)
        if wl_gaps:  # workload contributes only if it had at least one cell with headroom
            gaps.append(sum(wl_gaps) / len(wl_gaps))

    metrics["combined_score"] = sum(gaps) / len(gaps) if gaps else 0.0

    artifacts = {"hit_rate_table": "\n".join(rows)}
    if errors:
        artifacts["candidate_errors"] = "\n".join(errors)
    return metrics, artifacts


def evaluate(program_path):
    """Full evaluation over all workloads (OpenEvolve entry point)."""
    try:
        metrics, artifacts = _evaluate(program_path, WORKLOADS)
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "full_traceback": traceback.format_exc(),
                "suggestion": "Check the program imports/syntax and the score() return value.",
            },
        )


def evaluate_stage1(program_path):
    """Cheap cascade gate: one workload (churn), to reject broken programs fast."""
    try:
        metrics, artifacts = _evaluate(program_path, ["churn"])
        # Stage-1 fitness is just the single-workload gap; keep the same key.
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": str(e)},
            artifacts={"error_type": type(e).__name__, "error_message": str(e)},
        )


def evaluate_stage2(program_path):
    """Full evaluation (cascade stage 2)."""
    return evaluate(program_path)


if __name__ == "__main__":
    # Manual smoke test against whatever program path is given.
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else str(_THIS_DIR / "initial_program.py")
    t0 = time.time()
    result = evaluate(path)
    dt = time.time() - t0
    print(json.dumps(result.metrics, indent=2))
    print(result.artifacts.get("hit_rate_table", ""))
    print(f"\nevaluate() took {dt:.2f}s")
