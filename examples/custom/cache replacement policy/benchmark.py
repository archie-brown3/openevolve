"""
Framework-agnostic benchmark for cache replacement policies.

Scores a candidate score() function by signed skill versus the best human
heuristic (B = max(LRU, LFU, ARC)) across synthetic workloads and seeds.

  evaluate_policy(score_fn, split="train") -> BenchmarkResult

No OpenEvolve imports. Run standalone:
    python benchmark.py --selfcheck
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import libcachesim as lcs
from libcachesim import PluginCache, Request

sys.path.insert(0, str(Path(__file__).resolve().parent / "resources"))
import trace_generators  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_SEEDS  = [1, 2, 3]
TEST_SEEDS   = [4, 5]
CAPACITIES   = [64, 256]
EVAL_LENGTH  = 8_000
HEADROOM_EPS = 0.02   # exclude workloads where OPT - B < this
WORKLOADS    = ["temporal_locality", "stable_frequency", "scan", "churn"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _hit_rate(cache_obj, trace: list[int]) -> float:
    hits = sum(cache_obj.get(Request(obj_id=k, obj_size=1)) for k in trace)
    return hits / len(trace)


def _next_access_times(trace: list[int]) -> list[int]:
    """Precompute virtual next-access time for each position (required for Belady)."""
    NEVER = 10 ** 9
    nxt: list[int] = [NEVER] * len(trace)
    last: dict[int, int] = {}
    for i in range(len(trace) - 1, -1, -1):
        k = trace[i]
        nxt[i] = last.get(k, NEVER)
        last[k] = i
    return nxt


def _hit_rate_belady(cap: int, trace: list[int]) -> float:
    nxt = _next_access_times(trace)
    cache = lcs.Belady(cache_size=cap)
    hits = sum(
        cache.get(Request(obj_id=k, obj_size=1, next_access_vtime=nxt[i]))
        for i, k in enumerate(trace)
    )
    return hits / len(trace)


def _make_plugin(score_fn: Callable, cap: int) -> PluginCache:
    """Wrap a stateless score() function as a libCacheSim PluginCache."""
    def init_hook(_):
        return {"__now__": 0}

    def hit_hook(data, req):
        data["__now__"] += 1
        meta = data[req.obj_id]
        meta[1] = data["__now__"]   # last_access
        meta[2] += 1                # freq

    def miss_hook(data, req):
        now = data["__now__"]
        data[req.obj_id] = [now, now, 1, req.obj_size]  # [insert, last_access, freq, size]
        data["__now__"] += 1

    def eviction_hook(data, _req):
        now = data["__now__"]
        victim = min(
            (k for k in data if k != "__now__"),
            key=lambda k: score_fn(
                now - data[k][0], data[k][2], now - data[k][1],
                data[k][3], cap, now,
            ),
        )
        del data[victim]  # must remove here; remove_hook is not called after eviction
        return victim

    def remove_hook(data, obj_id):
        data.pop(obj_id, None)  # guard for explicit removes

    def free_hook(data):
        data.clear()

    return PluginCache(
        cache_size=cap,
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
    )


# ---------------------------------------------------------------------------
# Precomputed baselines (computed once per worker process on first use)
# ---------------------------------------------------------------------------
_baseline_cache: dict[tuple, dict] = {}  # {(workload, seed, cap): {lru, lfu, arc, opt, s3fifo}}


def _baselines(workload: str, seed: int, cap: int) -> dict:
    key = (workload, seed, cap)
    if key not in _baseline_cache:
        trace = trace_generators.generate_all(seed=seed, length=EVAL_LENGTH)[workload]
        _baseline_cache[key] = {
            "lru":    _hit_rate(lcs.LRU(cache_size=cap),    trace),
            "lfu":    _hit_rate(lcs.LFU(cache_size=cap),    trace),
            "arc":    _hit_rate(lcs.ARC(cache_size=cap),    trace),
            "opt":    _hit_rate_belady(cap, trace),          # requires next_access_vtime
            "s3fifo": _hit_rate(lcs.S3FIFO(cache_size=cap), trace),
        }
    return _baseline_cache[key]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass
class WorkloadResult:
    hr_cand:  float
    hr_lru:   float
    hr_lfu:   float
    hr_arc:   float
    hr_opt:   float
    hr_s3fifo: float
    skill:    float | None   # None = excluded (headroom gate)
    excluded: bool


@dataclass
class BenchmarkResult:
    combined_score: float
    per_workload:   dict[str, WorkloadResult] = field(default_factory=dict)
    n_retained:     int = 0
    split:          str = "train"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def evaluate_policy(score_fn: Callable, split: str = "train",
                    workloads: list[str] | None = None) -> BenchmarkResult:
    """Score a candidate score() function against all baselines.

    Args:
        score_fn: callable with signature score(age, freq, recency, size, cap, now) -> float
        split: "train" (seeds 1-3) or "test" (seeds 4-5)
        workloads: subset of WORKLOADS (default: all four)
    """
    seeds = TRAIN_SEEDS if split == "train" else TEST_SEEDS
    wl_list = workloads or WORKLOADS

    all_skills: list[float] = []
    wl_results: dict[str, WorkloadResult] = {}

    for workload in wl_list:
        # Accumulate hit rates over seeds x capacities, then average
        sums = {"cand": 0.0, "lru": 0.0, "lfu": 0.0, "arc": 0.0, "opt": 0.0, "s3fifo": 0.0}
        n_cells = len(seeds) * len(CAPACITIES)

        for seed in seeds:
            trace = trace_generators.generate_all(seed=seed, length=EVAL_LENGTH)[workload]
            for cap in CAPACITIES:
                refs = _baselines(workload, seed, cap)
                hr_cand = _hit_rate(_make_plugin(score_fn, cap), trace)
                sums["cand"]   += hr_cand
                sums["lru"]    += refs["lru"]
                sums["lfu"]    += refs["lfu"]
                sums["arc"]    += refs["arc"]
                sums["opt"]    += refs["opt"]
                sums["s3fifo"] += refs["s3fifo"]

        avgs = {k: v / n_cells for k, v in sums.items()}
        B   = max(avgs["lru"], avgs["lfu"], avgs["arc"])
        denom = avgs["opt"] - B

        if denom < HEADROOM_EPS:
            skill = None
            excluded = True
        else:
            skill = (avgs["cand"] - B) / denom   # signed, no clamp
            excluded = False
            all_skills.append(skill)

        wl_results[workload] = WorkloadResult(
            hr_cand   = avgs["cand"],
            hr_lru    = avgs["lru"],
            hr_lfu    = avgs["lfu"],
            hr_arc    = avgs["arc"],
            hr_opt    = avgs["opt"],
            hr_s3fifo = avgs["s3fifo"],
            skill     = skill,
            excluded  = excluded,
        )

    combined = sum(all_skills) / len(all_skills) if all_skills else 0.0
    return BenchmarkResult(
        combined_score = combined,
        per_workload   = wl_results,
        n_retained     = len(all_skills),
        split          = split,
    )


def format_table(result: BenchmarkResult) -> str:
    lines = [
        f"{'workload':<20} {'cand':>6} {'lru':>6} {'lfu':>6} {'arc':>6} "
        f"{'opt':>6} {'s3fifo':>7} {'skill':>8}",
        "-" * 73,
    ]
    for name, w in result.per_workload.items():
        skill_str = "excluded" if w.excluded else f"{w.skill:+.4f}"
        lines.append(
            f"{name:<20} {w.hr_cand:6.4f} {w.hr_lru:6.4f} {w.hr_lfu:6.4f} "
            f"{w.hr_arc:6.4f} {w.hr_opt:6.4f} {w.hr_s3fifo:7.4f} {skill_str:>8}"
        )
    lines.append(
        f"\ncombined_score = {result.combined_score:+.6f}  "
        f"(mean over {result.n_retained}/{len(result.per_workload)} workloads, "
        f"split={result.split})"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-check (acceptance test for Stage 1)
# ---------------------------------------------------------------------------
def _selfcheck():
    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"PASS  {name}")
            passed += 1
        else:
            print(f"FAIL  {name}: {detail}")
            failed += 1

    lru_score  = lambda age, freq, recency, size, cap, now: -recency
    lfu_score  = lambda age, freq, recency, size, cap, now: freq          # anti-gaming test
    gdsf_score = lambda age, freq, recency, size, cap, now: freq / (recency + 1)

    # 1. Determinism
    r1 = evaluate_policy(lru_score)
    r2 = evaluate_policy(lru_score)
    check("Determinism", r1.combined_score == r2.combined_score,
          f"{r1.combined_score} != {r2.combined_score}")

    # 2. Anti-gaming: LFU-only policy must score negative (regresses temporal_locality)
    r_lfu = evaluate_policy(lfu_score)
    check("Anti-gaming (LFU-only < 0)", r_lfu.combined_score < 0,
          f"combined_score = {r_lfu.combined_score:+.4f}")

    # 3. Smoothness: GDSF blend must score strictly between LRU seed and skill=1
    r_gdsf = evaluate_policy(gdsf_score)
    r_lru  = evaluate_policy(lru_score)
    # GDSF should beat LRU seed (which is baseline on temporal but negative on others)
    check("Smoothness (GDSF > LRU seed)", r_gdsf.combined_score > r_lru.combined_score,
          f"GDSF={r_gdsf.combined_score:+.4f} LRU={r_lru.combined_score:+.4f}")

    # 4. Sign sanity on LRU seed: skill <= 0 everywhere (LRU never beats its own baselines).
    #    ARC can beat LRU on temporal_locality, so skill can be negative there too.
    w_tl = r_lru.per_workload.get("temporal_locality")
    w_sf = r_lru.per_workload.get("stable_frequency")
    check("Sign sanity: temporal_locality skill <= 0 (or excluded)",
          w_tl is None or w_tl.excluded or w_tl.skill <= 0.001,
          f"skill={w_tl.skill if w_tl else 'N/A'}")
    check("Sign sanity: stable_frequency skill < 0",
          w_sf is not None and not w_sf.excluded and w_sf.skill < 0,
          f"skill={w_sf.skill if w_sf else 'N/A'}")

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        ok = _selfcheck()
        sys.exit(0 if ok else 1)

    # Standalone: score the LRU seed and print the table
    lru_score = lambda age, freq, recency, size, cap, now: -recency
    t0 = time.time()
    result = evaluate_policy(lru_score)
    print(format_table(result))
    print(f"\n(took {time.time() - t0:.2f}s)")
