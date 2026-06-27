# Stage 1 — Technical Implementation Plan (Milestone M0)

**Goal:** replace the degenerate fitness (see `FINDINGS.md` F2) with a
**framework-agnostic benchmark** that scores a policy by **signed skill versus the best
human heuristic**, robustly (multi-seed, train/test, headroom-gated), and add **S3-FIFO**
as a reported modern baseline.

**Outcome:** a trustworthy, reusable measuring instrument. No evolution-framework code
depends on it; `evaluator.py` becomes a thin OpenEvolve adapter.

---

## Architecture

```
trace_generators.py   (existing) — seeded synthetic workloads
        │
benchmark.py   (NEW) — the scientific instrument:
        │   reference policies (LRU, LFU, ARC, S3-FIFO, Belady)
        │   signed-skill fitness, headroom gate, multi-seed, train/test
        │   evaluate_policy(simulate_fn, split="train") -> BenchmarkResult
        │   __main__ --selfcheck
        ▼
evaluator.py   (REWORK) — thin OpenEvolve adapter:
            loads candidate's simulate(); calls benchmark.evaluate_policy();
            maps BenchmarkResult -> EvaluationResult(metrics, artifacts)
```

The benchmark takes a **`simulate_fn(trace, capacity) -> hit_rate`** (what the candidate
already exposes), so it is agnostic to how the policy was produced.

---

## Tasks (chunked)

### T1.1 — Create `benchmark.py` skeleton + move reference policies
- New module `benchmark.py`. Move `_lru`, `_lfu`, `_arc`, `_belady` out of
  `evaluator.py` into here (single source of truth).
- Define `BenchmarkResult` (dataclass): `combined_score`, `per_workload` (dict of
  hit rates for cand/LRU/LFU/ARC/S3FIFO/OPT + skill), `matches_best_count`,
  `distance_to_opt`, `split`.
- **Done when:** `from benchmark import evaluate_policy` imports cleanly; existing
  baselines reproduce the numbers in `resources/baselines.md`.

### T1.2 — Add S3-FIFO reference policy
- Implement `_s3fifo(trace, capacity)` (pure Python, ~30 lines): small FIFO (S, ~10%) +
  main FIFO (M) + ghost FIFO (G); objects get a 2-bit frequency counter; on eviction
  from S, promote to M if accessed else move to G; demote from M with the counter.
- Reported-only (not part of `B`).
- **Done when:** S3-FIFO runs on all four workloads and returns sane hit rates
  (≥ LRU on scan; documented in a quick comparison print).

### T1.3 — Signed-skill fitness + headroom gate
- `B(w) = max(LRU, LFU, ARC)` per workload.
- `skill(w) = (hr_cand − B) / (OPT − B)`; **no lower clamp** (negative allowed).
- **Exclude** workloads with `OPT − B < HEADROOM_EPS` (default 0.02) from the mean.
- `combined_score = mean(skill over retained workloads)` (may be negative).
- **Done when:** the LRU seed scores ≈0 on workloads where LRU is best and **negative**
  where LFU/ARC beat it; `scan` is auto-excluded.

### T1.4 — Multi-seed averaging + train/test split
- `TRAIN_SEEDS = [1,2,3]`, `TEST_SEEDS = [4,5]` (configurable). Average hit rates over
  the split's seeds × capacities before computing skill.
- `evaluate_policy(simulate_fn, split="train")` is the optimisation path;
  `split="test"` is report-only for held-out evaluation.
- **Done when:** averaging is deterministic (same seeds → identical result) and the
  score lattice is visibly finer than the old quantized one.

### T1.5 — Preserve MAP-Elites feature metrics
- Continue returning `stable_frequency_hitrate` and `churn_hitrate` (cand hit rates) so
  OpenEvolve's `feature_dimensions` still bin on them.
- **Done when:** `evaluator.py` output contains those keys plus `combined_score`.

### T1.6 — `benchmark.py --selfcheck` (verification harness)
Four assertions (the Stage-1 acceptance test):
1. **Determinism:** evaluate the LRU seed twice → identical `combined_score`.
2. **Anti-gaming:** a frequency-only policy (`score = frequency`) → `combined_score < 0`
   (it regresses temporal_locality).
3. **Smoothness:** a GDSF-like blend (`frequency + α/(recency+1)`) scores **strictly
   between** the LRU seed and an artificial perfect-match upper reference.
4. **Sign sanity:** LRU seed skill = 0 on temporal_locality (LRU is best there),
   negative on stable_frequency (LFU/ARC better).
- **Done when:** `python benchmark.py --selfcheck` prints PASS for all four.

### T1.7 — Rework `evaluator.py` + sync docs/config
- `evaluator.py`: load candidate `simulate`; call `benchmark.evaluate_policy(...,
  split="train")`; map to `EvaluationResult` (metrics incl. `combined_score` + feature
  metrics; artifacts = per-workload table incl. S3-FIFO + skill + matches_best_count).
- Keep `evaluate_stage1` (cheap one-workload gate) / `evaluate_stage2 = evaluate`.
- Update `IMPLEMENTATION.md §A.4/§A.5` to the new fitness; note `scan` is gated out;
  add a one-line `config.yaml` comment.
- **Done when:** `python evaluator.py initial_program.py` runs and matches the
  benchmark's direct result for the seed.

---

## Verification (Stage-1 acceptance)
- `python benchmark.py --selfcheck` → all four checks PASS.
- `python evaluator.py initial_program.py` → deterministic signed score for the LRU
  seed; per-workload table present and auditable.
- `python -c "from openevolve.config import load_config; load_config('config.yaml')"`
  still loads.

## Out of scope for Stage 1 (later milestones)
- W-TinyLFU / SIEVE baselines and libCacheSim integration → M2.
- Real traces (FIU/MSR) → M2.
- Stateful policy interface (`on_access`/`on_evict`) → M4/M5.
- Any change to the evolution loop itself (this stage only changes *measurement*).

## Risks / notes
- The LRU seed will now score **negative overall** (it loses to LFU/ARC on frequency).
  That is correct and intended — it means the optimiser finally has a gradient to climb
  toward "best-of-all". Re-baseline expectations in `IMPLEMENTATION.md`.
- ARC reference is the simplified standard implementation; cross-check against
  libCacheSim at M2.
