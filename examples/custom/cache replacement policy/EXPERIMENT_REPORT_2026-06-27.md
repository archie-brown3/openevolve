# Cache-Replacement Policy Evolution — Experiment Report

**Run:** `openevolve_20260627_155635` · **Date:** 2026-06-27 ·
**Duration:** ~11 min (15:56:35 → 16:07:52) · **Iterations:** 50 requested

## Verdict (TL;DR)

The **infrastructure works end-to-end** and the local LLM produced **genuine,
varied logic mutations** (a real step up from the Phase-1 0.6B failure). However,
the headline result — best `combined_score = 0.284` vs the LRU-equivalent seed's
`0.0` — is **largely a measurement artifact, not a genuine improvement**. The
"best" evolved policy is actually *worse than LRU on two of four workloads*. The run
surfaced two concrete, fixable problems: a **degenerate fitness normalization on the
`scan` workload**, and a **`max_code_length` cap that wasted 20% of iterations**.

This is a successful *plumbing* run and a useful *negative* result on methodology —
not yet evidence that evolution discovers better cache heuristics.

---

## 1. Setup

| Component | Value |
|---|---|
| Model | `unsloth/Qwen3.5-9B-MTP-GGUF` (Q4_K_M), thinking disabled server-side |
| Inference | llama.cpp `llama-server`, CUDA, `-ngl 99`, `n_parallel=4`, OpenAI endpoint :8080 |
| Hardware | RTX 3070 8 GB; ~6.5 GB VRAM in use; **40–45 tok/s** gen, **1786 tok/s** prompt |
| Evolve target | single function `score(age, frequency, recency, size, cache_size, now)` |
| Seed | `return -recency` (== LRU, scores 0.0 by construction) |
| Workloads | 4 synthetic: temporal_locality, stable_frequency, scan, churn (seeded) |
| Fitness | mean over workloads of `clamp((hr_cand − hr_lru)/(hr_opt − hr_lru), 0, 1)` |
| Config | 50 iters, 4 islands, diff-based, `max_code_length=5000`, MAP-Elites on `stable_frequency_hitrate` × `churn_hitrate` |

---

## 2. Run statistics

- **Programs evaluated/stored:** 41 of 50 iterations.
- **Failed iterations (10):** #2, 7, 11, 18, 21, 23, 27, 32, 36, 41 — **all** the same
  cause: *"Generated code exceeds maximum length (>5000 chars)."* The model kept
  appending verbose docstrings/comments past the cap → the whole iteration was
  discarded. **20% of compute wasted.**
- **Context errors:** **0** this run (the `500: Context size exceeded` you saw earlier
  was the *previous* run on the 8 192 window; this run's prompts (~5 747 tokens) fit).
- **LLM calls:** 50 × HTTP 200. No timeouts, no refusals.
- **Best-program timeline:** `0.2742` at iter 3 → `0.2844` at iter 17 → **no further
  improvement for 33 iterations.** The 0.2844 solution was *re-discovered* at iters
  34, 39, 44, 50 → the population converged to a fixed point.

### Score distribution (all evaluations)
| combined_score | count | meaning |
|---|---|---|
| 0.0000 | 35 | no better than LRU (or regressed) |
| 0.1250 | 8 | ~½ of one workload's gap |
| 0.2500 | 16 | exactly one workload "fully closed" |
| 0.2742 | 3 | one workload + a little |
| 0.2844 | 11 | the best cluster |

The scores are **quantized** (0.0, 0.125, 0.25, …). That is the first red flag —
see §4.

---

## 3. The best program

Iteration 17, generation 2, `combined_score = 0.2844`:

```python
def score(age, frequency, recency, size, cache_size, now):
    return frequency * 4 + recency / (age + 1)
```

The model's own comment claims an "Enhanced Hybrid ARC-like policy." In reality the
`frequency * 4` term dominates → this is effectively **LFU with a tiny recency
tie-breaker**. Note the recency term is also *semantically backwards* for our
convention (high `recency` = stale = should be penalised, but it's added
positively); it barely matters because frequency dwarfs it. The model produced
plausible-sounding prose that **misdescribes** what the code does.

### Per-workload reality (best vs LRU)
| workload | LRU (avg) | best (avg) | Δ vs LRU | OPT (avg) |
|---|---|---|---|---|
| temporal_locality | ~0.630 | **0.169** | **−0.461** ✗ | 0.705 |
| stable_frequency | 0.583 | 0.6025 | +0.020 ✓ | 0.721 |
| scan | 0.487 | 0.4938 | **+0.007** (noise) | 0.494 |
| churn | ~0.313 | **0.210** | **−0.103** ✗ | 0.603 |

**The "best" policy is worse than LRU on temporal_locality and churn**, and its only
real gain is a modest LFU-style win on stable_frequency. It is not a better
generalist — it traded recency-awareness for frequency-awareness.

---

## 4. Why `combined_score = 0.284` is misleading (the core finding)

Decomposing the best program's fitness by workload (gap closed, clamped to [0,1]):

| workload | gap closed | contribution |
|---|---|---|
| temporal_locality | 0.0 (regressed) | 0.00 |
| stable_frequency | (0.6025−0.583)/(0.721−0.583) ≈ **0.14** | 0.035 |
| scan | (0.4938−0.487)/(0.494−0.487) ≈ **0.96** | 0.24 |
| churn | 0.0 (regressed) | 0.00 |
| **mean** | | **≈ 0.28** |

**~85% of the score comes from `scan`.** But `scan`'s entire LRU→OPT headroom is
only **0.007 hit-rate** (LRU 0.487, OPT 0.494) — i.e. *noise-level*. A change of
0.0068 — essentially meaningless in practice — registers as "96% of the gap closed"
because the denominator is tiny. The discrete score clusters (0.25 = "one workload
maxed") are the fingerprint of this: fitness is being driven by which near-saturated
workload happens to flip, not by real, large improvements.

**Consequence:** the optimiser was rewarded for gaming a degenerate metric rather
than for finding a genuinely better policy. This is the single most important thing
to fix before drawing any scientific conclusion.

---

## 5. What worked (keep these)

1. **Full loop is solid:** sample → prompt → llama.cpp → diff apply → evaluate →
   MAP-Elites store → checkpoint. 50/50 calls succeeded.
2. **Real mutations.** Unlike the Phase-1 0.6B (byte-identical cargo-cult edits), the
   9B produced behaviourally diverse policies: `stable_frequency_hitrate` spanned
   0.129–0.6025, `churn_hitrate` 0.207–0.312 across the population. The model is a
   capable mutation operator for this task.
3. **Thinking-disabled serving** gave clean `content` (no `<think>` leakage) — the
   diff parser worked.
4. **Deterministic evaluation:** seeded traces → identical code always scores
   identically (the seed gate reproduced LRU exactly at 0.0). Good for future DPO
   label quality.
5. **GPU throughput** ample: ~11 min for 50 iterations, mostly LLM latency.

---

## 6. Problems found (ranked by impact)

1. **Degenerate fitness on low-headroom workloads (critical).** `scan` (and the
   saturated `temporal_locality@256` cell) have ~0 LRU→OPT headroom, so the
   normalized gap is dominated by noise. **Fix:** exclude any workload whose
   `hr_opt − hr_lru < ε` (e.g. 0.02) from the mean *entirely*, and/or re-tune the
   `scan` generator so it has real, differentiating headroom (§7).
2. **`max_code_length=5000` wasted 20% of iterations.** The model writes long
   comments and overflows the cap. **Fix:** raise to ~8000–10000 and/or instruct
   "no comments, return one expression" in the system message.
3. **Backwards / regressing exploration.** The best policy *regressed* on 2/4
   workloads yet still scored highest — a direct symptom of (1). With a corrected
   metric, such policies would score ~0 and be pruned.
4. **Early convergence / stagnation.** Best found by iter 17, then 33 idle
   iterations re-discovering the same point. Suggests under-exploration (single-
   expression search space is small; temperature/island diversity not paying off).
5. **Uneven islands.** island_generations `[8, 8, 1, 11]`; island 2 held a single
   program — migration/seeding spread work unevenly across only ~50 iters.

---

## 7. Recommendations (next run)

**Must-fix before trusting any score:**
- **Harden the fitness.** In `evaluator.py`, drop workloads with
  `hr_opt − hr_lru < 0.02` from `combined_score` (don't let a 0.007-headroom
  workload contribute). Re-run the seed gate: it should still be 0.0.
- **Fix the `scan` workload** in `trace_generators.py` so LRU genuinely
  under-performs OPT by a meaningful margin (e.g. larger hot set relative to
  capacity, or repeated scans), giving real scan-resistance headroom — otherwise
  drop `scan` from fitness and keep it as a reported diagnostic only.

**Should-fix:**
- Raise `max_code_length` to 8000; add "return a single expression, no comments" to
  the system message to stop comment-bloat overflows.
- Consider weighting fitness toward the high-headroom workloads (`churn` +0.25–0.34,
  `stable_frequency` +0.12–0.16) which carry the real signal.
- Add explicit anti-regression: penalise policies that fall below LRU on any
  high-headroom workload, so an LFU-style specialist can't "win" by trashing
  temporal_locality.

**Nice-to-have:**
- More iterations (150–200) with higher temperature or `num_islands` tuning to fight
  the iter-17 plateau.
- Try `Qwen2.5-Coder-7B-Instruct` (non-thinking, code-specialised) — likely cleaner
  diffs and no comment-bloat, no `--chat-template-kwargs` needed.

---

## 8. Appendix — provenance

- Best program: `openevolve_output/best/best_program.py` (id `8c032b50…`, iter 17).
- Checkpoints: `openevolve_output/checkpoints/checkpoint_{10,20,50}` (41 programs in
  archive at ckpt 50; `best_program_id = 8c032b50…`).
- Logs: `openevolve_output/logs/openevolve_20260627_155635.log` (run),
  `…_155635_llama_server_logs.log` (server).
- Reference hit rates (seed gate, caps 64/256) used for the Δ tables above are in
  `resources/baselines.md`.
