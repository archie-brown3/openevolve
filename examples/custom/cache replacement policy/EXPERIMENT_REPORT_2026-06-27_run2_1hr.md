# Cache-Replacement Policy Evolution — 1-Hour Run Analysis

**Run:** `openevolve_20260627_163104` · 16:31:04 → 17:31:41 (**60.6 min**, fixed time
budget) · **272 iterations** · stock OpenEvolve · model Qwen3.5-9B (thinking off) via
llama.cpp.

## Verdict

**Not promising — but the key correction is that the results are not "noisy," they
are *deterministically degenerate*.** Evaluation is seeded and exact (a given program
always scores identically), so there is no stochastic noise. The problem is that the
**fitness function is a bad proxy**: it is dominated by an artifact, quantized into a
coarse step-function, and *blind* to the very dimension the model spends the hour
varying. The optimiser found its best region by **iteration 6** and then sat flat for
the remaining 244 iterations. Stock OpenEvolve ran cleanly; the experiment's *metric*,
not the engine, is what makes the output unusable.

---

## 1. Run health — clean (config fixes worked)

| metric | value |
|---|---|
| iterations attempted | 272 |
| LLM calls (HTTP 200 / non-200) | 276 / **0** |
| length-rejections | **0** (was 10/50 last run; `max_code_length 8000` fixed it) |
| context-exceeded / eval errors / retries | 0 / 0 / 0 |
| throughput | ~4.5 iters/min; ~40–45 tok/s gen |

Operational note: this run shared one `openevolve_output/` and one llama.cpp server
with **two other runs that hung at 15:53 and stayed alive 2h+** (now killed). The
shared checkpoints are cross-run contaminated — this analysis is reconstructed from
the `163104` **log + timestamp-filtered program records**, not the global `best/` dir.

---

## 2. Trajectory — plateau at iteration 6

| iteration | best combined_score |
|---|---|
| 3 | 0.0267 |
| 4 | 0.2500 |
| **6** | **0.3620** |
| 75 | 0.3654 |
| 154 | 0.3663 |
| 250 | **0.3671** (final) |

**+0.005 over the final 244 iterations.** The hour bought nothing after iteration 6.

---

## 3. Best program & why the score is mostly artifact

Iteration 250, `combined_score = 0.3671`:
```python
return frequency + (3.5 / (recency + 1)) - (age / (now + 1))
```
A frequency-dominant hybrid with a small recency term and an age penalty.

| workload | LRU | best | Δ vs LRU | gap closed | contribution |
|---|---|---|---|---|---|
| temporal_locality | ~0.630 | **0.268** | **−0.362** ✗ | 0 (clamped) | 0.00 |
| stable_frequency | 0.583 | **0.650** | +0.066 ✓ | 0.48 | **0.12** (real) |
| scan | 0.487 | 0.494 | +0.007 (noise) | 0.96 | **0.24** (artifact) |
| churn | ~0.313 | **0.230** | **−0.083** ✗ | 0 (clamped) | 0.00 |

**~66% of the best score is the `scan` artifact** (0.007 hit-rate over a 0.007
headroom → reads as "96% closed"). The only genuine gain is an LFU-style win on
`stable_frequency`, **bought by regressing temporal_locality and churn below LRU.**
This is not a better generalist; it's a frequency specialist the metric overrates.

---

## 4. Search collapsed into coefficient-tweaking (the flat landscape, proven)

198 distinct expressions were generated, but the **top ~14 are all the same template**
with one constant tweaked:

```
frequency + (k / (recency + 1)) - (age / (now + 1))      k ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 10.0}
```
All score **0.365–0.367**, all with `stable_frequency=0.65`, `churn=0.23`, `scan`
saturated. Only `temporal_locality` moves (0.24 → 0.30) — **and the fitness cannot see
it**, because temporal_locality stays below the LRU floor where the gap clamps to 0.

So the model *was* exploring (temporal_locality varied across runs), but **every
variation it made was invisible to `combined_score`.** 244 iterations of zero gradient.

---

## 5. Prompt strategy (stock) & confabulations

The run used **full-rewrite** prompts (`full_rewrite_user`, ~15.5 k chars / ~5.7 k
tokens). Structure:
- **Current Program Information**: fitness, MAP-Elites coordinates, and auto-generated
  *Focus areas* — e.g. *"Fitness declined 0.3663 → 0.3634. Consider revising"*,
  *"Consider simplifying — code length exceeds 500 characters."*
- **Evolution History**: last 2 attempts with metrics, each labelled *"Outcome: Mixed
  results."*
- **Top Performing Programs**: full source of the top programs — **ranked by the
  artifact-dominated `combined_score`** — embedded verbatim *including the fixed
  `simulate` harness*, repeated in every exemplar (most of the token bloat).

Two pathologies this creates:
1. **The prompt teaches the artifact.** "Top" exemplars are the scan-artifact winners,
   so the model is steered to imitate them — the fitness flaw propagates into the
   prompt.
2. **Confabulated rationales.** Evolved docstrings assert claims the *same prompt's
   metrics contradict* — e.g. *"Recency (w=3.5): Dominant factor for temporal
   locality"* while `temporal_locality_hitrate = 0.27` (far below LRU's 0.63), and the
   recency term `3.5/(recency+1)` is negligible once recency > ~3. The model narrates
   plausible stories rather than reasoning from the numbers it was given. The
   "simplify / code exceeds 500 chars" hint also actively nudges it toward trivial
   coefficient edits instead of structural change.

---

## 6. Fitness pathologies that make results unusable (the core answer)

Precisely characterised and quantified:

1. **Tiny-denominator artifact (`scan`).** Headroom LRU→OPT ≈ 0.007 (noise). The
   normalised gap turns a meaningless 0.007 change into ~0.96 → **66% of the best
   score**. The optimiser is rewarded for winning a non-signal.
2. **LRU-floor clamping makes the metric blind.** `gap = clamp((hr−lru)/(opt−lru),
   0,1)` → any workload *below* LRU contributes exactly 0, regardless of *how far*
   below. The model's entire variation (temporal_locality 0.24–0.30, churn changes)
   lives below the LRU floor and is therefore **invisible** — no gradient, hence the
   iteration-6 plateau.
3. **Quantization → step-function landscape.** Scores cluster on a coarse lattice
   (0.0, 0.125, 0.25, 0.27, 0.28, 0.36, 0.367) because fitness = sum of a few discrete
   per-workload gap states. An EA cannot hill-climb a staircase; it must *stumble* onto
   the right discrete combo, which it did at iteration 6.
4. **It is not stochastic noise — it is a degenerate proxy.** Evaluation is
   deterministic (seeded traces; the seed reproduces LRU at exactly 0.0). So "noisy and
   unusable" is more precisely **low-resolution + artifact-dominated + floor-blind**.
   This distinction matters: you can't fix it with more trials/averaging; you must
   redesign the metric.
5. **MAP-Elites diversity collapses.** Feature axes are `stable_frequency_hitrate ×
   churn_hitrate`; both pin (0.65, 0.23) for the whole winning family → the archive
   concentrates in ~one cell, so island/quality-diversity pressure does nothing.
6. **`cascade_thresholds: [0.0]`** admits every non-crashing program — no real gating,
   so compute isn't focused.

---

## 7. Recommendations (the stock baseline → feature-branch comparison)

**Redesign the fitness (must-do — nothing is comparable until this is fixed):**
- Replace clamp-at-LRU with a **signed, continuous** reward per workload, e.g.
  `(hr_cand − hr_lru) / (hr_opt − hr_lru)` **without** the lower clamp (allow negative),
  so regressions are *seen* and the landscape has a gradient. Average (optionally
  weighted by headroom).
- **Drop or floor near-zero-headroom workloads** (`scan`, headroom 0.007): exclude
  when `hr_opt − hr_lru < 0.02`, or fix the `scan` generator to create real LRU
  scan-eviction headroom. Keep it as a reported diagnostic, not a fitness term.
- Consider a smoother target than per-workload gap (e.g. mean normalised hit-rate, or
  distance-to-OPT) to de-quantise the landscape.

**Prompt strategy (stock weaknesses to note vs the paper-feature branch):**
- Stop embedding the fixed `simulate` in every exemplar — show only the `score()` body
  (cuts ~70% of prompt tokens, lets you raise exemplar count or context).
- Add **explicit per-workload, vs-LRU feedback** ("temporal_locality 0.27 < LRU 0.63 —
  you regressed it") to counter confabulation and give the model the gradient the
  scalar hides.
- Rank exemplars by a *fixed, hardened* fitness, not the artifact score.

**Process / methodology (so stock vs feature-branch is valid):**
- **Canonical run protocol**: unique `output_dir` per run, one dedicated server, fixed
  seed, no concurrent runs. Today's contamination (3 overlapping runs, shared dir,
  hung processes) must not recur or the comparison is meaningless.
- Re-run the seed gate after any evaluator change (seed must still score 0.0).

**For the feature branch (EoH / and the others to confirm):** their contributions
(reflection, diversity mechanisms, richer prompt strategies) should be benchmarked on
the **hardened metric** — on the current metric they would also just climb the scan
artifact, and any "improvement" would be confabulated.

---

## 8. Provenance
- Run log: `openevolve_output/logs/openevolve_20260627_163104.log` (snapshot in
  scratchpad `run163104.log`).
- Best program (this run): id `42ca6e15-…`, iter 250, in
  `checkpoints/checkpoint_260/programs/`. (Global `best/` dir is cross-run contaminated
  — do not use.)
- Prompts/responses: persisted in each program JSON's `prompts` field (`log_prompts`
  default on) — used for §5.
- Reference LRU/OPT hit rates: `resources/baselines.md`.
