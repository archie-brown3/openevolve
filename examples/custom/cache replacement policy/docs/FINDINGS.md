# Findings — LLM-Evolved Cache Replacement Policies

Concrete, paper-ready findings from the stock-OpenEvolve experiments (June 2026).
Each is stated as **claim → evidence → why it matters**. Raw run records are in the
`EXPERIMENT_REPORT_*` files; this document distils them.

Setup common to all runs: evolve a stateless eviction scoring function
`score(age, frequency, recency, size, cache_size, now)` (evict the lowest-scored
resident); fixed Python simulator; four seeded synthetic workloads
(temporal-locality, stable-frequency, scan, churn); local Qwen3.5-9B (Q4_K_M,
thinking disabled) served by llama.cpp on an 8 GB RTX 3070.

---

## F1 — A stateless scoring function cannot beat the best human heuristic on hit-rate on this suite

**Claim.** Within the stateless scoring-function policy class, there is essentially no
room to exceed the best classical heuristic on these workloads — so "outperform human
heuristics on hit-rate" is not an achievable objective as originally framed.

**Evidence.** Reference hit rates at capacity 300 (`resources/baselines.md`):

| workload | LRU | LFU | ARC | best human | Belady OPT | headroom (best→OPT) |
|---|---|---|---|---|---|---|
| temporal_locality | 0.697 | 0.191 | 0.694 | 0.697 | 0.709 | **0.012** |
| stable_frequency | 0.692 | 0.742 | 0.742 | 0.742 | 0.807 | 0.065 |
| scan | 0.390 | 0.398 | 0.398 | 0.398 | 0.398 | **0.000** |
| churn | 0.580 | 0.182 | 0.578 | 0.580 | 0.831 | **0.251** |

Two workloads are saturated (best human ≈ optimal), `stable_frequency` is already
owned by LFU, and the one workload with real headroom (`churn`) requires *state* (a
record of past behaviour) that a stateless score cannot hold — even ARC fails to
capture it.

**Why it matters.** It reframes the contribution: the realistic, well-posed question
is **"can evolution find one stateless rule that *matches* the best human heuristic
across all regimes at once"** (a generalist claim), not "beat them." Beating SOTA is a
rung-2 (stateful) goal. This is a defensible scoping decision, not a failure.

---

## F2 — The naive "gap-closed vs LRU" fitness is degenerate

**Claim.** The first fitness function — mean over workloads of
`clamp((hr − hr_LRU)/(hr_OPT − hr_LRU), 0, 1)` — is not a usable optimisation signal.
It is artifact-dominated, blind to regressions, and quantized into a near-flat
landscape.

**Evidence (from the 1-hour run, best policy `frequency + 3.5/(recency+1) −
age/(now+1)`, combined_score 0.367):**

| workload | best policy | vs LRU | gap closed | contribution to score |
|---|---|---|---|---|
| temporal_locality | 0.268 | **−0.36** (regressed) | 0 (clamped) | 0.00 |
| stable_frequency | 0.650 | +0.07 | 0.48 | 0.12 (real) |
| scan | 0.494 | +0.007 (noise) | 0.96 | **0.24 (artifact)** |
| churn | 0.230 | −0.08 (regressed) | 0 (clamped) | 0.00 |

Three compounding defects:
1. **Tiny-denominator artifact:** `scan` headroom is 0.007 (noise level), so a
   meaningless 0.007 gain reads as "96 % of the gap closed" → **66 % of the best
   score** comes from a non-signal.
2. **Floor-clamping hides regressions:** any workload below LRU contributes exactly 0
   regardless of how far below, so the metric cannot see the policy *getting worse*.
3. **Quantization:** scores live on a coarse lattice (0.0, 0.125, 0.25, 0.27, 0.28,
   0.36, 0.367) — a step function, not a gradient.

**Why it matters.** This is a transferable methodological lesson for LLM-driven
heuristic design: **normalising against a near-optimal baseline with a clamp creates a
blind, gameable fitness.** The fix (adopted: signed skill vs best-of-baselines, with a
headroom gate and multi-seed averaging — see F5) is itself a contribution.

---

## F3 — Model capability and fitness landscape interact: a capable model is wasted by a flat metric

**Claim.** Mutation quality is gated by model capability, but a degenerate fitness
landscape then prevents a capable model from making progress.

**Evidence.**
- **0.6 B model (earlier phase):** produced byte-identical or comment-only edits — no
  change to executable logic in either diff or full-rewrite mode. Capability floor not
  met.
- **9 B model (this experiment):** produced **198 distinct** score expressions with
  genuinely diverse behaviour (e.g. `stable_frequency_hitrate` spanned 0.13–0.65). The
  capability floor *is* met.
- **But** under the degenerate fitness (F2), the search **converged by iteration 6**
  and then spent 244 more iterations tweaking a single constant: the top ~14 programs
  are all `frequency + (k/(recency+1)) − age/(now+1)` with `k ∈ {1.5…10}`, all scoring
  0.365–0.367. The fitness could not see the only thing the model was varying
  (`temporal_locality`, which sat below the LRU floor), so there was no gradient to
  climb.

**Why it matters.** "The model is too weak" and "the model produced nothing useful"
are different failures. Here the model was adequate; the **measurement** wasted it.
Diagnosis must separate the operator (LLM) from the objective (fitness).

---

## F4 — Hardware throughput: feasible locally; bottleneck is LLM latency, not cost

**Claim.** The experiment is compute-feasible on a single 8 GB consumer GPU at zero
token cost; the limiting resource is wall-clock LLM latency.

**Evidence.** Qwen3.5-9B (Q4_K_M, full GPU offload) on the RTX 3070:
- ~270 iterations/hour (~13 s/iteration), ~40–45 tokens/s generation, 1786 tokens/s
  prompt processing, ~6.5 GB VRAM.
- Policy evaluation is ~1.5 s (CPU) — negligible next to the LLM call.
- Local inference incurs **no token/API cost**; overnight runs buy search for free.
- The first 50-iteration run wasted 10 iterations (20 %) on `max_code_length`
  overflow (verbose comments); raising the cap to 8000 eliminated this (0 failures in
  the 272-iteration run).

**Why it matters.** Scaling is a wall-clock/parallelism problem, not a budget problem.
The path to larger search (rung 2) is **distributed inference** across multiple
machines (e.g. university lab boxes over SSH, round-robin endpoints) and/or a faster
smaller model for breadth — not more expensive hardware.

---

## F5 — Methodological corrections adopted (the "method" for the writeup)

From F1–F3, the experiment method is revised as follows (implemented in Stage 1):
1. **Objective:** match the best human heuristic per workload (generalist), not beat it
   (rung 1); beating SOTA is deferred to a stateful policy class (rung 2).
2. **Signed skill vs best-of-baselines:** `skill(w) = (hr − B(w))/(OPT(w) − B(w))`
   with `B = max(LRU,LFU,ARC)`, **no lower clamp** (regressions are negative → gradient
   restored).
3. **Headroom gate:** exclude workloads with `OPT − B < 0.02` (removes the `scan`
   artifact automatically).
4. **Multi-seed averaging + train/test split:** de-quantises the landscape and measures
   generalisation rather than luck on one trace.
5. **Benchmark/evolver separation:** the fitness becomes a standalone, framework-
   agnostic instrument any evolver (OpenEvolve, EoH, custom) can call — so results are
   comparable across frameworks.
6. **Honest reporting bar:** report against modern policies (S3-FIFO, W-TinyLFU,
   SIEVE) and Belady, even though optimisation targets only the classical crux.

**Why it matters.** These corrections are the experimental-design contribution; the
broken-then-fixed fitness is itself a reportable result about evaluating LLM-evolved
heuristics.

---

## Bibliography

**LLM + evolution for algorithm/heuristic design (method):**
- AlphaEvolve — Google DeepMind, 2025. *A coding agent for scientific and algorithmic
  discovery.* (OpenEvolve is an open re-implementation.)
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- FunSearch — Romera-Paredes et al., *Mathematical discoveries from program search with
  large language models*, Nature, 2023. https://www.nature.com/articles/s41586-023-06924-6
- EoH — Liu et al., *Evolution of Heuristics: Towards Efficient Automatic Algorithm
  Design Using Large Language Model*, ICML 2024 (Oral). https://arxiv.org/abs/2401.02051 ·
  code: https://github.com/FeiLiu36/EoH
- MAP-Elites — Mouret & Clune, *Illuminating search spaces by mapping elites*, 2015.
  https://arxiv.org/abs/1504.04909

**Cache replacement — classical baselines & oracle:**
- Belady, *A study of replacement algorithms for a virtual-storage computer*, IBM
  Systems Journal, 1966 (optimal offline / MIN — the ceiling).
- Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache*, USENIX FAST
  2003 (the conceptual crux).
- GDSF (Greedy-Dual-Size-Frequency) and LRFU (Lee et al.) — human-designed
  *scoring-function* policies; the class our evolved `score()` belongs to.

**Cache replacement — learned policies (closest related work; rung-2 company):**
- LeCaR — Vietri et al., USENIX HotStorage 2018 (regret over LRU+LFU; beats ARC).
  https://www.usenix.org/conference/hotstorage18/presentation/vietri
- CACHEUS — Rodriguez et al., USENIX FAST 2021 (scan/churn experts).
  https://par.nsf.gov/servlets/purl/10251571
- LRB — Song et al., *Learning Relaxed Belady for CDN Caching*, USENIX NSDI 2020.
- GL-Cache — Yang et al., *Group-level learning for efficient and high-performance
  caching*, USENIX FAST 2023. https://www.usenix.org/conference/fast23/presentation/yang-juncheng

**Cache replacement — modern simple/strong policies (the reporting bar):**
- S3-FIFO — Yang et al., *FIFO queues are all you need for cache eviction*, SOSP 2023.
  https://yazhuozhang.com/assets/publication/sosp23-s3fifo.pdf
- SIEVE — Zhang et al., *SIEVE is simpler than LRU*, USENIX NSDI 2024.
  https://www.usenix.org/publications/loginonline/sieve-cache-eviction-can-be-simple-effective-and-scalable
- TinyLFU / W-TinyLFU — Einziger et al., *TinyLFU: A Highly Efficient Cache Admission
  Policy*, ACM ToS 2017 (basis of Caffeine). https://dl.acm.org/doi/10.1145/3149371

**Tooling:**
- libCacheSim — high-performance C cache simulator + Python bindings (implements the
  above + Belady); for real-trace evaluation. https://github.com/1a1a11a/libCacheSim
