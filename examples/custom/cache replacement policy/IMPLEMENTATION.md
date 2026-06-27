# Implementation Plan — Cache Replacement Policy Evolution

This document is the engineer-facing technical plan. It has three parts:

- **Part A — Experiment design**: the three experiment files, the evolvable
  interface, the simulator, the evaluator, fitness, and `config.yaml`.
- **Part B — Codebase context & code trace**: how OpenEvolve's engine runs one
  iteration, for a reader new to the codebase. Part A references it as **§B**.
- **Part C — Hardware, inference provider & model**: the local-first serving setup
  on an 8 GB GPU and the chosen model.

Companion files: [`README.md`](README.md) (overview),
[`resources/baselines.md`](resources/baselines.md) (policy pseudocode),
[`resources/trace_generators.py`](resources/trace_generators.py) (workloads),
[`resources/REFERENCES.md`](resources/REFERENCES.md) (sources).

---

## Part A — Experiment design

### A.1 Components & data flow

Three files in this directory, following the OpenEvolve example convention verified
in `examples/shipped/function_minimization/`:

```
initial_program.py   seed score() inside EVOLVE-BLOCK + FIXED simulator
       │  (engine mutates only the EVOLVE-BLOCK — see §B.3)
       ▼
evaluator.py         loads program → runs simulate() over the workload suite
       │             → compares vs LRU/LFU/ARC/Belady → returns EvaluationResult
       ▼
config.yaml          models, islands, MAP-Elites features, cascade thresholds
```

Per-iteration data flow: **trace → simulator(score) → hit rate → fitness**. The
trace suite and the reference policies live on the evaluator side; the LLM only
ever sees and edits `score()`.

### A.2 The `score` interface contract (the evolvable surface)

```python
# EVOLVE-BLOCK-START
def score(age, frequency, recency, size, cache_size, now) -> float:
    """Higher score = more valuable = keep. Lowest-scored resident is evicted.
    age        = now - insert_time
    frequency  = total access count while resident
    recency    = now - last_access_time
    size       = item size (1 in the uniform-cost experiments)
    cache_size = capacity (constant within a run)
    now        = current logical timestamp (access index)
    """
    return -recency   # seed == LRU
# EVOLVE-BLOCK-END
```

**Why this boundary.** A single pure function is the smallest unit that fully
expresses the EA target from the project image (`score(age, frequency, recency,
cache_size) → evict lowest`). Three properties make it robust:

1. **Signature stability.** The fixed simulator *calls* `score()` with named args.
   The LLM cannot drift the contract (rename params, change arity) without breaking
   the call, which the evaluator catches and scores 0. This is the lesson from the
   Phase-1 run where weak models emitted plausible-looking but malformed edits
   (see §C.3): keep the harness fixed and the mutable surface tiny.
2. **Known-good seed.** `score = -recency` reproduces LRU *exactly* — verified: the
   score-policy and the LRU reference return identical hit rates on all four
   workloads (0.697 / 0.692 / 0.390 / 0.580 at capacity 300). Iteration 0 is never
   degenerate.
3. **Auditability.** A winning policy is a short arithmetic expression a human can
   read and a future step can port to Rust/C++ (README roadmap).

### A.3 Fixed simulator (in `initial_program.py`, outside the EVOLVE-BLOCK)

A trace-driven LRU-style simulator that delegates the eviction choice to `score()`:

```text
state: cache = { key -> Meta(insert_time, last_access, freq, size) }, capacity C
simulate(trace, capacity):
    hits = 0
    for now, key in enumerate(trace):
        if key in cache:
            hits += 1; cache[key].last_access = now; cache[key].freq += 1
        else:
            if len(cache) >= capacity:
                victim = argmin over residents of
                         score(now-m.insert_time, m.freq, now-m.last_access,
                               m.size, capacity, now)
                del cache[victim]
            cache[key] = Meta(insert_time=now, last_access=now, freq=1, size=1)
    return hits / len(trace)
```

Exposed entry point: `simulate(trace, capacity) -> float`. The `argmin` is O(C) per
miss; with C ≤ a few hundred and 20 k-access traces this is well under a second per
workload — acceptable for thousands of evaluations. (If profiling later demands it,
the evaluator can cap trace length or memoize; the simulator stays fixed so the
optimisation never leaks into the evolvable surface.)

### A.4 Evaluator (`evaluator.py`)

Mirrors the house pattern (`evaluate(program_path) -> EvaluationResult`, see §B.6):

1. Import the candidate program (`importlib`), confirm `simulate` and `score` exist;
   on failure return `combined_score: 0.0` + diagnostic artifacts (as
   `function_minimization/evaluator.py` does).
2. Build the workload suite via `trace_generators.generate_all(seed)` and a small
   set of capacities (e.g. `[100, 300]`).
3. For each (workload, capacity): run the candidate `simulate`, plus the **fixed
   reference** `lru/lfu/arc/belady` implementations (pseudocode in
   `resources/baselines.md`). These references live in the evaluator and are
   **never** evolvable.
4. Compute per-workload **gap closed** and the aggregate fitness (§A.5).
5. Return `EvaluationResult(metrics=..., artifacts=...)` where artifacts include the
   per-workload hit-rate table so the LLM sees exactly where it wins/loses.

**Cascade (optional, see §B.6).** Implement `evaluate_stage1` as a cheap smoke test
(one workload, one capacity, must run + beat a trivial floor) and `evaluate_stage2 =
evaluate` for the full suite. Gate via `cascade_evaluation` in `config.yaml`.

### A.5 Fitness function

Per workload, normalise the candidate hit rate into the LRU→OPT band:

```
gap_closed(w) = clamp( (hr_cand(w) - hr_lru(w)) / (hr_opt(w) - hr_lru(w)), 0, 1 )
combined_score = mean_w  gap_closed(w)
```

`0.0` = "no better than LRU"; `1.0` = "matched the Belady oracle". This makes scores
comparable across workloads of very different absolute hit rate and directly rewards
closing the headroom that exists (`churn` and `stable_frequency` carry most of it;
`temporal_locality`/`scan` are near-saturated "don't regress" constraints — see the
table in `resources/baselines.md`). Guard the degenerate `hr_opt == hr_lru` case
(treat gap as already closed → contribute 1.0, so a near-saturated workload can't
drag the mean down for a policy that simply ties LRU).

Also emit raw per-workload hit rates and the baseline/OPT references as extra
metrics + artifacts (not just the scalar) — they drive both the prompt feedback and
the MAP-Elites features.

### A.6 MAP-Elites features

`combined_score` is fitness; the **feature dimensions** are for *diversity* (which
cells of the archive a program occupies), not fitness. Use per-workload hit rates so
the archive retains **specialists** rather than collapsing to one averaged
generalist:

```yaml
database:
  feature_dimensions:
    - stable_frequency_hitrate   # must be returned by the evaluator as a metric
    - churn_hitrate
  feature_bins: 10
```

These two are the high-headroom, policy-discriminating axes (frequency-sensitivity
vs. adaptivity). Returning them as metrics is required for OpenEvolve to bin on them.
Built-in `complexity`/`diversity` remain available if we prefer code-shape diversity
instead.

### A.7 `config.yaml` settings (rationale)

Start from the existing default `config.yaml` and set:

| setting | value | why |
|---|---|---|
| `diff_based_evolution` | `true` | `score()` edits are small, local diffs |
| `max_code_length` | ~5000 | the evolvable surface is tiny; discourage bloat |
| `database.num_islands` | 4–5 | maintain diverse heuristic lineages |
| `database.population_size` | 200–1000 | small per-eval cost; afford a big archive |
| `database.feature_dimensions` | per-workload hitrates (§A.6) | keep specialists |
| `evaluator.cascade_evaluation` | `true` (+ stage fns) | kill broken programs cheaply |
| `evaluator.cascade_thresholds` | `[0.0, 0.5]` | stage-1 only requires "runs & ties LRU-ish" |
| `evaluator.parallel_evaluations` | 2–4 | CPU-bound sim; modest parallelism |
| `evaluator.timeout` | 60 | a candidate with a pathological `score` can't hang the run |

Model/endpoint settings (`llm.*`, `api_base`) are covered in **§C**.

### A.8 Alternatives considered & rejected

- **Rust/C++ now** — rejected for v1: per-evaluation compile latency multiplied
  across thousands of iterations dominates, and harness complexity rises. The
  evolved expression ports trivially later (README roadmap).
- **Evolve a full policy class** (`on_access`/`on_evict`/admission, cacheus-style) —
  more expressive (ghost lists, sketches) but a much larger surface that weak local
  models break easily; deferred until the scoring-function track plateaus.
- **Simulator owned by the evaluator** (bare `score()` in the program) — rejected:
  having the program export a *fixed* `simulate` wrapper that the evaluator calls is
  the strongest guard against signature drift (§A.2).

---

## Part B — Codebase context & code trace

For a reader new to OpenEvolve, here is one iteration end-to-end through the engine
modules this experiment touches. Pointers are `file:line` at time of writing; verify
against current source.

### B.1 Entrypoint & orchestration
`openevolve-run.py:6` → `openevolve/cli.py:main()` (`cli.py:174`) →
`openevolve/api.py:run_evolution()` (`api.py:33`) constructs
`openevolve/controller.py:OpenEvolve` (`controller.py:56`), whose `run()`
(`controller.py:249`) builds a
`openevolve/process_parallel.py:ProcessParallelController` (`controller.py:330`).
Parallelism is a `ProcessPoolExecutor` (`process_parallel.py:11`) — each worker runs
iterations against a database snapshot (the "process worker pattern" in CLAUDE.md).

### B.2 Config
`openevolve/config.py:Config.from_yaml()` (`config.py:434`, via `load_config`
`config.py:494`) parses **our `config.yaml`** into the `Config` dataclass that every
component reads (`llm`, `database`, `evaluator`, `prompt`).

### B.3 The worker loop (the heart)
`process_parallel.py:run_evolution()` (`:472`) drives iterations; each calls
`_run_iteration_worker()` (`:134`), which:
1. **Samples** a parent + inspirations from the island —
   `database.sample_from_island()` (`process_parallel.py:806` →
   `database.py:403`; internals `_sample_parent` `database.py:1270`,
   `_sample_inspirations` `database.py:1554`).
2. **Builds the prompt** — `iteration.py:73` calls
   `prompt/sampler.py:PromptSampler.build_prompt()` (`sampler.py:51`), templated by
   `prompt/templates.py`.
3. **Calls the LLM** — `iteration.py:92`
   `llm_ensemble.generate_with_context()` (§B.4).
4. **Applies the edit** — diff mode parses SEARCH/REPLACE via
   `extract_diffs`/`apply_diff_blocks` (`iteration.py:99,116`); full-rewrite mode via
   `parse_full_rewrite` (`iteration.py:147`). See §B.5.
5. **Evaluates** the child — `iteration.py:166`
   `evaluator.evaluate_program(child_code, child_id)` (§B.6).
6. **Stores** the child (code, parent_id, generation, metrics, prompts) back into the
   database (§B.7).

### B.4 LLM client — the single integration seam
`openevolve/llm/ensemble.py` fans out to `openevolve/llm/openai.py:OpenAILLM`
(`openai.py:47`). It is a plain OpenAI-compatible client: `base_url = self.api_base`
(`openai.py:62,87`), `generate_with_context()` (`openai.py:108`), `_call_api()`
(`openai.py:212`). **This `api_base` is the only place a model is reached** — point
it at local vLLM or a hosted API (§C). Note `openai.py:172` special-cases the Gemini
base URL; a local vLLM endpoint is *not* in any special-case list, so standard
`temperature`/`top_p`/`max_tokens` are sent as-is (correct for our use).

### B.5 Diff application — a gotcha worth knowing
`openevolve/utils/code_utils.py`: `apply_diff` (`:40`), `extract_diffs` (`:78`),
`apply_diff_blocks` (`:243`), `parse_full_rewrite` (`:95`). **Gotcha:** a
SEARCH block that doesn't match the parent is **silently skipped** — no error, no
log — so a weak model can "succeed" while changing nothing (observed in Phase-1,
§C.3). Implication for us: the evaluator must score behaviour, never trust that an
edit landed, and a no-op child simply re-scores as its parent.

### B.6 Evaluation & cascade
`openevolve/evaluator.py:Evaluator` (`:32`) loads **our `evaluator.py`** and calls
`evaluate_program`. With `cascade_evaluation: true` it expects `evaluate_stage1`
(/`stage2`) and warns if they're missing (`evaluator.py:101-119`) — then falls back
to direct `evaluate`. Our evaluator returns
`openevolve/evaluation_result.py:EvaluationResult(metrics, artifacts)`; `metrics`
must contain `combined_score` (fitness) plus the per-workload metrics that
`feature_dimensions` bin on.

### B.7 Database — MAP-Elites + islands
`openevolve/database.py:ProgramDatabase` (`:113`) holds per-island MAP-Elites grids
(`island_feature_maps` `:129`), island populations (`:142`), generation counters
(`:146`), and per-island best (`:158`). `add()` (`:211`) inserts a program into the
right feature cell; migration periodically shares elites between islands. Checkpoints
persist the whole structure for resume.

### B.8 What this experiment must supply
Only the right column is ours; everything in §B.1–B.7 is reused unchanged:

| engine expects | we provide |
|---|---|
| a program with an EVOLVE-BLOCK | `initial_program.py` (`score` + fixed `simulate`) |
| `evaluate(program_path)` → `EvaluationResult` | `evaluator.py` |
| metrics incl. `combined_score` + feature metrics | fitness §A.5, features §A.6 |
| `config.yaml` | §A.7 + §C model block |
| an `api_base` endpoint | local vLLM / API (§C) |

---

## Part C — Hardware, inference provider & model (local-first)

### C.1 Hardware budget (the binding constraint)
RTX 3070, **8 GB** VRAM, **shared with the desktop** (~0.8 GB baseline), so the
usable budget is ~7 GB for weights + KV cache + activations. This rules out
unquantised 7 B+ models and makes a **4-bit quantised ~4 B** model the sweet spot.

### C.2 Two providers, one seam
OpenEvolve reaches a model only through `llm.api_base` (§B.4). We support two
interchangeable backends and **lead with local**:

- **Local (primary):** vLLM serving an OpenAI-compatible endpoint at
  `http://localhost:8000/v1`, `api_key: "EMPTY"` (the OpenAI client needs a
  non-empty string; vLLM ignores it). Config goes in a `config_local.yaml`.
- **API (fallback / sanity baseline):** the Gemini OpenAI-compat endpoint used by
  the shipped examples — a capability ceiling to confirm the *experiment* is sound
  when the local model is the bottleneck. Drop-in by swapping `api_base` + key.

### C.3 Model choice & rationale
**Phase-1 finding (decisive):** Qwen3-0.6B was **too weak** — it emitted valid
SEARCH/REPLACE *syntax* but degenerate *content* (echoes, comment relabels),
changing **zero executable logic** in both diff and full-rewrite modes. The
bottleneck was **model capability**, not the prompt or parser. So we specify a
current, capable *code* model that still fits 8 GB at 4-bit.

- **Primary: `Qwen3.5-4B`, 4-bit (AWQ/GPTQ), ~2.5 GB weights.** Released Mar 2026,
  successor to the Qwen2.5-Coder line; strong recent coding (LiveCodeBench v6 ≈ 55.8),
  256 K context, vLLM-compatible, and supports `enable_thinking: False` — matching
  the proven Qwen3 serving recipe (clean non-thinking output for the diff parser).
  ~2.5 GB leaves comfortable headroom on the 3070 for KV cache + desktop.
- **Lighter alternative: `Phi-4-mini` (3.8 B, Q4 ≈ 3.5 GB)** if the 4 B quant proves
  unstable under the installed vLLM (there is a known vLLM Qwen3-4B-quant issue —
  treat as a risk and keep this on standby).
- **Quantisation note:** if no official `Qwen3.5-4B` AWQ exists yet, quantise with
  AutoAWQ or llm-compressor (vLLM supports AWQ). 
- **Capability bar is *lower* here than Phase-1:** the cache `score()` target is a
  short arithmetic expression, not a multi-branch search algorithm — so a 4 B-class
  coder has real headroom to produce *meaningful* mutations, where 0.6 B could not.

### C.4 Serving recipe (adapt the verified flags)
From the working setup on this hardware:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve <qwen3.5-4b-awq> \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enforce-eager --max-model-len 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.55 --port 8000
```
Why these: `VLLM_USE_FLASHINFER_SAMPLER=0` avoids the FlashInfer JIT/CUDA-13 nvcc
failure; `--enforce-eager` skips CUDA-graph capture (~0.5 GB); low
`--gpu-memory-utilization` + small `--max-num-seqs` avoid OOM on the shared 8 GB;
disabling thinking keeps diff output clean. Matching `config_local.yaml` must set
`api_base: http://localhost:8000/v1`, `api_key: "EMPTY"`, and `max_tokens` **<**
`max-model-len − prompt` (else a 400). Operational gotchas (don't `pkill -f vllm`
from the launching shell; foreground `sleep` is blocked) are recorded in the
project's local-vLLM notes.

### C.5 Tie-in
This experiment is the **mutation-operator workload** for the larger self-improvement
loop (evolve → harvest preference pairs from the program DB → DPO-train the local
model → repeat). Keeping the local endpoint as the primary backend is what makes that
loop possible; the API backend is only a correctness check.
