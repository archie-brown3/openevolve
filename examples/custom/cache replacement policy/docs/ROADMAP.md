# Roadmap — LLM-Evolved Cache Replacement Policies

## Objective

Use LLM + evolutionary search (OpenEvolve now; framework-agnostic later) to
automatically design cache-replacement policies, evaluated rigorously against human
heuristics and the Belady optimum.

- **Rung 1 — stateless (now):** evolve one scoring function
  `score(age, frequency, recency, size, cache_size, now)` that **matches the best human
  heuristic on every workload at once** (a single rule doing the job of LRU+LFU+ARC).
- **Rung 2 — stateful (headline):** evolve a policy *with memory*
  (`on_access`/`on_evict` + internal state) that **beats ARC and the modern bar**
  (S3-FIFO, W-TinyLFU) — the territory of LeCaR/CACHEUS/GL-Cache.

## Design principle: separate the BENCHMARK from the EVOLVER

The scientific instrument is the **benchmark**: `evaluate(policy) → {skill, table}` —
deterministic, reusable, with no dependency on the evolution framework. The **evolver**
only *proposes* policies and calls the benchmark. This keeps measurement constant while
the framework is swapped (OpenEvolve → EoH → custom), making cross-framework results
comparable. See `STAGE1_TECHNICAL_PLAN.md` for the concrete split.

## Baseline policy: ARC is the crux; the bar is modern

- **Conceptual crux:** ARC (FAST 2003) — adaptive recency/frequency.
- **Optimisation target** `B = max(LRU, LFU, ARC)` — an achievable bar for a stateless
  rule.
- **Reported modern bar:** S3-FIFO (SOSP'23), W-TinyLFU (Caffeine), SIEVE (NSDI'24),
  Belady OPT — to position results vs 2023–24 SOTA honestly.
- **Real-trace baselines (M2+):** via libCacheSim (no reimplementation).

---

## Milestones (each is a small experiment that de-risks the next)

### M0 — Fix the fitness / build the benchmark *(Stage 1 — see STAGE1_TECHNICAL_PLAN.md)*
Signed skill vs best human, headroom gate, multi-seed, train/test; benchmark/evolver
split; add S3-FIFO as a reported baseline.
- **Exit criteria:** self-check passes — deterministic; an LFU-only policy that
  regresses locality scores **negative**; a GDSF-like blend scores smoothly between the
  LRU seed and a perfect match (landscape no longer quantized).

### M1 — Stateless concept validation
Short run (200–500 iters, 9B) with the new fitness.
- **Question:** does evolution now find a *real* stateless rule that closes signed skill
  (matches best-of-{LRU,LFU}, approaches ARC)?
- **Exit criteria:** best policy has positive mean skill *and* does not regress below
  LRU on any high-headroom workload; result recorded in `FINDINGS.md`.

### M2 — Generalisation & real traces
Hold out test seeds; then evaluate the M1 winner on a fixed **FIU/MSR** trace slice via
libCacheSim, against LRU/LFU/ARC/S3-FIFO/W-TinyLFU/OPT.
- **Exit criteria:** the synthetic-evolved rule's ranking holds on held-out + real
  traces (or we document where it fails to transfer — also a result).

### M3 — Specialists (quality-diversity)
Use MAP-Elites to evolve a *portfolio* of per-workload specialists.
- **Exit criteria:** for each workload, a specialist that matches/beats the best simple
  human rule on that regime; archive coverage analysed.

### M4 — Minimal state (rung-2 entry)
Extend the interface with ONE memory feature (e.g. a bounded ghost list of recent
evictions, or a single adaptive parameter).
- **Question:** can one memory feature beat the stateless ceiling on churn?
- **Exit criteria:** a policy using the memory feature beats the best M1 stateless
  policy on churn without regressing elsewhere. De-risks M5.

### M5 — Full stateful policy-class evolution (beat ARC)
Evolve `on_access`/`on_evict` + arbitrary internal state behind a fixed simulator +
guardrails (timeouts, capacity/contract validation, fenced mutable region).
- **Exit criteria:** a policy that matches/beats ARC across workloads and is competitive
  with S3-FIFO/W-TinyLFU on real traces; compared head-to-head with LeCaR/CACHEUS.

### M6 — Scale & orchestrate
Distributed inference across lab machines; longer budgets; optional SOTA-model
reflection module.
- **Exit criteria:** a scaling curve (search budget → policy quality); reproducible
  multi-machine setup.

---

## Hardware path (feasibility)

- **M0–M3:** single 8 GB RTX 3070 is sufficient. ~270 iters/hr at zero token cost;
  overnight runs buy search for free. Bottleneck is LLM latency, not money or
  evaluation (F4 in `FINDINGS.md`).
- **Throughput levers:** smaller/faster model (e.g. Qwen2.5-Coder-3B) for breadth vs 9B
  for quality; longer wall-clock; more parallelism.
- **M5–M6 — distributed inference:** run a llama.cpp server on each SSH-accessible lab
  machine and round-robin requests across endpoints → throughput scales ≈ linearly with
  machines (weak boxes run a smaller model). Needs a thin endpoint load-balancer or
  multi-`api_base` support in the evolver.
- **Hybrid SOTA orchestrator (optional, future):** bulk mutation stays local/free; call
  a strong API model occasionally for reflection/meta-prompting (few calls = low spend).
  Mirrors AlphaEvolve's cheap-breadth + strong-depth ensemble.
- **Rung 2 cost:** statefulness does not raise LLM cost (still code mutation + cheap
  eval); it raises *search difficulty* → more iterations → more wall-clock/machines, not
  bigger hardware.

## Rung-2 interface sketch (for M4/M5)

```python
class Policy:
    def __init__(self, capacity):
        ...                       # internal state: queues, ghost lists, counters, params
    def on_access(self, key) -> None:
        ...                       # update state on hit/insert
    def on_evict(self) -> key:
        ...                       # choose victim using state; must return a resident key
```
The simulator (fixed) drives these; the benchmark validates the contract (cache never
exceeds capacity; `on_evict` returns a resident; bounded time) and scores violations 0.
Only the class body is mutable.
