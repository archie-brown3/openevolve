# References & Resources

Curated, annotated pointers for designing and extending the cache-replacement
evolution experiment. Intentionally **links + notes only** — no large vendored
repos. Clone what you need locally when you reach the relevant roadmap step.

## Reference implementations to borrow from

### sylab/cacheus — Python policies (start here)
- Repo: <https://github.com/sylab/cacheus>
- Why: clean, readable **Python** implementations of LRU, LFU, ARC, LIRS, LeCaR,
  and CACHEUS over a common trace-driven simulator. Closest match to our setup.
- What to borrow:
  - The `lru`, `lfu`, `arc` reference policies → validate our evaluator's baseline
    implementations against theirs on a shared trace.
  - The **LeCaR** weight-update (regret-minimisation over a recency vs frequency
    expert pair) and **CACHEUS** scan/churn handling → inspiration for what a
    *learned* `score()` might rediscover. Our evolved policy is stateless per
    eviction, so treat these as conceptual targets, not drop-in code.
  - Their trace format / driver loop → cross-check our synthetic generator output.
- Note: the repo bundles **no synthetic generator and no traces** — `lib/traces.py`
  is only a *reader* for external trace files, and `lib/pollutionator.py` is a
  pollution *metric*, not a workload source. Real traces are downloaded separately
  (see "Trace sources" below). Our `trace_generators.py` is the synthetic suite the
  repo lacks, built to the CACHEUS workload taxonomy (LRU-friendly / LFU-friendly /
  scan / churn).

### libCacheSim — high-performance C library + Python bindings (for scale)
- Repo: <https://github.com/1a1a11a/libCacheSim>
- Why: production-grade C cache simulator with a Python interface and many
  built-in policies (LRU, LFU, ARC, S3-FIFO, TinyLFU, GL-Cache, Belady). Use it
  when we graduate from synthetic traces to **realistic, large-scale** traces
  where pure-Python simulation is too slow.
- What to borrow:
  - Its **Belady/OPT** implementation as an independent oracle cross-check.
  - Its trace readers (CSV, oracleGeneral binary) → ingest Twitter/real traces.
  - Bundled **Twitter cluster traces** in CSV (see below).
- Note: building needs a C toolchain (CMake, glib). Defer until the real-trace
  roadmap step.

## Algorithms / papers

- **ARC** — Megiddo & Modha, *ARC: A Self-Tuning, Low Overhead Replacement Cache*,
  USENIX FAST 2003. The adaptive recency/frequency baseline.
- **LeCaR** — Vietri et al., *Driving Cache Replacement with ML-based LeCaR*,
  USENIX HotStorage 2018. RL/regret over a 2-expert (LRU, LFU) ensemble.
- **CACHEUS** — Rodriguez et al., *Learning Cache Replacement with CACHEUS*,
  USENIX FAST 2021. Extends LeCaR for scan + churn; the sylab repo is its artifact.
- **TinyLFU / W-TinyLFU** — Einziger et al., *TinyLFU: A Highly Efficient Cache
  Admission Policy* (admission via Count-Min Sketch; basis of Caffeine).
- **S3-FIFO** — Yang et al. (2023), simple scan-resistant 3-queue FIFO; strong
  modern baseline, implemented in libCacheSim.

## Trace sources & formats

Community-standard traces, roughly in the order we'd adopt them. The numbered list
is the *menu*; the **phased integration plan** below is how/when we actually use them.

1. **Synthetic (Phase 1, now)** — `trace_generators.py` in this folder. Four
   patterns (temporal-locality, stable-frequency, scan, churn). Deterministic, fast,
   designed so the baselines diverge. This is where evolution starts.
2. **FIU traces** — real-world storage I/O **block** traces (Florida International
   University: web/mail/file servers; the `SRC_Map` set). The literature's reference
   set — used by the **LeCaR and CACHEUS** papers, so it lets us plot our evolved
   policy on the same axes as published baselines. Mirrored in the **SNIA IOTTA**
   repository: <https://iotta.snia.org/traces/>. Records are roughly
   `timestamp, pid, process, LBA, size, op(R/W)`; use the **LBA** as the key.
3. **MSR Cambridge** — Microsoft Research, 36 enterprise server volumes, 1 week.
   The standard storage benchmark; also on SNIA IOTTA. Same block-trace shape (LBA
   key). Good breadth of patterns across volumes.
4. **CloudVPS / CloudCache** — VPS and application (webserver, moodle) caching
   traces referenced by the cacheus repo; one-day spans, public. Useful secondary
   real-world checks.
5. **Twitter cluster traces** — production **key-value** cache traces, CSV, bundled
   with libCacheSim and published at <https://github.com/twitter/cache-trace>.
   Format: `timestamp, key, key_size, value_size, client_id, operation, ttl`. Use
   `key` as the access key. (KV, not block — a different domain, optional.)
6. **CloudPhysics / SPEC CPU** — **not public** (CloudPhysics, FAST'15) or heaviest
   to set up (SPEC CPU2006/2017 memory traces). Skip unless specifically needed.

### Phased integration plan (synthetic → real-trace validation)

The real traces are **out of scope for Phase 1** (proof of concept). They enter as a
*validation* phase once evolution produces a winning `score()`:

- **Phase 1 — evolve (synthetic only).** Discover heuristics fast on the four seeded
  workloads. Fitness is noise-free and the baselines provably diverge. No trace
  downloads, no parsing, no C deps. Get the evolution loop itself working first.
- **Phase 2 — validate (FIU + MSR, frozen).** Take the *winning, unchanged*
  `score()` and evaluate it on a fixed slice of FIU + MSR. This is the external-
  validity result for the dissertation: does a synthetic-evolved policy generalise
  to real traces, head-to-head with LRU/LFU/ARC/LeCaR/CACHEUS on the same data the
  papers use? Hold the slice constant so the comparison is reproducible.
  - **Speed:** real traces are millions of requests; our pure-Python O(C)-per-miss
    simulator is too slow here. Use **libCacheSim** (C engine, has Belady + the
    baselines) for the real-trace runs, or down-sample to a fixed window.
- **Phase 3 — co-evolve (optional).** *Only if* Phase 2 shows synthetic-evolved
  policies don't generalise: fold a frozen real-trace slice into the evaluator's
  fitness mix and re-evolve. Watch fitness noise — keep the slice fixed and seeded.

### Mapping any trace to our contract
The simulator only needs an ordered `list[int]` of keys. For each source, extract
the identifying field (block **LBA** for FIU/MSR/CloudVPS, **key** for Twitter,
memory address for SPEC), intern it to an int, and feed the sequence. A ~20-line
loader per source under `resources/` is all it takes when Phase 2 arrives — the
`list[int]` contract means **the evolved `score()` and simulator never change**.
