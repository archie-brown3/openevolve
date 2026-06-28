# Stage 1 — Technical Implementation Plan (M0: Fix the Measuring Instrument)

**Status: IMPLEMENTED** (2026-06-28)

**Goal:** replace the degenerate fitness (see `FINDINGS.md` F2) with a **signed-skill
metric vs the best human heuristic**, using **libCacheSim** for validated, citable
baselines and a PluginCache adapter to run the evolved policy in the same engine.

---

## Architecture

```
resources/trace_generators.py   (unchanged) — seeded synthetic workloads
        │
        ▼
benchmark.py   (NEW) — the scientific instrument:
        │   libCacheSim baselines (LRU/LFU/ARC/Belady/S3-FIFO) via per-request get()
        │   PluginCache adapter wraps score() in the same libCacheSim engine
        │   signed-skill fitness, headroom gate, multi-seed, train/test split
        │   evaluate_policy(score_fn, split="train") -> BenchmarkResult
        │   __main__ --selfcheck (5 acceptance tests)
        ▼
evaluator.py   (REWRITTEN) — thin OpenEvolve adapter (~60 lines):
            loads score() from candidate; calls benchmark.evaluate_policy();
            maps BenchmarkResult -> EvaluationResult(metrics, artifacts)
```

**Evolved candidate interface:** `initial_program.py` now defines only `score()`.
The `simulate()` harness is removed — benchmark.py provides it via PluginCache.

---

## Key implementation decisions

### 1. libCacheSim per-request API (no temp files)

libCacheSim has no in-memory reader, but every cache object has a `.get(Request)` method
that processes one access at a time. We use:
```python
hits = sum(cache.get(Request(obj_id=k, obj_size=1)) for k in trace)
hit_rate = hits / len(trace)
```
No temp CSV files needed. `Request(obj_id=k, obj_size=1)` is sufficient for all
policies except Belady.

### 2. Belady requires next_access_vtime

`lcs.Belady` uses the `next_access_vtime` field of each Request to make oracle decisions.
Without it, hit rates are garbage (< 0.09 on temporal_locality vs correct 0.70). Fix:
```python
def _next_access_times(trace) -> list[int]:
    NEVER = 10**9
    nxt = [NEVER] * len(trace); last = {}
    for i in range(len(trace)-1, -1, -1):
        nxt[i] = last.get(trace[i], NEVER); last[trace[i]] = i
    return nxt
# then: Request(obj_id=k, obj_size=1, next_access_vtime=nxt[i])
```

### 3. PluginCache: eviction_hook must delete the victim from data

When `eviction_hook` returns a victim id, `remove_hook` is NOT automatically called.
The victim must be `del data[victim]` inside `eviction_hook` or it stays in the tracking
dict and causes a "object not found in cache" runtime error on the next eviction.

### 4. Baseline precomputation (once per worker process)

Baselines (LRU/LFU/ARC/Belady/S3FIFO × 4 workloads × 3 seeds × 2 caps = 120 runs) are
computed on first access and cached in a module-level dict. Cost: ~8s once per worker.
Each subsequent candidate evaluation only runs 24 PluginCache calls: ~4s.

### 5. signed-skill fitness

Per workload (averaged over seeds × caps):
```
B     = max(lru, lfu, arc)        # best human heuristic
skill = (hr_cand - B) / (OPT - B) # signed, no clamp (negative = regression)
```
Exclude workloads where `OPT - B < 0.02` (headroom gate). `combined_score = mean(skill)`.

**LRU seed (`-recency`) scores -0.358** on train seeds {1,2,3}:
- temporal_locality: -0.150 (ARC beats LRU → B=ARC=0.622 > LRU=0.611)
- stable_frequency:  -0.923 (LFU/ARC >> LRU → B=0.649, LRU=0.584 far below)
- scan: **excluded** (LFU≈ARC≈OPT, headroom near zero)
- churn: 0.000 (LRU=B=0.312; huge OPT gap 0.601 → only upside from here)

Evolution starts at -0.358 and gains a real gradient: beating ARC on temporal_locality
or LFU/ARC on stable_frequency earns positive skill on those workloads.

---

## Tasks completed

- **T1.0** `pip install libcachesim==0.3.3.post4` ✓  
  Verified: LRU per-request matches hand-rolled exactly; full baseline sweep ≤ 10s.

- **T1.1** `benchmark.py` (new, 170 lines) ✓  
  `evaluate_policy(score_fn, split) -> BenchmarkResult` — no OpenEvolve imports.

- **T1.2** libCacheSim baselines: LRU, LFU, ARC, Belady (with next_access_vtime), S3FIFO ✓

- **T1.3** Signed-skill fitness + headroom gate + `format_table()` artifact ✓

- **T1.4** Multi-seed (TRAIN_SEEDS={1,2,3}, TEST_SEEDS={4,5}), 2 caps ✓

- **T1.5** MAP-Elites features preserved: `stable_frequency_hitrate`, `churn_hitrate` ✓

- **T1.6** `benchmark.py --selfcheck`: 5 checks, all PASS ✓  
  (determinism, anti-gaming, smoothness, sign sanity ×2)

- **T1.7** `evaluator.py` rewritten (~60 lines, was 310 lines) ✓  
  `initial_program.py` simplified to score() only ✓  
  `config.yaml` system message updated ✓

---

## Verification

```bash
# All pass:
python benchmark.py --selfcheck
python evaluator.py initial_program.py   # → combined_score ≈ -0.358, no errors
python -c "from openevolve.config import load_config; load_config('config.yaml')"
```

Expected output for LRU seed:
```
workload               cand    lru    lfu    arc    opt  s3fifo    skill
temporal_locality    0.6108 0.6108 0.1614 0.6224 0.6996  0.5571  -0.1499
stable_frequency     0.5836 0.5836 0.6465 0.6493 0.7205  0.6538  -0.9227
scan                 0.4875 0.4875 0.4937 0.4937 0.4937  0.4937 excluded
churn                0.3123 0.3123 0.2106 0.3107 0.6015  0.3017  +0.0000
combined_score = -0.357562
```

---

## Out of scope (later milestones)

- W-TinyLFU / SIEVE / LeCaR / GL-Cache baselines → M2
- Real traces (FIU/MSR CSV/oracleGeneral) → M2 (TraceReader already supported)
- Stateful rung-2 interface (`on_access`/`on_evict` with ghost lists) → M4
- Changes to OpenEvolve prompt strategy / engine → separate branch

## Next step (M1)

Short 200–500 iteration run to confirm evolution finds a policy with `combined_score > 0`
(i.e., genuinely beats max(LRU,LFU,ARC) on at least one retained workload). The huge
churn gap (OPT=0.60, B=LRU=0.31, headroom=0.29) is the main opportunity for rung-1.
