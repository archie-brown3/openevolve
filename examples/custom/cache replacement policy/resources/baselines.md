# Baseline & Oracle Reference Implementations

These are the policies the **evaluator** runs alongside the evolved scoring
function to produce reference hit rates. They are implemented as fixed, trusted
code *inside* `evaluator.py` — they are **never** part of the evolvable surface.
The evolved policy is scored by how much of the **LRU → Belady(OPT)** gap it
closes (see `IMPLEMENTATION.md` §A, "Fitness").

All policies consume the same trace contract: a `list[int]` of keys (see
`trace_generators.py`). Uniform object cost is assumed (`size == 1`), so capacity
is measured in number of items.

The evolved-policy interface (kept identical across all docs):

```python
def score(age, frequency, recency, size, cache_size, now) -> float:
    """Higher score = more valuable = keep. The lowest-scored resident is evicted.
    age        = now - insert_time      (how long the item has lived in cache)
    frequency  = total access count of the item while resident
    recency    = now - last_access_time (how long since last touched)
    size       = item size (1 in the uniform-cost experiments)
    cache_size = capacity (constant within a run)
    now        = current logical timestamp (access index)
    """
```

`score = -recency` reproduces LRU exactly — that is the seed program, verified
against the LRU reference (identical hit rates on all four workloads).

---

## LRU — Least Recently Used

Evict the item whose last access is furthest in the past. Bets on short-term
temporal locality. O(1) per access with an ordered map.

```text
on access(k):
    if k in cache: move k to MRU end; hit
    else:
        if full: evict LRU end
        insert k at MRU end; miss
```
Score-function equivalent: `score = -recency`.

## LFU — Least Frequently Used

Evict the item with the smallest total access count (tie-break by recency).
Rewards stable long-run popularity; pathologically bad when popularity shifts
(cache fills with stale once-hot items) or under churn.

```text
on access(k):
    if k in cache: freq[k] += 1; hit
    else:
        if full: evict argmin(freq[x], then oldest)
        freq[k] = 1; miss
```
Score-function approximation: `score = frequency` (with a recency tie-break).

## ARC — Adaptive Replacement Cache

Maintains two LRU lists — T1 (seen once recently) and T2 (seen ≥2 times) — plus
ghost lists B1/B2 of recently evicted keys. A target `p` for |T1| adapts on ghost
hits: a B1 hit grows T1 (favor recency), a B2 hit grows T2 (favor frequency).
Self-tunes between recency and frequency; resists single-access flooding. O(1)
per access, overhead independent of cache size.

Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead Replacement Cache"
(FAST '03). The evaluator implements the standard T1/T2/B1/B2 + `replace()` form.
*Not* directly expressible as a stateless `score()` — it is a baseline only.

## Belady OPT (MIN) — Oracle Upper Bound

Evict the resident whose **next** access is furthest in the future (∞ if never
reused again). Requires knowing the future, so it is not deployable — it exists
purely to bound achievable hit rate and quantify how much headroom a policy
leaves on the table.

```text
precompute next_use[i] for every position i   # next index where trace[i] recurs
on access at time i for key k:
    if k in cache: hit
    else:
        if full: evict argmax(next_use of resident)   # furthest future use
        insert k with next_use[i]; miss
```

---

## Why these four traces differentiate the policies

Measured hit rates from the premise check (`trace_generators.generate_all()`,
seed 42, capacity 300; "seed" is the evolved-interface LRU seed):

| workload            |   LRU |   LFU |   ARC |  seed |   OPT | LRU→OPT gap |
|---------------------|------:|------:|------:|------:|------:|------------:|
| temporal_locality   | 0.697 | 0.191 | 0.694 | 0.697 | 0.709 |       0.012 |
| stable_frequency    | 0.692 | 0.742 | 0.742 | 0.692 | 0.807 |       0.115 |
| scan                | 0.390 | 0.398 | 0.398 | 0.390 | 0.398 |       0.008 |
| churn               | 0.580 | 0.182 | 0.578 | 0.580 | 0.831 |       0.251 |

Reading: **no single policy wins.** LFU dominates `stable_frequency` yet collapses
on `temporal_locality` (0.19) and `churn` (0.18); LRU/ARC are robust generalists
but leave large frequency/adaptivity gains unclaimed. `stable_frequency` and
`churn` carry most of the evolvable headroom; `temporal_locality` and `scan` are
"don't regress" constraints where LRU is already near-optimal.
