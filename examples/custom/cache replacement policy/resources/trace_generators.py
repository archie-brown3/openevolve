"""
Synthetic trace generators for the cache-replacement-policy evolution experiment.

Each generator returns a deterministic list of integer keys (an access trace).
The four workloads correspond to the patterns named in the experiment design and
are chosen because the classical/adaptive baselines (LRU, LFU, ARC) diverge on
them -- which is exactly what gives an evolved scoring function room to win:

  - temporal_locality : recency-biased reuse            -> LRU-friendly
  - stable_frequency  : static skewed popularity (Zipf) -> LFU-friendly
  - scan              : hot set polluted by one-time sweeps -> scan-resistance matters
  - churn             : working set rotates over time       -> adaptivity matters

Design notes
------------
* Everything is seeded (numpy Generator). Same seed -> identical trace, so
  evaluation is reproducible and fitness is not polluted by trace noise.
* Keys are plain ints. A trace is a list[int]; that is the only contract the
  simulator / evaluator depends on.
* `generate_all(seed)` returns the canonical benchmark suite as an ordered dict
  {workload_name: trace}. The evaluator imports this; the __main__ block below
  prints summary statistics so you can eyeball the traces standalone.

Run standalone:
    python trace_generators.py
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# Default suite parameters. Kept modest so a full multi-policy evaluation over
# all workloads x a few capacities stays well under a second.
# ---------------------------------------------------------------------------
DEFAULT_LENGTH = 20_000  # accesses per trace
DEFAULT_SEED = 42


def _zipf_keys(rng: np.random.Generator, n_keys: int, length: int, s: float) -> np.ndarray:
    """Sample `length` keys in [0, n_keys) from a Zipf(s) popularity law.

    Rank r (1-indexed) gets weight 1/r**s. Lower keys are more popular.
    """
    ranks = np.arange(1, n_keys + 1)
    weights = 1.0 / np.power(ranks, s)
    weights /= weights.sum()
    return rng.choice(n_keys, size=length, p=weights)


def temporal_locality(
    length: int = DEFAULT_LENGTH,
    n_keys: int = 5_000,
    window: int = 200,
    reuse_prob: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Recency-biased stream: a key just seen is likely to be seen again soon.

    With probability `reuse_prob` the next access is drawn (uniformly) from the
    last `window` distinct keys; otherwise a fresh key from the universe is used.
    This is the canonical case where LRU shines.
    """
    rng = np.random.default_rng(seed)
    trace: list[int] = []
    recent: list[int] = []
    fresh = iter(rng.permutation(n_keys).tolist() * (length // n_keys + 2))
    for _ in range(length):
        if recent and rng.random() < reuse_prob:
            key = recent[rng.integers(len(recent))]
        else:
            key = next(fresh)
        trace.append(int(key))
        recent.append(int(key))
        if len(recent) > window:
            recent.pop(0)
    return trace


def stable_frequency(
    length: int = DEFAULT_LENGTH,
    n_keys: int = 5_000,
    zipf_s: float = 1.1,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Static skewed popularity (Zipf). Popularity never shifts, so counting
    accesses (LFU) pays off and pure recency (LRU) wastes the cache on
    one-hit-wonder tails."""
    rng = np.random.default_rng(seed + 1)
    return _zipf_keys(rng, n_keys, length, zipf_s).astype(int).tolist()


def scan(
    length: int = DEFAULT_LENGTH,
    hot_set: int = 50,
    scan_span: int = 4_000,
    scan_period: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """A small hot set accessed most of the time, periodically interrupted by a
    long one-time sequential sweep over fresh keys. Naive LRU evicts the hot set
    during a sweep (which never pays off); scan-resistant policies protect it.

    Hot keys occupy [0, hot_set). Scan keys are unique and disjoint.
    """
    rng = np.random.default_rng(seed + 2)
    trace: list[int] = []
    next_scan_key = hot_set
    while len(trace) < length:
        # Burst of repeated hot-set access (high reuse).
        for _ in range(scan_period):
            if len(trace) >= length:
                break
            trace.append(int(rng.integers(hot_set)))
        # One-time sequential sweep that pollutes the cache.
        for _ in range(scan_span):
            if len(trace) >= length:
                break
            trace.append(next_scan_key)
            next_scan_key += 1
    return trace


def churn(
    length: int = DEFAULT_LENGTH,
    working_set: int = 500,
    shift: int = 50,
    epoch: int = 1_000,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Rotating working set: every `epoch` accesses the active window of keys
    slides forward by `shift`, so previously hot keys go cold and new ones turn
    hot. Pure LRU/LFU lag the rotation; an adaptive policy tracks it. This is the
    churn pattern CACHEUS targets."""
    rng = np.random.default_rng(seed + 3)
    trace: list[int] = []
    base = 0
    for i in range(length):
        if i and i % epoch == 0:
            base += shift
        key = base + int(rng.integers(working_set))
        trace.append(key)
    return trace


def generate_all(seed: int = DEFAULT_SEED, length: int = DEFAULT_LENGTH) -> "OrderedDict[str, list[int]]":
    """Return the canonical benchmark suite as {name: trace}. This is the entry
    point the evaluator imports."""
    return OrderedDict(
        [
            ("temporal_locality", temporal_locality(length=length, seed=seed)),
            ("stable_frequency", stable_frequency(length=length, seed=seed)),
            ("scan", scan(length=length, seed=seed)),
            ("churn", churn(length=length, seed=seed)),
        ]
    )


def _reuse_distance_stats(trace: list[int]) -> tuple[float, float]:
    """Mean and median stack reuse distance (distinct keys since last access).
    Infinite for first-time accesses; those are excluded from the average."""
    last_pos: dict[int, int] = {}
    dists: list[int] = []
    for pos, key in enumerate(trace):
        if key in last_pos:
            # distinct keys touched since the previous access of `key`
            seen = set(trace[last_pos[key] + 1 : pos])
            dists.append(len(seen))
        last_pos[key] = pos
    if not dists:
        return float("inf"), float("inf")
    arr = np.array(dists)
    return float(arr.mean()), float(np.median(arr))


def _summarize(name: str, trace: list[int]) -> str:
    unique = len(set(trace))
    # reuse distance is O(n * distinct) above; sample to keep __main__ snappy
    sample = trace[: min(len(trace), 4_000)]
    mean_rd, med_rd = _reuse_distance_stats(sample)
    return (
        f"{name:<20} length={len(trace):>7}  unique={unique:>6}  "
        f"reuse_dist(mean/median, first 4k)={mean_rd:7.1f}/{med_rd:6.1f}"
    )


if __name__ == "__main__":
    print(f"Synthetic benchmark suite (seed={DEFAULT_SEED}):\n")
    suite = generate_all()
    for name, trace in suite.items():
        print("  " + _summarize(name, trace))

    # Determinism check: regenerating with the same seed must be identical.
    again = generate_all()
    identical = all(suite[k] == again[k] for k in suite)
    print(f"\nDeterministic across two generations with same seed: {identical}")
