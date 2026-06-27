"""
Seed program for the cache-replacement-policy evolution experiment.

OpenEvolve mutates ONLY the `score()` function inside the EVOLVE-BLOCK. The fixed
`simulate()` harness below calls it on every eviction and must not be changed — it
is the contract the evaluator depends on (see IMPLEMENTATION.md §A).

`score(...)` returns a value where HIGHER = more valuable = keep; the lowest-scored
resident is evicted. The seed `-recency` reproduces LRU exactly, so generation 0 is a
known-good baseline (the evaluator scores it 0.0 = no gain over LRU).
"""

# EVOLVE-BLOCK-START
def score(age, frequency, recency, size, cache_size, now):
    """Higher score = keep; lowest-scored resident is evicted.
    age        = now - insert_time
    frequency  = total access count while resident
    recency    = now - last_access_time
    size       = item size (1 in the uniform-cost experiments)
    cache_size = capacity (constant within a run)
    now        = current logical timestamp
    """
    return -recency  # seed == LRU
# EVOLVE-BLOCK-END


def simulate(trace, capacity):
    """Fixed trace-driven simulator. Returns the hit rate of `score()` on `trace`
    at the given cache `capacity`. Not evolved."""
    cache = {}  # key -> [insert_time, last_access, freq, size]
    hits = 0
    for now, key in enumerate(trace):
        m = cache.get(key)
        if m is not None:
            hits += 1
            m[1] = now
            m[2] += 1
        else:
            if len(cache) >= capacity:
                victim = min(
                    cache,
                    key=lambda k: score(
                        now - cache[k][0], cache[k][2], now - cache[k][1],
                        cache[k][3], capacity, now,
                    ),
                )
                del cache[victim]
            cache[key] = [now, now, 1, 1]
    return hits / len(trace)


if __name__ == "__main__":
    # Tiny self-check: seed must behave like LRU on a simple reuse pattern.
    trace = [1, 2, 3, 1, 2, 3, 4, 1, 2, 3]
    print("hit rate @cap2:", simulate(trace, 2))