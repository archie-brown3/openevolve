∫# OpenEvolve Experiments

Tracks experiment runs against the OpenEvolve framework. Each section is a separate run.

**Baseline problem:** Circle packing n=26, target sum\_radii = 2.635 (AlphaEvolve paper result).  
**Initial program score:** ~0.959 sum\_radii (target ratio ≈ 0.364).

---

## Experiment: Qwen/Qwen3-0.6B-GGUF (local llama.cpp) — Circle Packing n=26

**Date:** 2026-06-13  
**Model:** Qwen/Qwen3-0.6B-GGUF via llama.cpp at `localhost:8080`  
**Config:** `experiments/circle_packing_local_vs_cloud/config_local.yaml`  
**Iterations:** 30  
**Wall time:** 11m 52s  
**Avg time / iteration:** 23.7s  

### Fitness Curve

Target: 2.635 (AlphaEvolve). Baseline: ~0.959 (ratio ≈ 0.364).

| Checkpoint | Best sum\_radii | Target ratio | Generation |
|:----------:|:--------------:|:------------:|:---------:|
|         30 |         0.9598 |       0.3642 |         0 |

**Final best:** 0.9598 sum\_radii (36.42% of target 2.635)

### Mutation Acceptance & Quality

| Metric | Value |
|:-------|------:|
| Total iterations scheduled | 30 |
| Invalid LLM outputs (parse failed) | 28 (93.3%) |
| Valid programs produced (in trace) | 2 |
| Mutations improving parent score | 0 / 2 (0.0%) |

### Convergence

No clear plateau within 30 iterations — the run may still have headroom to improve.

### Mutation Diversity

Code similarity between consecutive best programs (lower similarity = more exploration):

| Span | Similarity | Diversity |
|:-----|----------:|----------:|

### Throughput

- Total wall time: 11m 52s for 30 iterations
- Average time per iteration: 23.7s
- Token counts not directly available; use `llama-server --metrics` and `GET /metrics` for per-request token data.

### Verdict: **USELESS**

The local model fails to make meaningful progress: too many invalid outputs and/or no fitness improvement beyond the baseline. Not viable for autonomous evolution on this problem. Consider using it only as a low-weight secondary ensemble member under a capable cloud model.

---

## Experiment: Qwen/Qwen3-0.6B-GGUF (local llama.cpp) — Circle Packing n=26

**Date:** 2026-06-13  
**Model:** Qwen/Qwen3-0.6B-GGUF via llama.cpp at `localhost:8080`  
**Config:** `experiments/circle_packing_local_vs_cloud/config_local.yaml`  
**Iterations:** 30  
**Wall time:** 9m 3s  
**Avg time / iteration:** 18.1s  

### Fitness Curve

Target: 2.635 (AlphaEvolve). Baseline: ~0.959 (ratio ≈ 0.364).

| Checkpoint | Best sum\_radii | Target ratio | Generation |
|:----------:|:--------------:|:------------:|:---------:|
|         30 |         0.9598 |       0.3642 |         0 |

**Final best:** 0.9598 sum\_radii (36.42% of target 2.635)

### Mutation Acceptance & Quality

| Metric | Value |
|:-------|------:|
| Total iterations scheduled | 30 |
| Invalid LLM outputs (parse failed) | 28 (93.3%) |
| Valid programs produced (in trace) | 7 |
| Mutations improving parent score | 0 / 7 (0.0%) |

### Convergence

No clear plateau within 30 iterations — the run may still have headroom to improve.

### Mutation Diversity

Code similarity between consecutive best programs (lower similarity = more exploration):

| Span | Similarity | Diversity |
|:-----|----------:|----------:|

### Throughput

- Total wall time: 9m 3s for 30 iterations
- Average time per iteration: 18.1s
- Token counts not directly available; use `llama-server --metrics` and `GET /metrics` for per-request token data.

### Verdict: **USELESS**

The local model fails to make meaningful progress: too many invalid outputs and/or no fitness improvement beyond the baseline. Not viable for autonomous evolution on this problem. Consider using it only as a low-weight secondary ensemble member under a capable cloud model.

---

## Experiment: Qwen/Qwen3-0.6B-MLX-bf16 (local llama.cpp) — Circle Packing n=26

**Date:** 2026-06-13  
**Model:** Qwen/Qwen3-0.6B-MLX-bf16 via llama.cpp at `localhost:8080`  
**Config:** `experiments/circle_packing_local_vs_cloud/config_local.yaml`  
**Iterations:** 30  
**Wall time:** 1m 18s  
**Avg time / iteration:** 2.6s  

### Fitness Curve

Target: 2.635 (AlphaEvolve). Baseline: ~0.959 (ratio ≈ 0.364).

| Checkpoint | Best sum\_radii | Target ratio | Generation |
|:----------:|:--------------:|:------------:|:---------:|
|         30 |         0.9598 |       0.3642 |         0 |

**Final best:** 0.9598 sum\_radii (36.42% of target 2.635)

### Mutation Acceptance & Quality

| Metric | Value |
|:-------|------:|
| Total iterations scheduled | 30 |
| Invalid LLM outputs (parse failed) | 28 (93.3%) |
| Valid programs produced (in trace) | 7 |
| Mutations improving parent score | 0 / 7 (0.0%) |

### Convergence

No clear plateau within 30 iterations — the run may still have headroom to improve.

### Mutation Diversity

Code similarity between consecutive best programs (lower similarity = more exploration):

| Span | Similarity | Diversity |
|:-----|----------:|----------:|

### Throughput

- Total wall time: 1m 18s for 30 iterations
- Average time per iteration: 2.6s
- Token counts not directly available; use `llama-server --metrics` and `GET /metrics` for per-request token data.

### Verdict: **USELESS**

The local model fails to make meaningful progress: too many invalid outputs and/or no fitness improvement beyond the baseline. Not viable for autonomous evolution on this problem. Consider using it only as a low-weight secondary ensemble member under a capable cloud model.

---

## Experiment: Qwen/Qwen3-0.6B-MLX-bf16 (local llama.cpp) — Circle Packing n=26

**Date:** 2026-06-13  
**Model:** Qwen/Qwen3-0.6B-MLX-bf16 via llama.cpp at `localhost:8080`  
**Config:** `experiments/circle_packing_local_vs_cloud/config_local.yaml`  
**Iterations:** 30  
**Wall time:** 9m 3s  
**Avg time / iteration:** 18.1s  

### Fitness Curve

Target: 2.635 (AlphaEvolve). Baseline: ~0.959 (ratio ≈ 0.364).

| Checkpoint | Best sum\_radii | Target ratio | Generation |
|:----------:|:--------------:|:------------:|:---------:|
|          5 |         0.9598 |       0.3642 |         0 |
|         30 |         0.9598 |       0.3642 |         0 |

**Final best:** 0.9598 sum\_radii (36.42% of target 2.635)

### Mutation Acceptance & Quality

| Metric | Value |
|:-------|------:|
| Total iterations scheduled | 30 |
| Invalid LLM outputs (parse failed) | 28 (93.3%) |
| Valid programs produced (in trace) | 14 |
| Mutations improving parent score | 0 / 14 (0.0%) |

### Convergence

No clear plateau within 30 iterations — the run may still have headroom to improve.

### Mutation Diversity

Code similarity between consecutive best programs (lower similarity = more exploration):

| Span | Similarity | Diversity |
|:-----|----------:|----------:|
| cp5 → cp30 | 100.0% | 0.0% |
| **Average** | **100.0%** | **0.0%** |

### Throughput

- Total wall time: 9m 3s for 30 iterations
- Average time per iteration: 18.1s
- Token counts not directly available; use `llama-server --metrics` and `GET /metrics` for per-request token data.

### Verdict: **USELESS**

The local model fails to make meaningful progress: too many invalid outputs and/or no fitness improvement beyond the baseline. Not viable for autonomous evolution on this problem. Consider using it only as a low-weight secondary ensemble member under a capable cloud model.

---
