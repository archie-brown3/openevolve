# Gaps and Action Items for the MSc Experiment

## What Must Be Built (none of this exists in any repo on disk)

### 1. Operator Infrastructure in OpenEvolve

OpenEvolve currently has a single LLM-call-per-iteration with no operator pool. You need:

- [ ] **Operator registry**: Name → function mapping. Borrow from EoH's dispatch pattern at `eoh_interface_EC.py:103-131`.
- [ ] **Mutation operators** (at least 2):
  - From ReEvo: `mutate()` at `reevo.py:441-458` — elitist-based, uses long-term reflection
  - From EoH: `m1` at `eoh_evolution.py:228-245` — modified version of existing
  - From EoH: `m2` at `eoh_evolution.py:247-264` — parameter tuning
- [ ] **Crossover operator** (at least 1):
  - From ReEvo: `crossover()` at `reevo.py:408-438` — requires short-term reflection first
  - From EoH: `e1` at `eoh_evolution.py:190-207` — diverse from multiple parents
- [ ] **Reflection operator**:
  - From ReEvo: `short_term_reflection()` at `reevo.py:358-378` — compares two individuals, identifies why one is better
  - From ReEvo: `long_term_reflection()` at `reevo.py:380-405` — summarises short-term reflections
- [ ] **Operator interface normalization**: All borrowed operators must share a uniform interface (input: selected parents + LLM client, output: new individual)

### 2. Cost Meter (required for all four conditions)

None of the three repos capture token/cost data. Every LLM call discards `response.usage`.

- [ ] **Cost wrapper**: Intercept every `chat.completions.create` response to capture:
  - `response.usage.prompt_tokens`
  - `response.usage.completion_tokens`
  - `response.usage.total_tokens` (where available)
- [ ] **Model pricing table**: Map model name → cost per 1K tokens (input and output prices differ)
- [ ] **Budget tracker**: Per-condition accumulator for total LLM-call cost
- [ ] **Budget enforcement**: Stop condition based on cost budget, not iteration count (this is the "honest cost accounting" requirement)

The OpenEvolve call site to instrument is `openevolve/llm/openai.py:218-232` (`_call_api`).

### 3. Four Selector Conditions (the novel surface)

#### Condition A: Random Scheduling (baseline)
- Uniform random selection from operator pool each iteration
- Equal-cost comparison against other conditions

#### Condition B: Best Single Operator
- Run each operator in isolation for the same budget
- Report which single operator dominates
- Serves as lower bound for what AOS must beat

#### Condition C: Numerical AOS (UCB/DMAB from Fialho)
- Implement from Fialho's published work (not from any repo on disk)
- UCB1 formula: `UCB(op) = mean_reward(op) + c * sqrt(ln(N) / n(op))`
- Or DMAB: sliding window + statistical test for operator switch
- Track per-operator reward (improvement in fitness / tokens spent)

#### Condition D: Semantic LLM-Based Selector (adapted from HiFo-Prompt/LAOS)
- No code on disk — build from concept
- Reference: A2DEPT's softmax scheduler + your extension with reflection escalation
- Key mechanism (from your Obsidian design notes):
  - Default: cheap state-signals (per-operator recent success rate)
  - Escalation trigger: weight-vector entropy exceeds threshold → LLM reflection on which operator to use now
  - After reflection: bias selector for N steps, then revert to cheap-signal
- Data needed: per-operator history (what was tried, what worked, what didn't) passed as context to the reflection LLM call

### 4. LLM Client Normalization

Three repos use three different call paths:
- OpenEvolve: `openai.OpenAI()` SDK → `client.chat.completions.create()`
- ReEvo: same SDK path
- EoH: raw `http.client.HTTPSConnection` → JSON parsing

- [ ] **Unify to one path** (recommend OpenEvolve's path since it has retry logic, async, and error handling)
- [ ] **Wrap with cost meter** at the unified call site

### 5. TSP-Constructive Problem Setup

ReEvo has the most complete TSP constructive benchmark:
- Problem: `reevo/problems/tsp_constructive/eval.py` — evaluates `select_next_node()` heuristic
- Dataset: synthetic instances in `problems/tsp_constructive/dataset/`
- Function signature: `select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix) -> next_node`

- [ ] **Port evaluator to OpenEvolve's interface**: OpenEvolve expects an `evaluate(program_path)` function; ReEvo's eval calls `from gpt import select_next_node_v2`. Bridge these.
- [ ] **Seed heuristic**: Write or borrow an initial constructive heuristic (e.g., nearest-neighbor) with `EVOLVE-BLOCK-START` markers

### 6. Dependency Compatibility

All three repos are compatible in a shared Python >=3.11 environment. No version conflicts. The additive deps (hydra-core, scipy, numba, joblib) do not conflict with OpenEvolve's core deps.

---

## What Already Exists and Can Be Used

| Component | Source | File:Line |
|---|---|---|
| Program dataclass | OpenEvolve | `openevolve/database.py:44` |
| Island-based population | OpenEvolve | `openevolve/database.py:113` |
| MAP-Elites grid | OpenEvolve | `openevolve/database.py:129` |
| Evaluator (cascade) | OpenEvolve | `openevolve/evaluator.py:32` |
| Parallel controller | OpenEvolve | `openevolve/process_parallel.py:335` |
| Prompt templates | OpenEvolve | `openevolve/prompt/templates.py:23-93` |
| Prompt sampler | OpenEvolve | `openevolve/prompt/sampler.py:21` |
| LLM ensemble | OpenEvolve | `openevolve/llm/ensemble.py:17` |
| Crossover prompt | ReEvo | `prompts/common/crossover.txt` |
| Mutation prompt | ReEvo | `prompts/common/mutation.txt` |
| EoH operator prompts | EoH | `eoh_evolution.py:36-118` |
| TSP eval | ReEvo | `problems/tsp_constructive/eval.py` |
| TSP config | ReEvo | `cfg/problem/tsp_constructive.yaml` |

---

## Files Saved

- [x] `experiments/recon/recon-summary.md` — full Phase 1 report
- [x] `experiments/recon/recon-interface-references.md` — file:line index
- [x] `experiments/recon/recon-gaps.md` — this file
