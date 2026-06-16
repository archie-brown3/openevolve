# Phase 1 — Repository Reconnaissance

## Overview

Four repositories were examined on disk at `/Users/archiebrown/`. The HiFo-Prompt codebase is **not present** on disk; it exists only as notes in the Obsidian vault. A2DEPT's operator selection mechanism is described in Obsidian notes and serves as the closest prior work for the novel semantic-selector condition.

---

## 1. OpenEvolve (`/Users/archiebrown/openevolve`)

### Installability
**Yes.** `pip install -e ".[dev]"` via setuptools/pyproject.toml. Python >=3.10.

**Dependencies** (`pyproject.toml:14-20`):
- `openai>=1.0.0`, `pyyaml>=6.0`, `numpy>=1.22.0`, `tqdm>=4.64.0`, `flask`, `dacite>=1.9.2`

All are current. No staleness issues. Dev deps include `pytest>=7.0.0`, `black>=22.0.0`, `mypy>=0.950`.

### Heuristic Representation (`openevolve/database.py:44-111`)
```python
@dataclass
class Program:
    id: str
    code: str
    changes_description: str = ""
    language: str = "python"
    parent_id: Optional[str] = None
    generation: int = 0
    timestamp: float = time.time()
    iteration_found: int = 0
    metrics: Dict[str, float] = {}
    complexity: float = 0.0
    diversity: float = 0.0
    metadata: Dict[str, Any] = {}
    prompts: Optional[Dict[str, Any]] = None
    artifacts_json: Optional[str] = None       # JSON for small artifacts
    artifact_dir: Optional[str] = None         # path for large artifacts
    embedding: Optional[List[float]] = None    # for novelty rejection
```

A heuristic = a code string + metric dict + lineage metadata + optional artifacts.

### Population Management (`openevolve/database.py:113+`)
`ProgramDatabase` implements MAP-Elites + island-based evolution:
- Multiple islands (`num_islands`, default 5), each with its own feature grid
- Per-island feature maps (`island_feature_maps`) map feature-coord keys to best program ID
- Archive of elite programs (`archive`)
- Global `best_program_id` tracking
- Lazy migration between islands based on generation counts
- `sample()` at line 382: three strategies (exploration, exploitation, weighted)
- `sample_from_island()` at line 403: thread-safe island-specific sampling

### Operators
**None in the traditional GA sense.** OpenEvolve has a single mechanism: the LLM generates either:
1. Diff-based edits (`<<<<<<< SEARCH...>>>>>>> REPLACE` pattern, default)
2. Full program rewrites

No crossover. No multi-mutation operator pool. No operator selection layer. The single-operator iteration logic is at `openevolve/iteration.py:36-231`:
- Sample parent/inspirations from database
- Build prompt via `PromptSampler`
- Call LLM
- Parse response (diffs or full rewrite)
- Evaluate child program
- Return `Result` with child program

### Selection/Sampling
- `database.sample()` at `database.py:382-401`: exploration_ratio / exploitation_ratio / weighted sampling
- Parent selection only — no operator selection

### LLM Calls
`LLMEnsemble` at `openevolve/llm/ensemble.py:17-94` wraps multiple `OpenAILLM` instances (`openevolve/llm/openai.py:47-292`).

Call site at `openevolve/iteration.py:112-115`:
```python
llm_response = await llm_ensemble.generate_with_context(
    system_message=prompt["system"],
    messages=[{"role": "user", "content": prompt["user"]}],
)
```

The `_call_api` at `openai.py:218-232` calls `self.client.chat.completions.create(**params)` and returns only `response.choices[0].message.content` — **discarding the `usage` field with token counts.**

**No token/cost tracking at the call site.** This must be addressed for honest cost accounting.

---

## 2. ReEvo (`/Users/archiebrown/reevo`)

### Installability
**Pipe-mode only** — run via `python main.py` (Hydra entry point). Not packaged for `pip install -e`. Uses `uv.lock` for deps. Python >=3.11.

**Dependencies** (`pyproject.toml:15-38`):
- `openai>=1.8.0` (lockfile: 1.68.0), `pyyaml>=6.0.1`, `hydra-core>=1.3.2`, `numpy>=1.26.3`, `scipy>=1.11.4`, `pydantic>=2.5.3`, `tqdm>=4.66.3`

### Heuristic Representation (`reevo.py:109-115`)
Plain dict, not a class:
```python
individual = {
    "stdout_filepath": str,    # path to stdout file
    "code_path": str,          # path to code file
    "code": str,               # the heuristic code
    "response_id": int,
    "exec_success": bool,      # whether execution succeeded
    "obj": float,              # objective value (minimization)
    "traceback_msg": str,      # error message if failed
}
```
No lineage tracking beyond the dict. `best_code_overall` and `elitist` stored separately.

### Population Management (`reevo.py:48-138`)
- Simple list of dicts. No MAP-Elites. No islands. No archive beyond `elitist`.
- `rank_select()` at line 274-299: probability proportional to rank
- `random_select()` at line 302-323: equal probability
- Selection filters out non-executing individuals

### Operators Confirmed (all in `reevo.py`)

| Operator | Method | Lines | Type | Signature |
|---|---|---|---|---|
| Short-term reflection | `short_term_reflection()` | 358-378 | Reflection | `(population: list[dict]) -> tuple[list[list[dict]], list[str], list[str]]` |
| Crossover | `crossover()` | 408-438 | Crossover | `(short_term_reflection_tuple: tuple) -> list[dict]` |
| Long-term reflection | `long_term_reflection()` | 380-405 | Reflection | `(short_term_reflections: list[str]) -> None` |
| Mutation | `mutate()` | 441-458 | Mutation | `() -> list[dict]` |

**Crossover prompt** (`prompts/common/crossover.txt`): Takes worse_code, better_code, and reflection. Prompts LLM to write improved `func_name_v2`. Outputs code only in Python code block.

**Mutation prompt** (`prompts/common/mutation.txt`): Takes prior reflection + elitist_code. Prompts LLM to write mutated `func_name_v2`.

**Evolve loop** (`reevo.py:461-488`): select → short_reflection → crossover → evaluate → update_iter → long_reflection → mutate → evaluate → update_iter.
**Operators are called in a FIXED SEQUENCE, not selected.** There is no operator selection mechanism.

### LLM Calls
`BaseClient.multi_chat_completion()` at `utils/llm_client/base.py:44-85`. Uses `concurrent.futures.ThreadPoolExecutor` for parallel calls. `OpenAIClient` at `utils/llm_client/openai.py:13-36` wraps `openai.OpenAI()`.

Call site at `openai.py:33-36`:
```python
response = self.client.chat.completions.create(
    model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
)
return response.choices
```

**No token/cost tracking** — returns only `response.choices[i].message.content`.

---

## 3. EoH (`/Users/archiebrown/EoH`)

### Installability
**Yes.** `pip install eoh/` via setuptools at `eoh/setup.py`. Python >=3.10.

**Dependencies** (`eoh/setup.py:11-14`):
- `numpy`, `numba`, `joblib`
- Notably: **no `openai` in setup.py** — EoH uses raw `http.client.HTTPSConnection` for API calls, not the OpenAI SDK.

### Heuristic Representation (`eoh/src/eoh/methods/eoh/eoh_interface_EC.py:103-109`)
```python
offspring = {
    'algorithm': str,    # NL description of algorithm
    'code': str,         # Python implementation
    'objective': float,  # fitness value
    'other_inf': None    # unused
}
```
Algorithm description is explicitly stored alongside code.

### Population Management (`eoh/src/eoh/methods/eoh/eoh.py:66-184`)
- Simple list of dicts, sorted by objective
- Management via `pop_greedy` (top-pop_size) or `ls_greedy`/`ls_sa` for local search
- Configurable: `ec_pop_size`, `ec_n_pop` (number of generations/populations), `ec_m` (parents per crossover)

### Operators Confirmed (all in `eoh/src/eoh/methods/eoh/eoh_evolution.py`)

| Operator | Method | Lines | Type | Description |
|---|---|---|---|---|
| i1 | `i1()` | 171-188 | Init | Generate novel algorithm from scratch |
| e1 | `e1(parents)` | 190-207 | Crossover | Diverse new algorithm from multiple parents |
| e2 | `e2(parents)` | 209-226 | Crossover | Identify backbone idea, create new from it |
| m1 | `m1(parents)` | 228-245 | Mutation | Modified version of one existing algorithm |
| m2 | `m2(parents)` | 247-264 | Mutation | Parameter tuning of existing algorithm |
| m3 | `m3(parents)` | 266-282 | Mutation | Simplify/enhance generalization |

**Signatures** (from `eoh_interface_EC.py:103-131` dispatch):
- `e1(parents: list[dict]) -> [code, algorithm]` — multiple parents
- `e2(parents: list[dict]) -> [code, algorithm]` — multiple parents
- `m1(parents: list[dict]) -> [code, algorithm]` — single parent (parents[0])
- `m2(parents: list[dict]) -> [code, algorithm]` — single parent (parents[0])
- `m3(parents: list[dict]) -> [code, algorithm]` — single parent (parents[0])

### Operator Selection (`eoh/src/eoh/methods/eoh/eoh.py:144-155`)
Default operators for EoH: `['e1', 'e2', 'm1', 'm2']` (from `utils/getParas.py:72`).

**Probabilistic weight-gating** (line 150):
```python
if (np.random.rand() < op_w):
    parents, offsprings = interface_ec.get_algorithm(population, op)
```

Each operator has an independent weight (`ec_operator_weights`, default all 1.0). **This is NOT mutually exclusive** — multiple operators can fire in a single generation. Operators are iterated in fixed order, each independently gated by its weight. This is a primitive scheduling strategy, not a proper AOS.

### Selection Methods (`eoh/src/eoh/methods/selection/`)
- `prob_rank.py` (default): rank-proportional selection — `1 / (rank + 1 + len(pop))`
- `roulette_wheel.py`: fitness-proportional
- `tournament.py`: size-2 tournament
- `equal.py`: uniform random

### LLM Calls
`InterfaceAPI` at `eoh/src/eoh/llm/api_general.py:5-51`. Uses **raw `http.client.HTTPSConnection`**, not the OpenAI SDK:
```python
conn = http.client.HTTPSConnection(self.api_endpoint)
conn.request("POST", "/v1/chat/completions", payload_explanation, headers)
res = conn.getresponse()
data = res.read()
json_data = json.loads(data)
response = json_data["choices"][0]["message"]["content"]
```

**Very fragile**: simple retry loop (max 5), no async, no streaming, **no token/cost tracking**. `InterfaceLLM` at `eoh/src/eoh/llm/interface_LLM.py:4-56` adds local LLM support but same call path.

---

## 4. HiFo-Prompt / A2DEPT Operator Selection

### Status
**No code on disk.** The A2DEPT paper's operator selection is described in Obsidian notes at:
- `/Users/archiebrown/Obsidian-Vault/MSc Dissertation/Reading/A2DEPT/A2DEPT_CODE_TRACE.md`
- `/Users/archiebrown/Obsidian-Vault/MSc Dissertation/Reading/A2DEPT/A2DEPT Feature Analysis.md`

No A2DEPT repository exists anywhere under `/Users/archiebrown/`.

### A2DEPT's Operator Selection Mechanism (from Obsidian notes)

**Adaptive Softmax Scheduler** (Eq. 5):
```
P(op | n) = exp(ω_op / τ) / Σ exp(ω_k / τ)
```
- `ω_op`: accumulated weight per operator
- `τ`: temperature parameter controlling exploration/exploitation balance
- Produces a probability distribution over operators

**Credit Assignment** (Eq. 6):
- Compute normalized improvement: `r̄ = (S_new - S_parent) / |S_parent|`
- If `r̄ ≥ 0`: increase `ω_op` by `r̄`
- Otherwise: decrease `ω_op` by `λ·|r̄|` (penalty coefficient `λ = 0.8`)
- Updated weight vector is inherited by the child node (lineage-aware)

**Temperature Schedule** (Eq. 3):
- Geometric cooling: `T_{t+1} = α·T_t`
- Re-annealing (`+ΔT`) when no global improvement for `N_stall` generations
- Balances exploration vs exploitation over the run

**Operator Pool**: `{m₁, m₂}` selected via softmax; `e₁` (crossover) triggered conditionally.

### Design Intent (from user's Obsidian notes)
`Scheduler Research Direction — Cheap Signals + Reflection Escalation.md` positions a semantic LLM-based selector that:
- Uses cheap state-signals by default
- Escalates to reflection-based reasoning at ambiguous decision points
- Triggers escalation on entropy over the operator weight vector
- Contrasts with A2DEPT's purely numeric adaptation

---

## 5. Dependency Compatibility Matrix

| Concern | OpenEvolve | ReEvo | EoH | Conflict? |
|---|---|---|---|---|
| Python | >=3.10 | >=3.11 | >=3.10 | **Shared env must be >=3.11** |
| `openai` | >=1.0.0 | >=1.8.0 | N/A (raw HTTP) | No — compatible |
| `hydra-core` | None | >=1.3.2 | None | No — additive |
| `numpy` | >=1.22.0 | >=1.26.3 | latest | No — higher bound wins |
| `pyyaml` | >=6.0 | >=6.0.1 | None | No — compatible |
| `scipy` | None | >=1.11.4 | None | No — additive |
| `numba` | None | None | required | No — additive |
| `joblib` | None | None | required | No — additive |
| LLM client | `openai` SDK | `openai` SDK | raw `http.client` | **Three different call paths** |

---

## 6. Key Gaps for Your Experiment

### 6.1 No Operator Pool in OpenEvolve
OpenEvolve has exactly one mechanism: LLM-based diff/full-rewrite edits. You need to add:
- Mutation variants from ReEvo (`mutate()`) and EoH (`m1`, `m2`, `m3`)
- Crossover from ReEvo (`crossover()` with reflection) or EoH (`e1`, `e2`)
- Operator registration infrastructure (operator name → function + prompt mapping)

### 6.2 No Cost Tracking Anywhere
None of the three repos capture token usage or API cost. Specifically:
- OpenEvolve `_call_api()` at `openai.py:225`: discards `response.usage`
- ReEvo `_chat_completion_api()` at `openai.py:33`: discards `response.usage`
- EoH `api_general.py:43`: parses only `json_data["choices"][0]["message"]["content"]`

You need to instrument every LLM call site with a wrapper that:
- Captures `response.usage.prompt_tokens`, `completion_tokens`, `total_tokens`
- Computes cost based on model-specific pricing
- Accumulates per-condition budgets for honest comparison

### 6.3 No Selector Layer in Any Repo
- **ReEvo**: Fixed sequence (reflection → crossover → mutate)
- **EoH**: Independent weight-gating per operator (not mutually exclusive)
- **OpenEvolve**: No operators to select from
- **A2DEPT**: Code not on disk, only described in notes

You will build all four conditions from scratch:
1. Random scheduling over operator pool
2. Best single operator (baseline)
3. Numerical AOS (UCB/DMAB from Fialho)
4. Semantic LLM-based selector (adapted from HiFo-Prompt/LAOS concepts)

### 6.4 LLM Client Heterogeneity
Three different call mechanisms. For consistent cost accounting, you should normalize to one client path (recommend OpenEvolve's `openai` SDK approach) and wrap it with a cost meter.
