# Operator-Selection Layer (the "router")

This package adds an **adaptive operator router** on top of stock OpenEvolve. An
*operator* is a named prompt strategy; a *selector* (the router) decides which
operator to apply each iteration and learns from the fitness gain it produces.
The whole layer is **off by default** — when `operators.enabled` is `False`,
behaviour is byte-for-byte identical to stock OpenEvolve (every iteration uses
the default `diff_user` / `full_rewrite_user` template).

## Why

Stock OpenEvolve uses a single prompt template for every mutation. Different
prompt strategies ("just diff it" vs "reflect first, then diff") suit different
programs and different stages of a run. Rather than pick one globally, the router
treats each strategy as a **bandit arm** and lets an online algorithm allocate
iterations toward whichever strategy is currently producing the largest fitness
gains — a numerical Adaptive Operator Selection (AOS) scheme.

## Components

| File | Class / fn | Responsibility |
|------|-----------|----------------|
| `pool.py` | `OperatorPool` | Registry mapping `operator_id → template_key`. `"baseline" → None` (use OpenEvolve's default template); `"reflect_rewrite" → "op_reflect_rewrite"`. |
| `selector.py` | `OperatorSelector` (ABC), `RandomSelector` | Selector interface (`select()` / `update(operator_id, reward)`) and a uniform-random control baseline that ignores reward. |
| `bandit.py` | `UCBSelector` | UCB1 bandit. Plays each arm once, then maximises `mean_i + c·sqrt(2·ln t / N_i)`. Keeps an incremental sample mean per arm. |
| `cost_meter.py` | `CostMeter` | Per-operator token accounting (prompt/completion tokens, call counts). |
| `__init__.py` | `build_selector(name, ids, ucb_c, seed)` | Factory: `"random"` → `RandomSelector`, `"ucb"` → `UCBSelector`. |
| `../prompts/defaults/op_reflect_rewrite.txt` | — | The `reflect_rewrite` operator's template: diagnose the biggest weakness in two-three sentences, then emit targeted SEARCH/REPLACE diffs. |

## Configuration

```yaml
operators:
  enabled: false                 # master switch; false ⇒ stock OpenEvolve
  selector: "ucb"                # "random" | "ucb"
  operator_ids: ["baseline", "reflect_rewrite"]
  ucb_c: 1.414                   # UCB1 exploration constant
```

Parsed into `OperatorConfig` (`openevolve/config.py`) via dacite, like every
other config block.

## Control flow

All router state lives on the **parent controller process**
(`ProcessParallelController`), so no locking is needed; workers are stateless and
only receive the resolved `template_key`.

1. **Submit** (`_submit_iteration`): if enabled, `operator_id =
   selector.select()`, then `template_key = pool.resolve(operator_id)`. Both are
   passed to the worker.
2. **Worker** (`_run_iteration_worker`): forwards `template_key` to
   `build_prompt` (a `None` key falls through to the default template). The
   chosen `operator_id` and the call's token `usage` are returned on the
   `SerializableResult`.
3. **Collect** (result loop): the operator id round-trips on the result itself.
   Token usage (when present) is metered. The selector is then credited:
   - **error / discard** → `update(operator_id, 0.0)`
   - **success** → `update(operator_id, fit(child) − fit(parent))`, where
     `fit = get_fitness_score(metrics, feature_dimensions)`. The operator id is
     also stamped onto `child.metadata["operator_id"]`.
4. **Teardown**: logs `selector.stats()` (per-arm pulls + mean reward) and
   `cost_meter.totals()`.

## Current scope & known limitations

The layer is intentionally a thin experiment slice. Honest list of what it does
*not* yet do:

- **Reward asymmetry (failures vs. regressions).** A discarded/errored attempt
  is credited `0.0`, but a *successful* child that regresses scores a *negative*
  reward (`fit(child) − fit(parent) < 0`). So the bandit currently rates an
  outright failure **above** a small genuine regression, which can bias arm
  selection toward failure-prone operators. Reward shaping (e.g. clamp at
  `max(0, gain)`, or assign failures a small negative penalty, or normalise
  gains to `[0, 1]`) would fix the ordering. This is the most important open
  issue.
- **Reward scale is not normalised.** Raw fitness gains vary by orders of
  magnitude across problems, but `ucb_c` is a fixed constant. The
  exploration/exploitation balance is therefore problem-dependent. A
  scale-adaptive bandit or per-run reward normalisation would make `ucb_c`
  transferable.
- **Stationarity assumption.** UCB1 assumes stationary arm rewards, but
  evolution is highly non-stationary — the best operator early (broad rewrites)
  differs from the best operator late (fine diffs). A discounted / sliding-window
  UCB, or Thompson sampling, would track the moving optimum better.
- **Delayed, batched feedback.** With parallel workers, many `select()` calls
  happen before any `update()` lands, so the UCB index is stale within a batch.
  Batched-bandit variants or optimistic in-flight accounting would help.
- **Global, context-free routing.** One selector is shared across all islands
  and generations. Operators likely perform differently per island / per feature
  region; a contextual bandit keyed on those signals could route better.
- **Cost is measured but not used.** `CostMeter` records tokens but never feeds
  selection — there is no cost-aware "reward per token" objective yet.
  Additionally, token usage is only as good as the LLM layer's reporting:
  providers that omit `usage` (or a client that drops it) leave the meter with
  call counts only.
- **No checkpoint/resume of router state.** Bandit counts and means live on the
  controller, not in the database snapshot, so they reset on resume.
- **Static operator registry.** Operators are hard-coded in
  `DEFAULT_OPERATORS`; adding one means adding a template file and a registry
  entry. Config-defined operators/templates would make this open-ended.
- **`reflect_rewrite` assumes diff-based evolution.** Its template mandates the
  SEARCH/REPLACE contract, so it is only meaningful when
  `diff_based_evolution: true`.
