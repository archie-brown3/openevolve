# Interface Reference Index — File:Line Citations

Every claim in the recon report is grounded to a specific file:line. This document provides a quick-lookup index.

---

## OpenEvolve

| Concept | Location |
|---|---|
| Program dataclass | `openevolve/database.py:44-111` |
| ProgramDatabase class | `openevolve/database.py:113` |
| Database.add() | `openevolve/database.py:211` |
| Database.sample() | `openevolve/database.py:382` |
| Database.sample_from_island() | `openevolve/database.py:403` |
| Island structure | `openevolve/database.py:129,142` |
| MAP-Elites feature coords | `openevolve/database.py:834` |
| Island migration | `openevolve/database.py:148-149` |
| Iteration main loop | `openevolve/iteration.py:36-231` |
| LLM call in iteration | `openevolve/iteration.py:112-115` |
| OpenAILLM class | `openevolve/llm/openai.py:47` |
| API call (_call_api) | `openevolve/llm/openai.py:218-232` |
| LLMEnsemble class | `openevolve/llm/ensemble.py:17` |
| ProcessParallelController | `openevolve/process_parallel.py:335` |
| Worker iteration | `openevolve/process_parallel.py:134-332` |
| Controller entry point | `openevolve/controller.py:56` |
| Config dataclasses | `openevolve/config.py:51-418` |
| PromptSampler | `openevolve/prompt/sampler.py:21` |
| TemplateManager | `openevolve/prompt/templates.py:175` |
| pyproject.toml (deps) | `pyproject.toml:13-20` |
| setup.py (installable) | `setup.py:1-3` |

---

## ReEvo

| Concept | Location |
|---|---|
| ReEvo class init | `reevo.py:12-48` |
| Individual dict structure | `reevo.py:109-115` |
| Population init | `reevo.py:105-138` |
| Evaluate population | `reevo.py:173-227` |
| Short-term reflection | `reevo.py:358-378` |
| Crossover | `reevo.py:408-438` |
| Long-term reflection | `reevo.py:380-405` |
| Mutation | `reevo.py:441-458` |
| Evolve main loop | `reevo.py:461-488` |
| Selection: rank_select | `reevo.py:274-299` |
| Selection: random_select | `reevo.py:302-323` |
| Crossover prompt template | `prompts/common/crossover.txt:1-15` |
| Mutation prompt template | `prompts/common/mutation.txt:1-11` |
| BaseClient (LLM) | `utils/llm_client/base.py:11-85` |
| OpenAIClient | `utils/llm_client/openai.py:13-36` |
| TSP eval (problem) | `problems/tsp_constructive/eval.py:1-99` |
| pyproject.toml (deps) | `pyproject.toml:15-38` |
| Main entry point | `main.py:1-65` |

---

## EoH

| Concept | Location |
|---|---|
| EOH class | `eoh/src/eoh/methods/eoh/eoh.py:8-185` |
| EOH.run() main loop | `eoh/src/eoh/methods/eoh/eoh.py:77-184` |
| Operator selection (weights) | `eoh/src/eoh/methods/eoh/eoh.py:144-155` |
| Individual dict structure | `eoh/src/eoh/methods/eoh/eoh_interface_EC.py:103-109` |
| Operator dispatch (_get_alg) | `eoh/src/eoh/methods/eoh/eoh_interface_EC.py:103-131` |
| Evolution class (operators) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:5-282` |
| i1 (init) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:171-188` |
| e1 (crossover) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:190-207` |
| e2 (crossover) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:209-226` |
| m1 (mutation) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:228-245` |
| m2 (mutation) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:247-264` |
| m3 (mutation) | `eoh/src/eoh/methods/eoh/eoh_evolution.py:266-282` |
| Selection: prob_rank | `eoh/src/eoh/methods/selection/prob_rank.py:1-6` |
| Selection: tournament | `eoh/src/eoh/methods/selection/tournament.py:1-12` |
| LLM: InterfaceAPI | `eoh/src/eoh/llm/api_general.py:5-51` |
| LLM: InterfaceLLM | `eoh/src/eoh/llm/interface_LLM.py:4-56` |
| Paras class (config) | `eoh/src/eoh/utils/getParas.py:1-132` |
| Default operators | `eoh/src/eoh/utils/getParas.py:72` |
| Methods dispatch | `eoh/src/eoh/methods/methods.py:5-51` |
| setup.py (deps) | `eoh/setup.py:1-17` |

---

## HiFo-Prompt / A2DEPT (from Obsidian notes)

| Concept | Location |
|---|---|
| Adaptive softmax scheduler (Eq. 5) | `Obsidian/MSc Dissertation/Reading/A2DEPT/A2DEPT_CODE_TRACE.md` (section b) |
| Credit assignment (Eq. 6) | `Obsidian/MSc Dissertation/Reading/A2DEPT/A2DEPT_CODE_TRACE.md` (section e) |
| Temperature schedule (Eq. 3) | `Obsidian/MSc Dissertation/Reading/A2DEPT/A2DEPT_CODE_TRACE.md` (section e) |
| Operator pool {m₁, m₂, e₁} | `Obsidian/MSc Dissertation/Reading/A2DEPT/A2DEPT_CODE_TRACE.md` (section b) |
| UCB1 bandit reference | `Obsidian/MSc Dissertation/Reading/A2DEPT/A2DEPT Feature Analysis.md:88,187` |
| Design notes (cheap signals) | `Obsidian/MSc Dissertation/Design/Scheduler Research Direction — Cheap Signals + Reflection Escalation.md:1-95` |
