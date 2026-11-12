# 13 · Agent Workflows (design how agents think, act and complete tasks)

> **Goal:** Master the five composable patterns that solve most problems *before* you need a fully autonomous agent.

| Pattern | Shape | Use when | Example here |
|---|---|---|---|
| **1. Prompt chaining** | A → gate → B → C | Task splits into fixed sequential steps | Extract facts → draft release note → polish |
| **2. Routing** | classify → specialised handler | Distinct input categories need different handling | Route a question to *data*, *code* or *general* handler |
| **3. Parallelisation** | fan-out → aggregate (sectioning or voting) | Independent sub-tasks, or you want several opinions | Review `train.py` for security / performance / readability in parallel |
| **4. Orchestrator–workers** | LLM plans sub-tasks dynamically → workers → synthesise | Sub-tasks aren't known in advance | Lesson 17 |
| **5. Evaluator–optimizer** | generate → critique → revise loop | Clear quality criteria, iterative refinement helps | Write a model card, critique, improve until it passes |

```
Chaining:      [LLM] → (check) → [LLM] → [LLM]
Routing:       [classifier] ─┬→ [data handler]
                             ├→ [code handler]
                             └→ [general handler]
Parallel:      ┌→ [LLM security] ┐
       input ──┼→ [LLM perf]     ├→ [aggregate]
               └→ [LLM style]    ┘
Eval-optimize: [generator] ⇄ [evaluator]  (until PASS or N rounds)
```

**Workflows vs agents:** in a workflow *your code* owns control flow → predictable, cheaper, testable. Combine them: a workflow whose steps are small agents is very common in production.

## Hands-on
```bash
python level2_agent_essentials/13_agent_workflows/workflow_patterns.py            # all patterns
python level2_agent_essentials/13_agent_workflows/workflow_patterns.py routing    # just one
```
Patterns: `chaining`, `routing`, `parallel`, `evaluator`.

## Exercises
1. Add a gate to the chain: if the extracted facts list is empty, stop early.
2. Add a fourth route `deployment` that answers using `read_repo_file("deployment/Dockerfile")`.
3. Use **voting**: run the same "is this code safe?" check 3× in parallel and take the majority.
