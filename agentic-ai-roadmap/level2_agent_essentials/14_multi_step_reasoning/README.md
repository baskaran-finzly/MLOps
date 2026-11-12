# 14 · Multi-Step Reasoning

> **Goal:** Get agents to think in logical sequences — and know which techniques still matter with modern reasoning models.

## Techniques

| Technique | Idea | Status with modern models |
|---|---|---|
| **Chain-of-thought (CoT)** | "Think step by step" before answering | Built in: models now *think* natively (adaptive thinking). You control depth with **effort**. |
| **Adaptive thinking + effort** | Model decides how much to reason; `effort` = low…max sets the budget | **Primary lever today.** Use `display: "summarized"` to see a summary of the reasoning. |
| **Interleaved thinking** | Reasoning *between* tool calls ("result looks odd, let me check X") | Automatic with adaptive thinking — makes agents much better at multi-step tool use. |
| **Decomposition** | Break a hard question into sub-questions, solve each, combine | Still valuable for very large tasks; it's what planners (lesson 15) and orchestrators (17) do. |
| **Self-consistency** | Sample several solutions, take the majority | Useful for high-stakes answers; costs N× tokens. |
| **Self-verification** | Ask the model to check its answer (or verify with a tool / test) | Cheap and effective, especially with a *tool* as the checker (calculator, unit test). |
| **ReAct** | Interleave Reason → Act → Observe | The agent loop from lesson 11. |

**Rule of thumb:** raise effort before adding prompting tricks; verify with tools rather than with more words.

## Hands-on
```bash
python level2_agent_essentials/14_multi_step_reasoning/reasoning.py
```
1. Same logic puzzle at `low` vs `high` effort with a visible thinking summary.
2. Explicit decomposition into sub-questions.
3. Self-consistency voting (3 parallel samples).
4. Self-verification with the calculator tool.

## Exercises
1. Find a question where `low` effort is wrong and `high` is right. Note the token difference.
2. Make the decomposition step return a Pydantic list of sub-questions instead of free text.
3. Add a *verifier* that re-runs the final number through `calculator` and flags mismatches.
