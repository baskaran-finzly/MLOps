# 15 · Action Planning (plan, adapt, retry intelligently)

> **Goal:** Make agents plan before acting, track progress, recover from failures and re-plan.

## Why plan?
A pure ReAct agent decides one step at a time — fine for short tasks, but on long tasks it can wander, repeat itself or forget the goal. An explicit plan gives **direction, progress tracking and a place to adapt**.

## Plan-and-Execute (with re-planning)
```
goal ─► PLANNER (structured plan: steps + success criteria)
            │
            ▼
      for each step:  EXECUTOR (a small tool-using agent) ─► result
            │                │
            │            failed / surprising?
            │                ▼
            │         RE-PLANNER (revise remaining steps, given what we learned)
            ▼
      SYNTHESISER ─► final answer
```

## Retry strategies (from cheapest to most expensive)
1. **Transient errors** (rate limits, timeouts) → exponential backoff (SDK does this for you).
2. **Bad tool input** → return the error as `is_error: true`; the model fixes the call itself.
3. **Step failed** → retry the step once with the error message as extra context.
4. **Plan is wrong** → re-plan the remaining steps.
5. **Still stuck** → stop and escalate to a human, with a summary of what was tried.

## Tips
- Use **structured outputs** for plans (Pydantic) so your code can iterate over steps reliably.
- Give each step a **success criterion** — it makes "did it work?" checkable.
- Keep a **scratchpad/progress log** the executor can read (state lives in your code, not only in the prompt).
- Cap re-plans (e.g. 2) to avoid infinite loops.

## Hands-on
```bash
python level2_agent_essentials/15_action_planning/plan_execute_replan.py
```
Goal: *"Write a short sales-targeting brief"* over the dataset. The planner creates steps; an executor agent runs them with tools; a deliberately missing column triggers an error → retry → re-plan.

## Exercises
1. Print the plan as a checklist and tick steps off as they complete.
2. Add a "human approval of the plan" step before execution.
3. Persist the plan + progress to `plan_state.json` so the run can resume after a crash.
