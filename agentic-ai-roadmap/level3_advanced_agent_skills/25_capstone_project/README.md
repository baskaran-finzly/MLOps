# 25 · Capstone: the "MLOps Copilot" agent

> **Goal:** Combine everything into one portfolio-worthy project built on *this* repository.

## The brief
Build an agent that a data-science team can talk to (and that also runs on its own) to keep the Wellness Tourism ML system healthy:

> *"Is our model still good? What changed in the data? What should we fix in the pipeline? Tell the team in Slack."*

## Required capabilities → where you learned them

| Capability | Lessons |
|---|---|
| Answers questions about the repo's code with citations (RAG over the codebase) | 04-07 |
| Analyses the dataset & model metrics with tools (incl. MLflow runs) | 09, 11 |
| Uses a workflow for routine reports, an agent for open-ended questions | 13 |
| Plans multi-step investigations and re-plans on failure | 14, 15 |
| Remembers team preferences and past findings | 16 |
| Delegates to specialists (data, code, MLOps) in parallel | 17 |
| Has an eval set with ≥ 20 cases and a CI quality gate | 18 |
| Input/action/output guardrails; approval before any write action | 19 |
| Exposes its tools as an MCP server | 20 |
| Posts a weekly digest to Slack (dry-run by default) | 21 |
| Runs nightly on a schedule over a task queue | 22 |
| Uses prompt caching + effort tuning; tracks cost per run | 23 |
| Deployed as a containerised API with logs and health checks | 24 |

## Milestones
1. **MVP (week 1)** – FastAPI `/ask` endpoint (lesson 24) + tools for dataset stats, repo search and file reading. 10 eval cases.
2. **Smarter (week 2)** – add a **data-drift tool**: compare a new CSV batch to the training data (mean/std shifts, new categories, PSI). Add memory and planning.
3. **Team player (week 3)** – multi-agent review (`/review` endpoint), Slack digest, MCP server.
4. **Autonomous (week 4)** – nightly run via GitHub Actions: drift check → if drift > threshold, open a task "retrain recommended" and alert Slack. Eval in CI. Cost dashboard.

## Stretch goals
- A `retrain` tool that triggers the existing GitHub Actions pipeline (`workflow_dispatch`) — **only after human approval**.
- A Streamlit chat UI next to the existing `deployment/app.py`.
- Compare your hand-rolled agent with a LangGraph or Claude Agent SDK version on the same eval.

## Definition of done
- [ ] `eval_harness`-style eval with ≥ 20 cases, pass rate ≥ 85 %, run in CI
- [ ] No write action without approval; prompt-injection case in the eval passes
- [ ] p95 latency and cost per request reported
- [ ] One-command local run and one-command Docker run documented
- [ ] A short write-up: architecture diagram, trade-offs, what you'd do next

## Rubric (self-assessment)
| | 1 – basic | 3 – solid | 5 – excellent |
|---|---|---|---|
| **Usefulness** | Answers simple questions | Finds real issues in the repo | Proposes fixes the team would merge |
| **Reliability** | Works on happy path | Eval ≥ 80 % | Eval ≥ 90 %, pass^3 tracked |
| **Safety** | Basic system prompt | Approval gates | Layered guards + adversarial evals |
| **Engineering** | Script | Service + Docker | CI/CD, observability, cost controls |
