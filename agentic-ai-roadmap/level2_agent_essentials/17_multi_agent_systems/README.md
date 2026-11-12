# 17 · Multi-Agent Systems

> **Goal:** Make several specialised agents work together on a task too big or too broad for one.

## Why more than one agent?
- **Specialisation** – focused instructions + a small, relevant tool set per agent beat one agent with 30 tools.
- **Context isolation** – each sub-agent reads lots of material in its *own* context and returns only a summary.
- **Parallelism** – independent sub-tasks run at the same time.
- **Checks & balances** – a reviewer agent catches the worker's mistakes.

**Costs:** many more tokens, harder debugging, coordination failures. Use it when the task genuinely fans out.

## Architectures

| Pattern | Shape | Example |
|---|---|---|
| **Orchestrator–workers (supervisor)** | Lead agent plans, delegates to workers, synthesises | Lead → data analyst + code auditor → report (this lesson) |
| **Pipeline / assembly line** | A → B → C, each a specialist | Researcher → writer → editor |
| **Handoff / router** | Agent transfers the conversation to a better-suited agent | Triage → billing agent |
| **Debate / reviewer** | Agents critique each other until agreement | Proposer ⇄ critic |
| **Hierarchical** | Supervisors of supervisors | Large research systems |

## Design rules
1. **Delegate with a full brief** – goal, context, what to return, output format, limits. Sub-agents know *nothing* you don't tell them.
2. **Return compressed results** – ask workers for concise findings, not transcripts.
3. **Give each worker only the tools it needs.**
4. **Cap fan-out and depth** (e.g. ≤ 4 workers, no recursive spawning) to control cost.
5. **Use a cheaper model for simple workers**, the strongest one for the lead.

## Hands-on
```bash
python level2_agent_essentials/17_multi_agent_systems/orchestrator_workers.py
```
A **lead agent** gets the task *"Assess whether our wellness-tourism model and data are ready for a marketing campaign"*. It decides which specialists to call through a `delegate` tool:
- `data_analyst` (dataset tools) · `code_auditor` (repo tools) · `mlops_reviewer` (repo tools, pipeline focus)

Workers run **in parallel**, and the lead writes the final report.

## Exercises
1. Add a `critic` worker that reviews the lead's draft before it's returned.
2. Run workers on `claude-haiku-5-5` (set `WORKER_MODEL`) — compare cost and quality.
3. Rebuild the same crew in CrewAI or LangGraph and compare.
