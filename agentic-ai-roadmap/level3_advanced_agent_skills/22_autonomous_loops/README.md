# 22 · Autonomous Loops (agents that run and update tasks on their own)

> **Goal:** Run an agent unattended — on a schedule or continuously — working through a task queue, recording progress and stopping safely.

## Anatomy of an autonomous agent
```
            ┌─────────────── scheduler (cron / GitHub Actions / Managed Agents deployment) ──────────────┐
            ▼                                                                                            │
   load state (tasks.json) ─► pick next pending task ─► agent works on it (tools) ─► update task status ─┘
                                        ▲                       │
                                        │                       ├─► may add follow-up tasks
                                        └──── checkpoint ◄──────┘
   stop when: queue empty │ max tasks per run │ budget spent │ repeated failures │ human "stop" flag
```

## Must-haves for unattended agents
| Concern | Practice |
|---|---|
| **State** | Persist task queue + progress outside the model (JSON/DB). The model's context is not a database. |
| **Checkpointing** | Save after every task so a crash resumes instead of restarting. |
| **Stop conditions** | Max tasks per run, token/cost budget, max retries per task, kill switch file/flag. |
| **Bounded autonomy** | Agent may add follow-up tasks, but with a cap (e.g. ≤ 3 per run) — no infinite self-expansion. |
| **Retries** | Failed tasks get `attempts += 1`; after N attempts → `blocked`, needs a human. |
| **Observability** | Log every run; send a summary (Slack, lesson 21) when the run ends. |
| **Scheduling** | `cron`, a GitHub Actions `schedule:` trigger (like `pipeline.yml`), or a hosted scheduler such as Claude Managed Agents scheduled deployments. |

## Hands-on
```bash
python level3_advanced_agent_skills/22_autonomous_loops/autonomous_task_agent.py --reset   # seed tasks.json
python level3_advanced_agent_skills/22_autonomous_loops/autonomous_task_agent.py           # process up to 3 tasks
python level3_advanced_agent_skills/22_autonomous_loops/autonomous_task_agent.py --watch 300   # loop every 5 min
touch level3_advanced_agent_skills/22_autonomous_loops/STOP                                  # kill switch
```
Open `tasks.json` between runs to watch statuses change (`pending → done / blocked`) and new follow-up tasks appear. Reports land in `reports/`.

### Schedule it with GitHub Actions
```yaml
on:
  schedule: [{cron: "17 6 * * 1-5"}]   # weekdays 06:17 UTC
jobs:
  agent:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r agentic-ai-roadmap/requirements.txt
      - run: python agentic-ai-roadmap/level3_advanced_agent_skills/22_autonomous_loops/autonomous_task_agent.py
        env: {ANTHROPIC_API_KEY: "${{ secrets.ANTHROPIC_API_KEY }}"}
```

## Exercises
1. Add a `priority` field and always pick the highest-priority pending task.
2. Post a run summary to Slack using the lesson-21 tool.
3. Add a budget: stop the run once cumulative tokens exceed 200 000.
