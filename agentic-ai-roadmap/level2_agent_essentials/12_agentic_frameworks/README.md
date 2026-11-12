# 12 · Agentic Frameworks

> **Goal:** Know the main frameworks, what they give you over a hand-written loop, and pick sensibly.

You wrote the loop in lesson 11. Frameworks package that loop plus extras: state management, persistence, tracing, multi-agent orchestration, integrations.

## Landscape (late 2026)

| Framework | Mental model | Strengths | Pick it when |
|---|---|---|---|
| **Anthropic SDK Tool Runner** | Decorated Python functions + auto loop | Tiny, official, per-turn hooks for approval/logging | You want the lesson-11 agent with ~70 % less code |
| **Claude Agent SDK** | Claude Code as a library: built-in file/bash/web tools, subagents, hooks, permissions | Batteries-included coding / file-system agents | Agents that read/write files and run commands |
| **Claude Managed Agents** | Anthropic hosts the loop *and* a sandbox per session | No infra, versioned agents, schedules, long sessions | Hosted, scheduled or long-running agents |
| **LangGraph** | Agent = explicit **state graph** (nodes, edges, checkpoints) | Fine-grained control, persistence, human-in-the-loop, time travel | Complex, stateful, production workflows |
| **LangChain** | Chains + huge integration library | Loaders, retrievers, vector stores | You need many integrations quickly |
| **CrewAI** | A **crew** of role-playing agents with tasks | Very fast multi-agent prototypes | Role-based teams ("researcher", "writer") |
| **AutoGen / AG2** | Agents that **converse** with each other | Research, conversational multi-agent patterns | Experimenting with agent-to-agent chat |
| **LlamaIndex Agents** | Agents over your indexed data | Strong RAG integration | Document-heavy agents |

### How to choose
1. Start with the **SDK tool runner** (or your own loop) — you'll understand everything.
2. Need explicit graphs, checkpoints, resumability → **LangGraph**.
3. Need file/shell/coding superpowers → **Claude Agent SDK**.
4. Need it hosted, scheduled, sandboxed → **Managed Agents**.
5. Want a quick role-play multi-agent demo → **CrewAI**.

> Frameworks change fast; the concepts (loop, tools, memory, planning, evaluation) don't. Learn concepts, rent frameworks.

## Hands-on
```bash
# Same Data Analyst agent as lesson 11, using the SDK tool runner
python level2_agent_essentials/12_agentic_frameworks/tool_runner_agent.py

# Optional: the same agent in LangGraph
pip install langgraph langchain-anthropic
python level2_agent_essentials/12_agentic_frameworks/langgraph_agent.py
```

## Exercises
1. Diff `tool_runner_agent.py` against lesson 11. What did the framework take over?
2. Add an approval prompt inside a tool function (`input("Allow? y/n")`) — human-in-the-loop with no loop code.
3. Build a 2-agent CrewAI crew (researcher + writer) that produces a short report on the dataset.
