# 20 · Custom Toolkits (equip agents with APIs or Python tools)

> **Goal:** Package tools so any agent — yours, Claude Desktop, Claude Code, LangGraph — can use them, via the **Model Context Protocol (MCP)**.

## From functions to toolkits
Lessons 09-19 passed Python functions straight to the model. That's fine inside one app, but every new agent has to re-wire them. A **toolkit** is a reusable, versioned bundle of tools with a stable interface.

## MCP in one picture
```
 ┌──────────── MCP host (your agent, Claude Desktop, Claude Code, IDE…) ────────────┐
 │  MCP client ───────── JSON-RPC over stdio / HTTP ─────────► MCP server            │
 └───────────────────────────────────────────────────────────  (this lesson)        │
                                                                 ├─ tools     (functions the model can call)
                                                                 ├─ resources (read-only data: files, rows)
                                                                 └─ prompts   (reusable prompt templates)
```
Write the server once → plug it into any MCP-capable host. Thousands of ready-made servers exist (GitHub, Slack, Notion, Postgres, Google Drive…).

## Toolkit design checklist
- **Cohesive scope** – one domain per toolkit ("mlops-toolkit"), 5-15 tools.
- **Model-friendly names & docstrings** – they *are* the prompt.
- **Validated inputs** (types, enums), **helpful errors**, **compact outputs**.
- **Read vs write** – separate safe read tools from side-effecting ones; mark/gate the latter.
- **Auth** – read secrets from env vars on the server side; never pass them through the model.
- **Version & test** – unit-test tools without any LLM (see `tests/`).

## Hands-on
```bash
pip install "mcp[cli]>=2" "anthropic[mcp]"

# 1) The MCP server: exposes this repo's dataset + code tools
python level3_advanced_agent_skills/20_custom_toolkits/mlops_mcp_server.py   # (waits on stdio; Ctrl+C)

# 2) An agent that connects to that server and uses its tools
python level3_advanced_agent_skills/20_custom_toolkits/mcp_client_agent.py "Which city tier converts best, and where is the model trained?"
```

### Use the same server from Claude Code or Claude Desktop
```bash
claude mcp add mlops-toolkit -- python /absolute/path/to/agentic-ai-roadmap/level3_advanced_agent_skills/20_custom_toolkits/mlops_mcp_server.py
```

## Exercises
1. Add a `model_metrics` tool that reads the latest MLflow run (`mlflow.search_runs`).
2. Add an MCP **resource** `dataset://schema` and an MCP **prompt** `weekly_report`.
3. Connect a public MCP server (e.g. the GitHub one) alongside yours in the same agent.
