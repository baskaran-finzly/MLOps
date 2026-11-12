# 16 · Agent Memory (recall past actions)

> **Goal:** Give agents short-term and long-term memory so they stay coherent and learn across sessions.

The API is stateless — *all* memory is something your code stores and feeds back.

## Types of memory

| Type | Human analogy | Implementation | Lifetime |
|---|---|---|---|
| **Short-term / working** | What you're thinking about now | The `messages` list (conversation + tool results) | One session |
| **Summary memory** | "Where were we?" | Periodically summarise old turns; keep summary + recent turns. Or server-side **compaction**. | One long session |
| **Long-term semantic** | Facts you know ("user prefers bullet points") | Key-value / document store the agent writes & reads via tools | Across sessions |
| **Episodic** | Memories of past events ("last run, column X was missing") | Log of past runs/outcomes, retrieved by similarity | Across sessions |
| **Procedural** | Skills / how-tos | Instructions, playbooks, Agent Skills files loaded on demand | Across sessions |

## Design choices
- **What to remember** – let the agent decide via a `save_memory` tool, or extract facts automatically after each session.
- **How to recall** – load everything (small stores), keyword/embedding search (large stores), or recency.
- **Context hygiene** – long tool outputs bloat context. Use *context editing* (clear old tool results) or *compaction* (server summarises old turns) in long-running agents.
- **Privacy** – memory is data: allow users to view/delete it, don't store secrets, scope memory per user.

> Anthropic also ships a ready-made **memory tool** (`{"type": "memory_20250818", "name": "memory"}`) where the model reads/writes files in a `/memories` directory that *you* store. The script below builds the same idea with two plain tools so you see every moving part.

## Hands-on
```bash
python level2_agent_essentials/16_agent_memory/memory_agent.py          # interactive chat
python level2_agent_essentials/16_agent_memory/memory_agent.py --demo   # scripted 2-session demo
```
- **Session 1:** you tell the agent preferences/facts → it saves them to `memory_store.json`.
- **Session 2 (fresh process, empty conversation):** it recalls them.
- **Summary memory** kicks in when the conversation grows past a threshold.

## Exercises
1. Add a `forget_memory(key)` tool and a `/memories` command that prints the store.
2. Replace keyword recall with embedding search from lesson 05.
3. Add **episodic memory**: after each answer, store `{question, tools_used, outcome}` and retrieve similar past episodes at the start of a new task.
