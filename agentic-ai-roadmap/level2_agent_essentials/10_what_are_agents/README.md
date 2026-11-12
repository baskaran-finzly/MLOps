# 10 · What Are AI Agents?

> **Goal:** Know precisely what an agent is, how it differs from a chatbot or a workflow, and when *not* to build one.

## Definition
An **AI agent** is an LLM that **decides its own next action in a loop** — calling tools, observing results, and continuing until the goal is met.

```
            ┌──────────────────────────────────────────┐
            ▼                                          │
 goal ─► [ LLM: think → choose action ] ─► tool call ─► observe result
                     │
                     └─► "done" ─► final answer
```

That loop (often called **ReAct**: *Reason + Act*) is the whole trick. Everything else — memory, planning, multi-agent — is a refinement of it.

## The spectrum of autonomy

| Level | What decides the steps? | Example in this repo's world | Lesson |
|---|---|---|---|
| **Single LLM call** | Nobody – one shot | "Summarise `train.py`" | 01-03 |
| **RAG** | Fixed: retrieve → answer | "How is the model deployed?" | 06-07 |
| **Workflow** | *Your code* (fixed graph of LLM calls) | Review PR → if risky, run checks → write report | 13 |
| **Agent** | *The model* (dynamic loop) | "Find why conversion dropped and propose a fix" | 11+ |
| **Multi-agent** | Several models coordinating | Analyst agent + reviewer agent + writer agent | 17 |

## Building blocks of every agent
1. **Model** – the reasoning engine.
2. **Tools** – how it acts on the world (lesson 09).
3. **Instructions** – system prompt: role, goal, constraints, when to stop.
4. **Memory** – short-term (conversation) and long-term (stored facts) (lesson 16).
5. **Planning** – breaking goals into steps, re-planning on failure (lesson 15).
6. **Guardrails** – limits on what it may do (lesson 19).
7. **Evaluation** – proof that it works (lesson 18).

## Should you build an agent? (check all four)
- **Complexity** – is the task multi-step and hard to specify in advance?
- **Value** – does the outcome justify higher cost and latency?
- **Viability** – can the model actually do this task well?
- **Cost of error** – can mistakes be caught and undone (tests, review, rollback)?

If any answer is "no", use a simpler level from the table. **Start simple; add autonomy only when it pays.**

## Self-check
- What's the difference between a workflow and an agent? *(who controls the control flow)*
- Why does an agent need a max-steps limit? *(loops can run forever and burn money)*
- Which of your daily MLOps tasks would benefit from an agent, and which just need a script?

➡️ Next: build one from scratch in lesson 11.
