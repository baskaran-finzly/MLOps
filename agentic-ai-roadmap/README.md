# 🗼 Agentic AI Roadmap — from GenAI basics to production agents

A hands-on course that follows the *"How To Start Learning AI Agents"* lighthouse:
**Level 1 → GenAI & RAG basics · Level 2 → AI agent essentials · Level 3 → advanced agent skills.**

Every lesson has a short **README** (concepts → hands-on → exercises) and **runnable Python code**.
Instead of toy data, everything works on **this MLOps repository**: the wellness-tourism dataset in `data/`
and the pipeline code in `model_building/`, `deployment/`, `hosting/` and `.github/`. So the agents you build answer real questions about a real ML system.

---

## 🧭 The roadmap

### Level 1 · GenAI & RAG Basics — *"make the model useful with your data"*
| # | Lesson | You'll build | Key skills |
|---|---|---|---|
| 01 | [GenAI Introduction](level1_genai_rag_basics/01_genai_intro) | First API calls | GenAI vs classic ML, messages → content blocks |
| 02 | [Basics of LLMs](level1_genai_rag_basics/02_llm_basics) | Token / context experiments | Tokens, context window, statelessness, effort |
| 03 | [Prompt Engineering](level1_genai_rag_basics/03_prompt_engineering) | Prompt pattern lab | Roles, XML tags, few-shot, structured JSON |
| 04 | [LLM Preprocessing](level1_genai_rag_basics/04_llm_preprocessing) | Repo chunker *(offline)* | Clean, chunk, overlap, metadata, contextual chunks |
| 05 | [Embeddings](level1_genai_rag_basics/05_embeddings) | Sparse vs dense search *(offline)* | Vectors, cosine similarity, hybrid search |
| 06 | [RAG Fundamentals](level1_genai_rag_basics/06_rag_fundamentals) | Q&A over this repo | Retrieve → augment → generate, citations |
| 07 | [Vector Databases](level1_genai_rag_basics/07_vector_databases) | Chroma-backed RAG | Persistence, filters, ANN indexes |
| 08 | [API Wrappers](level1_genai_rag_basics/08_api_wrappers) | Same call 4 ways | Raw HTTP vs SDK vs LangChain, streaming, errors |
| 09 | [Tool Integration](level1_genai_rag_basics/09_tool_integration) | Manual tool-use round trip | Function calling, tool schemas, server tools |

### Level 2 · AI Agent Essentials — *"let the model decide what to do next"*
| # | Lesson | You'll build | Key skills |
|---|---|---|---|
| 10 | [What Are AI Agents?](level2_agent_essentials/10_what_are_agents) | — (concepts) | Agent loop, autonomy spectrum, when *not* to use agents |
| 11 | [Build Your First Agent](level2_agent_essentials/11_build_first_agent) | Data Analyst Agent from scratch | ReAct loop, tool results, step limits |
| 12 | [Agentic Frameworks](level2_agent_essentials/12_agentic_frameworks) | Same agent with SDK tool runner + LangGraph | LangChain, LangGraph, CrewAI, AutoGen, Claude Agent SDK |
| 13 | [Agent Workflows](level2_agent_essentials/13_agent_workflows) | 4 workflow patterns | Chaining, routing, parallelisation, evaluator-optimizer |
| 14 | [Multi-Step Reasoning](level2_agent_essentials/14_multi_step_reasoning) | Reasoning lab | Adaptive thinking, decomposition, self-consistency, verification |
| 15 | [Action Planning](level2_agent_essentials/15_action_planning) | Plan → execute → re-plan agent | Structured plans, retries, re-planning |
| 16 | [Agent Memory](level2_agent_essentials/16_agent_memory) | Agent that remembers across sessions | Short-term, summary, long-term memory |
| 17 | [Multi-Agent Systems](level2_agent_essentials/17_multi_agent_systems) | Lead + 3 parallel specialists | Orchestrator–workers, delegation briefs |
| 18 | [Agent Evaluation](level2_agent_essentials/18_agent_evaluation) | Eval harness (+ MLflow) | Code graders, LLM-as-judge, trajectory, pass^k |
| 19 | [Safety & Guardrails](level2_agent_essentials/19_safety_guardrails) | Guarded agent | Injection defence, approval gates, PII redaction, grounding |

### Level 3 · Advanced Agent Skills — *"make it real, reliable and running"*
| # | Lesson | You'll build | Key skills |
|---|---|---|---|
| 20 | [Custom Toolkits](level3_advanced_agent_skills/20_custom_toolkits) | MCP server + MCP client agent | Model Context Protocol, reusable toolkits |
| 21 | [Real-World Integration](level3_advanced_agent_skills/21_real_world_integration) | Slack digest agent, remote MCP | Slack/Notion/Gmail, dry-runs, idempotency |
| 22 | [Autonomous Loops](level3_advanced_agent_skills/22_autonomous_loops) | Task-queue agent on a schedule | State, checkpoints, stop conditions, cron |
| 23 | [Optimize Performance](level3_advanced_agent_skills/23_optimize_performance) | Cost/latency benchmark | Caching, effort, routing, parallelism, streaming |
| 24 | [Deploy to Production](level3_advanced_agent_skills/24_deploy_to_production) | FastAPI agent service + Docker | Auth, rate limits, logging, CI quality gates |
| 25 | [Capstone: MLOps Copilot](level3_advanced_agent_skills/25_capstone_project) | Your portfolio project | Everything together |

> **About the whiteboard numbering:** the original image lists *11. Agentic Frameworks* twice, skips 5 and 18, and puts *19 Action Planning* / *20 Safety* above *12 Build Your First Agent*. This course keeps all of its topics, adds the missing **Embeddings** step, and orders them so each lesson builds on the previous one:
>
> | Whiteboard | This course |
> |---|---|
> | 1-4, 6-9 | 01-04, 06-09 (05 = Embeddings, added) |
> | 10 What are agents · 11 Frameworks · 12 First agent | 10 · **12** · **11** (build one by hand *before* using a framework) |
> | 13 Workflows · 14 Memory · 15 Evaluation · 16 Reasoning · 17 Multi-agent · 19 Planning · 20 Safety | 13 · 16 · 18 · 14 · 17 · 15 · 19 |
> | 21 Integration · 22 Autonomous loops · 23 Toolkits · 24 Optimize · 25 Deploy | 21 · 22 · **20** · 23 · 24 (+ 25 Capstone) |

---

## 🚀 Setup

```bash
cd agentic-ai-roadmap
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then put your key in it, or just:
export ANTHROPIC_API_KEY=sk-ant-...                     # get one at console.anthropic.com
```

Run any lesson from inside `agentic-ai-roadmap/`:
```bash
python level1_genai_rag_basics/01_genai_intro/hello_claude.py
```

Check everything is wired up (no API key or network needed):
```bash
python -m pytest tests -q
```

**Model:** lessons use `claude-opus-5-5` by default (set in `common/llm.py`). To experiment with a cheaper/faster model for a run:
```bash
AGENT_MODEL=claude-sonnet-5-5 python level2_agent_essentials/11_build_first_agent/data_analyst_agent.py
```
On supported models, `common/llm.py` also turns on **server-side refusal fallbacks** (`fallbacks="default"`): if a safety classifier declines a request, the API retries it on a fallback model within the same call.

**Cost tip:** most Level 1 calls cost fractions of a cent. Agent lessons make several calls per run; multi-agent (17) and evaluation (18) runs are the most expensive. Lesson 23 shows how to measure and reduce cost.

---

## 📁 Layout

```
agentic-ai-roadmap/
├── common/                       shared code used by every lesson
│   ├── llm.py                    client, model, create()/ask()/parse() helpers
│   ├── corpus.py                 loads + cleans + chunks this repo's files (the RAG knowledge base)
│   ├── retriever.py              TF-IDF retriever (offline, no downloads)
│   └── tools.py                  reusable agent tools: dataset stats, conversion rates, calculator, repo search/read
├── level1_genai_rag_basics/      lessons 01-09
├── level2_agent_essentials/      lessons 10-19
├── level3_advanced_agent_skills/ lessons 20-25
└── tests/                        offline tests for chunking, retrieval and tools
```

---

## 🗓️ Suggested 8-week study plan (≈ 5-7 h/week)

| Week | Lessons | Milestone |
|---|---|---|
| 1 | 01-03 | Comfortable with the API and prompting |
| 2 | 04-07 | RAG chatbot over this repo |
| 3 | 08-11 | Your first hand-written agent |
| 4 | 12-14 | Know frameworks + workflow patterns |
| 5 | 15-17 | Planning, memory, multi-agent |
| 6 | 18-19 | Eval set + guardrails (do **not** skip these) |
| 7 | 20-23 | MCP toolkit, Slack integration, autonomous loop, optimisation |
| 8 | 24-25 | Deployed service + capstone |

**How to study each lesson:** read the README (15 min) → run the script and read its output (15 min) → read the code (20 min) → do at least one exercise (1 h+). The exercises are where the learning happens.

## 📚 Further reading
- Anthropic: *Building effective agents* (workflow patterns), prompt engineering guide, tool use docs — docs.claude.com / platform.claude.com
- Model Context Protocol — modelcontextprotocol.io
- LangGraph, CrewAI and AutoGen documentation for framework deep-dives
