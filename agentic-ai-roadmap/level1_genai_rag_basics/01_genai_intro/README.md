# 01 · GenAI Introduction

> **Goal:** Understand what Generative AI is, where it is used, and make your first LLM call.

## Concepts

| Term | Meaning |
|---|---|
| **Generative AI** | Models that *create* new content (text, code, images, audio) rather than only classifying or predicting a number. |
| **LLM** (Large Language Model) | A neural network trained on huge text corpora to predict the next token. Claude, GPT, Gemini, Llama are LLMs. |
| **Prompt** | The input you send the model: instructions + context + question. |
| **Completion / response** | What the model generates back. |
| **Foundation model** | A general model you adapt to many tasks with prompting, RAG or fine-tuning instead of training from scratch. |

### Traditional ML vs GenAI (you already know traditional ML from this repo!)

| | `model_building/train.py` (XGBoost) | GenAI / LLM |
|---|---|---|
| Input | Fixed tabular features | Free-form text / images |
| Output | A class or number (`ProdTaken` = 0/1) | Free-form text, code, JSON, tool calls |
| Training | You train it on your data | Pre-trained by a lab; you *prompt* it |
| Adapting to a new task | Retrain | Change the prompt |

### Where GenAI is used
Chat assistants · code generation · summarisation · search over documents (RAG) · data extraction · customer support · **agents** that take actions (the rest of this course).

## Hands-on

```bash
python level1_genai_rag_basics/01_genai_intro/hello_claude.py
```

The script makes three calls that show three GenAI uses: explain, generate, transform.

## Exercises
1. Change the prompt so Claude explains the *XGBoost model in this repo* to a non-technical manager.
2. Add a `system` prompt that makes every answer start with a one-line TL;DR.
3. Print `response.usage` — how many input/output tokens did each call use?

## Key takeaway
An LLM call is just: **messages in → content blocks out**. Everything else in this course (RAG, tools, agents) is built on top of that one call.
