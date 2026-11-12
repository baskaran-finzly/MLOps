# 08 · API Wrappers (direct SDK vs LangChain / LlamaIndex)

> **Goal:** Know the layers between your code and the model, and when each is worth it.

```
 your app
   │
   ├── Framework (LangChain, LlamaIndex, Haystack…)   ← abstractions: chains, retrievers, loaders, agents
   │       │
   ├── Official SDK (anthropic, openai…)              ← typed client, retries, streaming, tool helpers
   │       │
   └── Raw HTTP  POST https://api.anthropic.com/v1/messages
```

| Layer | Pros | Cons | Use when |
|---|---|---|---|
| **Raw HTTP** | No deps, any language | You handle retries, streaming, types | Shell scripts, unsupported languages |
| **Official SDK** | Typed, retries & timeouts built in, newest features first, easy to debug | You write your own glue | **Default.** Most production apps |
| **LangChain** | Huge integration catalogue (loaders, vector DBs, tools), swap providers | Extra abstraction to learn/debug, version churn | Prototyping, lots of integrations |
| **LlamaIndex** | Best-in-class data ingestion & RAG indexes | Less general for non-RAG apps | Document-heavy RAG |

**Advice:** learn the raw SDK first (this course does). Frameworks are much easier once you know what they wrap.

## Production concerns the SDK handles for you
- **Retries** on 429/5xx with backoff (`max_retries`), **timeouts**, typed **errors** (`RateLimitError`, `APIConnectionError`…), **streaming** for long outputs, request IDs for support tickets.

## Hands-on
```bash
python level1_genai_rag_basics/08_api_wrappers/raw_vs_framework.py
pip install langchain-anthropic      # optional
python level1_genai_rag_basics/08_api_wrappers/raw_vs_framework.py --langchain
```
Same question four ways: raw HTTP, SDK, SDK streaming, (optionally) LangChain — plus proper error handling.

## Exercises
1. Wrap the SDK call in your own `LLMClient` class with logging of tokens + latency per call.
2. Trigger an error on purpose (bad model name) and confirm which exception class you get.
3. Build the lesson-06 RAG pipeline with LlamaIndex and compare lines of code.
