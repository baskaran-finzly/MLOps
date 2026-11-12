# 04 · LLM Preprocessing (clean → chunk → format)

> **Goal:** Turn raw files into clean, right-sized, well-labelled chunks an LLM (and a retriever) can use.

Garbage in → garbage out applies even more to LLMs: noisy text wastes tokens and confuses retrieval.

## The pipeline

```
raw files ──► load ──► clean ──► chunk ──► attach metadata ──► (embed → store)  [lessons 05-07]
```

| Step | What to do | Watch out for |
|---|---|---|
| **Load** | Read PDFs, HTML, code, CSV… into text. (`pypdf`, `beautifulsoup4`, `unstructured`) | Encoding errors, tables lost in PDFs |
| **Clean** | Strip boilerplate, nav bars, repeated headers, excess whitespace; normalise unicode. | Don't strip meaningful structure (headings, code indentation) |
| **Chunk** | Split into pieces small enough to retrieve precisely, big enough to make sense alone. | Too small = no context. Too big = noisy retrieval & cost |
| **Overlap** | Repeat ~10-20 % of the previous chunk so a sentence isn't cut in half. | Too much overlap = duplicate results |
| **Metadata** | Source file, section, chunk index, date, author… | Needed for citations and filtering |

### Chunking strategies
- **Fixed-size** (N chars/tokens) – simple baseline.
- **Recursive / structural** – split on headings → paragraphs → sentences (what `common/corpus.py::chunk_text` does with blank lines).
- **Semantic** – split where the topic changes (embedding similarity drops).
- **Code-aware** – split on functions/classes.
- **Contextual chunking** – ask an LLM to prepend a 1-sentence summary of *where this chunk sits in the document* before embedding. Big retrieval gains; shown in the script.

## Hands-on
```bash
python level1_genai_rag_basics/04_llm_preprocessing/chunking.py          # offline: no API key needed
python level1_genai_rag_basics/04_llm_preprocessing/chunking.py --contextual   # + LLM contextual headers
```

## Exercises
1. Change `max_chars` to 400 and 3000. How many chunks? Read a few — which size "makes sense alone"?
2. Write a `chunk_python(source)` that splits on top-level `def`/`class` using the `ast` module.
3. Add a `"language"` metadata field (`python`/`yaml`/`markdown`) based on the file extension.
