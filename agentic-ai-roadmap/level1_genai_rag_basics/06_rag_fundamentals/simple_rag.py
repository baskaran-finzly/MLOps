"""Lesson 06 – a complete RAG pipeline in ~40 lines, over this repo's code.

Run:  python level1_genai_rag_basics/06_rag_fundamentals/simple_rag.py ["your question"]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ask, banner, require_api_key
from common.retriever import TfidfRetriever, format_context

RAG_SYSTEM = (
    "You answer questions about a software repository using ONLY the documents provided. "
    "Cite sources inline like [model_building/train.py#3]. If the documents don't contain the answer, "
    "say \"I couldn't find that in the repository\" rather than guessing."
)


def rag_answer(question: str, retriever: TfidfRetriever, k: int = 5) -> tuple[str, list]:
    results = retriever.search(question, k=k)  # 1) retrieve
    prompt = (  # 2) augment
        f"<documents>\n{format_context(results)}\n</documents>\n\n"
        f"<question>{question}</question>"
    )
    return ask(prompt, system=RAG_SYSTEM, max_tokens=2000), results  # 3) generate


if __name__ == "__main__":
    require_api_key()
    question = " ".join(sys.argv[1:]) or "Which ML algorithm is trained, and where is the trained model uploaded?"
    retriever = TfidfRetriever()

    banner(f"Question: {question}")

    banner("WITHOUT RAG (model guesses from general knowledge)")
    print(ask(f"In the 'Wellness Tourism MLOps' GitHub repo: {question}", max_tokens=1000))

    banner("WITH RAG (grounded in retrieved chunks)")
    answer, results = rag_answer(question, retriever)
    print("Retrieved:", ", ".join(f"{c.id} ({s:.2f})" for c, s in results))
    print()
    print(answer)
