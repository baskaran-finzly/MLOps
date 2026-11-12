"""Offline tests: no API key or network needed.

Run from agentic-ai-roadmap/:  python -m pytest tests -q
"""

import json
import os
import py_compile
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")  # lets the client construct; no calls are made

from common import tools as T  # noqa: E402
from common.corpus import _line_tail, build_chunks, chunk_text  # noqa: E402
from common.retriever import TfidfRetriever  # noqa: E402


def test_every_lesson_script_compiles():
    scripts = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    assert len(scripts) > 20
    for p in scripts:
        py_compile.compile(str(p), doraise=True)


def test_chunking_respects_size_and_line_boundaries():
    text = "\n\n".join(f"paragraph {i}\nline two of paragraph {i}" for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 + 60 for c in chunks)
    assert all(not c.startswith(("aragraph", "ine")) for c in chunks)  # no half-words at the start


def test_line_tail():
    assert _line_tail("abc\ndef\nghi", 6) == "ghi"
    assert _line_tail("", 10) == ""
    assert _line_tail("abc", 0) == ""


def test_corpus_and_retriever_find_relevant_file():
    chunks = build_chunks()
    assert {c.metadata["source"] for c in chunks} >= {"model_building/train.py", "deployment/Dockerfile"}
    top = TfidfRetriever(chunks).search("Dockerfile base image python", k=3)
    assert any(c.metadata["source"] == "deployment/Dockerfile" for c, _ in top)


def test_dataset_tools():
    cols = json.loads(T.dataset_columns())
    assert cols["rows"] > 1000 and "ProdTaken" in cols["columns"]
    assert "stats" in json.loads(T.dataset_stats("Age"))
    assert "top_values" in json.loads(T.dataset_stats("Occupation"))
    rates = json.loads(T.conversion_rate_by("CityTier"))
    assert all(0 <= v["conversion_rate"] <= 1 for v in rates.values())


def test_calculator_is_safe():
    assert float(T.calculator("(2 + 3) * 4")) == 20
    for bad in ("__import__('os').system('ls')", "open('x')", "a + 1"):
        with pytest.raises(ValueError):
            T.calculator(bad)


def test_execute_reports_errors_instead_of_raising():
    out, is_error = T.execute("dataset_stats", {"column": "NotAColumn"})
    assert is_error and "Valid columns" in out
    out, is_error = T.execute("read_repo_file", {"path": "../../etc/passwd"})
    assert is_error
    out, is_error = T.execute("no_such_tool", {})
    assert is_error


def test_tool_schemas_are_strict_json_schema():
    for fn in T.ALL_FUNCTIONS:
        s = T.schema(fn)
        assert s["name"] == fn.__name__ and s["description"]
        assert s["input_schema"]["type"] == "object"
        assert s["input_schema"].get("additionalProperties") is False
