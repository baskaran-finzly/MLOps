"""Reusable tools that the tool-use and agent lessons give to the model.

Each tool is a plain, typed Python function with a Google-style docstring.
- `beta_tool(fn)` turns it into a tool the SDK's tool runner can call.
- `schema(fn)` gives the raw JSON definition for hand-written agent loops.
So the function itself is the single source of truth.
"""

from __future__ import annotations

import ast
import json
import operator
from functools import lru_cache

import pandas as pd
from anthropic import beta_tool

from .llm import REPO_ROOT

DATASET = REPO_ROOT / "data" / "wellness_tourism_dataset.csv"
MAX_TOOL_OUTPUT = 6000  # characters – keep tool results small; they cost tokens every turn


@lru_cache(maxsize=1)
def _df() -> pd.DataFrame:
    df = pd.read_csv(DATASET)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


@lru_cache(maxsize=1)
def _retriever():
    from .retriever import TfidfRetriever

    return TfidfRetriever()


# --------------------------------------------------------------------------- data tools
def dataset_columns() -> str:
    """List the columns of the wellness-tourism customer dataset with their dtypes and row count.

    Use this first when you need to know what data is available.
    """
    df = _df()
    return json.dumps({"rows": len(df), "columns": {c: str(t) for c, t in df.dtypes.items()}})


def dataset_stats(column: str) -> str:
    """Summary statistics for one column of the wellness-tourism dataset.

    Numeric columns return count/mean/std/min/quartiles/max and missing count.
    Categorical columns return the top value counts and missing count.

    Args:
        column: Exact column name, e.g. "Age", "MonthlyIncome", "Occupation", "ProdTaken".
    """
    df = _df()
    if column not in df.columns:
        raise ValueError(f"Unknown column '{column}'. Valid columns: {', '.join(df.columns)}")
    s = df[column]
    out = {"column": column, "missing": int(s.isna().sum())}
    if pd.api.types.is_numeric_dtype(s):
        out["stats"] = {k: round(float(v), 3) for k, v in s.describe().items()}
    else:
        out["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(10).items()}
    return json.dumps(out)


def conversion_rate_by(group_by: str) -> str:
    """Purchase conversion rate (mean of ProdTaken) and customer count for each value of a column.

    Useful for questions like "which occupation buys the most packages?".
    Numeric columns with many values are bucketed into 5 quantile bins.

    Args:
        group_by: Column to group by, e.g. "Occupation", "CityTier", "Age".
    """
    df = _df()
    if group_by not in df.columns:
        raise ValueError(f"Unknown column '{group_by}'. Valid columns: {', '.join(df.columns)}")
    key = df[group_by]
    if pd.api.types.is_numeric_dtype(key) and key.nunique() > 10:
        key = pd.qcut(key, 5, duplicates="drop").astype(str)
    g = df.groupby(key, observed=True)["ProdTaken"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    return json.dumps({str(k): {"conversion_rate": round(r["mean"], 3), "customers": int(r["count"])} for k, r in g.iterrows()})


# --------------------------------------------------------------------------- math tool
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Only numbers and + - * / ** % ( ) are allowed")


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly. Use this instead of doing math in your head.

    Args:
        expression: Arithmetic using numbers and + - * / ** % ( ), e.g. "(0.32 - 0.18) / 0.18 * 100".
    """
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


# --------------------------------------------------------------------------- repo tools
def search_repo(query: str, k: int = 4) -> str:
    """Keyword/semantic search over this MLOps repository's code and config files.

    Returns the best-matching chunks with their source file. Use it to find where something is implemented.

    Args:
        query: What to look for, e.g. "hyperparameter grid" or "upload model to Hugging Face".
        k: Number of chunks to return (1-8).
    """
    from .retriever import format_context

    return format_context(_retriever().search(query, k=max(1, min(int(k), 8))))[:MAX_TOOL_OUTPUT]


def list_repo_files() -> str:
    """List the files in the MLOps repository (excluding .git, data and this course folder)."""
    skip = {".git", "agentic-ai-roadmap", "data", "__pycache__"}
    files = [
        str(p.relative_to(REPO_ROOT)) for p in REPO_ROOT.rglob("*")
        if p.is_file() and not (skip & set(p.relative_to(REPO_ROOT).parts))
    ]
    return "\n".join(sorted(files))


def read_repo_file(path: str) -> str:
    """Read a text file from the MLOps repository (truncated to ~6000 characters).

    Args:
        path: Path relative to the repo root, e.g. "deployment/Dockerfile".
    """
    target = (REPO_ROOT / path).resolve()
    if REPO_ROOT not in target.parents or not target.is_file():  # block ../ path traversal
        raise ValueError(f"'{path}' is not a file inside the repository. Use list_repo_files to see valid paths.")
    text = target.read_text(encoding="utf-8", errors="ignore")
    return text if len(text) <= MAX_TOOL_OUTPUT else text[:MAX_TOOL_OUTPUT] + f"\n…[truncated, {len(text):,} chars total]"


# --------------------------------------------------------------------------- registries
ALL_FUNCTIONS = [dataset_columns, dataset_stats, conversion_rate_by, calculator, search_repo, list_repo_files, read_repo_file]
FUNCTIONS_BY_NAME = {f.__name__: f for f in ALL_FUNCTIONS}


def schema(fn) -> dict:
    """Raw JSON tool definition ({name, description, input_schema}) for a manual loop."""
    return beta_tool(fn).to_dict()


def runner_tools(*fns):
    """Wrap functions for `client.beta.messages.tool_runner(tools=...)`."""
    return [beta_tool(f) for f in (fns or ALL_FUNCTIONS)]


def execute(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run a tool by name for a manual loop. Returns (result_text, is_error)."""
    fn = FUNCTIONS_BY_NAME.get(name)
    if fn is None:
        return f"Unknown tool '{name}'", True
    try:
        return str(fn(**tool_input)), False
    except Exception as exc:  # report errors back to the model so it can recover
        return f"Error: {exc}", True
