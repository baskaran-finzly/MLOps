"""Shared helpers used by every lesson.

Every lesson imports from here so the model name, client setup and the
"get the text out of a response" logic live in exactly one place.

Credentials: the Anthropic SDK reads ANTHROPIC_API_KEY from the environment
(or an `ant auth login` profile). Never hard-code a key in a lesson file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anthropic

# Model used by every lesson. Override for a run with:
#   AGENT_MODEL=claude-sonnet-5-5 python some_lesson.py
MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")

# Models that support server-side refusal fallbacks (`fallbacks="default"`).
# If a safety classifier declines a request, the API transparently retries it
# on a suitable fallback model inside the same call.
_FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-2",
    "claude-fable-5-1",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Repo root (…/MLOps) – lessons use the repo's own MLOps code as a knowledge base.
REPO_ROOT = Path(__file__).resolve().parents[2]

client = anthropic.Anthropic()


def fallback_kwargs(model: str = MODEL) -> dict:
    """Extra request kwargs that turn on server-side refusal fallbacks."""
    if model in _FALLBACK_MODELS:
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}
    return {}


def create(**kwargs):
    """Thin wrapper over `client.beta.messages.create`.

    Fills in the default model / max_tokens and enables refusal fallbacks.
    Accepts every normal Messages API argument (system, messages, tools, ...).
    """
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("max_tokens", 16000)
    extra = fallback_kwargs(kwargs["model"])
    if "betas" in kwargs and "betas" in extra:
        kwargs["betas"] = list({*kwargs["betas"], *extra.pop("betas")})
    return client.beta.messages.create(**extra, **kwargs)


def text_of(response) -> str:
    """Join all text blocks of a response (ignores thinking / tool_use blocks)."""
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        return f"[model declined this request – category: {category}]"
    return "".join(b.text for b in response.content if b.type == "text").strip()


def ask(prompt: str, system: str | None = None, **kwargs) -> str:
    """One-shot question -> answer string. The simplest possible LLM call."""
    if system:
        kwargs["system"] = system
    response = create(messages=[{"role": "user", "content": prompt}], **kwargs)
    return text_of(response)


def parse(output_model, prompt: str | None = None, *, messages: list | None = None, **kwargs):
    """Structured output: returns a validated instance of the Pydantic `output_model`.

    The API constrains the reply to the model's JSON schema, and the SDK validates
    and parses it. Returns None if the model declined (stop_reason == "refusal").
    """
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("max_tokens", 16000)
    messages = messages or [{"role": "user", "content": prompt}]
    response = client.beta.messages.parse(
        output_format=output_model, messages=messages, **fallback_kwargs(kwargs["model"]), **kwargs
    )
    return response.parsed_output


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def require_api_key() -> None:
    """Friendly early exit when no credentials are configured."""
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        cfg = Path.home() / ".config" / "anthropic"
        if not cfg.exists():
            sys.exit(
                "No Anthropic credentials found.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   (get one at console.anthropic.com)\n"
                "  or run:  ant auth login"
            )
