"""Model + tracing configuration from environment variables.

  LLM_PROVIDER   ollama (default) | groq
  LLM_MODEL      default llama3.2 for ollama, llama-3.3-70b-versatile for groq
  OLLAMA_BASE_URL  default http://localhost:11434
  LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST  -> enables Langfuse tracing
"""

from __future__ import annotations

import os


def get_model(temperature: float = 0.0):
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"), temperature=temperature)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.environ.get("LLM_MODEL", "llama3.2"),
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=temperature,
        )
    raise ValueError(f"unknown LLM_PROVIDER {provider!r}")


def tracing_callbacks() -> list:
    """Day 6 · Langfuse callback handler, if keys are configured. Every node/LLM/tool call becomes a span."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def flush_tracing() -> None:
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        from langfuse import get_client

        get_client().flush()
