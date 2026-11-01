"""
core/llm.py - the connection to the local language model.

BACKGROUND: WHAT IS OLLAMA?
---------------------------
Ollama is a program that runs on your machine, downloads open-source language
models (Llama, Qwen, Mistral, ...), and exposes them over a small local web
API at http://localhost:11434. Nothing leaves your computer and there is no
API key.

LangChain's `ChatOllama` class is a thin wrapper around that local web API. It
gives us a uniform object with an `.invoke(messages)` method, the same shape
LangChain uses for every other model provider.

WHY WRAP IT IN THIS FILE?
-------------------------
Both agents need a model. If each agent constructed its own `ChatOllama` with
its own hard-coded settings, changing the model name would mean editing every
agent. Instead every agent calls `build_chat_model()` from here.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from langchain_ollama import ChatOllama

from core.config import get_settings


# ---------------------------------------------------------------------------
# 1. Building the chat model
# ---------------------------------------------------------------------------

def build_chat_model(
    *,
    model: str | None = None,
    temperature: float | None = None,
    **extra: Any,
) -> ChatOllama:
    """
    Create a `ChatOllama` object pointed at the local Ollama server.

    All arguments are optional - with no arguments you get the model and
    temperature from your `.env` file, which is what both agents normally use.

    Args:
        model:
            Override the model name for this one call. Useful if you ever want
            one agent on a small fast model and another on a bigger one.
        temperature:
            Override the creativity dial for this one call. The Quiz Maker
            uses a higher value than the Tutor, because inventing varied quiz
            questions benefits from randomness while grading does not.
        **extra:
            Any other keyword argument is passed straight through to
            `ChatOllama` (for example `num_predict=512` to cap output length).

    Returns:
        A ready-to-use `ChatOllama`. Call `.invoke([...])` on it with a list of
        LangChain messages to get a reply.

    Note:
        Creating this object does NOT contact Ollama. The first network call
        happens when you `.invoke()` it. So a bad model name shows up as an
        error at the first message, not here - which is exactly why the
        `scripts/check_ollama.py` script in step 1 exists.
    """
    settings = get_settings()

    # The `if x is None` checks (rather than `if not x`) matter: a caller
    # passing `temperature=0.0` means "exactly zero", and `not 0.0` is True,
    # which would wrongly fall back to the default.
    return ChatOllama(
        model=settings.ollama_model if model is None else model,
        base_url=settings.ollama_base_url,
        temperature=(
            settings.ollama_temperature if temperature is None else temperature
        ),
        # `client_kwargs` reaches the underlying HTTP client. A generous
        # timeout matters because a local model on a CPU-only laptop can take
        # a minute or more to answer.
        client_kwargs={"timeout": settings.ollama_timeout},
        **extra,
    )


# ---------------------------------------------------------------------------
# 2. Health checks
#
# These talk to Ollama's HTTP API directly with Python's built-in `urllib`,
# instead of going through LangChain. That is on purpose: when something is
# broken we want to know whether the problem is Ollama itself or our LangChain
# code, and these functions test only the former.
# ---------------------------------------------------------------------------

class OllamaNotReachable(RuntimeError):
    """Raised when the Ollama server does not answer at all."""


def list_installed_models(timeout: float = 10.0) -> list[str]:
    """
    Ask Ollama which models are downloaded on this machine.

    Calls `GET /api/tags`, which returns JSON shaped like:

        {"models": [{"name": "llama3.1:8b", "size": 4920753328, ...}, ...]}

    Returns:
        A list of model names, e.g. `["llama3.1:8b", "qwen2.5:7b"]`.
        An empty list means Ollama is running but you have not pulled
        any model yet.

    Raises:
        OllamaNotReachable: if the server is not running or not responding.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        # URLError covers "connection refused" (server not started), DNS
        # failures and timeouts. We re-raise as our own exception type with a
        # message that tells the user what to actually do about it.
        raise OllamaNotReachable(
            f"Could not reach Ollama at {settings.ollama_base_url}.\n"
            f"Underlying error: {error}\n\n"
            "Things to try:\n"
            "  1. Is it installed?   ollama --version\n"
            "  2. Is it running?     systemctl status ollama\n"
            "  3. Start it manually: ollama serve\n"
        ) from error

    # `.get("models") or []` handles both a missing key and an explicit null.
    #
    # Each entry has a "model" key on current Ollama versions and a "name" key
    # on older ones (they hold the same value). We accept either so this works
    # across versions, and skip any entry that somehow has neither.
    models = []
    for entry in payload.get("models") or []:
        name = entry.get("model") or entry.get("name")
        if name:
            models.append(name)
    return models


def check_model_available(model: str | None = None) -> tuple[bool, list[str]]:
    """
    Check whether the configured model has actually been pulled.

    Args:
        model: model name to look for; defaults to the configured one.

    Returns:
        A `(found, installed_models)` pair. We return the full list too so the
        caller can print a helpful "did you mean?" message.
    """
    settings = get_settings()
    wanted = settings.ollama_model if model is None else model

    installed = list_installed_models()

    # Ollama model names carry a tag after a colon, like "llama3.1:8b". If the
    # user configured a bare name ("llama3.1") Ollama treats it as
    # "llama3.1:latest", so we accept either spelling.
    candidates = {wanted, wanted if ":" in wanted else f"{wanted}:latest"}
    found = any(name in candidates for name in installed)

    return found, installed
