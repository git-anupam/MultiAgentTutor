"""
core/config.py - one place where all settings live.

WHY THIS FILE EXISTS
--------------------
Hard-coding things like the model name in the middle of the code is painful:
when you want to try a different model you have to hunt through every file.
Instead, every tunable value is read here, once, from environment variables
(which `python-dotenv` loads out of the `.env` file for us).

HOW TO USE IT
-------------
    from core.config import get_settings

    settings = get_settings()
    print(settings.ollama_model)

`get_settings()` is cached, so calling it from twenty different files still
only reads the environment once and always returns the same object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Find the project root (the folder that contains this `core/` folder) so that
# the `.env` file is found no matter which directory you run Python from.
#
#   __file__                        -> .../MultAgentChatbot/core/config.py
#   Path(__file__).resolve()        -> absolute version of that path
#   .parent                         -> .../MultAgentChatbot/core
#   .parent.parent                  -> .../MultAgentChatbot      <- what we want
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Read `.env` into the process environment.
#   override=False  -> a variable you already exported in your shell wins over
#                      the one in the file. That lets you do a quick one-off
#                      experiment like:
#                          OLLAMA_MODEL=qwen2.5:7b python chat.py
load_dotenv(PROJECT_ROOT / ".env", override=False)


# ---------------------------------------------------------------------------
# Small helper functions.
#
# Environment variables are ALWAYS strings (or missing). These helpers convert
# them into the Python types we actually want, with a fallback default.
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str) -> str:
    """Read a text setting. Blank or missing both fall back to `default`."""
    value = os.getenv(name)
    # `value.strip()` guards against a line like `OLLAMA_MODEL=   ` in .env.
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_float(name: str, default: float) -> float:
    """Read a number setting, ignoring junk instead of crashing."""
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError:
        # A typo in .env should not stop the whole program, so we warn and
        # carry on with the default.
        print(f"[config] WARNING: {name}={raw!r} is not a number; using {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Read a true/false setting. Accepts true/yes/1/on in any capitalisation."""
    raw = _env_str(name, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# The settings object.
#
# A `dataclass` is just a class whose boilerplate (__init__, __repr__, ...) is
# written for you. `frozen=True` makes instances read-only, which means no
# other part of the program can accidentally change a setting at runtime.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Every configurable value in the project, in one immutable object."""

    # --- Ollama (the local language model) ---------------------------------
    ollama_model: str          # e.g. "llama3.1:8b"
    ollama_base_url: str       # e.g. "http://localhost:11434"
    ollama_temperature: float  # 0.0 = predictable, 1.0 = creative
    ollama_timeout: float      # seconds to wait before giving up

    # --- PostgreSQL (conversation checkpoints, step 3) --------------------
    postgres_uri: str

    # --- MLflow (tracing, step 4) -----------------------------------------
    mlflow_tracking_uri: str
    mlflow_experiment: str
    mlflow_enabled: bool

    # --- Terminal logging (see core/logging_setup.py) ---------------------
    log_level: str             # "DEBUG", "INFO" or "WARNING"

    # --- Paths -------------------------------------------------------------
    # Where the Markdown system prompts live. Stored here so that the prompt
    # loader never has to guess at paths.
    project_root: Path
    prompts_dir: Path

    def describe(self) -> str:
        """
        A short, human-readable summary printed at startup.

        Deliberately does not print the Postgres password, so you can paste
        this output into a bug report safely.
        """
        safe_pg = self.postgres_uri
        if "@" in safe_pg:
            # "postgresql://user:password@host/db" -> "postgresql://***@host/db"
            scheme, rest = safe_pg.split("://", 1)
            safe_pg = f"{scheme}://***@{rest.split('@', 1)[1]}"

        return (
            f"  model        : {self.ollama_model}\n"
            f"  ollama url   : {self.ollama_base_url}\n"
            f"  temperature  : {self.ollama_temperature}\n"
            f"  postgres     : {safe_pg}\n"
            f"  mlflow       : {self.mlflow_tracking_uri} "
            f"(experiment={self.mlflow_experiment}, enabled={self.mlflow_enabled})\n"
            f"  log level    : {self.log_level}"
        )


# ---------------------------------------------------------------------------
# The single entry point.
#
# `@lru_cache` remembers the result of the first call and returns that same
# object forever after. This is the standard Python way to make a "singleton"
# without writing a singleton class.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build the Settings object from environment variables (cached)."""
    return Settings(
        # --- Ollama --------------------------------------------------------
        # qwen2.5:7b proved more reliable than llama3.1:8b at tool calls and
        # JSON during testing (llama3.1 often wrote tool calls as plain text).
        ollama_model=_env_str("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_temperature=_env_float("OLLAMA_TEMPERATURE", 0.2),
        ollama_timeout=_env_float("OLLAMA_TIMEOUT", 180.0),

        # --- Postgres ------------------------------------------------------
        # Port 5433 on purpose: see the comment in docker-compose.yml.
        postgres_uri=_env_str(
            "POSTGRES_URI",
            "postgresql://studyhelper:studyhelper@localhost:5433/studyhelper",
        ),

        # --- MLflow --------------------------------------------------------
        # A SQLite file in the project folder. The path is absolute so the
        # same file is used no matter which folder you run a script from.
        # ("sqlite:///" + "/home/..." gives the four slashes SQLite wants for
        # an absolute path.)
        mlflow_tracking_uri=_env_str(
            "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
        ),
        mlflow_experiment=_env_str("MLFLOW_EXPERIMENT", "study-helper"),
        mlflow_enabled=_env_bool("MLFLOW_ENABLED", True),

        # --- Logging -------------------------------------------------------
        log_level=_env_str("LOG_LEVEL", "INFO"),

        # --- Paths ---------------------------------------------------------
        project_root=PROJECT_ROOT,
        prompts_dir=PROJECT_ROOT / "agents" / "prompts",
    )
