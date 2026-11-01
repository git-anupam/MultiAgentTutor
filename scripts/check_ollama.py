"""
scripts/check_ollama.py - the step 1 smoke test.

Run this BEFORE anything else:

    python scripts/check_ollama.py

It answers four questions, in order, and stops at the first failure:

    1. Is the Ollama server running and reachable?
    2. Is the model named in `.env` actually downloaded?
    3. Can we send a message through LangChain and get a reply?
    4. Does this model support TOOL CALLING?

Question 4 is the one beginners get caught by. Agent 2 (the Quiz Maker) has a
tool, and tool calling only works on models that were trained for it. If
check 4 fails, the whole project still runs, but step 6 will not - so it is
much better to find out now than three steps later.

This file is deliberately standalone: it imports only `core.config` and
`core.llm`, so nothing about agents, Postgres or MLflow can break it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make `import core.config` work when this file is run as
# `python scripts/check_ollama.py`.
#
# Python puts the *script's own folder* (`scripts/`) on the import path, not
# the project root, so `core` would not be found. Inserting the project root
# at position 0 fixes that. Every script under `scripts/` starts this way.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.llm import (  # noqa: E402
    OllamaNotReachable,
    build_chat_model,
    check_model_available,
)


# ---------------------------------------------------------------------------
# Tiny printing helpers, so the output is easy to read in a terminal.
# ---------------------------------------------------------------------------

def heading(text: str) -> None:
    print(f"\n\033[1m{text}\033[0m")       # bold


def ok(text: str) -> None:
    print(f"  \033[32m[PASS]\033[0m {text}")  # green


def fail(text: str) -> None:
    print(f"  \033[31m[FAIL]\033[0m {text}")  # red


def warn(text: str) -> None:
    print(f"  \033[33m[WARN]\033[0m {text}")  # yellow


# ---------------------------------------------------------------------------
# A throwaway tool, used only by check 4.
#
# The `@tool` decorator turns a normal Python function into a LangChain tool:
# it reads the function name, the type hints and the docstring, and builds a
# JSON schema that the model can understand. The real tool, `get_topic_facts`,
# lives in tools/topic_facts.py - this one just needs to exist so we can see
# whether the model is willing to call a tool at all.
# ---------------------------------------------------------------------------

@tool
def add_two_numbers(a: int, b: int) -> int:
    """Add two whole numbers together and return the result."""
    return a + b


# ---------------------------------------------------------------------------
# The four checks. Each returns True on success, False on failure.
# ---------------------------------------------------------------------------

def check_1_server_reachable() -> bool:
    """Can we talk to the Ollama HTTP API at all?"""
    heading("1. Ollama server reachable?")
    try:
        found, installed = check_model_available()
    except OllamaNotReachable as error:
        fail(str(error))
        return False

    ok(f"Ollama answered. {len(installed)} model(s) installed.")
    for name in installed:
        print(f"         - {name}")
    # Stash the result so check 2 does not have to ask again.
    check_1_server_reachable.found = found      # type: ignore[attr-defined]
    check_1_server_reachable.installed = installed  # type: ignore[attr-defined]
    return True


def check_2_model_present() -> bool:
    """Is the model from `.env` one of the installed ones?"""
    settings = get_settings()
    heading(f"2. Configured model '{settings.ollama_model}' downloaded?")

    found = getattr(check_1_server_reachable, "found", False)
    installed = getattr(check_1_server_reachable, "installed", [])

    if found:
        ok(f"'{settings.ollama_model}' is available.")
        return True

    fail(f"'{settings.ollama_model}' is not installed.")
    print(f"\n  Fix it with:\n      ollama pull {settings.ollama_model}\n")
    if installed:
        print("  Or point OLLAMA_MODEL in your .env at one you already have:")
        for name in installed:
            print(f"      OLLAMA_MODEL={name}")
    return False


def check_3_basic_chat() -> bool:
    """Can LangChain send a message and read a reply?"""
    heading("3. Basic chat through LangChain?")
    print("  (first run can take a while - the model is loading into memory)")

    try:
        model = build_chat_model()
        # One message in, one message out. `HumanMessage` is LangChain's way of
        # saying "this text came from the user".
        reply = model.invoke(
            [HumanMessage(content="Reply with exactly the word: pong")]
        )
    except Exception as error:  # noqa: BLE001 - we want to report anything
        fail(f"The call raised {type(error).__name__}: {error}")
        return False

    text = (reply.content or "").strip()
    ok(f"Model replied: {text[:120]!r}")
    return True


def check_4_tool_calling() -> bool:
    """Will the model ask us to run a tool? (Needed by Agent 2 in step 6.)"""
    heading("4. Tool calling supported?")

    try:
        model = build_chat_model(temperature=0.0)

        # `.bind_tools([...])` is the key line. It hands the tool's JSON
        # schema to the model, so the model can answer with "please run
        # add_two_numbers(a=17, b=25)" instead of plain text.
        model_with_tools = model.bind_tools([add_two_numbers])

        reply = model_with_tools.invoke(
            [HumanMessage(content="What is 17 plus 25? Use the tool.")]
        )
    except Exception as error:  # noqa: BLE001
        fail(f"The call raised {type(error).__name__}: {error}")
        print("\n  A 'does not support tools' error here means you need a")
        print("  tool-capable model. Good choices:")
        print("      ollama pull llama3.1:8b")
        print("      ollama pull qwen2.5:7b")
        return False

    # When a model decides to use a tool, LangChain fills in `.tool_calls`:
    # a list of dicts like {"name": "add_two_numbers", "args": {"a": 17, ...}}
    # and the message's text content is usually empty.
    tool_calls = getattr(reply, "tool_calls", None) or []

    if not tool_calls:
        warn("The model answered with plain text instead of calling the tool.")
        print(f"         text was: {(reply.content or '').strip()[:120]!r}")
        print("\n  This is not always fatal - small models sometimes ignore")
        print("  tools on easy questions. But if step 6 misbehaves later,")
        print("  switch to llama3.1:8b or qwen2.5:7b.")
        return False

    call = tool_calls[0]
    ok(f"Model requested tool {call['name']!r} with args {call['args']}")

    # Prove the round trip works by actually running the function.
    result = add_two_numbers.invoke(call["args"])
    ok(f"We ran it locally and got {result} (expected 42)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run all four checks and return a shell exit code (0 = all good)."""
    settings = get_settings()

    print("\n=== Ollama connection check ===")
    print("\nConfiguration currently in effect:")
    print(settings.describe())

    # Checks 1 and 2 are prerequisites - no point continuing without them.
    if not check_1_server_reachable():
        print("\nStopped: fix Ollama, then run this script again.\n")
        return 1
    if not check_2_model_present():
        print("\nStopped: pull the model, then run this script again.\n")
        return 1

    chat_ok = check_3_basic_chat()
    if not chat_ok:
        print("\nStopped: the model is installed but will not answer.\n")
        return 1

    # Check 4 is a warning, not a hard stop: steps 2-5 of the build do not
    # need tools, so you can carry on and fix this before step 6.
    tools_ok = check_4_tool_calling()

    heading("Summary")
    if tools_ok:
        print("  Everything works. You are ready for step 2.\n")
        return 0

    print("  Chat works, tool calling does not.")
    print("  Steps 2-5 will be fine; swap the model before step 6.\n")
    return 0


if __name__ == "__main__":
    # `sys.exit` with the return code makes the script usable in shell
    # pipelines, e.g. `python scripts/check_ollama.py && python chat.py`.
    sys.exit(main())
