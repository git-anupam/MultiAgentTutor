"""
tools/topic_facts.py - the `get_topic_facts` tool used by the Quiz Maker.

WHAT IT DOES
------------
Looks a topic up in `tools/data/topic_facts.json`, a small set of local study
notes, and returns the facts as text. The Quiz Maker bases its questions on
those facts, so the quiz is grounded in known-good material instead of
whatever the model happens to remember.

Using a local JSON file (rather than, say, a web search) keeps the project
offline and free, and makes the tool's output predictable - which is handy
when you are learning what a tool call looks like. To add a topic, edit the
JSON file; no Python changes needed.

WHAT THE MODEL SEES
-------------------
The model never sees this Python code. It sees only the tool's SCHEMA,
which LangChain builds from three things below:

    name         "get_topic_facts"            (first argument to @tool)
    description  the function's docstring
    arguments    the `TopicFactsInput` class: one field, `topic`, a string,
                 with its own description

So the docstring and the field description are not just comments - they are
instructions to the model about when and how to use the tool. Write them
clearly.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from core.logging_setup import get_logger

log = get_logger("tool")

_FACTS_FILE = Path(__file__).resolve().parent / "data" / "topic_facts.json"


# ---------------------------------------------------------------------------
# The tool's argument schema.
#
# A pydantic `BaseModel` describes the arguments: their names, types and
# descriptions. LangChain turns this into the JSON schema sent to the model:
#
#   {"type": "object",
#    "properties": {"topic": {"type": "string",
#                             "description": "The subject to look up, ..."}},
#    "required": ["topic"]}
#
# It also VALIDATES what the model sends: if the model forgets `topic`, the
# call fails with a clear error instead of running with garbage.
# ---------------------------------------------------------------------------

class TopicFactsInput(BaseModel):
    """Arguments for get_topic_facts."""

    topic: str = Field(
        description=(
            "The subject to look up, in a few words, "
            "e.g. 'photosynthesis', 'gravity', 'the water cycle'."
        )
    )


# ---------------------------------------------------------------------------
# Loading the notes file (once, then cached - same idea as core/prompts.py).
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_facts() -> dict:
    """Read topic_facts.json. Shape: {topic: {"aliases": [...], "facts": [...]}}."""
    return json.loads(_FACTS_FILE.read_text(encoding="utf-8"))


def _find_topic(requested: str) -> str | None:
    """
    Match what the model asked for to a topic in the notes file.

    Tries, in order:
      1. exact name or alias          "gravity", "gravitation"
      2. one contains the other       "photosynthesis in plants" -> "photosynthesis"
    Everything is compared in lowercase, so "Gravity" works too.
    """
    wanted = requested.strip().lower()
    if not wanted:
        return None  # an empty string is "contained in" everything; skip it
    facts = _load_facts()

    for name, entry in facts.items():
        if wanted == name or wanted in entry.get("aliases", []):
            return name

    for name, entry in facts.items():
        for candidate in [name, *entry.get("aliases", [])]:
            if candidate in wanted or wanted in candidate:
                return name

    return None


# ---------------------------------------------------------------------------
# The tool itself.
#
# `@tool("get_topic_facts", args_schema=TopicFactsInput)` wraps the function
# in a LangChain `StructuredTool` object. That object has:
#     .name, .description, .args   - what gets sent to the model
#     .invoke({"topic": "..."})    - how our code runs it
# ---------------------------------------------------------------------------

@tool("get_topic_facts", args_schema=TopicFactsInput)
def get_topic_facts(topic: str) -> str:
    """Look up trusted study notes about a school topic.

    Call this BEFORE writing quiz questions, and base the questions on the
    facts it returns. Returns a bullet list of facts, or a message saying no
    notes exist for that topic.
    """
    match = _find_topic(topic)

    if match is None:
        known = ", ".join(sorted(_load_facts()))
        log.info("get_topic_facts(%r): no notes found", topic)
        # Returned as normal text, not raised as an error: the model reads
        # this and can still carry on sensibly.
        return (
            f"No stored notes for '{topic}'. Topics with notes: {known}. "
            "Write the quiz from your own general knowledge, and keep the "
            "questions basic and well-known."
        )

    facts = _load_facts()[match]["facts"]
    log.info("get_topic_facts(%r): matched %r, returning %d facts", topic, match, len(facts))
    return f"Notes on {match}:\n" + "\n".join(f"- {fact}" for fact in facts)
