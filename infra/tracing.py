"""
infra/tracing.py - records a detailed MLflow trace of every message.

BACKGROUND: WHAT IS A TRACE?
----------------------------
The terminal log (core/logging_setup.py) prints one line per step while you
chat. A TRACE is the detailed, permanent version: for each message you send,
MLflow saves a tree of SPANS, one span per piece of work, each with its
inputs, outputs, timing and labels:

    chat_turn                   you typed "what is gravity?"
    └── LangGraph               the graph run
        ├── router              the router node
        │   └── route           which agent it chose, and why
        │       └── ChatOllama
        └── tutor               the tutor node
            └── agent:tutor     the Tutor's model call         agent=tutor
                └── ChatOllama  exactly what went to/from Ollama, token counts

A quiz request has more branches: the Quiz Maker finding the topic, one
`ChatOllama` span per round of the tool-calling loop with a tool span in
between, and the handoff back to the Tutor:

    quiz_maker
    ├── quiz_maker:find_topic      the topic it found, e.g. "gravity"
    ├── agent:quiz_maker
    │   ├── ChatOllama             the model asks for get_topic_facts
    │   ├── tool:get_topic_facts   our code runs it   called_by=quiz_maker
    │   └── ChatOllama             the model writes the quiz
    └── handoff:quiz_maker->tutor
    tutor                          presents the quiz (no model call)

You open the traces in a web page, `mlflow ui`, and click through them.

WHERE DO THE SPANS COME FROM?
-----------------------------
Two sources, which nest together into one tree:

1. AUTOMATIC: `mlflow.langchain.autolog()` (below) hooks into LangChain and
   LangGraph. Without any extra code it records the graph run, every node,
   every model call and every tool call. That is where the `LangGraph`,
   `tutor` and `ChatOllama` spans above come from.

2. MANUAL: a few spans we create ourselves with `mlflow.start_span(...)`,
   where we want a clearer name or extra labels:
       chat_turn      - one per user message; labelled with the thread id
                        (`chat_turn_span()` below)
       agent:<name>   - one per agent turn; labelled with WHICH agent ran
                        (agents/base.py: call_model)
       tool:<name>    - one per tool call; labelled with which agent called
                        it (agents/base.py: run_tool)
       clarification  - a clarifying question and the user's answer
                        (core/clarify.py)
       route          - the router's choice and reason (core/router.py)
       handoff:<a>-><b> - one agent handing work to another (core/handoff.py)
       quiz_maker:find_topic, tutor:grade_quiz - steps inside those agents

WHERE ARE TRACES STORED?
------------------------
In a single SQLite database file, `mlflow.db`, in the project folder. SQLite
is a database that lives in one file - no server to run. Recording needs
nothing else; viewing needs the `mlflow ui` command (see the README).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import mlflow
from mlflow.entities import SpanType

from core.config import get_settings
from core.logging_setup import get_logger

log = get_logger("trace")


def setup_tracing() -> bool:
    """
    Point MLflow at our trace store and switch on automatic tracing.

    Call once, near the start of a script, before building the graph.

    Returns:
        True if tracing is on, False if MLFLOW_ENABLED=false in `.env`.
    """
    settings = get_settings()

    if not settings.mlflow_enabled:
        # Makes every `mlflow.start_span(...)` in the code a harmless no-op,
        # so the rest of the program runs unchanged with tracing off.
        mlflow.tracing.disable()
        log.info("tracing disabled (MLFLOW_ENABLED=false)")
        return False

    # MLflow prints its own INFO messages (database setup, "Flushing the
    # async trace logging queue..." at exit). Useful for MLflow developers,
    # noise in a chat window - show only its warnings and errors.
    logging.getLogger("mlflow").setLevel(logging.WARNING)

    # WHERE to store traces (our SQLite file).
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    # WHICH experiment to file them under. An "experiment" is just a named
    # folder in the MLflow UI; it is created on first use.
    mlflow.set_experiment(settings.mlflow_experiment)

    # Turn on automatic spans for LangChain and LangGraph (see the module
    # docstring, "AUTOMATIC").
    mlflow.langchain.autolog()

    log.info(
        "tracing to %s (experiment %r)",
        settings.mlflow_tracking_uri, settings.mlflow_experiment,
    )
    return True


@contextmanager
def chat_turn_span(
    thread_id: str, user_text: str, answers_question: bool = False
) -> Iterator[mlflow.entities.Span]:
    """
    Wrap one user message in a top-level `chat_turn` span.

    Everything that happens inside the `with` block - the graph run, the
    agents, the model calls - becomes a child of this span, so each message
    you send shows up in the UI as ONE trace.

    Args:
        thread_id: the conversation this message belongs to.
        user_text: what the user typed.
        answers_question: True when the message is the answer to an agent's
            clarifying question (step 5), i.e. it RESUMES a paused run rather
            than starting a new one. Recorded as an attribute so you can tell
            the two kinds of trace apart in the UI.

    Usage:
        with chat_turn_span(thread_id, user_text) as span:
            result = graph.invoke(...)
            span.set_outputs({"reply": ...})
    """
    with mlflow.start_span(name="chat_turn", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"user_message": user_text})
        span.set_attributes({
            "thread_id": thread_id,
            "answers_clarifying_question": answers_question,
        })

        # ONE TRACE = ONE MESSAGE, NOT ONE CONVERSATION. A conversation is
        # several traces that share a thread id. We record that link twice:
        #
        #   tags      a label on the whole trace; filter the Traces tab with
        #             tags.thread_id = 'chat-20261101-143210'
        #   metadata  "mlflow.trace.session" is MLflow's own field for
        #             "which conversation does this trace belong to"; newer
        #             MLflow UIs use it to group a conversation's traces
        mlflow.update_current_trace(
            tags={"thread_id": thread_id},
            metadata={"mlflow.trace.session": thread_id},
        )

        yield span

    # Only reached if the block finished without an error. (If it failed,
    # MLflow still saves the trace, marked with status ERROR.)
    log.info("recorded trace %s", span.trace_id)
