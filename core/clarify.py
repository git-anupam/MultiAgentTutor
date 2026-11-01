"""
core/clarify.py - the CLARIFYING QUESTION flow: pause, ask the user, resume.

THE PROBLEM
-----------
Sometimes an agent cannot do its job yet: "explain it" (explain WHAT?),
"make me a quiz" (on WHAT topic?). Guessing gives a bad answer. A better
agent asks a follow-up question, waits for the reply, and then carries on.

THE FLOW, STEP BY STEP
----------------------
    You:   "explain it"

    1. AGENT NODE (e.g. tutor)
       The model sees its prompt, which says: if the request is too vague,
       reply with exactly `CLARIFY: <question>`. It replies
           "CLARIFY: Which topic would you like me to explain?"
       `parse_clarification()` spots the marker, so instead of a normal reply
       the agent stores the question in the state:
           pending_question = {"agent": "tutor", "question": "Which topic...?"}

    2. ROUTING: `route_after_agent()` (core/handoff.py, the edge after every
       agent) sees `pending_question` is set and sends the graph to the
       ask_user node instead of END.

    3. ASK_USER NODE calls LangGraph's `interrupt(...)`.
       interrupt() STOPS the graph right there. The checkpointer saves
       everything to Postgres, `graph.invoke` returns, and the chat prints the
       question. The graph is now PAUSED - and stays paused even if you quit
       the program and come back tomorrow.

    You:   "photosynthesis"

    4. RESUME. Because the thread is paused, the chat does not send this as a
       new message. It sends `Command(resume="photosynthesis")` instead (see
       `graph_input()`). LangGraph loads the paused checkpoint and re-enters
       the ask_user node; this time `interrupt(...)` RETURNS the answer.

    5. The ask_user node adds your answer to the conversation and jumps back
       to the agent that asked (`Command(goto="tutor")`).

    6. AGENT NODE again. It now sees: "explain it" / "Which topic?" /
       "photosynthesis", and answers properly. The run carries on as normal
       (to END, or to another agent if a handoff is in progress - e.g. the
       Quiz Maker asked for a topic while working for the Tutor).

    NOTE: the user's answer does NOT go through the router (core/router.py).
    The paused run already knows which agent asked.

TWO WAYS TO DECIDE TO ASK
-------------------------
Step 1 above is the Tutor's way: the MODEL decides, by replying with the
"CLARIFY:" marker. The Quiz Maker decides in CODE instead: a small model
call extracts the topic, and `if not topic:` it builds the same CLARIFY
reply itself (agents/quiz_maker.py explains why). From step 2 on, both use
exactly the same pause-and-resume machinery below.

WHY IS ASK_USER A SEPARATE NODE?
--------------------------------
A key fact about `interrupt()`: when the graph resumes, the node that paused
RUNS AGAIN FROM ITS FIRST LINE, and this time interrupt() returns the
answer instead of stopping.

If the agent called the model and then interrupt() in the same node, resuming
would call the model a second time. The model might phrase things
differently, or not ask at all, and your answer would be lost. Keeping the
model call (agent node) apart from the pause (ask_user node) makes the
re-run harmless: ask_user only reads the question from the state and pauses.

This file works for ANY agent. A new agent only needs the CLARIFY rule in
its prompt file; nothing here, and nothing in the graph wiring, changes.
"""

from __future__ import annotations

import re

import mlflow
from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt
from mlflow.entities import SpanType

from core.logging_setup import get_logger
from core.state import ChatState

log = get_logger("clarify")

# The name of the pausing node in the graph.
ASK_USER = "ask_user"


# ---------------------------------------------------------------------------
# 1. Spotting a clarifying question in the model's reply
# ---------------------------------------------------------------------------

# Matches a line like "CLARIFY: Which topic?" and captures the question.
# Small local models are not perfectly obedient, so the pattern also accepts
# decorations they sometimes add: "**CLARIFY:** ...", "- Clarify: ...",
# lowercase, or a few words of chatter on the lines before it.
#   ^[\s*_>#-]*     optional Markdown junk at the start of the line
#   CLARIFY\s*:     the marker itself
#   \s*\**\s*       optional bold-closing asterisks
#   (?P<question>.+)$   the rest of the line = the question
_CLARIFY_LINE = re.compile(
    r"^[\s*_>#-]*CLARIFY\s*:\s*\**\s*(?P<question>.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_clarification(reply_text: str) -> str | None:
    """
    If the model's reply is a clarifying question, return just the question.

    Examples:
        "CLARIFY: Which topic?"        -> "Which topic?"
        "**CLARIFY:** Which topic?**"  -> "Which topic?"
        "Photosynthesis is how..."     -> None   (a normal answer)
    """
    match = _CLARIFY_LINE.search(reply_text or "")
    if match is None:
        return None
    question = match.group("question").strip().strip("*").strip()
    return question or None


# ---------------------------------------------------------------------------
# 2. The node that pauses
#
# (The decision to come here - "is a question pending?" - is made by
# `route_after_agent()` in core/handoff.py, together with the handoff
# decisions, so all "where next?" logic is in one function.)
# ---------------------------------------------------------------------------

def ask_user_node(state: ChatState) -> Command:
    """
    Pause the graph until the user answers the pending question.

    Remember: this function runs TWICE for every question (see "WHY IS
    ASK_USER A SEPARATE NODE?" above):
        run 1 - interrupt() stops the graph; nothing after it runs
        run 2 - after the user answers; interrupt() returns the answer
    That is why you will see the "waiting" log line twice.
    """
    pending = state["pending_question"]
    log.info("waiting for the user to answer %s's question", pending["agent"])

    # THE PAUSE. Whatever we pass in (`pending`) is handed to the caller of
    # graph.invoke, so the chat knows what to show the user.
    answer = interrupt(pending)

    # Only reached on run 2, after `Command(resume=answer)`.
    log.info("resumed: user answered %r, returning to %s", answer, pending["agent"])

    # A small span so the trace shows the question/answer pair clearly.
    with mlflow.start_span(name="clarification", span_type=SpanType.UNKNOWN) as span:
        span.set_attributes({"asked_by": pending["agent"]})
        span.set_inputs({"question": pending["question"]})
        span.set_outputs({"answer": answer})

    # `Command` lets a node update the state AND choose the next node in one
    # go. We go back to whichever agent asked, so this works for every agent
    # without listing them here.
    return Command(
        goto=pending["agent"],
        update={
            "messages": [HumanMessage(content=answer)],
            "pending_question": None,
        },
    )


# ---------------------------------------------------------------------------
# 3. Helpers for the chat program
# ---------------------------------------------------------------------------

def pending_question(graph, config) -> dict | None:
    """
    If this thread is paused on a question, return it; otherwise None.

    Asks the checkpointer for the latest snapshot and looks for an interrupt.
    The returned dict is whatever ask_user_node passed to `interrupt(...)`:
        {"agent": "tutor", "question": "Which topic...?"}
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def graph_input(graph, config, user_text: str):
    """
    Turn what the user typed into the right input for `graph.invoke`.

        thread is paused  -> Command(resume=user_text)   (answers the question)
        otherwise         -> {"messages": [HumanMessage(user_text)]}   (new turn)
    """
    if pending_question(graph, config):
        return Command(resume=user_text)
    return {"messages": [HumanMessage(content=user_text)]}
