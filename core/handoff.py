"""
core/handoff.py - how one agent hands work to another.

BACKGROUND: WHAT IS A HANDOFF?
------------------------------
The router (core/router.py) picks the FIRST agent for each user message. But
sometimes that agent needs another agent's help partway through. Passing
control from one agent to another is a HANDOFF. We support two kinds:

1. CALL AND RETURN  ("pause, call another agent, get its output, continue")

       You:   make me a quiz on gravity
       router -> tutor
       tutor:       "HANDOFF: quiz_maker"     <- the Tutor pauses its work...
       quiz_maker:  writes the quiz           <- ...the Quiz Maker does the job...
       tutor:       presents the quiz         <- ...and control RETURNS to the Tutor

   Like calling a function: the caller is remembered on a RETURN STACK
   (`state["return_stack"]`), and when the called agent finishes, the
   graph goes back to whoever is on top of the stack.

2. TRANSFER  ("an agent finishes and triggers another")

       You:   make it harder
       router -> quiz_maker
       quiz_maker:  writes a harder quiz, then triggers the tutor
       tutor:       presents the quiz

   Like a relay race: nobody is waiting to get control back, the first
   agent just names who runs next.

HOW AN AGENT ASKS FOR A HANDOFF
-------------------------------
An agent's node returns its normal state changes plus one extra key:

    {"messages": [...],
     "handoff": {"to": "quiz_maker", "mode": "call",        (or "transfer")
                 "task": "3-question quiz on gravity"}}     (optional)

For agents with `can_hand_off=True` (the Tutor), `run_agent()` in
agents/base.py produces that key when the model replies

    HANDOFF: quiz_maker | 3-question quiz on gravity

- the same "marker line" idea as `CLARIFY:` in step 5. The Quiz Maker sets
it directly in code.

PASSING THE TASK ALONG
----------------------
The part after "|" is the TASK: a short job description for the receiving
agent. The wrapper stores it in `state["task"]`, and the receiving agent
reads it from there. All agents share the conversation history, so the
information is usually there anyway - but a small model can miss it in a
long history. Handing over an explicit task, like a colleague's note, is far
more reliable than hoping the next agent finds the request on its own.

WHO DOES THE BOOKKEEPING?
-------------------------
`agent_node()` below wraps every agent's node function. The wrapper:

    * clears `next_agent` when the agent it names starts running
    * pops the return stack when control comes back to the caller
    * turns a "handoff" request into `next_agent` (+ a stack push for "call")
      and stores its task in `task`
    * enforces a limit, so two agents cannot hand work back and forth forever

Then `route_after_agent()` reads the result to choose the next node. Agent
files never touch the stack or `next_agent` themselves, and neither file
here knows any agent's name - so a new agent can take part in handoffs with
no changes to this file.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Iterable

import mlflow
from langgraph.graph import END
from mlflow.entities import SpanType

from core.clarify import ASK_USER
from core.logging_setup import get_logger
from core.state import ChatState

if TYPE_CHECKING:
    # Only for type hints. A real import would be circular: agents/base.py
    # imports this file.
    from agents.base import AgentSpec

log = get_logger("handoff")

# Safety limit: the most handoffs allowed while answering one user message.
# The router resets the count for every new message.
MAX_HANDOFFS_PER_TURN = 3


# ---------------------------------------------------------------------------
# 1. Spotting "HANDOFF: <agent>" in a model reply
# ---------------------------------------------------------------------------

# Same tolerant style as the CLARIFY pattern in core/clarify.py: allows
# Markdown decoration and any capitalisation.
#   HANDOFF\s*:            the marker
#   (?P<agent>...)         the agent's name (may contain spaces: "Quiz Maker")
#   (?:\|\s*(?P<task>.+?))?  optionally "| <task>"
_HANDOFF_LINE = re.compile(
    r"^[\s*_>#-]*HANDOFF\s*:\s*\**\s*(?P<agent>[A-Za-z][\w -]*?)\s*\**\s*"
    r"(?:\|\s*(?P<task>.+?))?\s*[.!]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def parse_handoff(
    reply_text: str, allowed: Iterable[str]
) -> tuple[str | None, str | None, str]:
    """
    Look for a `HANDOFF: <agent> | <task>` line in a reply.

    Args:
        reply_text: the model's reply.
        allowed: agent names this agent may hand off to.

    Returns:
        (agent name or None, task or None, the reply with the HANDOFF line removed)

    Examples (allowed = ["quiz_maker"]):
        "HANDOFF: quiz_maker | quiz on gravity"  -> ("quiz_maker", "quiz on gravity", "")
        "HANDOFF: quiz_maker"                    -> ("quiz_maker", None, "")
        "Sure!\\nHANDOFF: Quiz Maker"             -> ("quiz_maker", None, "Sure!")
        "HANDOFF: wizard"                        -> (None, None, "")   unknown agent
        "Photosynthesis is..."                   -> (None, None, "Photosynthesis is...")
    """
    match = _HANDOFF_LINE.search(reply_text or "")
    if match is None:
        return None, None, reply_text

    # "Quiz Maker", "quiz-maker" -> "quiz_maker"
    name = re.sub(r"[\s-]+", "_", match.group("agent").strip().lower())
    task = (match.group("task") or "").strip() or None
    rest = (reply_text[:match.start()] + reply_text[match.end():]).strip()

    if name not in set(allowed):
        log.warning("model asked to hand off to unknown agent %r; ignoring", name)
        return None, None, rest
    return name, task, rest


# ---------------------------------------------------------------------------
# 2. The wrapper around every agent node
# ---------------------------------------------------------------------------

def agent_node(spec: "AgentSpec") -> Callable[[ChatState], dict]:
    """
    Wrap an agent's node function with the handoff bookkeeping.

    core/graph.py registers `agent_node(spec)` instead of `spec.node`, for
    every agent in the registry.
    """
    me = spec.name

    def node(state: ChatState) -> dict:
        changes: dict = {}
        stack = list(state.get("return_stack") or [])

        # --- Arriving --------------------------------------------------------
        # Handed to us by name? That request is now fulfilled.
        if state.get("next_agent") == me:
            changes["next_agent"] = None

        # Are we the caller on top of the stack? Then control is RETURNING to
        # us after a "call" handoff: pop ourselves off.
        if stack and stack[-1] == me:
            stack.pop()
            changes["return_stack"] = stack
            log.info("control returns to %s", me)

        # --- The agent's real work ------------------------------------------
        update = dict(spec.node(state))
        request = update.pop("handoff", None)   # not a state key; handled here

        # --- Leaving: did the agent ask to hand off? ------------------------
        if request:
            count = (state.get("handoffs_this_turn") or 0) + 1
            if count > MAX_HANDOFFS_PER_TURN:
                log.warning(
                    "%s asked to hand off to %s, but %d handoffs already happened "
                    "this turn; stopping here", me, request["to"], MAX_HANDOFFS_PER_TURN,
                )
            else:
                if request["mode"] == "call":
                    stack.append(me)              # remember who to come back to
                    changes["return_stack"] = stack
                changes["next_agent"] = request["to"]
                changes["handoffs_this_turn"] = count
                # The job description for the receiving agent (may be None).
                changes["task"] = request.get("task")

                log.info(
                    "%s -> %s (%s)%s", me, request["to"], request["mode"],
                    f", task: {request['task']!r}" if request.get("task") else "",
                )
                # A tiny span so the handoff is visible as its own box in MLflow.
                with mlflow.start_span(
                    name=f"handoff:{me}->{request['to']}", span_type=SpanType.CHAIN
                ) as span:
                    span.set_attributes({
                        "from_agent": me,
                        "to_agent": request["to"],
                        "mode": request["mode"],
                        "return_stack": stack,
                    })
                    span.set_inputs({"task": request.get("task")})

        # The agent's own changes win if both set the same key.
        return {**changes, **update}

    # A readable name in error messages and traces.
    node.__name__ = f"{me}_node"
    return node


# ---------------------------------------------------------------------------
# 3. Routing after an agent: where does the graph go next?
# ---------------------------------------------------------------------------

def route_after_agent(state: ChatState) -> str:
    """
    The conditional edge after every agent node. Checked in this order:

        1. a clarifying question is waiting   -> ask_user (pause; step 5)
        2. a handoff named the next agent     -> that agent
        3. someone is waiting on the stack    -> back to them (call-and-return)
        4. otherwise                          -> END (the reply is done)
    """
    if state.get("pending_question"):
        return ASK_USER
    if state.get("next_agent"):
        return state["next_agent"]
    stack = state.get("return_stack") or []
    if stack:
        return stack[-1]
    return END
