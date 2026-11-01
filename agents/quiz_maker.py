"""
agents/quiz_maker.py - Agent 2, the Quiz Maker.

WHAT THE QUIZ MAKER DOES
------------------------
Writes a short multiple-choice quiz on a topic:

    1. FIND THE TOPIC (`find_topic`): a small, separate model call that only
       extracts the topic, e.g. "gravity", as JSON. Then CODE decides:
       no topic -> ask the student a clarifying question (step 5 flow)
    2. calls its tool, `get_topic_facts`, to fetch study notes on the topic
       (the tool-calling loop lives in agents/base.py: call_model)
    3. writes the quiz as JSON, which we check and store in the state

WHY DOES CODE DECIDE WHETHER TO ASK, NOT THE MODEL?
---------------------------------------------------
The Tutor lets its model decide (it replies "CLARIFY: ..." when a request is
vague), and that works well for it. The Quiz Maker used to do the same, but
a 7-8B model with a tool attached kept asking "which topic?" even for "make
me a quiz on gravity" - it copied the example question from its prompt.

Splitting the job fixed it:
    * the model does what small models are good at: pull one fact
      ("gravity") out of some text, in a fixed JSON shape
    * plain Python makes the yes/no decision: `if not topic: ask`
So this project shows both styles of clarifying question: model-decided
(Tutor) and code-decided (Quiz Maker).

The Tutor then shows the questions to the student (without the answers) and
grades the replies. How control gets back to the Tutor depends on how the
Quiz Maker was reached (step 7, see core/handoff.py):

    the Tutor CALLED it ("HANDOFF: quiz_maker")
        -> the Tutor is waiting on the return stack; control returns to it
           automatically when the Quiz Maker finishes

    the ROUTER picked it directly (e.g. "make it harder")
        -> nobody is waiting, so the Quiz Maker TRANSFERS to the Tutor
           itself when it finishes ("finish and trigger another")

WHY STORE THE QUIZ AS STRUCTURED DATA?
--------------------------------------
If the quiz were just a chat message, the answers would be sitting in the
conversation for the student to read. Storing it in `state["quiz"]` as data
- question, choices, correct letter, explanation - lets the Tutor show only
the questions and still know the answers when grading.

HOW TO CHANGE ITS BEHAVIOUR
---------------------------
Edit `agents/prompts/quiz_maker.md` (writing the quiz) or
`agents/prompts/quiz_maker_topic.md` (finding the topic), then restart the
chat.
"""

from __future__ import annotations

import json
import re

import mlflow
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mlflow.entities import SpanType
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from agents.base import AgentSpec, call_model, clarification_update
from core.llm import build_chat_model
from core.logging_setup import get_logger
from core.prompts import load_prompt
from core.state import ChatState
from tools.topic_facts import get_topic_facts

log = get_logger("quiz_maker")

# The agent that presents finished quizzes to the student.
PRESENTER = "tutor"

# What the Quiz Maker asks when no topic can be found.
ASK_FOR_TOPIC = "Which topic would you like the quiz to be about?"


# ---------------------------------------------------------------------------
# 1. What a valid quiz looks like
#
# These pydantic classes describe the JSON we expect from the model. Calling
# `Quiz.model_validate(data)` checks every field (right types, nothing
# missing, answer letter in range) and raises an error if anything is off.
# ---------------------------------------------------------------------------

class QuizQuestion(BaseModel):
    """One multiple-choice question."""

    question: str
    # The choice texts, WITHOUT letters. Position gives the letter:
    # choices[0] is A, choices[1] is B, and so on. At most 4: the prompt asks
    # for exactly 4, and more than that has only ever meant a garbled reply.
    choices: list[str] = Field(min_length=2, max_length=4)
    # The letter of the correct choice, e.g. "B".
    answer: str
    # One sentence on why the answer is right; the Tutor uses it when grading.
    explanation: str = ""

    @field_validator("choices")
    @classmethod
    def _strip_choice_letters(cls, choices: list[str]) -> list[str]:
        """
        Remove letters the model put in front of a choice despite the prompt:
        "A) Oxygen", "B. Oxygen", "(C) Oxygen" -> "Oxygen". The letters are
        added back when the quiz is shown, so they would appear twice.
        Fixing this in code is more reliable than asking the model nicely.
        """
        cleaned = [re.sub(r"^\s*\(?[A-Fa-f][).:]\s+", "", choice).strip() for choice in choices]

        # Every choice must be real and different. A blank or repeated choice
        # means the model garbled the quiz; rejecting it is better than
        # showing the student a broken question.
        if any(not choice for choice in cleaned):
            raise ValueError("a choice is empty")
        if len({choice.lower() for choice in cleaned}) != len(cleaned):
            raise ValueError("two choices are the same")
        return cleaned

    @model_validator(mode="after")
    def _check_answer_letter(self) -> "QuizQuestion":
        """Accept "b", " B ", "B)" etc., and make sure the letter has a choice."""
        letter = self.answer.strip().upper()[:1]
        index = ord(letter) - ord("A") if letter else -1
        if not 0 <= index < len(self.choices):
            raise ValueError(f"answer {self.answer!r} does not match any of the choices")
        self.answer = letter
        return self


class Quiz(BaseModel):
    """A whole quiz: its topic and 1-5 questions."""

    topic: str
    questions: list[QuizQuestion] = Field(min_length=1, max_length=5)


# ---------------------------------------------------------------------------
# 2. Turning the model's text into a Quiz
# ---------------------------------------------------------------------------

def parse_quiz(text: str) -> dict | None:
    """
    Find the JSON in the model's reply and check it is a valid quiz.

    Models often wrap JSON in extra text or ```json fences, so we take
    everything from the first "{" to the last "}".

    Returns:
        The quiz as a plain dict (easy to save in a checkpoint), or None if
        there is no valid quiz in the text.
    """
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return Quiz.model_validate(json.loads(text[start:end + 1])).model_dump()
    except (json.JSONDecodeError, ValidationError) as error:
        log.warning("quiz JSON was not valid: %s", str(error).splitlines()[0])
        return None


def repair_quiz(text: str) -> dict | None:
    """
    Second chance: ask the model to rewrite its reply as valid quiz JSON.

    `format=Quiz.model_json_schema()` uses Ollama's STRUCTURED OUTPUT
    feature: Ollama only lets the model produce text that matches this JSON
    schema. We do not use it for the first attempt because it cannot be
    combined with tool calling - the model needs to be free to request a
    tool first.

    ONLY for replies that actually contain quiz questions. Structured output
    forces the model to produce a quiz no matter what it is given - fed an
    empty reply, it will happily invent a quiz on a random topic. So an
    empty or question-free reply is not repaired; it counts as a failure.
    """
    if "question" not in text.lower() or "?" not in text:
        log.warning("reply does not look like a quiz (%d characters); not repairing", len(text))
        return None

    log.info("asking the model to repair the quiz JSON")
    fixer = build_chat_model(temperature=0.0, format=Quiz.model_json_schema())
    reply = fixer.invoke([
        SystemMessage(content="Rewrite the quiz below as JSON matching the required schema. "
                              "'answer' is the letter of the correct choice (A = first)."),
        HumanMessage(content=text),
    ])
    return parse_quiz(reply.content)


def describe_quiz(quiz: dict) -> str:
    """The chat message announcing a new quiz. Deliberately leaves out the answers."""
    count = len(quiz["questions"])
    return f"I made a {count}-question quiz on {quiz['topic']}."


# ---------------------------------------------------------------------------
# 3. Finding the topic - and deciding in code whether to ask for it
# ---------------------------------------------------------------------------

# The only shape the topic-finding call may reply in: {"topic": "..."}.
_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {"topic": {"type": "string"}},
    "required": ["topic"],
}


def find_topic(state: ChatState) -> str:
    """
    Work out the quiz topic from the conversation. "" means none was given.

    A small model call with STRUCTURED OUTPUT (the same Ollama feature the
    router uses): the reply can only be {"topic": "..."}, so the model cannot
    wander off into asking questions or writing the quiz early.

    It sees:
      * the task passed with a handoff, if any ("quiz on gravity")
      * the previous quiz's topic, if any (for "make it harder")
      * the last few messages (for "quiz me on that", or the student's
        answer to "which topic?")
    """
    lines = []
    if state.get("task"):
        lines.append(f"Task from the agent that asked for the quiz: {state['task']}")
    if state.get("quiz"):
        lines.append(f"Topic of the previous quiz: {state['quiz']['topic']}")
    lines.append("\nEnd of the chat (latest last):")
    for message in state["messages"][-6:]:
        who = "Student" if message.type == "human" else (message.name or "assistant")
        lines.append(f"{who}: {message.content[:200]}")

    model = build_chat_model(temperature=0.0, format=_TOPIC_SCHEMA, num_predict=40)

    with mlflow.start_span(name="quiz_maker:find_topic", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"context": "\n".join(lines)})
        reply = model.invoke([
            SystemMessage(content=load_prompt("quiz_maker_topic.md")),
            HumanMessage(content="\n".join(lines)),
        ])
        try:
            topic = str(json.loads(reply.content).get("topic", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            topic = ""
        span.set_outputs({"topic": topic})

    log.info("topic: %s", repr(topic) if topic else "(none found)")
    return topic


def task_brief(state: ChatState, topic: str) -> str:
    """
    The "Your current task" note added to the quiz-writing prompt: the topic
    (found above), plus any details from a handoff task or the student's
    latest message, such as "harder" or "5 questions".
    """
    latest = next((m.content for m in reversed(state["messages"]) if m.type == "human"), "")
    lines = ["# Your current task", "", f"Topic: {topic}"]
    if state.get("task"):
        lines.append(f"Details from the agent that asked: {state['task']}")
    lines.append(f'The student\'s latest message: "{latest}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. The node
# ---------------------------------------------------------------------------

def quiz_maker_node(state: ChatState) -> dict:
    """
    The Quiz Maker's turn in the graph.

    Unlike the Tutor, it does not use `run_agent`, because its reply is not a
    chat message: it is quiz JSON that has to be checked and stored.
    """
    # 1. The topic. CODE decides whether to ask: no topic -> clarifying
    #    question, using the same pause-and-resume flow as the Tutor (step 5).
    #    After the student answers, this node runs again and find_topic sees
    #    the answer as the latest message.
    topic = find_topic(state)
    if not topic:
        return clarification_update(AGENT, AIMessage(content=f"CLARIFY: {ASK_FOR_TOPIC}"))

    # 2. Write the quiz: the model call, including the tool loop
    #    (get_topic_facts), with the topic spelled out.
    reply = call_model(AGENT, state["messages"], extra_instructions=task_brief(state, topic))

    quiz = parse_quiz(reply.content) or repair_quiz(reply.content)

    if quiz is None:
        log.warning("could not get a valid quiz from the model")
        return {
            "messages": [AIMessage(
                content="Sorry, I couldn't put a quiz together this time. "
                        "Please try again, maybe with a more specific topic.",
                name=AGENT.name,
            )],
            "pending_question": None,
        }

    # We already know the topic for certain; do not trust the model to copy
    # it correctly into the JSON.
    quiz["topic"] = topic

    log.info("quiz ready: %d questions on %r", len(quiz["questions"]), quiz["topic"])
    update = {
        "messages": [AIMessage(content=describe_quiz(quiz), name=AGENT.name)],
        "quiz": quiz,
        "pending_question": None,
    }

    # Who shows the quiz to the student? If an agent called us, it is waiting
    # on the return stack and gets control back automatically. If nobody is
    # waiting (the router picked us directly), hand over to the presenter
    # ourselves: a TRANSFER - "finish and trigger another agent".
    if not state.get("return_stack"):
        update["handoff"] = {"to": PRESENTER, "mode": "transfer"}
    return update


AGENT = AgentSpec(
    name="quiz_maker",
    # Read by the ROUTER. New-quiz requests go to the Tutor (which calls the
    # Quiz Maker); the router sends the Quiz Maker only requests to CHANGE the
    # quiz that was just given.
    description=(
        "Writes multiple-choice quizzes using study notes from its "
        "get_topic_facts tool. Choose it directly ONLY when the latest message "
        "explicitly asks to CHANGE the quiz just given (e.g. 'make it harder', "
        "'easier', '5 questions instead'). NOT for answers to a quiz."
    ),
    prompt_file="quiz_maker.md",
    node=quiz_maker_node,
    # A little more varied than the Tutor's default, so repeated quizzes on
    # the same topic are not identical - but still low, because the reply
    # must be valid JSON.
    temperature=0.4,
    tools=(get_topic_facts,),
    # Its model writes quiz JSON, so the chat should not stream it; the short
    # "I made a quiz" message is printed once the node finishes instead.
    stream_replies=False,
)
