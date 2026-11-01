"""
agents/tutor.py - Agent 1, the Tutor.

WHAT THE TUTOR DOES
-------------------
    step 2  answers study questions in plain language
    step 5  asks a clarifying question when a request is too vague
    step 7  hands quiz requests to the Quiz Maker (a "call" handoff), then
            presents the quiz it gets back and grades the student's answers

THE TUTOR'S TURN, IN ORDER
--------------------------
`tutor_node` checks four situations, first match wins:

    1. A new quiz is waiting to be shown (state["quiz"] without
       "presented")  ->  show the questions, WITHOUT the answers.
       Plain Python, not the model: instant, and cannot reveal an answer.

    2. The student's message contains answers to the quiz just shown
       ("1-B, 2-A, 3-C")  ->  GRADE them, also in plain Python.
       The model used to do this, and once marked "C" correct by matching
       it to the text of choice B. Comparing letters with an answer key is
       a job code does perfectly and instantly; a model call took minutes.

    3. Control just came back from another agent and there is nothing to
       present (e.g. the Quiz Maker could not make a quiz)  ->  say nothing;
       that agent's own message already explains what happened.

    4. Anything else  ->  a normal model turn (run_agent). If a quiz has been
       shown, the model also gets the ANSWER KEY as extra hidden
       instructions, so it can discuss the questions ("why was 2 wrong?")
       or grade answers written in a form the code did not recognise.

HOW TO CHANGE ITS BEHAVIOUR
---------------------------
Edit `agents/prompts/tutor.md`, then restart the chat.
"""

from __future__ import annotations

import re

import mlflow
from langchain_core.messages import AIMessage
from mlflow.entities import SpanType

from agents.base import AgentSpec, run_agent
from core.logging_setup import get_logger
from core.state import ChatState

log = get_logger("tutor")


def _letter(index: int) -> str:
    """0 -> "A", 1 -> "B", ..."""
    return chr(ord("A") + index)


def present_quiz(quiz: dict) -> str:
    """The quiz as the student sees it: questions and choices, no answers."""
    lines = [f"Here's your quiz on {quiz['topic']}!", ""]
    for number, item in enumerate(quiz["questions"], start=1):
        lines.append(f"{number}. {item['question']}")
        for index, choice in enumerate(item["choices"]):
            lines.append(f"   {_letter(index)}) {choice}")
        lines.append("")
    lines.append("Reply with your answers, for example: 1-B, 2-A, 3-C")
    return "\n".join(lines)


def answer_key(quiz: dict) -> str:
    """
    Hidden instructions for grading. Sent to the model only - never shown in
    the chat - and only once the quiz has been presented.
    """
    lines = [
        f"# Answer key (hidden from the student) for the quiz on {quiz['topic']} you gave",
        "",
    ]
    for number, item in enumerate(quiz["questions"], start=1):
        letter = item["answer"]
        correct_text = item["choices"][ord(letter) - ord("A")]
        lines.append(f"{number}. {item['question']}")
        lines.append(f"   Correct: {letter}) {correct_text}. Why: {item['explanation']}")
    if quiz.get("graded"):
        lines += [
            "",
            "This quiz has already been graded. Use the key only to explain the",
            "questions if the student asks about them.",
        ]
    else:
        lines += [
            "",
            "If the student's latest message answers this quiz, grade it with this key:",
            "for each question say whether they were right, give the correct answer",
            "with its short explanation, then give the score, e.g. 'Score: 2/3'.",
            "If their latest message is about something else, ignore this key.",
        ]
    return "\n".join(lines)


# Matches one answer such as "1-B", "2) a", "3: C", "1B", "2 = d".
#   (\d+)          the question number
#   \s*[-:.)=]?\s* an optional separator
#   \(?([A-Da-d])\)?   the letter, optionally in brackets
#   (?![A-Za-z])   NOT followed by another letter, so "2 apples" is not "2-A"
_ANSWER = re.compile(r"(\d+)\s*[-:.)=]?\s*\(?([A-Da-d])\)?(?![A-Za-z])")


def parse_answers(text: str, question_count: int) -> dict[int, str]:
    """
    Read quiz answers from the student's message.

        "1-B, 2-A, 3-C"  ->  {1: "B", 2: "A", 3: "C"}
        "what is gravity" -> {}   (no answers)
    """
    return {
        int(number): letter.upper()
        for number, letter in _ANSWER.findall(text)
        if 1 <= int(number) <= question_count
    }


def grade_quiz(quiz: dict, answers: dict[int, str]) -> tuple[str, int]:
    """
    Compare the student's answers with the answer key.

    Returns:
        (the feedback message, the number of correct answers)
    """
    def choice_text(item: dict, letter: str) -> str:
        return item["choices"][ord(letter) - ord("A")]

    lines = [f"Here are your results for the quiz on {quiz['topic']}:", ""]
    correct = 0
    for number, item in enumerate(quiz["questions"], start=1):
        right = item["answer"]
        given = answers.get(number)
        if given == right:
            correct += 1
            lines.append(f"{number}. Correct! {right}) {choice_text(item, right)}")
        else:
            if given is None:
                lines.append(f"{number}. No answer given.")
            elif ord(given) - ord("A") >= len(item["choices"]):
                lines.append(f"{number}. {given} is not one of the choices.")
            else:
                lines.append(f"{number}. Not quite - you chose {given}) {choice_text(item, given)}.")
            lines.append(f"   The answer is {right}) {choice_text(item, right)}.")
        if item.get("explanation"):
            lines.append(f"   Why: {item['explanation']}")
        lines.append("")

    total = len(quiz["questions"])
    lines.append(f"Score: {correct}/{total}")
    if correct == total:
        lines.append("Perfect score - great work!")
    elif correct * 2 >= total:
        lines.append("Good effort! Ask me about any question you'd like explained.")
    else:
        lines.append("Keep going - ask me to explain any of these, and try another quiz.")
    return "\n".join(lines), correct


def tutor_node(state: ChatState) -> dict:
    """The Tutor's turn in the graph (see "THE TUTOR'S TURN" above)."""
    quiz = state.get("quiz")

    # 1. A new quiz has arrived: present it.
    if quiz and not quiz.get("presented"):
        log.info("presenting the %d-question quiz on %r", len(quiz["questions"]), quiz["topic"])
        return {
            "messages": [AIMessage(content=present_quiz(quiz), name=AGENT.name)],
            # Mark it as shown, so the next turn grades instead of re-showing.
            "quiz": {**quiz, "presented": True},
        }

    # 2. Answers to the quiz just shown: grade them in code.
    last = state["messages"][-1]
    if quiz and quiz.get("presented") and not quiz.get("graded") and last.type == "human":
        answers = parse_answers(last.content, len(quiz["questions"]))
        if answers:
            feedback, correct = grade_quiz(quiz, answers)
            log.info("graded the quiz in code: %d/%d", correct, len(quiz["questions"]))
            # A span so the trace shows the grading step, its input and result.
            with mlflow.start_span(name="tutor:grade_quiz", span_type=SpanType.CHAIN) as span:
                span.set_inputs({"answers": answers, "answer_key": [q["answer"] for q in quiz["questions"]]})
                span.set_outputs({"score": f"{correct}/{len(quiz['questions'])}"})
            return {
                "messages": [AIMessage(content=feedback, name=AGENT.name)],
                # Marked graded, so later messages like "2 apples" are never
                # mistaken for answers to this quiz.
                "quiz": {**quiz, "graded": True},
            }

    # 3. Back from another agent with nothing to present: stay quiet.
    if last.type == "ai" and last.name != AGENT.name:
        log.info("back from %s with nothing to present; letting its message stand", last.name)
        return {}

    # 4. A normal turn. `AGENT` is defined at the bottom of this file; Python
    #    looks the name up when this function RUNS, so that is fine.
    return run_agent(AGENT, state, extra_instructions=answer_key(quiz) if quiz else "")


# The description of this agent. The registry (agents/registry.py) collects
# this object; the ROUTER reads `description` to decide when to pick it.
AGENT = AgentSpec(
    name="tutor",
    description=(
        "The main study helper. Explains topics and answers questions; handles "
        "requests for a NEW quiz (it asks the Quiz Maker to write one); presents "
        "quizzes and GRADES the student's quiz answers (messages like "
        "'1-B, 2-A, 3-C'). Use it for greetings and anything general too."
    ),
    prompt_file="tutor.md",
    node=tutor_node,
    can_hand_off=True,
)
