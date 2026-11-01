# What is an agent?

In this project an **agent** is a language model given a job. Concretely,
each agent is four things bundled together:

| Part | What it is | Where |
| --- | --- | --- |
| **System prompt** | Standing instructions: who it is, how to behave | `agents/prompts/<name>.md` |
| **Model call** | The conversation, plus that prompt, sent to the local model | `call_model()` in `agents/base.py` |
| **Node function** | The Python function LangGraph runs when it is this agent's turn | `<name>_node()` in `agents/<name>.py` |
| **Tools** (optional) | Python functions the model may ask us to run | `tools/` |

The two agents:

| Agent | Job | Tools |
| --- | --- | --- |
| **Tutor** (`agents/tutor.py`) | Answers questions, presents quizzes, grades answers | none, but it can hand work to the Quiz Maker |
| **Quiz Maker** (`agents/quiz_maker.py`) | Writes multiple-choice quizzes | `get_topic_facts` |

## The `AgentSpec`

Every agent file ends with one `AgentSpec` object describing the agent:

```python
AGENT = AgentSpec(
    name="quiz_maker",                 # node name, log name, message label
    description="Writes multiple-choice quizzes ...",   # read by the router
    prompt_file="quiz_maker.md",
    node=quiz_maker_node,
    tools=(get_topic_facts,),
    stream_replies=False,              # its output is JSON, not chat
)
```

`agents/registry.py` collects them all in one dictionary. The graph, the
router and the handoff code read the registry, never a specific agent, so
**adding Agent 3 takes three things**: `agents/<name>.py`,
`agents/prompts/<name>.md`, and one line in the registry.

## Shared history, separate instructions

All agents share **one conversation history** (`state["messages"]`), and
every message is labelled with the agent that wrote it. But each agent's
system prompt is added only on the way out to the model, never stored. So
the Quiz Maker sees what the Tutor said, but not the Tutor's instructions.

## A lesson from building it: let code do what code can

Small local models (7–8B parameters) are good at writing explanations and
bad at following strict rules.

Everything that needs **understanding the student** stays with a model:
the router choosing the agent for each message, the Tutor's explanations,
the Quiz Maker reading the topic, calling its tool and writing questions.

But several **mechanical jobs inside the agents** started as "ask the model
nicely" and became reliable only once plain Python took them over:

| Job | First attempt | What works |
| --- | --- | --- |
| Presenting a quiz | – | Python formats the stored quiz (instant; cannot leak answers) |
| Grading "1-B, 2-A, 3-C" | Model compared letters, and once marked a wrong answer right; 143 s | Python compares with the answer key; under 1 ms |
| Letters in choices ("A) A) …") | Prompt said "no letters" | A validator strips them |
| "Which topic?" for "quiz on gravity" | Model decided whether to ask, and kept asking | Model only extracts the topic; Python decides whether to ask |

The model does the language; code does the bookkeeping.
