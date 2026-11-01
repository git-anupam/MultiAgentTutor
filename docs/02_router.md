# What is a router?

With more than one agent, something has to decide **who answers**. The
**router** is the first node of every run (`core/router.py`):

```
START -> router -> tutor        "what is photosynthesis?"
                -> quiz_maker   "help me build a quiz on gravity"
                -> tutor        "1-B, 2-A, 3-C"
```

It runs again for **every new message**, so the conversation can switch
agents at any point. Your first message can even go straight to the Quiz
Maker, without the Tutor ever being involved.

## How it decides

It asks the model, with a prompt built from the registry:

1. Every agent's **name and `description`** (from `agents/registry.py`).
2. The last few messages, for context.
3. The student's **latest message**, set apart and labelled, so an older
   request ("make it harder") cannot be mistaken for the current one.

The reply is forced into this shape by Ollama's **structured output**
(`format=<JSON schema>`):

```json
{"reason": "The student is answering the quiz questions.", "agent": "tutor"}
```

The schema's `enum` lists only real agent names, so the router cannot invent
an agent. If anything still goes wrong, it falls back to the Tutor. The
reason appears in the log (`[router] -> tutor (...)`) and in the trace's
`route` span.

## Writing descriptions for the router

The router only knows what the descriptions tell it, so a description is
really an instruction to the router: say **when to pick this agent, and when
not to**. The Tutor's description mentions grading answers like
"1-B, 2-A, 3-C"; adding that example fixed a case where quiz answers were
sent to the Quiz Maker.

## When the router does NOT run

When an agent has asked a clarifying question and the run is paused, your
answer **skips the router** and goes straight back to the agent that asked.
See [clarifying questions](04_clarifying_questions.md).

## Cost

One short model call per message, roughly 5–40 seconds on a CPU. With a
single agent the router skips the call entirely.
