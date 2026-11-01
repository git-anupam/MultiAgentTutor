# How handoffs work

The router picks the **first** agent for a message. A **handoff** is one
agent passing control to another partway through. There are two kinds
(`core/handoff.py`).

## 1. Call and return

*"An agent pauses, calls another agent, gets its output, and continues."*

```
You: test me on what we just discussed
router -> tutor
tutor:       "HANDOFF: quiz_maker | quiz on photosynthesis"
             -> pushed onto the return stack: ["tutor"]
quiz_maker:  writes the quiz
             -> finishes; "tutor" is on top of the stack
tutor:       presents the quiz      (popped off the stack)
```

Like calling a function: the caller is remembered on a **return stack**
(`state["return_stack"]`), and when the called agent finishes, control goes
back to whoever is on top.

## 2. Transfer

*"An agent finishes and triggers another."*

```
You: help me build a quiz on gravity
router -> quiz_maker
quiz_maker:  writes the quiz; nobody is waiting on the stack,
             so it names the next agent itself: next_agent = "tutor"
tutor:       presents the quiz
```

Like a relay race: nobody is waiting to get control back.

## Passing the task along

A handoff can carry a **task**, the text after `|`:

```
HANDOFF: quiz_maker | 3-question quiz on gravity
```

It is stored in `state["task"]`, and the receiving agent reads it. All
agents share the conversation history, so the request is in there anyway,
but an explicit note from the calling agent is much more reliable than
hoping the next agent finds it.

## Who does the bookkeeping

Agent code only says **"hand this to X"**, by returning an extra key:

```python
{"messages": [...], "handoff": {"to": "quiz_maker", "mode": "call", "task": "..."}}
```

A wrapper, `agent_node()`, runs around every agent and does the rest:
clears `next_agent` when the named agent starts, pops the stack when control
returns, pushes the caller for a "call", and stops after 3 handoffs per
message so two agents cannot pass work back and forth forever.

Then the edge after every agent, `route_after_agent()`, picks the next node:

1. a clarifying question is waiting → `ask_user`
2. a handoff named an agent → that agent
3. someone is waiting on the stack → back to them
4. otherwise → `END`

Neither function names a specific agent, and the Tutor's list of agents it
can hand off to is built from the registry, so a new agent takes part in
handoffs automatically.
