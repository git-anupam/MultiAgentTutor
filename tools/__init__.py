"""
tools/ - Python functions that an agent's model can ask us to run.

BACKGROUND: WHAT IS A TOOL?
---------------------------
A language model can only produce text. It cannot look things up, do exact
maths, or read a file. A TOOL bridges that gap: a normal Python function,
plus a description the model can read (its name, what it does, what
arguments it takes). When the model decides it needs the tool, it does NOT
run it - it replies with a structured request like

    please call get_topic_facts with {"topic": "photosynthesis"}

and OUR code runs the function and sends the result back. The full loop is
in `call_model()` in agents/base.py.

Tools so far:

    topic_facts.py   get_topic_facts - study notes for a topic (Quiz Maker)

Tools are kept out of the agent files so any agent can be given any tool:
an agent lists the tools it may use in its `AgentSpec(tools=...)`.
"""
