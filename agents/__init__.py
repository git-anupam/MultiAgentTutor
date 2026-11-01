"""
agents/ - one file per agent.

Every agent file follows the same pattern, so once you have read one you can
read them all:

    1. a node function   - what the agent does when it is its turn
    2. an `AGENT` object - an `AgentSpec` (see base.py) describing the agent:
                           its name, a one-line description, its prompt
                           file, and the tools it may use

Files:

    base.py        the shared `AgentSpec` class, `call_model()` (including
                   the tool-calling loop) and `run_agent()`
    registry.py    the one list of every agent
    tutor.py       Agent 1: answers study questions
    quiz_maker.py  Agent 2: writes quizzes, using the get_topic_facts tool

Each agent's system prompt lives in `agents/prompts/<name>.md`, not in the
Python code, so it can be edited without touching Python.
"""
