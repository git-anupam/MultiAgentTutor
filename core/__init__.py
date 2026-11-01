"""
core/ - the shared plumbing of the application.

Nothing in here is specific to the Tutor or the Quiz Maker. These are the
building blocks every agent uses:

    config.py   reads settings (model name, database URL, ...) from `.env`
    llm.py      builds the connection to the local Ollama model
    logging_setup.py
                prints one line to the terminal per step of the flow
    prompts.py  loads each agent's system prompt from its Markdown file
    state.py    the shape of the conversation data the graph passes around
    clarify.py  the clarifying-question flow: pause, ask the user, resume
    handoff.py  how agents hand work to each other, and "where next?"
    router.py   picks which agent handles each new message
    graph.py    wires the router, the agents and ask_user into a graph

Connections to outside services (PostgreSQL, MLflow) live separately, in
`infra/`.

Keeping this separate from `agents/` means that adding a third agent later
never requires touching the plumbing.
"""
