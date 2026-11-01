"""
infra/ - connections to the outside services the app depends on.

    postgres.py   the PostgreSQL checkpointer (saves conversations), step 3
    tracing.py    MLflow tracing setup (records what happened), step 4

Kept apart from `core/` and `agents/` on purpose: if you ever swap Postgres
for another database, or MLflow for another tracing tool, this folder is the
only place that changes. No agent knows or cares where its conversation is
stored.
"""
