"""
core/logging_setup.py - prints one line to the terminal for each step of the flow.

WHY LOGGING?
------------
When you send a message, a lot happens that you never see: a prompt file is
loaded, a list of messages is built, the model is called, the reply is added
to the history. Logging makes those steps visible, live, in the terminal:

    12:04:31 [chat]   run started (2 messages in history)
    12:04:31 [prompt] loaded tutor.md from disk (812 characters)
    12:04:31 [tutor]  calling llama3.1:8b with 1 system + 3 conversation messages
    12:04:35 [tutor]  reply received in 4.2s (312 characters)
    12:04:35 [chat]   run finished

Logging and MLflow tracing (step 4) complement each other:
    * logging  -> a live, one-line-per-step view while you chat
    * tracing  -> a detailed, clickable record you can open afterwards

HOW TO CONTROL IT
-----------------
Set LOG_LEVEL in `.env`:

    LOG_LEVEL=INFO      the flow, one line per step (default)
    LOG_LEVEL=DEBUG     the flow PLUS the full text of every message sent to
                        the model - very useful for seeing exactly what an
                        agent "sees", but noisy
    LOG_LEVEL=WARNING   quiet: only problems

HOW TO USE IT IN A FILE
-----------------------
    from core.logging_setup import get_logger
    log = get_logger("tutor")         # the name shown in [brackets]
    log.info("something happened")
    log.debug("extra detail, only shown when LOG_LEVEL=DEBUG")

BACKGROUND: PYTHON'S `logging` MODULE IN 30 SECONDS
---------------------------------------------------
    * A LOGGER is a named object you call `.info(...)` / `.debug(...)` on.
    * Loggers form a tree by name: "studyhelper.tutor" is a child of
      "studyhelper". Messages travel up the tree to the parent.
    * A HANDLER on a logger decides WHERE messages go (here: the terminal).
    * A FORMATTER decides what each line LOOKS like.
    * The LEVEL decides which messages are shown: DEBUG < INFO < WARNING.
"""

from __future__ import annotations

import logging

from core.config import get_settings

# Every logger in this project lives under this parent name. That keeps our
# messages separate from the libraries' own loggers (LangChain, httpx, ...),
# which would otherwise flood the terminal.
_ROOT_NAME = "studyhelper"


class _ShortNameFormatter(logging.Formatter):
    """
    Prints `[tutor]` instead of the full logger name `[studyhelper.tutor]`.

    The name is padded to a fixed width so the messages line up in a column,
    which makes a long log much easier to scan.
    """

    def format(self, record: logging.LogRecord) -> str:
        short = record.name.removeprefix(f"{_ROOT_NAME}.")
        record.short_name = f"[{short}]".ljust(9)
        return super().format(record)


def setup_logging(level: str | None = None) -> None:
    """
    Turn on terminal logging. Call this once, at the top of a script's main().

    Args:
        level: "DEBUG", "INFO" or "WARNING" to override LOG_LEVEL from `.env`
            (e.g. `chat.py --quiet` passes "WARNING"). None = use `.env`.

    Calling it twice is harmless: the second call sees a handler is already
    attached and does nothing, so you never get every line printed twice.
    """
    root = logging.getLogger(_ROOT_NAME)
    if root.handlers:
        return

    # getattr(logging, "INFO") -> logging.INFO (the number 20). An unknown
    # value like "LOUD" falls back to INFO instead of crashing.
    level_name = (level or get_settings().log_level).upper()
    root.setLevel(getattr(logging, level_name, logging.INFO))

    handler = logging.StreamHandler()  # writes to the terminal (stderr)
    handler.setFormatter(
        _ShortNameFormatter("%(asctime)s %(short_name)s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(handler)

    # Do not pass our messages on to Python's global root logger, which might
    # print them a second time in a different format.
    root.propagate = False


def get_logger(component: str) -> logging.Logger:
    """
    Get the logger for one part of the program.

    Args:
        component: a short label shown in brackets, e.g. "tutor", "graph".

    It is safe to call this at import time, before `setup_logging()` has
    run: the logger exists immediately and starts printing once logging has
    been set up.
    """
    return logging.getLogger(f"{_ROOT_NAME}.{component}")
