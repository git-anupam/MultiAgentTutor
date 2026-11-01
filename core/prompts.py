"""
core/prompts.py - loads each agent's system prompt from a Markdown file.

BACKGROUND: WHAT IS A SYSTEM PROMPT?
------------------------------------
When we send a conversation to a chat model, the very first message can be a
special "system" message. The model treats it as standing instructions:
who it is, how to behave, what to focus on. The user never sees it.

Changing the system prompt is the main way to change an agent's personality
and job, so it deserves to be easy to edit.

WHY MARKDOWN FILES INSTEAD OF PYTHON STRINGS?
---------------------------------------------
If the prompt lived inside `agents/tutor.py` as a big string, tweaking one
sentence would mean editing Python code (and risking a syntax error). Keeping
it in `agents/prompts/tutor.md` means:

    * anyone can edit it, even without knowing Python
    * Markdown headings and bullet points make long prompts readable
    * the change shows up in version control as a plain-text diff

WHEN IS THE FILE READ?
----------------------
Once. The first time an agent asks for its prompt, `load_prompt()` reads the
file from disk and remembers the result (see `@lru_cache` below). Every later
call returns that remembered text without touching the disk again.

The consequence: if you edit a prompt file, restart the chat to pick up the
change.

AUTHOR NOTES INSIDE PROMPT FILES
--------------------------------
Anything inside an HTML comment, `<!-- like this -->`, is removed before the
prompt is sent to the model. That lets each prompt file carry explanations for
human readers (why a rule exists, what to try changing) without wasting the
model's attention on them.
"""

from __future__ import annotations

import re
from functools import lru_cache

from core.config import get_settings
from core.logging_setup import get_logger

log = get_logger("prompt")

# A regular expression that matches one HTML comment, including comments that
# span several lines.
#   <!--   the literal opening
#   .*?    any characters, as FEW as possible ("non-greedy"), so two separate
#          comments are matched separately instead of as one giant match
#   -->    the literal closing
# re.DOTALL lets `.` match newline characters too, which multi-line comments
# need.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


# `@lru_cache` remembers the return value for each `filename` it has seen, so
# each prompt file is read from disk only once per run. (It is the same tool
# `core/config.py` uses for `get_settings()`.) If the function raises an
# error, nothing is remembered, so fixing a typo and retrying still works.
@lru_cache(maxsize=None)
def load_prompt(filename: str) -> str:
    """
    Read a system prompt from `agents/prompts/<filename>`.

    Args:
        filename: just the file name, e.g. "tutor.md". The folder is taken
            from the settings, so callers never deal with paths.

    Returns:
        The prompt text with author comments removed and surrounding blank
        space trimmed, ready to be wrapped in a SystemMessage.

    Raises:
        FileNotFoundError: if the file does not exist. The message lists the
            files that DO exist, because the usual cause is a typo.
        ValueError: if the file exists but is empty once comments are removed.
            Sending an empty system prompt would silently turn the agent into
            a generic chatbot, which is a confusing bug to track down.
    """
    prompts_dir = get_settings().prompts_dir
    path = prompts_dir / filename

    if not path.is_file():
        available = sorted(p.name for p in prompts_dir.glob("*.md"))
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Prompt files that do exist: {available or 'none'}"
        )

    # encoding="utf-8" explicitly, so a prompt containing an accented letter
    # or an emoji reads the same on every machine.
    text = path.read_text(encoding="utf-8")
    text = _HTML_COMMENT.sub("", text).strip()

    if not text:
        raise ValueError(f"Prompt file {path} is empty (after removing comments).")

    # Because of @lru_cache, this line prints only the FIRST time each prompt
    # is loaded - a handy visible proof that the cache works.
    log.info("loaded %s from disk (%d characters)", filename, len(text))
    return text
