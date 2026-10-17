"""Greeting detection used to skip the planner LLM call.

Matching is token based, not substring based. The English greetings
("hi" / "hello" / "hey") used to be matched as plain substrings, so
ordinary sentences such as ``this is a test`` or ``which option``
were mistaken for greetings and silently skipped the planner.

Rules:
- ASCII words are compared whole: only ``hi`` / ``hello`` / ``hey``
  count as a greeting, ``history`` or ``this`` never do.
- CJK greetings have no word boundaries, so they are still located
  as substrings, but every CJK run in the message must be fully
  consumed by greetings and particles — leftover CJK text means the
  message carries real content.
- The message counts as a greeting only when it contains at least
  one greeting and no content words at all. Anything else (numbers,
  unknown words, CJK requests) falls through to normal planning.

The check is conservative on purpose: a false negative only costs an
extra planner call, while a false positive drops the plan entirely.
"""

import re


GREETING_WORDS = frozenset({"hi", "hello", "hey"})

# Words that may appear next to a greeting without turning the
# message into a real request ("hi there", "hello everyone").
GREETING_FILLER_WORDS = frozenset(
    {"there", "everyone", "folks", "team", "guys", "all"}
)

CJK_GREETINGS = (
    "你好",
    "您好",
    "哈喽",
    "早上好",
    "上午好",
    "下午好",
    "晚上好",
    "早安",
    "晚安",
    "嗨",
)

# Polite trailing characters that carry no content on their own
# ("你好呀", "嗨~啊").
CJK_PARTICLES = ("呀", "啊", "呢", "吧", "啦", "哦", "噢", "咯", "嘿", "喂")

_CJK_GREETING_SET = frozenset(CJK_GREETINGS)

# Longest first so "早上好" is consumed before any shorter fragment.
_CJK_STRIPPABLE = tuple(
    sorted(CJK_GREETINGS + CJK_PARTICLES, key=len, reverse=True)
)

# ASCII words / digits, or a run of CJK characters. Punctuation,
# whitespace and emoji are noise and are simply skipped.
_TOKEN = re.compile(r"[a-z0-9']+|[一-鿿]+")


def _consume_cjk_run(run: str) -> tuple[bool, str]:
    """Strip greetings and particles from a CJK run.

    Returns ``(found_greeting, leftover)``. Non-empty ``leftover``
    means the run still carries content that is not a greeting.
    """
    found = False
    leftover = run

    while True:
        for item in _CJK_STRIPPABLE:
            index = leftover.find(item)

            if index == -1:
                continue

            leftover = leftover[:index] + leftover[index + len(item):]

            if item in _CJK_GREETING_SET:
                found = True

            break
        else:
            return found, leftover


def is_greeting(message) -> bool:
    """Whether a message is a pure greeting.

    Greetings need no tool planning, so the planner LLM call can be
    skipped for them.
    """
    if not isinstance(message, str):
        return False

    text = message.strip().lower()

    if not text:
        return False

    found_greeting = False

    for token in _TOKEN.findall(text):
        if token[0].isascii():
            if token in GREETING_WORDS:
                found_greeting = True
            elif token in GREETING_FILLER_WORDS:
                continue
            else:
                # Any other word ("this", "option", "500") is content.
                return False

            continue

        matched, leftover = _consume_cjk_run(token)

        if leftover:
            return False

        if matched:
            found_greeting = True

    return found_greeting
