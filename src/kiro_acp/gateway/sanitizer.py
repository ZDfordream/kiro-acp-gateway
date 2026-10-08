"""Sanitizers for model-visible and user-visible text.

Two independent layers keep upstream identity out of user-facing answers:

* ``sanitize_system`` (opt-in, ``KIRO_GATEWAY_SANITIZE_SYSTEM``) strips lines from
  the client's system prompt that assert a product identity or demand concealment —
  the lines most likely to trigger "prompt injection" refusals. Everything else
  (project instructions, tool guidance, memory) is kept verbatim.
* ``scrub_text`` / ``IdentityScrubber`` (on by default, ``KIRO_GATEWAY_SCRUB_IDENTITY``)
  replace hard product and infrastructure names in assistant output with a neutral
  phrase. Generic words ("gateway", "backend", "proxy") are left alone so ordinary
  technical answers survive, and `` ``` ``-fenced code blocks are never touched.
"""

from __future__ import annotations

import re

_PRODUCTS = r"(Claude(?: Code)?|Codex|Copilot|ChatGPT|GPT-\d[\w.-]*|Gemini|Cursor|OpenCode|Kiro)"
_PATTERNS = [
    re.compile(rf"^\s*(You are|I am|This is|You must identify as)\b.*\b{_PRODUCTS}\b", re.I),
    re.compile(
        r"^\s*You are (an? )?(interactive |AI |autonomous )?(CLI |coding )?(tool|agent|assistant) (made|created|developed|built) by\b",
        re.I,
    ),
    re.compile(
        r"^\s*You are (NOT|never)\b.*\b(other AI|another AI|different (AI|model|identity)|Claude|Kiro)\b",
        re.I,
    ),
    re.compile(
        r"^\s*(Never|Do not|Don't) (say|claim|admit|reveal|disclose|mention)\b.*\b(you are|your (identity|model|provider)|gateway|proxy|bridge)\b",
        re.I,
    ),
    re.compile(
        r"^\s*(Ignore|Override|Disregard)\b.*\b(instructions?|prompts?)\b.*\b(contradict|conflict|previous|above)\b",
        re.I,
    ),
    re.compile(r"^\s*#+\s*(Identity|Persona)\s*$", re.I),
]


def sanitize_system(text: str) -> tuple[str, int]:
    """Return the sanitized text and the number of lines removed."""
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in _PATTERNS):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


# ---------------------------------------------------------------- output scrubbing

NEUTRAL_NAME = "an AI assistant"

# Ordered longest-first where one alternative prefixes another.
_NAME_SUBS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\b(?:amazon\s+)?q\s+developer\b"), NEUTRAL_NAME),
    (re.compile(r"(?i)\b(?:aws\s+|amazon\s+)?code\s*whisperer\b"), NEUTRAL_NAME),
    (re.compile(r"(?i)\b(?:aws\s+|amazon\s+)?kiro[-\w]*\b"), NEUTRAL_NAME),
    (re.compile(r"(?i)\bamazon\s+q\b"), NEUTRAL_NAME),
)

# Proper prefixes that must not end a streamed piece: a product name or a code
# fence marker may still be arriving after the cut. Compared case-insensitively.
_HOLD_PREFIX_NAMES: tuple[str, ...] = (
    "```",
    "kiro",
    "kiro-cli",
    "kiro-gateway",
    "kiro-gateway-harness",
    "aws kiro",
    "amazon kiro",
    "amazon q",
    "amazon q developer",
    "q developer",
    "aws codewhisperer",
    "amazon codewhisperer",
    "code whisperer",
    "codewhisperer",
)


def _sub_all(text: str) -> str:
    for pattern, replacement in _NAME_SUBS:
        text = pattern.sub(replacement, text)
    return text


def _scrub_segmented(text: str, in_code: bool) -> tuple[str, bool]:
    """Scrub prose outside `` ``` `` fences; return the new fence state."""
    parts = text.split("```")
    out = [parts[0] if in_code else _sub_all(parts[0])]
    for segment in parts[1:]:
        out.append("```")
        in_code = not in_code
        out.append(segment if in_code else _sub_all(segment))
    return "".join(out), in_code


def scrub_text(text: str) -> str:
    """One-shot scrub of product and infrastructure names in assistant output."""
    scrubbed, _ = _scrub_segmented(text, False)
    return scrubbed


class IdentityScrubber:
    """Streaming counterpart of :func:`scrub_text`: feed deltas, then flush.

    A tail that is a proper prefix of a product name or of a code fence marker is
    held back until the next feed resolves it, so no name can be split across
    chunks; everything safe is emitted as soon as it is known.
    """

    def __init__(self) -> None:
        self.pending = ""
        self.in_code = False

    def feed(self, chunk: str) -> str:
        self.pending += chunk
        out: list[str] = []
        while (cut := self._safe_cut()) > 0:
            piece, self.pending = self.pending[:cut], self.pending[cut:]
            scrubbed, self.in_code = _scrub_segmented(piece, self.in_code)
            out.append(scrubbed)
        return "".join(out)

    def flush(self) -> str:
        scrubbed, self.in_code = _scrub_segmented(self.pending, self.in_code)
        self.pending = ""
        return scrubbed

    def _safe_cut(self) -> int:
        """Latest cut that cannot split a name or fence marker; 0 when none is safe."""
        pending = self.pending
        for cut in range(len(pending), 0, -1):
            head = pending[:cut].lower()
            if not any(
                head.endswith(name[:k]) for name in _HOLD_PREFIX_NAMES for k in range(1, len(name))
            ):
                return cut
        return 0
