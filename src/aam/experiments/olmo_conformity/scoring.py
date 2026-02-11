from __future__ import annotations

import re
from typing import Optional


_SUBSCRIPT_SUPERSCRIPT_TRANS = str.maketrans(
    {
        # subscripts
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        # superscripts
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
    }
)

_DOTTED_ABBR_RE = re.compile(r"\b(?:[a-z]\.)+[a-z]\b")
_PUNCT_RE = re.compile(r"[.,;:!?'\"()\[\]{}*_`~]")
_WS_RE = re.compile(r"\s+")


def normalize_text_for_matching(text: Optional[str]) -> str:
    """
    Normalize text for deterministic string matching.

    Designed to reduce false negatives from formatting differences while keeping the
    matcher conservative for short/numeric answers.

    - Lowercase + strip
    - Collapse dotted abbreviations: "u.s." -> "us", "d.c." -> "dc"
    - Normalize unicode sub/superscript digits: "H₂O" -> "h2o"
    - Remove common punctuation / markdown wrappers and collapse whitespace
    """
    if not text:
        return ""

    normalized = str(text).lower().strip()
    normalized = normalized.translate(_SUBSCRIPT_SUPERSCRIPT_TRANS)
    normalized = _DOTTED_ABBR_RE.sub(lambda m: m.group(0).replace(".", ""), normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _WS_RE.sub(" ", normalized)
    return normalized.strip()


_GARBAGE_MARKERS = [
    "passage:",
    "question:",
    "user:",
    "assistant:",
    "system:",
    "\nuser",
    "\nassistant",
    "\n\nuser",
    "\n\nquestion",
    "article:",
    "movie title:",
    "movie plot:",
]


def _strip_after_think_end(text: str) -> str:
    """
    Keep only the post-</think> segment, if present.

    Empirically, OLMo variants frequently emit a closing </think> tag even when
    the corresponding <think> tag is absent; we treat the last closing tag as a
    delimiter for the final answer region.
    """
    t = str(text or "")
    low = t.lower()
    tag = "</think>"
    idx = low.rfind(tag)
    if idx == -1:
        return t
    return t[idx + len(tag) :].lstrip()


def parse_answer_text(raw_text: Optional[str]) -> str:
    """
    Best-effort extraction of the answer-bearing region from a raw model completion.

    Current rules:
      1) If a </think> delimiter exists, discard everything before it.
      2) Truncate at known garbage/hallucination markers (e.g., USER:/SYSTEM:/Passage:).
      3) If truncation yields empty, fall back to first non-empty line.
    """
    if not raw_text or not str(raw_text).strip():
        return ""

    text = _strip_after_think_end(str(raw_text).strip()).strip()
    low = text.lower()

    earliest = len(text)
    for marker in _GARBAGE_MARKERS:
        pos = low.find(marker)
        if pos > 0:
            earliest = min(earliest, pos)

    extracted = text[:earliest].strip()
    if extracted:
        return extracted

    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def is_refusal(text: Optional[str]) -> bool:
    t = (text or "").lower()
    return any(x in t for x in ["i can't", "i cannot", "sorry", "i'm unable", "i am unable", "as an ai"])


def evaluate_correctness(*, parsed_answer_text: str, ground_truth_text: Optional[str]) -> Optional[bool]:
    """
    Deterministic correctness scorer used by the runner.

    Notes:
    - For short/numeric answers: word-boundary/start/end matching to avoid substring false positives.
    - For longer ground truths: normalized substring containment.
    """
    if ground_truth_text is None:
        return None
    if not parsed_answer_text:
        return False

    text_norm = normalize_text_for_matching(parsed_answer_text)
    gt_norm = normalize_text_for_matching(ground_truth_text)
    if not gt_norm:
        return None

    is_short_or_numeric = len(gt_norm) <= 4 or gt_norm.isdigit()
    if is_short_or_numeric:
        if re.search(r"^" + re.escape(gt_norm) + r"(?:\b|$)", text_norm):
            return True
        if re.search(r"\b" + re.escape(gt_norm) + r"\b", text_norm):
            return True
        if re.search(r"(?:^|\b)" + re.escape(gt_norm) + r"$", text_norm):
            return True
        return False

    return gt_norm in text_norm

