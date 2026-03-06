from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


logger = logging.getLogger(__name__)


class OutputQualityLabel(str, Enum):
    EMPTY = "EMPTY"
    DEGENERATE_REPETITION = "DEGENERATE_REPETITION"
    PROMPT_LEAKAGE = "PROMPT_LEAKAGE"
    STRUCTURAL_GARBAGE = "STRUCTURAL_GARBAGE"
    INCOHERENT = "INCOHERENT"
    PARTIAL_VALID = "PARTIAL_VALID"
    VALID = "VALID"


@dataclass(frozen=True)
class SpecialTokenPattern:
    """
    Regex-based pattern for stripping leaked template / special tokens.

    These are intended to be conservative (high precision) because some tokens can
    collide with ordinary user text when expressed as plain words (e.g., "assistant:").
    """

    name: str
    pattern: str
    flags: int = re.IGNORECASE


def _default_special_token_patterns() -> Tuple[SpecialTokenPattern, ...]:
    # Note: patterns intentionally match both well-formed and partial/malformed tokens
    # (e.g., "<|endoftext" without a closing "|>").
    return (
        # Llama / Mistral chat templates
        SpecialTokenPattern("llama_inst", r"\[(?:/?INST)\]"),
        SpecialTokenPattern("llama_sys", r"<<\s*SYS\s*>>|<</\s*SYS\s*>>"),
        SpecialTokenPattern("bos_eos_s", r"</?s>"),
        # ChatML / OpenAI-like
        SpecialTokenPattern("chatml_im_start", r"<\|\s*im_start\s*\|>?"),
        SpecialTokenPattern("chatml_im_end", r"<\|\s*im_end\s*\|>?"),
        SpecialTokenPattern("chatml_im_sep", r"<\|\s*im_sep\s*\|>?"),
        # Llama3 / other HF special tokens
        SpecialTokenPattern("begin_of_text", r"<\|\s*begin_of_text\s*\|>?"),
        SpecialTokenPattern("end_of_text", r"<\|\s*end_of_text\s*\|>?"),
        SpecialTokenPattern("start_header_id", r"<\|\s*start_header_id\s*\|>?"),
        SpecialTokenPattern("end_header_id", r"<\|\s*end_header_id\s*\|>?"),
        SpecialTokenPattern("eot_id", r"<\|\s*eot_id\s*\|>?"),
        # Common "endoftext"
        SpecialTokenPattern("endoftext", r"<\|\s*endoftext\s*(?:\|>)?"),
        # Think tags (may appear across model families)
        SpecialTokenPattern("think_open", r"<\s*think\s*>"),
        SpecialTokenPattern("think_close", r"<\s*/\s*think\s*>"),
        # Phi / Qwen / generic role tokens
        SpecialTokenPattern("role_user", r"<\|\s*user\s*\|>?"),
        SpecialTokenPattern("role_assistant", r"<\|\s*assistant\s*\|>?"),
        SpecialTokenPattern("role_system", r"<\|\s*system\s*\|>?"),
        SpecialTokenPattern("role_end", r"<\|\s*end\s*\|>?"),
        # Gemma-style turn delimiters
        SpecialTokenPattern("gemma_start_of_turn", r"<\s*start_of_turn\s*>"),
        SpecialTokenPattern("gemma_end_of_turn", r"<\s*end_of_turn\s*>"),
        # Catch-all for "<|...|>"-style tokens (kept last; conservative max length).
        SpecialTokenPattern("hf_special_token_like", r"<\|[^>\n]{1,64}\|>?"),
    )


_ZERO_WIDTH_CHARS = {
    "\u200b",  # ZWSP
    "\u200c",  # ZWNJ
    "\u200d",  # ZWJ
    "\u2060",  # WORD JOINER
    "\ufeff",  # BOM
}

_BULLET_TRANSLATION = str.maketrans(
    {
        "•": "-",
        "‣": "-",
        "∙": "-",
        "·": "-",
        "●": "-",
        "◦": "-",
        "▪": "-",
        "–": "-",  # en dash
        "—": "-",  # em dash
        "−": "-",  # minus
    }
)

_CODE_FENCE_LINE_RE = re.compile(r"(?m)^\s*```[^\n]*\n?")
_CODE_FENCE_TAIL_RE = re.compile(r"(?m)^\s*```\s*$")
_HTML_TAG_RE = re.compile(r"<[A-Za-z!/][^>]{0,128}>")
_ESCAPED_BYTE_RE = re.compile(r"\\x[0-9A-Fa-f]{2}")
_ESCAPED_UNICODE_RE = re.compile(r"\\u[0-9A-Fa-f]{4}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HSPACE_RE = re.compile(r"[ \t\f\v\u00A0\u1680\u180e\u2000-\u200a\u202f\u205f\u3000]+")
_WS_AROUND_NL_RE = re.compile(r"[ \t]+\n")

# Tokenization for similarity/repetition/coherence checks.
_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

_EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "as",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "them",
    "his",
    "her",
    "their",
    "our",
    "your",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "may",
    "might",
    "must",
    "will",
    "just",
    "so",
    "than",
    "too",
    "very",
}


@dataclass(frozen=True)
class OutputParsingConfig:
    # --- Normalization behavior ---
    unicode_normalization_form: str = "NFC"
    strip_zero_width: bool = True
    normalize_bullets: bool = True
    strip_control_chars: bool = True
    normalize_escaped_whitespace: bool = True  # turn literal "\\n" into "\n"

    # Markup / structural stripping (plain-text tasks)
    expect_plain_text: bool = True
    strip_markdown_code_fences: bool = True
    strip_html_xml_tags: bool = True
    try_extract_text_from_json: bool = True

    # Special tokens stripping
    strip_special_tokens: bool = True
    special_token_patterns: Tuple[SpecialTokenPattern, ...] = field(default_factory=_default_special_token_patterns)

    # Whitespace normalization
    max_consecutive_newlines: int = 2

    # --- Prompt leakage ---
    prompt_ngram_n: int = 5
    prompt_min_token_len: int = 2
    prompt_ignore_stopwords: bool = True
    prompt_leakage_token_coverage_threshold: float = 0.40

    # Optional prompt prefix stripping (best-effort only)
    strip_prompt_prefix: bool = True
    strip_prompt_substrings: bool = True
    prompt_strip_min_chars: int = 40
    prompt_strip_max_removals: int = 3

    # --- Repetition detection ---
    max_consecutive_identical_tokens: int = 10
    unique_trigram_ratio_threshold: float = 0.30

    # Periodic cycling detection (metadata only; not part of decision cascade by default)
    cycle_max_period_tokens: int = 50
    cycle_min_repetitions: int = 3
    cycle_match_ratio_threshold: float = 0.90

    # --- Structural garbage / partial-valid ---
    structural_garbage_stripped_ratio_threshold: float = 0.50
    partial_valid_noise_ratio_threshold: float = 0.30

    # --- Coherence / incoherence ---
    coherence_logprob_threshold: float = -5.0
    coherence_min_chars: int = 80
    coherence_min_words_for_stopword_check: int = 20
    coherence_min_stopword_ratio: float = 0.04
    coherence_require_sentence_boundary: bool = True
    coherence_gibberish_token_no_vowel_ratio_threshold: float = 0.30


@dataclass(frozen=True)
class ClassifiedOutput:
    raw_text: str
    normalized_text: str
    label: OutputQualityLabel
    metadata: Dict[str, Any]


def _strip_zero_width(text: str) -> Tuple[str, int]:
    if not text:
        return text, 0
    removed = 0
    out_chars: List[str] = []
    for ch in text:
        if ch in _ZERO_WIDTH_CHARS:
            removed += 1
            continue
        out_chars.append(ch)
    return "".join(out_chars), removed


def _normalize_whitespace(text: str, *, max_newlines: int) -> str:
    if not text:
        return ""
    # Normalize newline styles first.
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse horizontal whitespace (including many unicode spaces) to ASCII space.
    t = _HSPACE_RE.sub(" ", t)
    # Remove trailing horizontal whitespace before newline to stabilize normalization.
    t = _WS_AROUND_NL_RE.sub("\n", t)
    # Collapse more than N newlines.
    if max_newlines >= 0:
        t = _MULTI_NL_RE.sub("\n" * max(1, int(max_newlines)), t)
    # Collapse multiple spaces (but keep newlines).
    t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def _strip_special_tokens(text: str, patterns: Sequence[SpecialTokenPattern]) -> Tuple[str, List[str], int]:
    if not text:
        return "", [], 0
    found: List[str] = []
    removed_chars = 0
    t = text
    for p in patterns:
        rx = re.compile(p.pattern, p.flags)
        matches = list(rx.finditer(t))
        if not matches:
            continue
        found.append(p.name)
        removed_chars += sum(len(m.group(0)) for m in matches)
        # Replace with space to avoid accidental token concatenation.
        t = rx.sub(" ", t)
    return t, found, removed_chars


def _strip_markdown_and_html(text: str, cfg: OutputParsingConfig) -> Tuple[str, int]:
    if not text:
        return "", 0
    removed = 0
    t = text
    if cfg.strip_markdown_code_fences:
        before = t
        t = _CODE_FENCE_LINE_RE.sub("", t)
        t = _CODE_FENCE_TAIL_RE.sub("", t)
        removed += max(0, len(before) - len(t))
    if cfg.strip_html_xml_tags:
        before = t
        t = _HTML_TAG_RE.sub(" ", t)
        removed += max(0, len(before) - len(t))
    return t, removed


def _try_extract_text_from_json(text: str) -> Optional[str]:
    s = (text or "").strip()
    if not s:
        return None
    # Only attempt if it looks like it starts with JSON.
    if not (s.startswith("{") or s.startswith("[")):
        return None

    # Extract first JSON object/array if extra text exists.
    m = re.search(r"[\{\[][\s\S]*[\}\]]", s)
    if not m:
        return None
    candidate = m.group(0)

    try:
        obj = json.loads(candidate)
    except Exception:
        return None

    # Heuristics: pull common "final answer" keys if a dict.
    if isinstance(obj, dict):
        for key in ("final", "answer", "response", "content", "result", "output", "text", "message"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                nested = val.get("content") or val.get("text") or val.get("answer")
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
        # Otherwise, return first non-empty string value.
        for val in obj.values():
            if isinstance(val, str) and val.strip():
                return val.strip()
    elif isinstance(obj, list):
        # Return the first non-empty string element.
        for el in obj:
            if isinstance(el, str) and el.strip():
                return el.strip()
    return None


def _detect_encoding_issues(text: str) -> Tuple[bool, float]:
    """
    Best-effort mojibake / encoding artifact detector.

    Returns (encoding_issues_detected, score in [0,1]).
    """
    if not text:
        return False, 0.0
    t = text
    # Strong signal: replacement char or control chars.
    if "\ufffd" in t:
        return True, 1.0
    if _CONTROL_CHAR_RE.search(t):
        return True, 1.0

    # Heuristic: common mojibake sequences from UTF-8 interpreted as Latin-1/Windows-1252.
    markers = [
        "Ã",  # e.g. "Ã©"
        "Â",  # e.g. "Â "
        "â€™",
        "â€œ",
        "â€\u009d",  # sometimes appears as two chars
        "â€“",
        "â€”",
        "ï¿½",
        "ðŸ",  # emoji mojibake
    ]
    hits = 0
    for m in markers:
        hits += t.count(m)
    # Normalize by length to a rough score.
    score = min(1.0, (hits * 8.0) / max(40.0, float(len(t))))
    return (score >= 0.25), score


def _tokenize_for_similarity(text: str) -> List[str]:
    # Conservative tokenization for similarity and repetition checks.
    # Keep only "word-ish" tokens to avoid punctuation dominating.
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _filter_prompt_tokens(tokens: Sequence[str], cfg: OutputParsingConfig) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if len(tok) < int(cfg.prompt_min_token_len):
            continue
        if cfg.prompt_ignore_stopwords and tok in _EN_STOPWORDS:
            continue
        out.append(tok)
    return out


def _prompt_leakage_coverage(
    *,
    output_text: str,
    prompt_text: str,
    cfg: OutputParsingConfig,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute token-coverage of output tokens that appear in prompt n-grams.

    Coverage is the fraction of output tokens that are part of at least one matched
    token n-gram (n = cfg.prompt_ngram_n).
    """
    out_tokens_raw = _tokenize_for_similarity(output_text)
    prompt_tokens_raw = _tokenize_for_similarity(prompt_text)
    out_tokens = _filter_prompt_tokens(out_tokens_raw, cfg)
    prompt_tokens = _filter_prompt_tokens(prompt_tokens_raw, cfg)

    if not out_tokens or not prompt_tokens:
        return 0.0, {
            "output_tokens": len(out_tokens_raw),
            "prompt_tokens": len(prompt_tokens_raw),
            "ngram_n": int(cfg.prompt_ngram_n),
            "matched_tokens": 0,
        }

    n = int(cfg.prompt_ngram_n)
    if n <= 1 or len(prompt_tokens) < n or len(out_tokens) < n:
        # Fallback: unigram overlap coverage over filtered tokens.
        prompt_set = set(prompt_tokens)
        matched = sum(1 for t in out_tokens if t in prompt_set)
        cov = matched / max(1, len(out_tokens))
        return float(cov), {
            "output_tokens": len(out_tokens),
            "prompt_tokens": len(prompt_tokens),
            "ngram_n": 1,
            "matched_tokens": matched,
        }

    prompt_ngrams = {tuple(prompt_tokens[i : i + n]) for i in range(0, len(prompt_tokens) - n + 1)}
    matched_mask = [False] * len(out_tokens)
    matched_ngrams = 0
    for i in range(0, len(out_tokens) - n + 1):
        ng = tuple(out_tokens[i : i + n])
        if ng in prompt_ngrams:
            matched_ngrams += 1
            for j in range(i, i + n):
                matched_mask[j] = True
    matched_tokens = int(sum(1 for x in matched_mask if x))
    coverage = matched_tokens / max(1, len(out_tokens))
    return float(coverage), {
        "output_tokens": len(out_tokens),
        "prompt_tokens": len(prompt_tokens),
        "ngram_n": n,
        "matched_tokens": matched_tokens,
        "matched_ngrams": matched_ngrams,
    }


def _max_consecutive_identical(tokens: Sequence[str]) -> int:
    if not tokens:
        return 0
    best = 1
    run = 1
    prev = tokens[0]
    for tok in tokens[1:]:
        if tok == prev:
            run += 1
            best = max(best, run)
        else:
            prev = tok
            run = 1
    return best


def _unique_ngram_ratio(tokens: Sequence[str], n: int) -> float:
    if n <= 0:
        return 1.0
    if len(tokens) < n:
        return 1.0
    total = len(tokens) - n + 1
    grams = [tuple(tokens[i : i + n]) for i in range(total)]
    uniq = len(set(grams))
    return float(uniq) / float(total) if total > 0 else 1.0


def _detect_token_cycle(tokens: Sequence[str], cfg: OutputParsingConfig) -> Tuple[bool, Optional[int], float]:
    """
    Detect periodic repetition patterns in token sequence.
    Returns (has_cycle, period_tokens, match_ratio).
    """
    L = len(tokens)
    if L < 2:
        return False, None, 0.0
    max_period = min(int(cfg.cycle_max_period_tokens), max(1, L // 2))
    for period in range(1, max_period + 1):
        reps = L / float(period)
        if reps < float(cfg.cycle_min_repetitions):
            continue
        matches = 0
        for i in range(L):
            if tokens[i] == tokens[i % period]:
                matches += 1
        match_ratio = matches / float(L)
        if match_ratio >= float(cfg.cycle_match_ratio_threshold):
            return True, period, float(match_ratio)
    return False, None, 0.0


def _coherence_heuristic(text: str, cfg: OutputParsingConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    Lightweight, deterministic incoherence heuristic.

    Returns (fails_coherence, details).
    """
    t = (text or "").strip()
    if len(t) < int(cfg.coherence_min_chars):
        return False, {"skipped": True, "reason": "too_short"}

    words = [w.lower() for w in _WORD_RE.findall(t)]
    if not words:
        return True, {"skipped": False, "reason": "no_words"}

    stopword_count = sum(1 for w in words if w in _EN_STOPWORDS)
    stopword_ratio = stopword_count / max(1, len(words))

    # "Gibberish" tokens: alphabetic tokens length>=4 with no vowel.
    vowels = set("aeiou")
    gibberish = 0
    long_words = 0
    for w in words:
        if len(w) < 4:
            continue
        long_words += 1
        if not any(ch in vowels for ch in w):
            gibberish += 1
    gibberish_ratio = (gibberish / long_words) if long_words else 0.0

    has_sentence_boundary = bool(re.search(r"[.!?]", t)) or ("\n" in t)

    fails = False
    reasons: List[str] = []
    if (
        len(words) >= int(cfg.coherence_min_words_for_stopword_check)
        and stopword_ratio < float(cfg.coherence_min_stopword_ratio)
    ):
        fails = True
        reasons.append("low_stopword_ratio")
    if bool(cfg.coherence_require_sentence_boundary) and not has_sentence_boundary and len(words) >= 25:
        fails = True
        reasons.append("no_sentence_boundary")
    if gibberish_ratio >= float(cfg.coherence_gibberish_token_no_vowel_ratio_threshold) and long_words >= 10:
        fails = True
        reasons.append("gibberish_tokens")

    return fails, {
        "skipped": False,
        "word_count": len(words),
        "stopword_ratio": float(stopword_ratio),
        "gibberish_ratio": float(gibberish_ratio),
        "has_sentence_boundary": bool(has_sentence_boundary),
        "reasons": reasons,
    }


def _normalize_for_pipeline(raw_text: str, cfg: OutputParsingConfig) -> Tuple[str, Dict[str, Any]]:
    """
    Normalize and strip common debris from raw model output.

    Returns (normalized_text, stats) where stats include per-step stripped counts.
    """
    raw = "" if raw_text is None else str(raw_text)
    stats: Dict[str, Any] = {
        "raw_length": len(raw),
        "stripped": {
            "zero_width_chars": 0,
            "control_chars": 0,
            "escaped_bytes": 0,
            "special_tokens_chars": 0,
            "markup_chars": 0,
            "json_extracted": False,
        },
        "special_tokens_found": [],
    }

    # Unicode normalization
    t = unicodedata.normalize(str(cfg.unicode_normalization_form), raw)

    # Clean explicit escaped sequences that show up in some model outputs.
    if cfg.normalize_escaped_whitespace:
        t = t.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

    # Strip common escaped byte/unicode artifacts, but keep this conservative.
    before = t
    t = _ESCAPED_BYTE_RE.sub(" ", t)
    t = _ESCAPED_UNICODE_RE.sub(" ", t)
    stats["stripped"]["escaped_bytes"] += max(0, len(before) - len(t))

    # Control characters / null bytes
    if cfg.strip_control_chars:
        before = t
        t = _CONTROL_CHAR_RE.sub(" ", t)
        stats["stripped"]["control_chars"] += max(0, len(before) - len(t))

    # Zero-width characters
    if cfg.strip_zero_width:
        t, removed = _strip_zero_width(t)
        stats["stripped"]["zero_width_chars"] += int(removed)

    # Bullet normalization
    if cfg.normalize_bullets:
        t = t.translate(_BULLET_TRANSLATION)

    # Special token stripping
    if cfg.strip_special_tokens:
        t, found, removed = _strip_special_tokens(t, cfg.special_token_patterns)
        stats["special_tokens_found"] = found
        stats["stripped"]["special_tokens_chars"] += int(removed)

    # Markup stripping for plain-text tasks
    if cfg.expect_plain_text:
        t, removed = _strip_markdown_and_html(t, cfg)
        stats["stripped"]["markup_chars"] += int(removed)

        if cfg.try_extract_text_from_json:
            extracted = _try_extract_text_from_json(t)
            if extracted is not None:
                # Extraction is treated as stripping markup-like wrapper.
                stats["stripped"]["json_extracted"] = True
                t = extracted

    # Whitespace normalization (final)
    t = _normalize_whitespace(t, max_newlines=int(cfg.max_consecutive_newlines))
    return t, stats


def _detect_valid_answer_substring(
    normalized_text: str,
    expected_answer_texts: Sequence[str],
) -> bool:
    if not expected_answer_texts:
        return False
    hay = normalized_text.lower()
    for ans in expected_answer_texts:
        if not ans or not str(ans).strip():
            continue
        a = str(ans).strip().lower()
        # Conservative: word-boundary match for short answers; substring for longer.
        if len(a) <= 4 or a.isdigit():
            if re.search(r"(?:^|\\b)" + re.escape(a) + r"(?:\\b|$)", hay):
                return True
        else:
            if a in hay:
                return True
    return False


def _confidence_from_margin(margin: float) -> float:
    # Map margin in [0,1] to a conservative confidence in [0,1].
    return max(0.0, min(1.0, float(margin)))


def _collapse_ws_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Collapse all whitespace to single spaces while keeping a map from each character
    in the collapsed string back to an index in the original string.
    """
    out_chars: List[str] = []
    idx_map: List[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                out_chars.append(" ")
                idx_map.append(i)
                in_space = True
            continue
        out_chars.append(ch)
        idx_map.append(i)
        in_space = False

    # Trim leading/trailing spaces in the collapsed view (keep maps aligned).
    collapsed = "".join(out_chars)
    if not collapsed:
        return "", []
    start = 0
    end = len(collapsed)
    while start < end and collapsed[start] == " ":
        start += 1
    while end > start and collapsed[end - 1] == " ":
        end -= 1
    return collapsed[start:end], idx_map[start:end]


def _strip_prompt_prefix_from_text(text: str, prompt: str) -> Tuple[str, bool]:
    """
    Best-effort prompt prefix stripping using whitespace-collapsed matching.
    """
    t = text or ""
    p = prompt or ""
    if not t.strip() or not p.strip():
        return t.strip(), False

    t_collapsed, t_map = _collapse_ws_with_map(t)
    p_collapsed, _p_map = _collapse_ws_with_map(p)
    if not t_collapsed or not p_collapsed:
        return t.strip(), False
    if not t_collapsed.startswith(p_collapsed):
        return t.strip(), False
    if len(p_collapsed) > len(t_map):
        return t.strip(), False

    cut_orig_idx = t_map[len(p_collapsed) - 1]
    remainder = t[cut_orig_idx + 1 :]
    return remainder.lstrip(), True


def _strip_prompt_substrings_from_text(
    text: str,
    prompt: str,
    *,
    min_prompt_chars: int,
    max_removals: int,
) -> Tuple[str, int]:
    """
    Best-effort removal of prompt substrings anywhere in the text using whitespace-collapsed matching.

    This is deliberately conservative:
    - only runs for prompts above `min_prompt_chars`
    - caps removals to avoid pathological loops
    """
    t = text or ""
    p = prompt or ""
    if not t.strip() or not p.strip():
        return t.strip(), 0
    if len(p) < int(min_prompt_chars):
        return t.strip(), 0

    removed_total = 0
    out = t
    for _ in range(max(0, int(max_removals))):
        out_collapsed, out_map = _collapse_ws_with_map(out)
        p_collapsed, _p_map = _collapse_ws_with_map(p)
        if not out_collapsed or not p_collapsed:
            break
        idx = out_collapsed.find(p_collapsed)
        if idx == -1:
            break
        # Map collapsed indices back to original indices.
        start_orig = out_map[idx]
        end_orig = out_map[idx + len(p_collapsed) - 1]
        removed_total += (end_orig + 1 - start_orig)
        out = (out[:start_orig] + " " + out[end_orig + 1 :]).strip()
    return out.strip(), int(removed_total)


def classify_output(
    *,
    raw_text: str,
    cfg: Optional[OutputParsingConfig] = None,
    system_prompt: str = "",
    user_prompt: str = "",
    expected_answer_texts: Sequence[str] = (),
    token_logprobs: Optional[Sequence[float]] = None,
) -> ClassifiedOutput:
    """
    Normalize and deterministically classify a single model output.

    The decision cascade is evaluated in-order (first match wins):
      1) EMPTY
      2) DEGENERATE_REPETITION
      3) PROMPT_LEAKAGE
      4) STRUCTURAL_GARBAGE
      5) INCOHERENT
      6) PARTIAL_VALID
      7) VALID
    """
    cfg = OutputParsingConfig() if cfg is None else cfg
    normalized_text_pre_strip, norm_stats = _normalize_for_pipeline(raw_text, cfg)

    # Prompt contamination detection uses normalized strings to reduce false mismatches.
    prompt_text = "\n".join([str(system_prompt or "").strip(), str(user_prompt or "").strip()]).strip()
    prompt_coverage, prompt_details = _prompt_leakage_coverage(
        output_text=normalized_text_pre_strip,
        prompt_text=prompt_text,
        cfg=cfg,
    )

    # Repetition metrics
    rep_tokens = _tokenize_for_similarity(normalized_text_pre_strip)
    max_run = _max_consecutive_identical(rep_tokens)
    trigram_ratio = _unique_ngram_ratio(rep_tokens, 3)
    has_cycle, cycle_period, cycle_match = _detect_token_cycle(rep_tokens, cfg)

    encoding_issues, encoding_score = _detect_encoding_issues(str(raw_text or ""))

    raw_len = int(norm_stats.get("raw_length") or 0)
    normalized_text_for_classification = normalized_text_pre_strip
    normalized_text = normalized_text_pre_strip

    # Optional prompt prefix stripping for downstream consumers (does not affect classification metrics).
    prompt_prefix_stripped = False
    prompt_substrings_stripped_chars = 0
    if cfg.strip_prompt_prefix and prompt_text:
        # Try combined prompt first, then individual prompts.
        stripped, ok = _strip_prompt_prefix_from_text(normalized_text, prompt_text)
        if ok:
            normalized_text = stripped
            prompt_prefix_stripped = True
        else:
            stripped, ok = _strip_prompt_prefix_from_text(normalized_text, str(system_prompt or ""))
            if ok:
                normalized_text = stripped
                prompt_prefix_stripped = True
            else:
                stripped, ok = _strip_prompt_prefix_from_text(normalized_text, str(user_prompt or ""))
                if ok:
                    normalized_text = stripped
                    prompt_prefix_stripped = True

    if cfg.strip_prompt_substrings and prompt_text:
        # Remove long prompt blocks that re-appear inside the output.
        stripped, removed = _strip_prompt_substrings_from_text(
            normalized_text,
            prompt_text,
            min_prompt_chars=int(cfg.prompt_strip_min_chars),
            max_removals=int(cfg.prompt_strip_max_removals),
        )
        if removed > 0:
            normalized_text = stripped
            prompt_substrings_stripped_chars += int(removed)
        # Also try individual prompts (some models regurgitate only system or only user prompt).
        for p in (str(system_prompt or ""), str(user_prompt or "")):
            stripped, removed = _strip_prompt_substrings_from_text(
                normalized_text,
                p,
                min_prompt_chars=int(cfg.prompt_strip_min_chars),
                max_removals=int(cfg.prompt_strip_max_removals),
            )
            if removed > 0:
                normalized_text = stripped
                prompt_substrings_stripped_chars += int(removed)

    normalized_len = len(normalized_text)
    total_stripped = max(0, raw_len - normalized_len)

    structural_stripped_chars = int(norm_stats["stripped"]["special_tokens_chars"]) + int(norm_stats["stripped"]["markup_chars"])
    # Encoding artifacts stripped are tracked separately (control/escaped/zero-width); count them too.
    structural_stripped_chars += int(norm_stats["stripped"]["control_chars"]) + int(norm_stats["stripped"]["escaped_bytes"])
    structural_stripped_chars += int(norm_stats["stripped"]["zero_width_chars"])

    structural_stripped_ratio = (structural_stripped_chars / float(raw_len)) if raw_len else 0.0

    valid_answer_found = _detect_valid_answer_substring(normalized_text, expected_answer_texts)

    # Coherence proxy (logprobs if available; else heuristic-only)
    used_logprobs = False
    avg_logprob: Optional[float] = None
    incoherent_logprob = False
    if token_logprobs:
        try:
            vals = [float(x) for x in token_logprobs]
            if vals:
                used_logprobs = True
                avg_logprob = sum(vals) / float(len(vals))
                incoherent_logprob = avg_logprob < float(cfg.coherence_logprob_threshold)
        except Exception:
            used_logprobs = False
            avg_logprob = None
            incoherent_logprob = False

    incoherent_heur, incoherent_details = _coherence_heuristic(normalized_text, cfg)

    # PARTIAL_VALID noise estimate: combine stripped ratio with prompt similarity and encoding signal.
    noise_ratio_estimate = max(structural_stripped_ratio, float(prompt_coverage), float(encoding_score))

    decision_rule = "VALID"
    decision_values: Dict[str, Any] = {}
    label: OutputQualityLabel = OutputQualityLabel.VALID
    confidence = 0.7

    # Decision cascade (first match wins)
    if not normalized_text_for_classification.strip():
        label = OutputQualityLabel.EMPTY
        decision_rule = "EMPTY"
        confidence = 1.0
    elif (trigram_ratio < float(cfg.unique_trigram_ratio_threshold)) or (max_run > int(cfg.max_consecutive_identical_tokens)):
        label = OutputQualityLabel.DEGENERATE_REPETITION
        decision_rule = "DEGENERATE_REPETITION"
        # Confidence: based on whichever signal is stronger.
        trig_margin = (float(cfg.unique_trigram_ratio_threshold) - trigram_ratio) / max(1e-6, float(cfg.unique_trigram_ratio_threshold))
        run_margin = (max_run - int(cfg.max_consecutive_identical_tokens)) / max(1.0, float(cfg.max_consecutive_identical_tokens))
        confidence = _confidence_from_margin(max(trig_margin, run_margin))
        decision_values = {
            "unique_trigram_ratio": float(trigram_ratio),
            "unique_trigram_ratio_threshold": float(cfg.unique_trigram_ratio_threshold),
            "max_consecutive_identical_tokens": int(max_run),
            "max_consecutive_identical_tokens_threshold": int(cfg.max_consecutive_identical_tokens),
        }
    elif prompt_coverage >= float(cfg.prompt_leakage_token_coverage_threshold):
        label = OutputQualityLabel.PROMPT_LEAKAGE
        decision_rule = "PROMPT_LEAKAGE"
        confidence = _confidence_from_margin(
            (prompt_coverage - float(cfg.prompt_leakage_token_coverage_threshold))
            / max(1e-6, 1.0 - float(cfg.prompt_leakage_token_coverage_threshold))
        )
        decision_values = {
            "prompt_similarity_score": float(prompt_coverage),
            "prompt_similarity_threshold": float(cfg.prompt_leakage_token_coverage_threshold),
            **prompt_details,
        }
    elif structural_stripped_ratio > float(cfg.structural_garbage_stripped_ratio_threshold):
        label = OutputQualityLabel.STRUCTURAL_GARBAGE
        decision_rule = "STRUCTURAL_GARBAGE"
        confidence = _confidence_from_margin(
            (structural_stripped_ratio - float(cfg.structural_garbage_stripped_ratio_threshold))
            / max(1e-6, 1.0 - float(cfg.structural_garbage_stripped_ratio_threshold))
        )
        decision_values = {
            "structural_stripped_ratio": float(structural_stripped_ratio),
            "structural_stripped_ratio_threshold": float(cfg.structural_garbage_stripped_ratio_threshold),
            "structural_stripped_chars": int(structural_stripped_chars),
            "raw_length": int(raw_len),
        }
    elif incoherent_logprob or incoherent_heur:
        label = OutputQualityLabel.INCOHERENT
        decision_rule = "INCOHERENT"
        # Confidence: logprob gives stronger signal if present, else heuristic.
        if incoherent_logprob and avg_logprob is not None:
            margin = (float(cfg.coherence_logprob_threshold) - float(avg_logprob)) / max(1e-6, abs(float(cfg.coherence_logprob_threshold)))
            confidence = _confidence_from_margin(margin)
        else:
            confidence = 0.6 if (not incoherent_details.get("skipped", False)) else 0.4
        decision_values = {
            "avg_token_logprob": avg_logprob,
            "logprob_threshold": float(cfg.coherence_logprob_threshold),
            "logprobs_used": bool(used_logprobs),
            "heuristic_details": incoherent_details,
        }
    elif valid_answer_found and noise_ratio_estimate > float(cfg.partial_valid_noise_ratio_threshold):
        label = OutputQualityLabel.PARTIAL_VALID
        decision_rule = "PARTIAL_VALID"
        confidence = _confidence_from_margin(
            (noise_ratio_estimate - float(cfg.partial_valid_noise_ratio_threshold))
            / max(1e-6, 1.0 - float(cfg.partial_valid_noise_ratio_threshold))
        )
        decision_values = {
            "valid_answer_found": True,
            "noise_ratio_estimate": float(noise_ratio_estimate),
            "noise_ratio_threshold": float(cfg.partial_valid_noise_ratio_threshold),
        }
    else:
        label = OutputQualityLabel.VALID
        decision_rule = "VALID"
        confidence = 0.8 if not encoding_issues else 0.6

    metadata: Dict[str, Any] = {
        # Required keys
        "raw_length": int(raw_len),
        "normalized_length": int(normalized_len),
        "chars_stripped_count": int(total_stripped),
        "repetition_score": float(trigram_ratio),
        "prompt_similarity_score": float(prompt_coverage),
        "special_tokens_found": list(norm_stats.get("special_tokens_found") or []),
        "encoding_issues_detected": bool(encoding_issues),
        "classification_confidence": float(max(0.0, min(1.0, float(confidence)))),
        # Extra traceability/debug keys
        "decision_rule": decision_rule,
        "decision_values": decision_values,
        "structural_stripped_ratio": float(structural_stripped_ratio),
        "max_consecutive_identical_tokens": int(max_run),
        "token_cycle_detected": bool(has_cycle),
        "token_cycle_period": cycle_period,
        "token_cycle_match_ratio": float(cycle_match),
        "encoding_issue_score": float(encoding_score),
        "logprobs_used": bool(used_logprobs),
        "avg_token_logprob": avg_logprob,
        "coherence_heuristic": incoherent_details,
        "valid_answer_found": bool(valid_answer_found),
        "prompt_prefix_stripped": bool(prompt_prefix_stripped) if prompt_text else None,
        "prompt_substrings_stripped_chars": int(prompt_substrings_stripped_chars) if prompt_text else None,
    }

    logger.debug("Output classified: %s rule=%s values=%s", label.value, decision_rule, decision_values)
    return ClassifiedOutput(raw_text=str(raw_text or ""), normalized_text=normalized_text, label=label, metadata=metadata)
