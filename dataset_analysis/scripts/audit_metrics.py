"""Pre-registered audit metrics for the Dolci dataset audit.

PRE-REGISTRATION NOTICE
-----------------------
All regex patterns, thresholds, and domain mappings in this module were
registered BEFORE any phase-level analysis ran on the full corpus (Task 2
of the audit plan).  Tasks 3, 5, and 7 import from here directly.

To change any pattern or threshold after Task 4 has run on the full SFT
corpus you MUST add an "amendment" comment that documents:
  - what changed
  - why (new evidence, methodological error, clarification)
  - who approved it

Do NOT modify patterns retroactively without that comment.

Exports
-------
CONSENSUS_RE            Multi-phrase regex for consensus/majority framing.
PEER_FRAME_RE           Regex for "Participant N:" style peer-attribution.
max_run                 Max contiguous run of identical word-tokens.
repeat_run_geq_k        Binary indicator: max_run >= k.
multi_turn_agreement_score  Agreement drift score over a message list.
canonical_test_domain   Map raw test-domain strings to {math, science,
                        history, general, preference, unmapped}.
canonical_sft_domain    Map Dolci SFT `domain` field to canonical labels.
extract_pair_with_meta  Wraps phase1_sft_audit.extract_pair + metadata.
extract_triplet_with_meta  Wraps phase2_dpo_audit.extract_triplet + metadata.
SOURCE_DATASET_TO_BER_DOMAIN  Placeholder dict (filled in Task 4).
"""
from __future__ import annotations

import re
from typing import Optional

from common import WORD_RE, AFFIRM_RE, prompt_response_ngram_overlap, count_hits
from phase1_sft_audit import extract_pair
from phase2_dpo_audit import extract_triplet


# ---------------------------------------------------------------------------
# CONSENSUS_RE  —  PRE-REGISTERED, do not edit after Task 4 runs
# ---------------------------------------------------------------------------
# Matches phrases that frame a claim as group-level consensus or majority
# agreement.  Intentionally does NOT match bare "I agree" or "we agree"
# (those are individual-level; see test_consensus_re_negative).
#
# Design choices:
#   - "all * agreed/agreed": universal-agreement frame
#   - "consensus (is|holds|shows)": explicit consensus label
#   - "everyone (said|agreed|thinks|believes)": universal-agreement with verb
#   - "the group decided": collective decision frame
#   - "majority of * (preferred|agreed|said|found|reported|indicated)": majority frame
#   - "peer consensus": explicit peer-group label
#   - Does NOT match "I agree", "we agree to disagree", standalone "agree"
CONSENSUS_RE = re.compile(
    r"""
    \b(
        all \s+ \w+ \s+ agreed               # "all participants agreed"
      | all \s+ agreed                        # "all agreed"
      | (?:the \s+)? consensus \s+ (?:is|holds|shows|was|suggests)
                                              # "the consensus is/holds"
      | peer \s+ consensus                    # "peer consensus"
      | everyone \s+ (?:said|agreed|thinks|believes|felt|concluded|stated)
                                              # "everyone said"
      | the \s+ group \s+ (?:decided|agreed|concluded|found|determined)
                                              # "the group decided"
      | (?:the \s+)? majority \s+ of \s+ \w+
        \s+ (?:preferred|agreed|said|found|reported|indicated|chose|selected)
                                              # "majority of respondents preferred"
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# PEER_FRAME_RE  —  PRE-REGISTERED
# ---------------------------------------------------------------------------
# Matches lines that attribute a statement to a numbered peer/participant,
# e.g. "Participant 1:", "Reviewer 3.", "User 7)"
PEER_FRAME_RE = re.compile(
    r"\b(?:participant|peer|respondent|user|voter|reviewer|annotator|rater)"
    r"\s+\d+\s*[:.\)]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Run-length metrics
# ---------------------------------------------------------------------------

def max_run(text: str) -> int:
    """Return the maximum contiguous run of identical word-tokens (lowercased).

    Uses WORD_RE from common to tokenize (consistent with other metrics).
    Returns 0 for empty / no-word input.
    """
    tokens = WORD_RE.findall(text.lower())
    if not tokens:
        return 0
    best = 1
    current = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
    return best


def repeat_run_geq_k(text: str, k: int) -> bool:
    """Return True iff max_run(text) >= k."""
    return max_run(text) >= k


# ---------------------------------------------------------------------------
# Multi-turn agreement score
# ---------------------------------------------------------------------------

def multi_turn_agreement_score(messages: list[dict]) -> float:
    """Score how much each non-first assistant turn agrees with the prior context.

    Score is computed over `len(asst_turns) - 1` non-first assistant turns,
    or returns 0.0 if there are fewer than 2 assistant turns or fewer than 4
    total messages.

    Algorithm
    ---------
    Requires >= 4 messages with >= 2 assistant turns; returns 0.0 otherwise.

    For each non-first assistant turn t (index >= 1 among assistant turns):
      Signal 1 — opens_with_affirmation: AFFIRM_RE matches first 80 chars.
      Signal 2 — ngram_echo: prompt_response_ngram_overlap(prior_user, t, n=4) > 0.3.
      Signal 3 — rare_token_echo: any word in prior_user (len >= 6, not in t-1
                 assistant text) appears in t (models echo the user's rare tokens).
      score_t = 1 if any signal fires, else 0.

    Returns mean(score_t) over all scored turns.
    """
    asst_turns: list[tuple[int, str]] = []   # (msg_index, content)
    user_turns: list[tuple[int, str]] = []

    for i, m in enumerate(messages):
        role = m.get("role", "")
        content = m.get("content", "") or ""
        if role == "assistant":
            asst_turns.append((i, content))
        elif role == "user":
            user_turns.append((i, content))

    # Need at least 4 messages total and at least 2 assistant turns
    if len(messages) < 4 or len(asst_turns) < 2:
        return 0.0

    scores = []
    for turn_idx, (msg_i, asst_text) in enumerate(asst_turns[1:], start=1):
        # Find the most-recent user message before this assistant turn
        prior_user = ""
        for u_i, u_text in reversed(user_turns):
            if u_i < msg_i:
                prior_user = u_text
                break

        # Find the previous assistant turn text
        prev_asst_text = asst_turns[turn_idx - 1][1]

        # Signal 1: affirmation prefix
        opens_affirm = bool(AFFIRM_RE.search(asst_text[:80]))

        # Signal 2: 4-gram overlap with prior user message
        ngram_echo = (
            prompt_response_ngram_overlap(prior_user, asst_text, n=4) > 0.3
            if prior_user else False
        )

        # Signal 3: rare-token echo from prior user
        # "Rare" = appears in prior_user but NOT in previous assistant turn
        rare_echo = False
        if prior_user:
            user_words = set(w for w in WORD_RE.findall(prior_user.lower()) if len(w) >= 6)  # rare-token min length: 6 (pre-registered, do not mutate without amendment comment)
            prev_words = set(WORD_RE.findall(prev_asst_text.lower()))
            novel_user_words = user_words - prev_words
            asst_words = set(WORD_RE.findall(asst_text.lower()))
            rare_echo = bool(novel_user_words & asst_words)

        scores.append(1 if (opens_affirm or ngram_echo or rare_echo) else 0)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Domain canonicalization
# ---------------------------------------------------------------------------

# Test-domain patterns (MMLU-style raw strings → canonical bucket)
_TEST_DOMAIN_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"math|algebra|calculus|arithmetic|geometry|statistics", re.I), "math"),
    (re.compile(r"physics|chemistry|biology|science|astronomy|geology|ecology", re.I), "science"),
    (re.compile(r"history|geography|politics|economics|social|civics|philosophy|law|"
                r"sociology|psychology|culture|religion", re.I), "history"),
    (re.compile(r"^general$", re.I), "general"),
    (re.compile(r"preference", re.I), "preference"),
]


def canonical_test_domain(raw: str) -> str:
    """Map a raw test-domain string to a canonical bucket.

    Canonical values: math, science, history, general, preference, unmapped.
    """
    for pattern, label in _TEST_DOMAIN_RULES:
        if pattern.search(raw):
            return label
    return "unmapped"


# SFT domain mapping — maps Dolci SFT `domain` field values exactly.
# IMPORTANT: "Other" and "Coding" and "Safety" map to "unmapped".
# Never silently coerce "Other" to "general" — that hides data about
# what fraction of the corpus is genuinely un-categorised.
_SFT_DOMAIN_MAP: dict[str, str] = {
    "math": "math",
    "Math": "math",
    "science": "science",
    "Science": "science",
    "chat": "general",
    "Chat": "general",
    # Intentionally unmapped — do not promote to 'general':
    "coding": "unmapped",
    "Coding": "unmapped",
    "safety": "unmapped",
    "Safety": "unmapped",
    "precise if": "unmapped",
    "Precise IF": "unmapped",
    "multilingual": "unmapped",
    "Multilingual": "unmapped",
    "other": "unmapped",
    "Other": "unmapped",
}


def canonical_sft_domain(raw: str) -> str:
    """Map a Dolci SFT `domain` field value to a canonical bucket.

    Canonical values: math, science, general, unmapped.
    "Other", "Coding", "Safety", "Precise IF", "Multilingual" → "unmapped".
    Case-insensitive lookup via normalised key.
    """
    if raw in _SFT_DOMAIN_MAP:
        return _SFT_DOMAIN_MAP[raw]
    # Fallback: case-insensitive normalised lookup
    normalised = raw.strip().lower()
    for k, v in _SFT_DOMAIN_MAP.items():
        if k.lower() == normalised:
            return v
    return "unmapped"


# ---------------------------------------------------------------------------
# extract_pair_with_meta
# ---------------------------------------------------------------------------

def extract_pair_with_meta(
    row: dict,
) -> Optional[tuple[str, str, dict]]:
    """Wrap phase1_sft_audit.extract_pair and augment with metadata.

    Returns (user_text, asst_text, meta) or None if the row is malformed.

    meta keys:
      source_dataset   — row["source_dataset"] or None
      domain_raw       — row["domain"] or None
      domain_canonical — canonical_sft_domain(domain_raw) or "unmapped"
    """
    pair = extract_pair(row)
    if pair is None:
        return None
    user, asst = pair
    domain_raw = row.get("domain")
    meta = {
        "source_dataset": row.get("source_dataset"),
        "domain_raw": domain_raw,
        "domain_canonical": canonical_sft_domain(domain_raw) if domain_raw is not None else "unmapped",
    }
    return user, asst, meta


# ---------------------------------------------------------------------------
# extract_triplet_with_meta
# ---------------------------------------------------------------------------
# DEVIATION FROM PLAN TEMPLATE (Task 1 spike finding):
# The plan assumed DPO rows would carry a `source_dataset` field (as SFT rows
# do). Task 1 verified that real Dolci-Instruct-DPO rows do NOT have this
# field. The available provenance fields are: chosen_model, rejected_model,
# preference_type, prompt_id. We set meta["source_dataset"] = None and surface
# the model/preference fields instead. Do not pretend source_dataset exists.

def extract_triplet_with_meta(
    row: dict,
) -> Optional[tuple[str, str, str, dict]]:
    """Wrap phase2_dpo_audit.extract_triplet and augment with metadata.

    Returns (prompt, chosen_text, rejected_text, meta) or None if malformed.

    meta keys:
      source_dataset   — always None (not present on Dolci-Instruct-DPO rows;
                         see deviation note above)
      chosen_model     — row["chosen_model"] or None
      rejected_model   — row["rejected_model"] or None
      preference_type  — row["preference_type"] or None
      prompt_id        — row["prompt_id"] or None
      domain_canonical — "unmapped" (DPO rows carry no domain field)
    """
    triplet = extract_triplet(row)
    if triplet is None:
        return None
    prompt, chosen, rejected = triplet
    meta = {
        # Intentionally None — real DPO rows have no source_dataset field.
        "source_dataset": None,
        "chosen_model": row.get("chosen_model"),
        "rejected_model": row.get("rejected_model"),
        "preference_type": row.get("preference_type"),
        "prompt_id": row.get("prompt_id"),
        "domain_canonical": "unmapped",
    }
    return prompt, chosen, rejected, meta


# ---------------------------------------------------------------------------
# Placeholder: filled in Task 4
# ---------------------------------------------------------------------------

SOURCE_DATASET_TO_BER_DOMAIN: dict[str, str] = {}
