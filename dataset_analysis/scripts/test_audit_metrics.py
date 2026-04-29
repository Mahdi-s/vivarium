"""Tests for audit_metrics.py — pre-registered metrics for Dolci data audit.

These tests were written BEFORE the implementation (TDD).
Regex patterns and thresholds must not change after Task 4 runs on the full
corpus without an "amendment" comment documenting the deviation.
"""
import pytest
from audit_metrics import (
    CONSENSUS_RE, PEER_FRAME_RE, max_run, repeat_run_geq_k,
    multi_turn_agreement_score, canonical_test_domain, canonical_sft_domain,
    extract_pair_with_meta, extract_triplet_with_meta,
)
from common import count_hits


# ---------------------------------------------------------------------------
# max_run / repeat_run_geq_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("a a a b", 3),
    ("X. X. X. X. X.", 5),
    ("normal text without any repeats", 1),
    ("", 0),
    ("yes yes yes yes yes yes yes", 7),
])
def test_max_run(text, expected):
    assert max_run(text) == expected


def test_repeat_run_geq_k():
    assert repeat_run_geq_k("a a a b", 3) is True
    assert repeat_run_geq_k("a a a b", 5) is False
    assert repeat_run_geq_k("", 3) is False


# ---------------------------------------------------------------------------
# CONSENSUS_RE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "All participants agreed that the answer is X.",
    "The consensus is that B is correct.",
    "Everyone said the answer was 42.",
    "The group decided that option C is best.",
    "The majority of respondents preferred A.",
    "Peer consensus holds that this is the right path.",
])
def test_consensus_re_positive(text):
    assert count_hits(CONSENSUS_RE, text) >= 1


@pytest.mark.parametrize("text", [
    "I agree with you.",
    "We agree to disagree.",
    "Everyone has different opinions.",
    "I think the answer is X.",
])
def test_consensus_re_negative(text):
    assert count_hits(CONSENSUS_RE, text) == 0


# ---------------------------------------------------------------------------
# PEER_FRAME_RE
# ---------------------------------------------------------------------------

def test_peer_frame_re():
    s = "Participant 1: A\nParticipant 2: A\nParticipant 3: A"
    assert len(PEER_FRAME_RE.findall(s)) == 3


def test_peer_frame_re_variants():
    texts = [
        "Reviewer 1: good",
        "Annotator 2. looks right",
        "User 3) agreed",
        "Rater 10: rated 5",
    ]
    for t in texts:
        assert len(PEER_FRAME_RE.findall(t)) >= 1, f"No match for: {t!r}"


# ---------------------------------------------------------------------------
# multi_turn_agreement_score
# ---------------------------------------------------------------------------

def test_multi_turn_agreement_basic():
    msgs = [
        {"role": "user", "content": "Is the answer A or B?"},
        {"role": "assistant", "content": "I'd say A."},
        {"role": "user", "content": "I think it's A. What about you?"},
        {"role": "assistant", "content": "Yes, you're right, A is correct."},
    ]
    score = multi_turn_agreement_score(msgs)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5


def test_multi_turn_agreement_short_conv():
    """Conversations with fewer than 4 messages → 0.0."""
    msgs = [
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hi."},
    ]
    assert multi_turn_agreement_score(msgs) == 0.0


def test_multi_turn_agreement_no_affirmation():
    msgs = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The result is 4."},
        {"role": "user", "content": "Really?"},
        {"role": "assistant", "content": "It can be shown algebraically that 2+2=4."},
    ]
    score = multi_turn_agreement_score(msgs)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# canonical_test_domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("High School Mathematics", "math"),
    ("High School Geography", "history"),
    ("High School Physics", "science"),
    ("general", "general"),
    ("preference_xyz", "preference"),
    ("absolute garbage string", "unmapped"),
])
def test_canonical_test_domain(raw, expected):
    assert canonical_test_domain(raw) == expected


# ---------------------------------------------------------------------------
# canonical_sft_domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Math", "math"),
    ("Science", "science"),
    ("Coding", "unmapped"),
    ("Chat", "general"),
    ("Other", "unmapped"),
])
def test_canonical_sft_domain(raw, expected):
    assert canonical_sft_domain(raw) == expected


def test_canonical_sft_domain_safety_unmapped():
    """Safety is not silently coerced to 'general'."""
    assert canonical_sft_domain("Safety") == "unmapped"


# ---------------------------------------------------------------------------
# extract_pair_with_meta
# ---------------------------------------------------------------------------

def test_extract_pair_with_meta():
    row = {
        "messages": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ],
        "source_dataset": "WildChat",
        "domain": "Chat",
    }
    out = extract_pair_with_meta(row)
    assert out is not None
    user, asst, meta = out
    assert user == "Q?" and asst == "A."
    assert meta["source_dataset"] == "WildChat"
    assert meta["domain_canonical"] == "general"   # Chat → general


def test_extract_pair_with_meta_missing_domain():
    row = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    }
    out = extract_pair_with_meta(row)
    assert out is not None
    _, _, meta = out
    assert meta["domain_raw"] is None
    assert meta["domain_canonical"] == "unmapped"


def test_extract_pair_with_meta_malformed():
    row = {"messages": []}
    assert extract_pair_with_meta(row) is None


# ---------------------------------------------------------------------------
# extract_triplet_with_meta
# ---------------------------------------------------------------------------

def test_extract_triplet_with_meta_dpo():
    """Real Dolci-Instruct-DPO schema: NO source_dataset field.
    Available keys: chosen, chosen_model, preference_type, prompt_id,
    rejected, rejected_model."""
    row = {
        "prompt_id": "abc",
        "chosen": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "good"},
        ],
        "rejected": [
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "bad"},
        ],
        "chosen_model": "gpt-4o-2024-08-06",
        "rejected_model": "Llama-3-8B",
        "preference_type": "delta_learning",
    }
    out = extract_triplet_with_meta(row)
    assert out is not None
    prompt, chosen, rejected, meta = out
    assert chosen == "good" and rejected == "bad"
    assert meta["chosen_model"] == "gpt-4o-2024-08-06"
    assert meta["rejected_model"] == "Llama-3-8B"
    assert meta["preference_type"] == "delta_learning"
    assert meta["prompt_id"] == "abc"
    # source_dataset absent on real DPO rows; meta must not pretend otherwise
    assert meta.get("source_dataset") in (None, "unknown")


def test_extract_triplet_with_meta_malformed():
    row = {"chosen": [], "rejected": []}
    assert extract_triplet_with_meta(row) is None
