"""Shared utilities for Dolci dataset audits.

All phase scripts reuse these helpers so the metric definitions are computed
identically across SFT / DPO / Think corpora.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
for p in (DATA, RAW, RESULTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)


def iter_rows(short_name: str, jsonl_fallback: str | None = None, limit: int | None = None):
    """Yield dict rows from either the full parquet snapshot (data/raw/<name>/)
    or the streaming JSONL smoke-test sample (data/<jsonl_fallback>.jsonl).

    Uses pyarrow for parquet to avoid loading whole shards into memory.
    """
    import json as _json
    raw_dir = RAW / short_name
    parquets = sorted(raw_dir.rglob("*.parquet")) if raw_dir.exists() else []
    n = 0
    if parquets:
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise RuntimeError(
                "pyarrow is required for the full-parquet path; "
                "pip install pyarrow"
            ) from e
        for shard in parquets:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    yield row
                    n += 1
                    if limit and n >= limit:
                        return
        return
    # JSONL fallback (smoke-test data)
    if jsonl_fallback:
        p = DATA / f"{jsonl_fallback}.jsonl"
        if p.exists():
            with p.open() as f:
                for line in f:
                    try:
                        yield _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    n += 1
                    if limit and n >= limit:
                        return
            return
    raise FileNotFoundError(
        f"no data found for '{short_name}'. Run scripts/download_full.py "
        f"(full parquet at data/raw/{short_name}/) or scripts/download_samples.py "
        f"(streaming JSONL at data/{jsonl_fallback}.jsonl)."
    )

# ---------- Structural token extraction ----------
# These are the "format fingerprints" an induction head can latch onto.
BULLET_RE = re.compile(r"(?m)^\s*[-*•]\s+")
NUM_LIST_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+")
CODEFENCE_RE = re.compile(r"```")
COLON_LINE_RE = re.compile(r"(?m)^[A-Za-z][^\n:]{0,40}:\s")  # "Participant 1:"
NEWLINE_RE = re.compile(r"\n")


def structural_fingerprint(text: str) -> dict:
    """Return a multiset of structural tokens present in `text`."""
    if not text:
        return {"bullet": 0, "numbered": 0, "heading": 0, "codefence": 0,
                "colon_line": 0, "newline": 0, "len": 0}
    return {
        "bullet": len(BULLET_RE.findall(text)),
        "numbered": len(NUM_LIST_RE.findall(text)),
        "heading": len(HEADING_RE.findall(text)),
        "codefence": len(CODEFENCE_RE.findall(text)),
        "colon_line": len(COLON_LINE_RE.findall(text)),
        "newline": len(NEWLINE_RE.findall(text)),
        "len": len(text),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def structural_jaccard(prompt: str, response: str) -> float:
    """Jaccard similarity over the *presence* of structural token classes."""
    pf = structural_fingerprint(prompt)
    rf = structural_fingerprint(response)
    pk = {k for k, v in pf.items() if k != "len" and v > 0}
    rk = {k for k, v in rf.items() if k != "len" and v > 0}
    return jaccard(pk, rk)


# ---------- Lexical affirmation / correction cues ----------
AFFIRM_PREFIXES = (
    r"^\s*(yes[,!.]|absolutely[,!.]?|certainly[,!.]?|of course[,!.]?|"
    r"sure[,!.]?|i agree|you[' ]re (right|correct)|great question|"
    r"here(?: is| are|\u2019s| is a)|here's)"
)
AFFIRM_RE = re.compile(AFFIRM_PREFIXES, re.IGNORECASE)

CORRECTION_RE = re.compile(
    r"\b(actually|however|in fact|it'?s worth noting|"
    r"the premise is (?:incorrect|wrong|mistaken)|that's not (?:quite )?right|"
    r"i (?:have to )?disagree|important to note|"
    r"on the contrary|this is (?:a )?(?:misconception|incorrect))\b",
    re.IGNORECASE,
)

SYCOPHANCY_RE = re.compile(
    r"\b(you(?:'re| are) (?:absolutely |completely |totally )?(?:right|correct)|"
    r"that'?s (?:a )?(?:great|excellent|wonderful|fantastic|brilliant) "
    r"(?:question|point|idea|observation)|"
    r"great question|wonderful question|i agree (?:with you|completely)|"
    r"as you (?:correctly |rightly )?(?:said|pointed out|noted))\b",
    re.IGNORECASE,
)


def affirm_prefix(text: str, window: int = 80) -> bool:
    return bool(text) and bool(AFFIRM_RE.search(text[:window]))


def count_hits(regex: re.Pattern, text: str) -> int:
    return 0 if not text else len(regex.findall(text))


# ---------- N-gram repetition (response mirrors prompt) ----------
WORD_RE = re.compile(r"\w+")


def ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def prompt_response_ngram_overlap(prompt: str, response: str, n: int = 4) -> float:
    """Fraction of response n-grams that appear verbatim in the prompt.
    This is the direct fingerprint of an induction-head copy."""
    pt = WORD_RE.findall(prompt.lower())
    rt = WORD_RE.findall(response.lower())
    rn = ngrams(rt, n)
    if not rn:
        return 0.0
    pn = set(ngrams(pt, n))
    hits = sum(1 for g in rn if g in pn)
    return hits / len(rn)


# ---------- IO helpers ----------
def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def summarise(values: Iterable[float]) -> dict:
    vs = [v for v in values if v is not None]
    if not vs:
        return {"n": 0}
    import statistics as st
    return {
        "n": len(vs),
        "mean": st.fmean(vs),
        "median": st.median(vs),
        "stdev": st.pstdev(vs) if len(vs) > 1 else 0.0,
        "min": min(vs),
        "max": max(vs),
    }
