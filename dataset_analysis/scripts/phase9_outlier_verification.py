"""Phase 9: extract top-1% outliers from phase5/phase6 results to MD/CSV
for manual qualitative review. Reviewer gives ground truth on whether the
audit metrics are flagging the right kind of content.

Inputs:
  results/phase5_instruct-sft_per_example.csv
  results/phase6_instruct-dpo_per_pair.csv

Outputs: results/phase9_outliers/
  sft_top1pct_max_run.md             (100 random samples from top 1% by max_run_response)
  sft_top1pct_struct_jaccard.md
  sft_top1pct_consensus_hits.md
  dpo_top1pct_delta_struct_jaccard.md  (rejected − chosen, both texts shown)
  dpo_top1pct_delta_max_run.md
  dpo_top1pct_delta_correction.md

Each MD: one section per row with idx, source_dataset, metric value,
truncated text (first 800 chars), and a checkbox `- [ ] this is a true positive`.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
RESULTS = ROOT / "results"
SFT_CSV = RESULTS / "phase5_instruct-sft_per_example.csv"
DPO_CSV = RESULTS / "phase6_instruct-dpo_per_pair.csv"
OUT_DIR = RESULTS / "phase9_outliers"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEXT_LIMIT = 800      # chars to include per text field in the MD
SAMPLE_N = 100        # deterministic sample size
SAMPLE_SEED = 0       # reproducible sampling

SFT_METRICS = [
    "max_run_response",
    "structural_jaccard",
    "consensus_hits_response",
]
DPO_METRICS = [
    "delta_struct_jaccard",
    "delta_max_run",
    "delta_correction",
]


# ---------------------------------------------------------------------------
# Step 1: top-1% selection
# ---------------------------------------------------------------------------

def select_top_1pct(
    csv_path: Path,
    metric_col: str,
) -> tuple[list[dict], float, int]:
    """Return (sampled_rows, threshold, pool_size).

    sampled_rows: up to SAMPLE_N deterministically sampled rows (seed=0)
                  from the top-1% pool.
    threshold:    the 99th-percentile value (all ties at this value included).
    pool_size:    number of rows at or above the threshold.
    """
    # Pass 1: read the metric column and find the 99th-percentile threshold.
    # We use a streaming approach to avoid loading all text columns.
    values: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get(metric_col, "")
            if raw in ("", "nan", "None"):
                continue
            try:
                values.append(float(raw))
            except ValueError:
                continue

    if not values:
        raise ValueError(f"No valid values found for metric '{metric_col}' in {csv_path}")

    # Sort to find 99th-percentile threshold (inclusive ties)
    values_sorted = sorted(values)
    n = len(values_sorted)
    # numpy-style: index = ceil(p * n) - 1, clamped
    p99_idx = max(0, int(0.99 * n) - 1)
    threshold = values_sorted[p99_idx]

    # Pass 2: collect all rows at or above threshold (with full row data).
    pool: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw = row.get(metric_col, "")
            if raw in ("", "nan", "None"):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if val >= threshold:
                pool.append(row)

    pool_size = len(pool)

    # Deterministic sample
    rng = random.Random(SAMPLE_SEED)
    sampled = rng.sample(pool, min(SAMPLE_N, pool_size))
    # Sort by idx for stable output ordering
    sampled.sort(key=lambda r: int(r.get("idx", 0)))

    return sampled, threshold, pool_size


# ---------------------------------------------------------------------------
# Step 2: fetch original text from parquet
# ---------------------------------------------------------------------------

def _build_idx_set(rows: list[dict]) -> set[int]:
    return {int(r["idx"]) for r in rows}


def fetch_sft_texts(idx_set: set[int]) -> dict[int, tuple[str, str]]:
    """Stream instruct-sft parquet rows; return {idx: (prompt, response)}."""
    # Import here to keep dependencies localised
    sys.path.insert(0, str(SCRIPTS_DIR))
    from common import iter_rows
    from audit_metrics import extract_pair_with_meta

    result: dict[int, tuple[str, str]] = {}
    remaining = set(idx_set)

    for enum_idx, row in enumerate(iter_rows("instruct-sft")):
        if enum_idx in remaining:
            extracted = extract_pair_with_meta(row)
            if extracted is not None:
                prompt, response, _meta = extracted
            else:
                prompt, response = "", ""
            result[enum_idx] = (prompt, response)
            remaining.discard(enum_idx)
            if not remaining:
                break

    # Fill missing (malformed rows) with empty strings
    for idx in idx_set:
        if idx not in result:
            result[idx] = ("", "")

    return result


def fetch_dpo_texts(idx_set: set[int]) -> dict[int, tuple[str, str, str]]:
    """Stream instruct-dpo parquet rows; return {idx: (prompt, chosen, rejected)}."""
    sys.path.insert(0, str(SCRIPTS_DIR))
    from common import iter_rows
    from audit_metrics import extract_triplet_with_meta

    result: dict[int, tuple[str, str, str]] = {}
    remaining = set(idx_set)

    for enum_idx, row in enumerate(iter_rows("instruct-dpo")):
        if enum_idx in remaining:
            extracted = extract_triplet_with_meta(row)
            if extracted is not None:
                prompt, chosen, rejected, _meta = extracted
            else:
                prompt, chosen, rejected = "", "", ""
            result[enum_idx] = (prompt, chosen, rejected)
            remaining.discard(enum_idx)
            if not remaining:
                break

    for idx in idx_set:
        if idx not in result:
            result[idx] = ("", "", "")

    return result


# ---------------------------------------------------------------------------
# Step 3: write Markdown
# ---------------------------------------------------------------------------

def _trunc(text: str, limit: int = TEXT_LIMIT) -> str:
    """Truncate to first `limit` characters; append ellipsis if cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def write_sft_md(
    out_path: Path,
    metric_name: str,
    sampled_rows: list[dict],
    texts: dict[int, tuple[str, str]],
    threshold: float,
    pool_size: int,
    total_rows: int,
    source_csv: str,
) -> None:
    """Write a Markdown outlier-review file for an SFT metric."""
    lines: list[str] = []

    lines.append(f"# phase9 outlier review: {metric_name}")
    lines.append("")
    lines.append(f"Source: {source_csv}")
    lines.append(
        f"Selection: top 1% by {metric_name}, sampled {len(sampled_rows)} deterministic "
        f"rows (seed={SAMPLE_SEED})"
    )
    lines.append(
        f"Tie handling: all rows with {metric_name} ≥ 99th-percentile threshold "
        f"({threshold}) were included in the pool ({pool_size:,} of {total_rows:,} rows = "
        f"{pool_size/total_rows*100:.2f}%); then {min(SAMPLE_N, pool_size)} sampled."
    )
    lines.append("")
    lines.append(
        "Reviewer instructions: tick `- [x]` for each row where the flagged content "
        "matches the audit's intent.  "
    )
    # Metric-specific guidance
    _sft_guidance = {
        "max_run_response": (
            "For `max_run_response` this should be a true word-repetition pattern in "
            "the response (e.g., a model repeating the same word many times in a row), "
            "NOT a list where a word like 'the' recurs non-consecutively."
        ),
        "structural_jaccard": (
            "For `structural_jaccard` this should be a case where the response visibly "
            "mirrors the prompt's structural format (both have bullets, both have "
            "numbered lists, etc.), consistent with an induction-head formatting prior."
        ),
        "consensus_hits_response": (
            "For `consensus_hits_response` this should be a response that explicitly "
            "frames a claim as group-level consensus or majority agreement (e.g., "
            "'all participants agreed', 'the consensus holds')."
        ),
    }
    lines.append(_sft_guidance.get(metric_name, ""))
    lines.append("")
    lines.append("---")
    lines.append("")

    for row_num, csv_row in enumerate(sampled_rows, start=1):
        idx = int(csv_row["idx"])
        source_dataset = csv_row.get("source_dataset", "unknown")
        metric_val = csv_row.get(metric_name, "")
        prompt, response = texts.get(idx, ("", ""))

        lines.append(f"## Row {row_num} — idx={idx}, source_dataset={source_dataset}, {metric_name}={metric_val}")
        lines.append("- [ ] true positive")
        lines.append("")
        lines.append("**Prompt (first 800 chars):**")
        lines.append("```")
        lines.append(_trunc(prompt))
        lines.append("```")
        lines.append("")
        lines.append("**Response (first 800 chars):**")
        lines.append("```")
        lines.append(_trunc(response))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path} ({len(sampled_rows)} rows)")


def write_dpo_md(
    out_path: Path,
    metric_name: str,
    sampled_rows: list[dict],
    texts: dict[int, tuple[str, str, str]],
    threshold: float,
    pool_size: int,
    total_rows: int,
    source_csv: str,
) -> None:
    """Write a Markdown outlier-review file for a DPO delta metric."""
    lines: list[str] = []

    lines.append(f"# phase9 outlier review: {metric_name}")
    lines.append("")
    lines.append(f"Source: {source_csv}")
    lines.append(
        f"Selection: top 1% by {metric_name}, sampled {len(sampled_rows)} deterministic "
        f"rows (seed={SAMPLE_SEED})"
    )
    lines.append(
        f"Tie handling: all rows with {metric_name} ≥ 99th-percentile threshold "
        f"({threshold}) were included in the pool ({pool_size:,} of {total_rows:,} rows = "
        f"{pool_size/total_rows*100:.2f}%); then {min(SAMPLE_N, pool_size)} sampled."
    )
    lines.append("")
    lines.append(
        "Reviewer instructions: tick `- [x]` for each row where the flagged content "
        "matches the audit's intent.  "
    )
    _dpo_guidance = {
        "delta_struct_jaccard": (
            "For `delta_struct_jaccard` (rejected − chosen): the rejected response "
            "should visibly mirror the prompt's structural format more than the chosen "
            "response does (e.g., rejected uses the same bullet style as the prompt, "
            "while chosen diverges)."
        ),
        "delta_max_run": (
            "For `delta_max_run` (rejected − chosen): the rejected response should "
            "contain a longer run of repeated identical word-tokens than the chosen "
            "response."
        ),
        "delta_correction": (
            "For `delta_correction` (rejected − chosen): the chosen response should "
            "contain more correction/factual-pushback phrases than the rejected "
            "response (delta = rejected_correction − chosen_correction, so a large "
            "positive value means rejected has more correction hits than chosen — "
            "which is unexpected and worth flagging)."
        ),
    }
    lines.append(_dpo_guidance.get(metric_name, ""))
    lines.append("")
    lines.append("---")
    lines.append("")

    for row_num, csv_row in enumerate(sampled_rows, start=1):
        idx = int(csv_row["idx"])
        source_dataset = csv_row.get("source_dataset", "unknown")
        chosen_model = csv_row.get("chosen_model", "unknown")
        rejected_model = csv_row.get("rejected_model", "unknown")
        preference_type = csv_row.get("preference_type", "unknown")
        metric_val = csv_row.get(metric_name, "")
        prompt, chosen, rejected = texts.get(idx, ("", "", ""))

        lines.append(
            f"## Row {row_num} — idx={idx}, source_dataset={source_dataset}, "
            f"{metric_name}={metric_val}"
        )
        lines.append(
            f"- chosen_model: {chosen_model}  |  rejected_model: {rejected_model}  "
            f"|  preference_type: {preference_type}"
        )
        lines.append("- [ ] true positive")
        lines.append("")
        lines.append("**Prompt (first 800 chars):**")
        lines.append("```")
        lines.append(_trunc(prompt))
        lines.append("```")
        lines.append("")
        lines.append("**Chosen response (first 800 chars):**")
        lines.append("```")
        lines.append(_trunc(chosen))
        lines.append("```")
        lines.append("")
        lines.append("**Rejected response (first 800 chars):**")
        lines.append("```")
        lines.append(_trunc(rejected))
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path} ({len(sampled_rows)} rows)")


# ---------------------------------------------------------------------------
# Step 4: main orchestration
# ---------------------------------------------------------------------------

def _count_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        return sum(1 for _ in reader)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- SFT metrics ----
    print("=== SFT outlier extraction ===")
    sft_total = _count_rows(SFT_CSV)
    print(f"  SFT total rows: {sft_total:,}")

    sft_metric_samples: dict[str, list[dict]] = {}
    sft_thresholds: dict[str, tuple[float, int]] = {}

    for metric in SFT_METRICS:
        print(f"  Selecting top 1% by {metric}...")
        sampled, threshold, pool_size = select_top_1pct(SFT_CSV, metric)
        sft_metric_samples[metric] = sampled
        sft_thresholds[metric] = (threshold, pool_size)
        print(f"    threshold={threshold}, pool={pool_size:,}, sampled={len(sampled)}")

    # Collect all SFT idx needed across all metrics
    all_sft_idx: set[int] = set()
    for rows in sft_metric_samples.values():
        all_sft_idx.update(_build_idx_set(rows))

    print(f"  Fetching text for {len(all_sft_idx)} unique SFT rows from parquet...")
    sft_texts = fetch_sft_texts(all_sft_idx)
    print(f"  Fetched {len(sft_texts)} SFT text entries.")

    # Write SFT MDs
    sft_md_names = {
        "max_run_response": "sft_top1pct_max_run.md",
        "structural_jaccard": "sft_top1pct_struct_jaccard.md",
        "consensus_hits_response": "sft_top1pct_consensus_hits.md",
    }
    for metric in SFT_METRICS:
        out_path = OUT_DIR / sft_md_names[metric]
        threshold, pool_size = sft_thresholds[metric]
        write_sft_md(
            out_path=out_path,
            metric_name=metric,
            sampled_rows=sft_metric_samples[metric],
            texts=sft_texts,
            threshold=threshold,
            pool_size=pool_size,
            total_rows=sft_total,
            source_csv=SFT_CSV.name,
        )

    # ---- DPO metrics ----
    print("\n=== DPO outlier extraction ===")
    dpo_total = _count_rows(DPO_CSV)
    print(f"  DPO total rows: {dpo_total:,}")

    dpo_metric_samples: dict[str, list[dict]] = {}
    dpo_thresholds: dict[str, tuple[float, int]] = {}

    for metric in DPO_METRICS:
        print(f"  Selecting top 1% by {metric}...")
        sampled, threshold, pool_size = select_top_1pct(DPO_CSV, metric)
        dpo_metric_samples[metric] = sampled
        dpo_thresholds[metric] = (threshold, pool_size)
        print(f"    threshold={threshold}, pool={pool_size:,}, sampled={len(sampled)}")

    all_dpo_idx: set[int] = set()
    for rows in dpo_metric_samples.values():
        all_dpo_idx.update(_build_idx_set(rows))

    print(f"  Fetching text for {len(all_dpo_idx)} unique DPO rows from parquet...")
    dpo_texts = fetch_dpo_texts(all_dpo_idx)
    print(f"  Fetched {len(dpo_texts)} DPO text entries.")

    # Write DPO MDs
    dpo_md_names = {
        "delta_struct_jaccard": "dpo_top1pct_delta_struct_jaccard.md",
        "delta_max_run": "dpo_top1pct_delta_max_run.md",
        "delta_correction": "dpo_top1pct_delta_correction.md",
    }
    for metric in DPO_METRICS:
        out_path = OUT_DIR / dpo_md_names[metric]
        threshold, pool_size = dpo_thresholds[metric]
        write_dpo_md(
            out_path=out_path,
            metric_name=metric,
            sampled_rows=dpo_metric_samples[metric],
            texts=dpo_texts,
            threshold=threshold,
            pool_size=pool_size,
            total_rows=dpo_total,
            source_csv=DPO_CSV.name,
        )

    print("\nDone. Output files:")
    for f in sorted(OUT_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
