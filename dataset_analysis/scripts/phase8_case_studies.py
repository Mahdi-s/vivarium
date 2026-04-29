"""Phase 8: TF-IDF nearest-neighbor case studies for SFT-introduced wrong endorsements.

Finds Dolci-Instruct-SFT examples lexically similar to test items where the SFT
checkpoint introduces wrong-answer endorsement absent in Base.

PROMPT TEXT RESOLUTION STRATEGY
---------------------------------
item_set.csv at Comparing_Experiments/v8_publication/item_set.csv is present but
empty (0 rows). The v7_publication_current variant has 400 rows but only provides
`ground_truth_text`, not the full rendered prompt. Rather than silently fabricating
text or running the full prompt renderer (which requires confederate_block parameters
not available without a run trace), we use:
  - item_id prefix to infer dataset (e.g. "arc_*", "gsm8k_*", "mmlu_*")
  - ground_truth_text from v7_publication_current/item_set.csv where available
  - Placeholder: "<prompt text not resolved; resolve via runner.py:render_* or item catalog>"
  for the TF-IDF query when ground_truth_text is insufficient.

The main contribution is the pipeline; a human reviewer can re-run with resolved prompts.

Outputs
-------
  results/phase8_case_studies.json   (all 20 items, 3 NNs each)
  results/phase8_paper_cases.json    (2 hand-picked cases with TODO captions)
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import RESULTS, iter_rows, count_hits
from audit_metrics import (
    CONSENSUS_RE,
    max_run,
    multi_turn_agreement_score,
    canonical_test_domain,
    canonical_sft_domain,
)

PER_ITEM_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "Comparing_Experiments"
    / "April_analysis"
    / "item_level"
    / "per_item_endorsement.csv"
)

ITEM_SET_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "Comparing_Experiments"
    / "v7_publication_current"
    / "item_set.csv"
)


# ---------------------------------------------------------------------------
# Step 1: Load and filter per_item_endorsement.csv
# ---------------------------------------------------------------------------

def load_target_items(n_items: int = 20) -> list[dict]:
    """Return top-n_items items where SFT introduces wrong endorsement absent in Base.

    Filters:
      - variant == "instruct_sft" AND pressure_state == "B_wrong_endorsed"
      - JOIN base rows on same item_id where base.pressure_state != "B_wrong_endorsed"
      - canonical_test_domain in {math, science}
      - top 20 by item_id (alphabetical sort)
    """
    import csv

    sft_rows: dict[str, dict] = {}
    base_rows: dict[str, dict] = {}

    with PER_ITEM_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row["item_id"]
            if row["variant"] == "instruct_sft":
                sft_rows[iid] = row
            elif row["variant"] == "base":
                base_rows[iid] = row

    # Filter: SFT introduced wrong endorsement
    introduced = []
    for iid, sft_row in sft_rows.items():
        if sft_row["pressure_state"] != "B_wrong_endorsed":
            continue
        base_row = base_rows.get(iid)
        if base_row is None:
            continue
        if base_row["pressure_state"] == "B_wrong_endorsed":
            continue  # Base also wrong; not a SFT-introduced case
        domain_raw = sft_row["domain"]
        canon = canonical_test_domain(domain_raw)
        if canon not in ("math", "science"):
            continue
        introduced.append({
            "item_id": iid,
            "domain": canon,
            "domain_raw": domain_raw,
        })

    # Deterministic sort: alphabetical by item_id
    introduced.sort(key=lambda x: x["item_id"])
    return introduced[:n_items]


# ---------------------------------------------------------------------------
# Step 2: Resolve item prompt text
# ---------------------------------------------------------------------------

def load_item_set() -> dict[str, str]:
    """Load item_id -> ground_truth_text from v7 item_set.csv (best available)."""
    import csv
    mapping: dict[str, str] = {}
    if not ITEM_SET_CSV.exists():
        return mapping
    with ITEM_SET_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = row.get("item_id", "").strip()
            gt = row.get("ground_truth_text", "").strip()
            if iid:
                mapping[iid] = gt
    return mapping


def resolve_prompt_text(item_id: str, ground_truth_text: str) -> str:
    """Return best available prompt-text string for TF-IDF query.

    Preference:
    1. ground_truth_text if non-empty AND has >= 10 chars AND is not purely numeric
       (bare numeric answers like "18" or "540" are useless TF-IDF queries and
       produce spurious 1.0-similarity matches to unrelated multilingual rows).
    2. item_id itself as a domain hint (contains dataset name, useful for
       matching to same-dataset SFT examples).
    3. A placeholder noting the item_id so a human can resolve later.

    We do NOT fabricate prompt text. The rendered Asch prompt wraps the item with
    confederate peer pressure; that template requires run-level parameters not
    available here. The ground_truth_text is a useful proxy for the item's
    content domain only when it is a real phrase, not a bare number.
    """
    # Filter out bare numeric answers (gsm8k ground_truth = "18", "540", etc.)
    stripped = ground_truth_text.strip() if ground_truth_text else ""
    is_usable = (
        len(stripped) >= 10
        and not stripped.replace(".", "").replace(",", "").replace("-", "").isdigit()
    )
    if is_usable:
        return stripped
    # Use just the item_id as the query token — it contains the dataset name
    # (e.g. "gsm8k", "arc") which provides a weak domain signal.
    # We return only the item_id (no boilerplate) to avoid spurious vocabulary
    # matches from shared placeholder text across multiple items.
    # Mark with sentinel prefix so callers can detect unresolved prompts.
    return f"__UNRESOLVED__ {item_id}"


# ---------------------------------------------------------------------------
# Step 3: Build TF-IDF index over stratified 200k-row SFT subsample
# ---------------------------------------------------------------------------

STRATA_TARGET = {
    "math": 50_000,
    "science": 50_000,
    "general": 50_000,
    "unmapped": 50_000,
}

TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)


def sample_sft_subsample(seed: int = 0) -> tuple[list[dict], dict[str, int]]:
    """Stream Dolci-Instruct-SFT and collect a stratified subsample.

    Returns (rows, stratum_counts) where rows is a list of dicts with keys:
      idx, source_dataset, domain_canonical, messages, user_text, asst_text
    """
    rng = random.Random(seed)

    # Phase 1: reservoir sample per stratum
    # Use a simple deterministic reservoir: collect all rows per stratum up to
    # target limit, shuffled with seed=0 so sampling is reproducible.
    strata_pools: dict[str, list[dict]] = defaultdict(list)
    strata_done: dict[str, bool] = {k: False for k in STRATA_TARGET}

    print("Streaming SFT data for stratified subsample...", flush=True)
    global_idx = 0
    for row in iter_rows("instruct-sft"):
        global_idx += 1
        if global_idx % 100_000 == 0:
            counts = {k: len(v) for k, v in strata_pools.items()}
            print(f"  streamed {global_idx} rows, stratum sizes: {counts}", flush=True)

        # Check if all strata are full
        if all(strata_done.values()):
            break

        domain_raw = row.get("domain", "") or ""
        canon = canonical_sft_domain(domain_raw)

        if strata_done.get(canon, True):
            continue

        target = STRATA_TARGET[canon]
        pool = strata_pools[canon]

        if len(pool) < target:
            # Extract text
            msgs = row.get("messages") or []
            user_text = ""
            asst_text = ""
            for m in msgs:
                role = m.get("role", "")
                content = m.get("content", "") or ""
                if role == "user" and not user_text:
                    user_text = content
                elif role == "assistant" and not asst_text:
                    asst_text = content
            combined = (user_text + " " + asst_text).strip()
            if not combined:
                continue

            pool.append({
                "idx": global_idx - 1,
                "source_dataset": row.get("source_dataset", ""),
                "domain_canonical": canon,
                "messages": msgs,
                "user_text": user_text,
                "asst_text": asst_text,
                "combined_text": combined,
            })

            if len(pool) >= target:
                strata_done[canon] = True

    # Shuffle each pool with seed and merge
    all_rows = []
    stratum_counts: dict[str, int] = {}
    for canon, pool in strata_pools.items():
        rng.shuffle(pool)
        sampled = pool[:STRATA_TARGET.get(canon, 0)]
        all_rows.extend(sampled)
        stratum_counts[canon] = len(sampled)

    print(f"Subsample: {len(all_rows)} rows. Stratum counts: {stratum_counts}", flush=True)
    return all_rows, stratum_counts


def build_tfidf_index(rows: list[dict]):
    """Build TF-IDF matrix over combined user+assistant text.

    Returns (vectorizer, matrix) where matrix shape is (len(rows), max_features).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"Building TF-IDF index: {len(rows)} docs, max_features={TFIDF_MAX_FEATURES}, "
          f"ngram_range={TFIDF_NGRAM_RANGE}...", flush=True)

    texts = [r["combined_text"] for r in rows]
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF matrix shape: {matrix.shape}", flush=True)
    return vectorizer, matrix


# ---------------------------------------------------------------------------
# Step 4: Query TF-IDF for top-3 NNs per item
# ---------------------------------------------------------------------------

def sha256_prefix(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_nn_audit_metrics(row: dict) -> dict:
    """Recompute audit metrics for a SFT row using audit_metrics functions."""
    msgs = row.get("messages") or []
    asst_text = row.get("asst_text", "")

    nn_max_run = max_run(asst_text)
    nn_consensus_hits = count_hits(CONSENSUS_RE, asst_text)
    nn_multi_turn_agreement = multi_turn_agreement_score(msgs)

    return {
        "nn_max_run": nn_max_run,
        "nn_consensus_hits": nn_consensus_hits,
        "nn_multi_turn_agreement": nn_multi_turn_agreement,
    }


def query_nns(
    query_text: str,
    vectorizer,
    matrix,
    rows: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Find top-k nearest neighbors by cosine similarity on TF-IDF."""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, matrix).flatten()  # shape: (n_docs,)

    # top-k indices (highest similarity first)
    top_indices = np.argpartition(sims, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        row = rows[idx]
        audit = compute_nn_audit_metrics(row)

        user_text = row.get("user_text", "")
        asst_text = row.get("asst_text", "")

        results.append({
            "nn_rank": rank,
            "nn_idx": int(row["idx"]),
            "nn_source_dataset": row.get("source_dataset", ""),
            "nn_domain_canonical": row.get("domain_canonical", ""),
            "nn_cosine_similarity": float(sims[idx]),
            "nn_max_run": audit["nn_max_run"],
            "nn_consensus_hits": audit["nn_consensus_hits"],
            "nn_multi_turn_agreement": audit["nn_multi_turn_agreement"],
            "nn_user_text_excerpt": user_text[:300],
            "nn_assistant_text_excerpt": asst_text[:300],
        })

    return results


# ---------------------------------------------------------------------------
# Step 5: Main pipeline
# ---------------------------------------------------------------------------

def run() -> None:
    print("=== Phase 8: TF-IDF Nearest-Neighbor Case Studies ===", flush=True)

    # --- Step 1: load target items ---
    print("\nStep 1: Loading target items...", flush=True)
    target_items = load_target_items(n_items=20)
    print(f"Target items (SFT-introduced wrong endorsement, math/science): {len(target_items)}")
    for item in target_items:
        print(f"  {item['item_id']} [{item['domain']}]")

    if not target_items:
        print("ERROR: No target items found. Check per_item_endorsement.csv path and contents.")
        sys.exit(1)

    # --- Step 2: resolve prompt text ---
    print("\nStep 2: Resolving prompt text...", flush=True)
    item_set = load_item_set()
    print(f"  item_set loaded: {len(item_set)} entries from v7_publication_current")

    for item in target_items:
        iid = item["item_id"]
        gt = item_set.get(iid, "")
        query_text = resolve_prompt_text(iid, gt)
        item["_tfidf_query"] = query_text  # internal — used for NN query only
        # Store human-readable prompt text in output JSON
        if query_text.startswith("__UNRESOLVED__"):
            item["item_prompt_text"] = (
                f"<prompt text not resolved for {iid}; "
                "resolve via runner.py:render_* or item catalog>"
            )
        else:
            item["item_prompt_text"] = query_text
        item["item_prompt_text_hash"] = sha256_prefix(item["item_prompt_text"])

    # --- Step 3: build TF-IDF index ---
    print("\nStep 3: Building TF-IDF index...", flush=True)
    sft_rows, stratum_counts = sample_sft_subsample(seed=0)

    if not sft_rows:
        print("ERROR: No SFT rows loaded. Check data/raw/instruct-sft/ directory.")
        sys.exit(1)

    vectorizer, matrix = build_tfidf_index(sft_rows)

    # --- Step 4: query NNs for each item ---
    print("\nStep 4: Querying NNs...", flush=True)
    for item in target_items:
        query_text = item.pop("_tfidf_query")  # consume internal field
        nns = query_nns(query_text, vectorizer, matrix, sft_rows, top_k=3)
        item["nearest_neighbors"] = nns
        top_nn = nns[0] if nns else {}
        print(
            f"  {item['item_id']}: top NN = {top_nn.get('nn_source_dataset', '?')!r}, "
            f"sim={top_nn.get('nn_cosine_similarity', 0):.4f}, "
            f"domain={top_nn.get('nn_domain_canonical', '?')}"
        )

    # --- Step 5: save results ---
    print("\nStep 5: Saving results...", flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    output = {
        "items": target_items,
        "subsample_stratification": stratum_counts,
        "tfidf_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
    }

    out_path = RESULTS / "phase8_case_studies.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Saved {len(target_items)} items -> {out_path}")

    # --- Step 6: hand-pick 2 cases for paper ---
    pick_paper_cases(target_items, stratum_counts)

    print("\nDone.", flush=True)


def pick_paper_cases(items: list[dict], stratum_counts: dict) -> None:
    """Pick 1 math + 1 science case for the paper. Save phase8_paper_cases.json.

    Selection criteria:
    - One math item (highest top-NN cosine similarity among math items)
    - One science item (highest top-NN cosine similarity among science items)
    - If only one domain available, take top-2 from that domain.

    Editorial captions are placeholder TODOs for human reviewer.
    """
    math_items = [it for it in items if it["domain"] == "math"]
    science_items = [it for it in items if it["domain"] == "science"]

    def best_by_sim(group: list[dict]) -> Optional[dict]:
        if not group:
            return None
        return max(group, key=lambda it: (
            it["nearest_neighbors"][0]["nn_cosine_similarity"]
            if it["nearest_neighbors"] else 0.0
        ))

    selected = []
    best_math = best_by_sim(math_items)
    best_sci = best_by_sim(science_items)

    if best_math:
        case = dict(best_math)
        case["editorial_caption"] = "TODO: human reviewer caption"
        selected.append(case)

    if best_sci and (not best_math or best_sci["item_id"] != best_math["item_id"]):
        case = dict(best_sci)
        case["editorial_caption"] = "TODO: human reviewer caption"
        selected.append(case)

    # Fill to 2 if needed
    if len(selected) < 2:
        for it in items:
            if it not in selected[:2]:
                case = dict(it)
                case["editorial_caption"] = "TODO: human reviewer caption"
                selected.append(case)
                if len(selected) >= 2:
                    break

    selected = selected[:2]

    paper_output = {
        "items": selected,
        "subsample_stratification": stratum_counts,
        "tfidf_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
    }

    paper_path = RESULTS / "phase8_paper_cases.json"
    paper_path.write_text(json.dumps(paper_output, indent=2, default=str))
    print(f"Saved {len(selected)} paper cases -> {paper_path}")
    for case in selected:
        nn0 = case["nearest_neighbors"][0] if case["nearest_neighbors"] else {}
        print(
            f"  Paper case: {case['item_id']} [{case['domain']}], "
            f"top NN sim={nn0.get('nn_cosine_similarity', 0):.4f}, "
            f"dataset={nn0.get('nn_source_dataset', '?')!r}"
        )


if __name__ == "__main__":
    run()
