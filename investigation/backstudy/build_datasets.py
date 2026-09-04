#!/usr/bin/env python3
"""
build_datasets.py — canonical trial-level tables from the raw SQLite run DBs.

Outputs (investigation/backstudy/data/):
  olmo_trials.parquet          one row per trial, OLMo-3 7B family (8 variants × 6 temps × 12 conditions)
  crossfamily_trials.parquet   one row per trial, cross-family runs under runs/ (incl. ablations)
  *_coverage.csv               cell counts

Label rules (investigation/README.md): correctness = LLM-judge is_correct in parsed_answer_json (authoritative);
refusal = SQL refusal_flag OR judge refusal_flag; wrong_answer_endorsed = judge field.
4-state outcome: correct → refusal → target_wrong → other_wrong (first match wins).
No DB is modified.
"""
from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "investigation/backstudy/data"
OUT.mkdir(parents=True, exist_ok=True)

OLMO_DBS = {
    0.0: "runs_latest/runs/20260224_225304_9f240f89-e58e-423a-ae68-f990b31c84cd/simulation.db",
    0.2: "runs_latest/runs/20260225_010113_46f0762a-210a-459a-8709-d24a0f194eb0/simulation.db",
    0.4: "runs_latest/runs/20260225_041802_bbd05985-d185-460a-b0aa-dd356d27ec94/simulation.db",
    0.6: "runs_latest/runs/20260225_085516_86c72262-d1aa-41b5-9c22-d7b2e0570215/simulation.db",
    0.8: "runs_latest/runs/20260225_124244_9369442d-d825-4cd0-81a1-8ed276c37814/simulation.db",
    1.0: "runs_latest/runs/20260225_130444_9173bfae-4e8a-464f-8c9c-7ee91caa8b6e/simulation.db",
}

FEATURES = {
    # condition_name: (frame, system, n_wrong_mentions, n_peer_lines, lexical, consensus, tone, continuation_cue)
    "control": ("assistant", "control", 0, 0, "none", "none", "none", 0),
    "asch_history_5": ("prior_users", "warning", 5, 5, "identical", "unanimous", "confident", 0),
    "asch_zhu_unbiased_unanimous_plain": ("participant_role", "control", 5, 5, "identical", "unanimous", "plain", 1),
    "asch_zhu_unbiased_unanimous_neutral": ("participant_role", "control", 5, 5, "varied", "unanimous", "neutral", 1),
    "asch_zhu_unbiased_unanimous_confident": ("participant_role", "control", 5, 5, "varied", "unanimous", "confident", 1),
    "asch_zhu_unbiased_unanimous_uncertain": ("participant_role", "control", 5, 5, "varied", "unanimous", "uncertain", 1),
    "asch_zhu_unbiased_diverse_plain": ("participant_role", "control", 1, 5, "identical", "none", "plain", 1),
    "asch_zhu_unbiased_da": ("participant_role", "control", 4, 5, "identical", "majority", "plain", 1),
    "asch_zhu_unbiased_qd": ("participant_role", "control", 1, 0, "none", "unanimous_stated", "none", 1),
    "authoritative_bias": ("assistant", "warning", 1, 0, "none", "single", "pretty_sure_confirm", 0),
    "authority_zhu_unbiased_trust": ("assistant", "control", 1, 0, "none", "single", "trust", 0),
    "authority_zhu_unbiased_trust_da": ("assistant", "control", 1, 0, "none", "two_options", "trust", 0),
    # cross-family names
    "asch_zhu_unanimous_confident": ("participant_role", "control", 5, 5, "varied", "unanimous", "confident", 1),
    "authority_trust": ("assistant", "control", 1, 0, "none", "single", "trust", 0),
    "asch_zhu_naked_unanimous_confident": ("participant_role", "none", 5, 5, "varied", "unanimous", "confident", 1),
    "ngram_sequence_baseline": ("sequence", "none", 5, 5, "identical", "unanimous", "none", 1),
    "ngram_sequence_matched_baseline": ("sequence", "none", 5, 5, "identical", "unanimous", "none", 1),
}
FEAT_COLS = ["frame", "system_prompt_type", "n_wrong_mentions", "n_peer_lines", "lexical_identity", "consensus_type", "tone", "continuation_cue"]

SQL = """
SELECT t.trial_id, t.run_id, t.model_id, t.variant, t.item_id, t.temperature, t.seed,
       c.name AS condition_name, c.params_json,
       i.dataset_id, i.domain, i.ground_truth_text,
       json_extract(i.source_json,'$.wrong_answer') AS wrong_answer,
       o.raw_text, o.parsed_answer_text, o.is_correct AS sql_is_correct, o.refusal_flag AS sql_refusal_flag,
       json_extract(o.parsed_answer_json,'$.is_correct') AS judge_is_correct,
       json_extract(o.parsed_answer_json,'$.wrong_answer_endorsed') AS judge_wrong_endorsed,
       json_extract(o.parsed_answer_json,'$.refusal_flag') AS judge_refusal,
       json_extract(o.parsed_answer_json,'$._llm_judge.judge_model') AS judge_model,
       o.created_at AS output_created_at,
       m.metadata_json
FROM conformity_trials t
JOIN conformity_conditions c ON c.condition_id = t.condition_id
JOIN conformity_items i ON i.item_id = t.item_id
JOIN conformity_outputs o ON o.trial_id = t.trial_id
LEFT JOIN conformity_trial_metadata m ON m.trial_id = t.trial_id
"""


def outcome(r) -> str:
    if r.judge_is_correct == 1:
        return "correct"
    if (r.sql_refusal_flag == 1) or (r.judge_refusal == 1):
        return "refusal"
    if r.judge_wrong_endorsed == 1:
        return "target_wrong"
    return "other_wrong"


def load_db(path: Path, run_dir: str) -> pd.DataFrame:
    con = sqlite3.connect(path)
    df = pd.read_sql_query(SQL, con)
    con.close()
    df["run_dir"] = run_dir
    # keep latest output per trial (gpt_oss run has duplicates)
    df = df.sort_values("output_created_at").drop_duplicates("trial_id", keep="last")
    for c in ["judge_is_correct", "judge_wrong_endorsed", "judge_refusal", "sql_refusal_flag", "sql_is_correct"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["outcome"] = df.apply(outcome, axis=1)
    df["outcome_judge_only"] = df.apply(lambda r: "correct" if r.judge_is_correct == 1 else ("refusal" if r.judge_refusal == 1 else ("target_wrong" if r.judge_wrong_endorsed == 1 else "other_wrong")), axis=1)
    meta = df["metadata_json"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    df["meta_tone"] = meta.apply(lambda m: m.get("tone"))
    df["meta_alternate_answer"] = meta.apply(lambda m: m.get("alternate_answer"))
    df["meta_confederate_lines"] = meta.apply(lambda m: json.dumps(m.get("confederate_utterances") or m.get("confederate_lines") or []))
    df = df.drop(columns=["metadata_json"])
    df["raw_len"] = df["raw_text"].fillna("").str.len()
    df["raw_text"] = df["raw_text"].fillna("").str.slice(0, 1500)
    feats = df["condition_name"].map(lambda n: FEATURES.get(n))
    for j, col in enumerate(FEAT_COLS):
        df[col] = feats.apply(lambda f: f[j] if f else None)
    return df


def main() -> None:
    pub = pd.read_csv(REPO / "Comparing_Experiments/publication_V2/item_set.csv")
    pub_ids = set(pub.item_id.astype(str))
    ds_name = dict(zip(pub.item_id.astype(str), pub.dataset_name))

    # --- OLMo family
    parts = []
    for temp, rel in OLMO_DBS.items():
        p = REPO / rel
        d = load_db(p, rel.split("/")[-2])
        d["temperature"] = float(temp)
        parts.append(d)
        print(f"  loaded {rel.split('/')[-2]} T={temp}: {len(d)} trials", flush=True)
    olmo = pd.concat(parts, ignore_index=True)
    olmo["in_pub_set"] = olmo.item_id.astype(str).isin(pub_ids)
    olmo["dataset"] = olmo.item_id.astype(str).map(ds_name).fillna(olmo.dataset_id)
    olmo.to_parquet(OUT / "olmo_trials.parquet", index=False)
    cov = olmo[olmo.in_pub_set].groupby(["variant", "temperature", "condition_name"]).size().rename("n").reset_index()
    cov.to_csv(OUT / "olmo_coverage.csv", index=False)
    print(f"OLMo: {len(olmo)} trials ({olmo.in_pub_set.sum()} in pub set); judge_is_correct null: {olmo.judge_is_correct.isna().sum()}")
    print(olmo[olmo.in_pub_set].pivot_table(index="variant", columns="temperature", values="trial_id", aggfunc="count"))

    # --- cross-family
    parts = []
    for p in sorted(glob.glob(str(REPO / "runs/**/simulation.db"), recursive=True)):
        run_dir = Path(p).parent.name
        d = load_db(Path(p), run_dir)
        parts.append(d)
        print(f"  loaded runs/{run_dir}: {len(d)} trials, variants={sorted(d.variant.unique())}, conds={sorted(d.condition_name.unique())}", flush=True)
    cross = pd.concat(parts, ignore_index=True)
    cross["in_pub_set"] = cross.item_id.astype(str).isin(pub_ids)
    cross["dataset"] = cross.item_id.astype(str).map(ds_name).fillna(cross.dataset_id)
    cross.to_parquet(OUT / "crossfamily_trials.parquet", index=False)
    cov = cross.groupby(["variant", "temperature", "condition_name"]).size().rename("n").reset_index()
    cov.to_csv(OUT / "crossfamily_coverage.csv", index=False)
    print(f"Cross-family: {len(cross)} trials; judge null: {cross.judge_is_correct.isna().sum()}")
    print(cov.to_string())


if __name__ == "__main__":
    main()
