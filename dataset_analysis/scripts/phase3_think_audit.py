"""Phase 3: correlate <think> buffer length with truth-rescue rate.

Data source: the project's own simulation.db files under runs/think/*/simulation.db.
Each trial stores:
  raw_text          -- full generation including the <think> ... </think> block
  parsed_answer_json -- structured judgement with is_correct / wrong_answer_endorsed

Hypothesis: a longer reasoning buffer mechanically severs the prompt-side
induction loop, lowering the wrong-answer endorsement rate under Asch/N-gram
pressure. We regress endorsement (0/1) on think-block token length, and
bucket by pressure condition.

Outputs:
  results/phase3_think_trials.csv
  results/phase3_think_summary.json
  results/phase3_think_scatter.png   (token length vs endorsement rate bucket)
"""
from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics as st
from pathlib import Path

from common import RESULTS, write_json

WORKSPACE = Path("/sessions/inspiring-gallant-lamport/mnt/abstractAgentMachine")
RUNS_DIRS = [WORKSPACE / "runs" / "think", WORKSPACE / "runs-think-hpc"]

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
WORD_RE = re.compile(r"\w+|[^\s\w]")  # cheap tokeniser proxy


def think_token_length(raw: str) -> int:
    m = THINK_RE.search(raw or "")
    if not m:
        # fallback: everything up to "</think>" even if the opening tag is implicit
        idx = (raw or "").find("</think>")
        if idx == -1:
            return 0
        return len(WORD_RE.findall(raw[:idx]))
    return len(WORD_RE.findall(m.group(1)))


def discover_dbs() -> list[Path]:
    dbs = []
    for root in RUNS_DIRS:
        if not root.exists():
            continue
        dbs.extend(root.rglob("simulation.db"))
    return sorted(set(dbs))


def load_condition_labels(conn: sqlite3.Connection) -> dict[str, str]:
    """Map condition_id -> human-readable pressure type."""
    try:
        rows = conn.execute(
            "select condition_id, name, pressure_type from conformity_conditions"
        ).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = conn.execute(
                "select condition_id, name from conformity_conditions"
            ).fetchall()
            rows = [(cid, n, n) for cid, n in rows]
        except sqlite3.OperationalError:
            return {}
    out = {}
    for cid, name, ptype in rows:
        label = (ptype or name or "unknown").lower()
        # normalise
        if "asch" in label or "group" in label or "participant" in label:
            label = "asch"
        elif "author" in label:
            label = "authoritative"
        elif "ngram" in label or "n-gram" in label or "string" in label or "base" in label:
            label = "ngram"
        out[cid] = label
    return out


def main():
    dbs = discover_dbs()
    if not dbs:
        raise SystemExit(f"no simulation.db found under {RUNS_DIRS}")

    out_csv = RESULTS / "phase3_think_trials.csv"
    header = ["db", "trial_id", "model_id", "variant", "condition",
              "think_tokens", "is_correct", "wrong_endorsed", "refusal"]
    per_condition = {}
    all_rows = []
    with out_csv.open("w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(header)
        for db in dbs:
            try:
                conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            except sqlite3.OperationalError:
                continue
            cond_map = load_condition_labels(conn)
            q = """
              select t.trial_id, t.model_id, t.variant, t.condition_id,
                     o.raw_text, o.parsed_answer_json, o.is_correct, o.refusal_flag
              from conformity_trials t
              join conformity_outputs o on o.trial_id = t.trial_id
            """
            try:
                rows = conn.execute(q).fetchall()
            except sqlite3.OperationalError:
                conn.close()
                continue
            for trial_id, model_id, variant, cond_id, raw, pj, is_correct, refusal in rows:
                cond = cond_map.get(cond_id, "unknown")
                tk = think_token_length(raw or "")
                endorsed = 0
                try:
                    pj_obj = json.loads(pj) if pj else {}
                    endorsed = int(pj_obj.get("wrong_answer_endorsed", 0) or 0)
                except (json.JSONDecodeError, TypeError):
                    endorsed = 0
                w.writerow([db.name, trial_id, model_id, variant, cond,
                            tk, int(is_correct or 0), endorsed,
                            int(refusal or 0)])
                all_rows.append((cond, tk, endorsed, int(is_correct or 0)))
                per_condition.setdefault(cond, []).append((tk, endorsed, int(is_correct or 0)))
            conn.close()

    # Aggregate: bucket by think_tokens decile per condition.
    def bucket_stats(rows):
        if not rows:
            return []
        rows = sorted(rows, key=lambda r: r[0])
        n = len(rows)
        bsize = max(1, n // 10)
        buckets = []
        for i in range(0, n, bsize):
            chunk = rows[i:i + bsize]
            tks = [r[0] for r in chunk]
            endorse = [r[1] for r in chunk]
            correct = [r[2] for r in chunk]
            buckets.append({
                "bucket_index": len(buckets),
                "n": len(chunk),
                "think_tokens_mean": st.fmean(tks),
                "think_tokens_min": min(tks),
                "think_tokens_max": max(tks),
                "wrong_endorsed_rate": st.fmean(endorse),
                "correct_rate": st.fmean(correct),
            })
        return buckets

    # Pearson-style correlation without scipy
    def corr(xs, ys):
        if len(xs) < 3:
            return None
        mx, my = st.fmean(xs), st.fmean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return None if dx == 0 or dy == 0 else num / (dx * dy)

    summary = {
        "n_dbs": len(dbs),
        "n_trials": len(all_rows),
        "conditions": {},
    }
    for cond, rows in per_condition.items():
        tks = [r[0] for r in rows]
        end = [r[1] for r in rows]
        summary["conditions"][cond] = {
            "n": len(rows),
            "wrong_endorsed_rate_overall": st.fmean(end) if end else 0.0,
            "think_tokens_mean": st.fmean(tks) if tks else 0.0,
            "corr(think_tokens, wrong_endorsed)": corr(tks, end),
            "deciles": bucket_stats(rows),
        }
    write_json(RESULTS / "phase3_think_summary.json", summary)
    print(f"[phase3] {len(all_rows)} trials across {len(dbs)} dbs -> {out_csv}")
    for cond, s in summary["conditions"].items():
        print(f"  {cond:14s} n={s['n']:5d}  endorse={s['wrong_endorsed_rate_overall']:.3f}"
              f"  corr={s['corr(think_tokens, wrong_endorsed)']}")

    # Scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for cond, s in summary["conditions"].items():
            xs = [b["think_tokens_mean"] for b in s["deciles"]]
            ys = [b["wrong_endorsed_rate"] for b in s["deciles"]]
            ax.plot(xs, ys, "o-", label=f"{cond} (n={s['n']})")
        ax.set_xlabel("<think> block length (tokens, decile mean)")
        ax.set_ylabel("Wrong-answer endorsement rate")
        ax.set_title("Think-buffer length vs conformity, by pressure type")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS / "phase3_think_scatter.png", dpi=140)
        print(f"[phase3] plot -> {RESULTS/'phase3_think_scatter.png'}")
    except Exception as e:
        print(f"[phase3] plot skipped: {e}")


if __name__ == "__main__":
    main()
