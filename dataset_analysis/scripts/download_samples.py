"""Stream a bounded sample from each Dolci training corpus and cache to JSONL.

Rationale: the full Dolci corpora are large (multi-GB). For a computational
audit of format mirroring we only need a representative sample (default 20k
examples per split). Streaming avoids downloading the full parquet shards.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import DATA, LOGS

DEFAULT_SAMPLES = {
    "allenai/Dolci-Instruct-SFT": ("sft", 20_000),
    "allenai/Dolci-Instruct-DPO": ("dpo", 20_000),
    "allenai/Dolci-Think-RL-7B": ("think_rl_7b", 5_000),
}


def stream_and_save(repo_id: str, tag: str, n: int) -> Path:
    from datasets import load_dataset
    out = DATA / f"{tag}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        print(f"[skip] {out} already exists ({out.stat().st_size/1e6:.1f} MB)")
        return out
    print(f"[fetch] streaming {n} rows from {repo_id}")
    ds = load_dataset(repo_id, split="train", streaming=True)
    written = 0
    with out.open("w") as f:
        for row in ds:
            f.write(json.dumps(row, default=str) + "\n")
            written += 1
            if written >= n:
                break
            if written % 2000 == 0:
                print(f"  ... {written}", flush=True)
    print(f"[done] {repo_id} -> {out} ({written} rows)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-sft", type=int, default=DEFAULT_SAMPLES["allenai/Dolci-Instruct-SFT"][1])
    p.add_argument("--n-dpo", type=int, default=DEFAULT_SAMPLES["allenai/Dolci-Instruct-DPO"][1])
    p.add_argument("--n-think", type=int, default=DEFAULT_SAMPLES["allenai/Dolci-Think-RL-7B"][1])
    p.add_argument("--skip-think", action="store_true")
    args = p.parse_args(argv)

    plan = [
        ("allenai/Dolci-Instruct-SFT", "sft", args.n_sft),
        ("allenai/Dolci-Instruct-DPO", "dpo", args.n_dpo),
    ]
    if not args.skip_think:
        plan.append(("allenai/Dolci-Think-RL-7B", "think_rl_7b", args.n_think))

    log = LOGS / "download.log"
    with log.open("a") as lf:
        for repo, tag, n in plan:
            try:
                path = stream_and_save(repo, tag, n)
                lf.write(f"ok\t{repo}\t{path}\t{n}\n")
            except Exception as e:
                print(f"[error] {repo}: {e}", file=sys.stderr)
                lf.write(f"err\t{repo}\t{e}\n")


if __name__ == "__main__":
    main()
