"""Download the FULL Dolci post-training corpora for the OLMo-3 7B lineage.

Datasets and on-disk sizes (parquet, uncompressed row counts vary):
  allenai/Dolci-Instruct-SFT      3.06 GB   (SFT stage, Instruct branch)
  allenai/Dolci-Instruct-DPO      0.81 GB   (DPO stage, Instruct branch)
  allenai/Dolci-Instruct-RL       0.48 GB   (RL stage,  Instruct branch)
  allenai/Dolci-Think-SFT-7B     36.14 GB   (SFT stage, Think branch)  <-- large
  allenai/Dolci-Think-DPO-7B      1.39 GB   (DPO stage, Think branch)
  allenai/Dolci-Think-RL-7B       1.89 GB   (RL stage,  Think branch)
  ---------------------------------------
  TOTAL                          ~43.8 GB

Uses huggingface_hub.snapshot_download so partial downloads resume cleanly.

USAGE (from the repo root):
    cd dataset_analysis
    # optional but recommended for speed and rate limits:
    export HF_TOKEN=hf_xxx
    python scripts/download_full.py                    # all six
    python scripts/download_full.py --skip think-sft   # skip the 36 GB one
    python scripts/download_full.py --only instruct-sft instruct-dpo

Each dataset is snapshot-downloaded into data/raw/<short_name>/ as parquet
shards. The phase scripts (phase1_sft_audit.py etc.) then iterate those
parquet files directly instead of the 2k-row streaming JSONL that the smoke
test used.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# short_name -> (repo_id, approx_gb)
DATASETS: dict[str, tuple[str, float]] = {
    "instruct-sft":  ("allenai/Dolci-Instruct-SFT",  3.06),
    "instruct-dpo":  ("allenai/Dolci-Instruct-DPO",  0.81),
    "instruct-rl":   ("allenai/Dolci-Instruct-RL",   0.48),
    "think-sft":     ("allenai/Dolci-Think-SFT-7B", 36.14),
    "think-dpo":     ("allenai/Dolci-Think-DPO-7B",  1.39),
    "think-rl":      ("allenai/Dolci-Think-RL-7B",   1.89),
}


def human_gb(gb: float) -> str:
    return f"{gb:.2f} GB"


def pull(short: str, repo: str) -> Path:
    target = RAW / short
    target.mkdir(parents=True, exist_ok=True)
    print(f"\n[{short}] -> {repo}")
    print(f"[{short}] destination: {target}")
    t0 = time.time()
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=["*.parquet", "*.json", "README.md"],
        max_workers=8,
        token=os.environ.get("HF_TOKEN"),
    )
    dt = time.time() - t0
    total = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    print(f"[{short}] done in {dt/60:.1f} min  on-disk={total/1e9:.2f} GB")
    return Path(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--only", nargs="+", choices=list(DATASETS),
                   help="download only these short names")
    p.add_argument("--skip", nargs="+", choices=list(DATASETS), default=[],
                   help="skip these short names")
    p.add_argument("--dry-run", action="store_true",
                   help="print plan and exit")
    args = p.parse_args(argv)

    names = args.only or list(DATASETS)
    names = [n for n in names if n not in args.skip]
    total_gb = sum(DATASETS[n][1] for n in names)

    print("Planned downloads:")
    for n in names:
        repo, gb = DATASETS[n]
        print(f"  {n:14s} {repo:40s} {human_gb(gb)}")
    print(f"  {'TOTAL':14s} {'':40s} {human_gb(total_gb)}")
    if not os.environ.get("HF_TOKEN"):
        print("\nNOTE: HF_TOKEN is not set. Public datasets still work but with "
              "lower rate limits. `export HF_TOKEN=hf_xxx` recommended.")
    if args.dry_run:
        return

    failures = []
    for n in names:
        repo, _ = DATASETS[n]
        try:
            pull(n, repo)
        except Exception as e:
            print(f"[{n}] FAILED: {e}", file=sys.stderr)
            failures.append((n, str(e)))

    print("\n================ summary ================")
    for n in names:
        target = RAW / n
        size = sum(pf.stat().st_size for pf in target.rglob("*") if pf.is_file())
        print(f"  {n:14s} {size/1e9:6.2f} GB  {target}")
    if failures:
        print("\nFailures:")
        for n, e in failures:
            print(f"  {n}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
