# Dataset Analysis: OLMo 3 Post-Training Audit (7B lineage)

Extension of the COLM 2026 paper (`colm2026_paper`) on prompt-induced
conformity in OLMo-3.

## Thesis

What the field diagnoses as a deep behavioral alignment failure
("sociological conformity" / "sycophancy") in LLMs is actually a shallow
mechanistic artifact: autoregressive sequence-copying via induction heads,
inadvertently weaponized by specific post-training data distributions.

The three experimental pressure conditions isolate this:

| Condition | Semantic Authority | Structural Repetition |
|-----------|:------------------:|:---------------------:|
| Authoritative ("I am sure it's X") | HIGH | ZERO |
| Asch Group ("Participant 1:X ... Participant 5:X") | HIGH | HIGH |
| N-gram Baseline ("String 1:X ... String 5:X") | ZERO | HIGH |

If conformity were social, Authoritative would dominate. It doesn't. The
N-gram and Asch conditions drive failures because they exploit contiguous
token-level repetition that induction heads are mathematically biased to
continue.

## What to run, in order

This is the workflow you run on your own machine. You have storage; the
scripts are prepared here, you execute them locally.

```bash
cd dataset_analysis

# optional but recommended for throughput and HF rate limits
export HF_TOKEN=hf_xxx

# 1. Download everything (~44 GB, resume-safe, parquet only)
python scripts/download_full.py                  # all six
#    or narrow down:
python scripts/download_full.py --only instruct-sft instruct-dpo instruct-rl
python scripts/download_full.py --skip think-sft # skip the 36 GB one
python scripts/download_full.py --dry-run        # just print the plan

# 2. Run every audit phase
bash scripts/run_all_full.sh
```

After `download_full.py` completes, *I want you to actually pull the full
datasets locally*. You have the disk for it, the analysis scripts are
wired to iterate parquet shards directly (see `common.py::iter_rows`),
and everything is resume-safe via `huggingface_hub.snapshot_download`.

## The six Dolci datasets (7B lineage)

Total on-disk footprint ~44 GB. Measured from the HF `siblings` metadata.

| short name | HF repo | size | role |
|---|---|---:|---|
| `instruct-sft` | `allenai/Dolci-Instruct-SFT` | 3.06 GB | SFT stage, Instruct branch |
| `instruct-dpo` | `allenai/Dolci-Instruct-DPO` | 0.81 GB | DPO stage, Instruct branch |
| `instruct-rl`  | `allenai/Dolci-Instruct-RL`  | 0.48 GB | RL (RLVR) stage, Instruct branch |
| `think-sft`    | `allenai/Dolci-Think-SFT-7B` | 36.14 GB | SFT stage, Think branch (the big one) |
| `think-dpo`    | `allenai/Dolci-Think-DPO-7B` | 1.39 GB | DPO stage, Think branch |
| `think-rl`     | `allenai/Dolci-Think-RL-7B`  | 1.89 GB | RL stage, Think branch |

Files land in `data/raw/<short_name>/` as parquet shards plus each repo's
`README.md` / config JSON (so you retain provenance).

## How many rows do you actually need?

You asked whether partial samples would suffice. Quick power analysis
against the metrics this project actually reports:

- **Proportions** (e.g. `P(resp_has_list | prompt_has_list)`, affirmative-
  prefix rate): the 95% CI half-width for a binomial is ≤ 1/√n. For
  ±1% precision you need **n ≈ 10,000**; for ±0.5% you need **n ≈ 40,000**.
- **Paired deltas on DPO** (y_l minus y_w on n-gram overlap etc.): mean
  deltas in your smoke-test were ~0.03–0.05 with stdev ~0.2. A paired
  t-test reaches p<0.001 at **n ≈ 5,000** pairs, but to stratify by
  sub-source (TULU-3, SciRIFF, Persona-IF, ...) you want **n ≈ 50,000+**
  so each stratum is above 1–2k.
- **RL prompts with repeat-frames** (rare-event rate): if the true rate
  is ~1%, you need **n ≈ 100,000** to estimate it within ±0.2%.

Practical recommendation:

| dataset | full size | statistically-adequate sample |
|---|---:|---:|
| instruct-sft | ~940k rows | full (3 GB is trivial) |
| instruct-dpo | ~270k pairs | full (<1 GB) |
| instruct-rl  | ~180k rows | full (<1 GB) |
| think-sft    | ~4.6M rows | **200k rows is plenty** for the audits here; full is fine too |
| think-dpo    | ~180k pairs | full (1.4 GB) |
| think-rl     | ~220k rows | full (1.9 GB) |

So: **take everything except `think-sft`, where 200k is sufficient but
you can grab the full 36 GB if you want source-level stratification**.
The `download_full.py` script pulls the full corpora by default; if you
want to cap `think-sft` instead, keep the existing `download_samples.py`
as a smaller alternative.

## The pipeline

- `scripts/common.py` — shared metrics (structural Jaccard, n-gram overlap,
  affirmation / sycophancy / correction regexes) + `iter_rows()` which
  transparently reads parquet shards from `data/raw/<name>/` or falls back
  to the smoke-test JSONL.
- `scripts/download_full.py` — snapshot-downloads the six Dolci datasets.
- `scripts/download_samples.py` — older streaming N-row sampler. Kept for
  smoke tests on machines without disk budget.
- `scripts/phase1_sft_audit.py` — formatting-trap audit on SFT corpora.
  Args: `--short-name instruct-sft|think-sft --tag <name>`.
- `scripts/phase2_dpo_audit.py` — chosen-vs-rejected delta audit on DPO.
  Args: `--short-name instruct-dpo|think-dpo --tag <name>`.
- `scripts/phase4_rl_audit.py` — prompt-frame and (where available)
  completion audit on RL corpora.
  Args: `--short-name instruct-rl|think-rl`.
- `scripts/phase3_think_audit.py` — correlates `<think>` block length
  with truth-rescue rate from your local `runs/think/**/simulation.db`
  (14k real trials).
- `scripts/phase3b_dummy_buffer.py` — killshot experiment (dummy-buffer
  injection). `--dry-run` verifies prompt construction; live mode is
  dispatched to your HPC harness.
- `scripts/run_all_full.sh` — orchestrates every phase after the download.

## Outputs

Everything lands in `results/`:
- `phase1_<name>_per_example.csv` + `phase1_<name>_summary.json`
- `phase2_<name>_per_pair.csv` + `phase2_<name>_summary.json`
- `phase4_<name>_per_row.csv` + `phase4_<name>_summary.json`
- `phase3_think_trials.csv`, `phase3_think_summary.json`,
  `phase3_think_scatter.png`
- `phase3b_dryrun.json`

## Dependencies

```bash
pip install datasets huggingface_hub pyarrow pandas numpy matplotlib scipy
```

`pyarrow` is the only addition beyond what the main repo already uses; it
is what lets the phase scripts stream parquet shards row-by-row without
materialising 36 GB in RAM.
