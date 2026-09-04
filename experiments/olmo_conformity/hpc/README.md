# HPC plan — belief-probe factorial across the OLMo-3 ladders

## 1. How the existing Slurm setup works (read from `configs/job_*.sh`)

| Piece | Convention |
|---|---|
| Account / partition | `ll_774_951`, `gpu`; 7B jobs 2×(l40s\|a100\|a40\|v100), 200G, 48 h; 32B jobs 2×a100 |
| Repo (small, 100 GB home quota) | `/home1/mahdisae/aam/abstractAgentMachine` — code + suite JSONs only |
| Venv | `/scratch1/mahdisae/aam_venv` |
| Scratch root | `/scratch1/mahdisae/olmo_experiments/` — `models/huggingface_cache` (HF weights), `runs/` (vivarium run dirs `YYYYMMDD_HHMMSS_<uuid>/simulation.db`), `mpl_cache/`, `hf_home/`, `torch_home/`, `xdg_cache/` |
| Path wiring | `configs/paths.json` (`models_dir`, `runs_dir`) is read when `run_expanded_experiments.py --hpc`; newer jobs also export `HF_HOME`/`TRANSFORMERS_CACHE`/… to scratch |
| Chaining | Manual: `job_*_run.sh` (trials) → `job_judge.sh` (LLM judge, resolves `run_id` from `Comparing_Experiments/runs_metadata.json` or `AAM_RUN_ID`) → `job_posthoc.sh` (interpretability). `job_resume.sh` / `*_resume_*.sh` re-enter a run by `--run-id`. `job_7b_think_sft_dpo.sh` runs two models on two GPUs in one job. |
| Parametrisation | Environment variables at `sbatch` time: `TEMPERATURE`, `SUITE`, `AAM_RUN_ID`, `RUNS_DIR` |
| Known gap | Every script hard-codes the repo/venv/scratch paths; nothing writes a manifest; outputs are only discoverable through `runs_metadata.json` |

The new jobs keep every convention (same account/partition/constraints, same venv, all caches on scratch, env-var parametrisation) and add: one folder per launch **TAG** on scratch, a manifest per run, resumable outputs, and a one-command bundle to bring back.

## 2. Scratch layout for the new experiments

```
/scratch1/mahdisae/olmo_experiments/belief_probe/<TAG>/
    <variant>.jsonl            one row per (item, condition); appended, resumable
    <variant>.parquet          consolidated at the end of each variant
    <variant>.manifest.json    model id, items, conditions, budget, seed, tool version, git commit
    logs/<variant>.log
    bundle/                    created by collect_bundle.sh (parquets without reasoning text, summary CSVs, md)
/scratch1/mahdisae/olmo_experiments/slurm_logs/<job>_<id>.out
```

Set `AAM_BELIEF_DIR` to override the root; `TAG` groups a launch batch (use a new TAG per batch, never reuse).

## 3. What to run — one model per 48 h reservation

Every job loads **one checkpoint** and does everything for it inside one reservation: the 400-item × 32-condition
belief factorial (batched generation; Think checkpoints budget-forced at `THINK_BUDGET`, default 2,048 reasoning
tokens, with the natural-closure flag recorded), activation capture at three positions (answer slot, last token of
context+GT, last token of context+wrong; 10 layers by default, `CAPTURE_LAYERS=all` for every layer), the in-job
analysis (summary/contrasts; per-layer truth and belief-flip probes with shuffled-label and leave-one-dataset-out
controls; the train-on-control/test-under-pressure erasure test), and a causal steering pass along the learned pressure
direction with a random-direction control (`STEER=0` to skip; `STEER_ITEMS`, `STEER_ALPHAS`). Method details and the
literature basis: `investigation/backstudy/ACTIVATION_AUDIT.md`. Everything is written under
`/scratch1/mahdisae/olmo_experiments/belief_probe/<TAG>/`.

The tool stops itself cleanly at `TIME_BUDGET_H` (default 44 h, exit 75) and the job re-submits itself with
`--dependency=afterany` to resume from the jsonl (`AUTO_RESUBMIT=0` to disable). No row is ever recomputed.

```bash
cd /home1/mahdisae/aam/abstractAgentMachine/experiments/olmo_conformity/hpc

# Pilot (10 items/dataset): every checkpoint gets its own job; minutes for 7B, ~1–2 h for think/32B
TAG=pilot_0905 ITEMS=10 ./submit_ladder.sh

# Full (400 items). Expected per-job wall time on an A100 (batch 16):
#   7B instruct-path (base, instruct_sft, instruct_dpo, instruct, rl_zero)   ~2–4 h
#   7B think ladder (think_sft, think_dpo, think) at THINK_BUDGET=2048         ~12–20 h
#   32B instruct / base (2 x A100)                                             ~6–10 h
#   32B think ladder (2 x A100) at THINK_BUDGET=2048                           ~30–45 h  (auto-resumes if it spills over)
TAG=full_0905 ./submit_ladder.sh
TAG=full_0905 ONLY="think_dpo_32b" THINK_BUDGET=4096 ./submit_ladder.sh   # e.g. a deeper budget for one checkpoint

# Pack when squeue shows the TAG's jobs done
TAG=full_0905 ./collect_bundle.sh
```

Single jobs: `TAG=... VARIANT=instruct_dpo sbatch --gpus-per-task=1 --constraint='a100|l40s|a40' job_belief_one.sh`
(32B: `--gpus-per-task=2 --constraint=a100 --mem=200G`). Knobs are listed in the script header.

**Optional (old pipeline):** `MODEL=think_sft sbatch job_think_rerun8k_7b.sh` re-runs the Think ladder through the
vivarium runner at 8,192 tokens for judge-labelled free generations (4 core conditions, ~20–40 h per model). Not
required for the study.

## 4. What to give back

1. `<TAG>_bundle.tar.gz` from `collect_bundle.sh` (tens of MB): per-variant parquet (rows = item × condition, columns = forced-answer log-odds for GT/wrong/alternates in each context, greedy readout + flags, think closure/tokens), `probe_<variant>.csv` (per-layer belief/frame probe AUC, pressure-direction projections), the manifests, the per-variant logs, and `belief_probe_summary.csv` / `belief_probe_contrasts.csv` / `belief_probe.md`.
2. The `slurm_logs/AAM_BELIEF_*.out` files for any job that did not finish (they say which variant/context it stopped at; the jsonl resumes from there if re-submitted with the same TAG).
3. Unpack into `investigation/hpc_results/<TAG>/` in the repo; I take it from there (`analysis_belief_probe.py --data-dir investigation/hpc_results/<TAG>` and `analysis_policy_curve.py`).

For the external drive (not for git, not needed for the next analysis round): `<TAG>/*.jsonl` (with reasoning text), `<TAG>/*.steer.jsonl`, and `<TAG>/activations/` (three positions × layers; ~4.5 GB per 7B checkpoint at 10 layers, ~15 GB at all layers). Keep them — the cross-checkpoint CKA/direction analysis (`analysis_cross_checkpoint.py`) reads them.

## 5. Which existing results to keep, and which are no longer accurate

| Asset | Verdict | Why |
|---|---|---|
| `runs_latest` OLMo-7B Instruct-path trials (base, instruct_sft, instruct_dpo, instruct; 12 conditions × 6 temperatures × 400 items, judge-labelled) | **Keep — primary behavioural backbone.** Re-score with the four-way outcome (done in `investigation/backstudy`). | Complete, judge coverage 100%, effects stable across temperatures (Kendall's W 0.8–0.95). |
| `runs_latest` Think-path trials (think_sft, think_dpo, think at 256 tokens) | **Discard for any claim.** | Truncated mid-reasoning; no answer ever emitted. Replace with the belief-probe think jobs (budget-forced) or the 8k re-run. |
| `runs-think-hpc` 7B re-runs (2,048 tokens) and `runs-32B` (4,096) | **Keep as a secondary check only**, quoting the closure rate. | 42–66% (7B) / 65–88% (32B) of traces close; abandonment 0.21–0.36. Usable with the caveat, not as headline numbers. |
| Cross-family API runs (11 models, 4 conditions, T ∈ {0, 0.6}) | **Keep for non-reasoning models** (Llama-3.x, Llama-4, GPT-4o-mini, Gemini-Flash-Lite, OLMo-3.1-32B-Instruct). **Discard reasoning-model rows** (think_32b, gpt-oss-20b, Claude-Sonnet-4 with `<think>`; 6–26% clean endings). | Truncation/`max_tokens` interaction with reasoning output. |
| Ablation runs (naked social / n-gram / matched-instruction n-gram; Llama-3.1-70B, OLMo-3.1-32B) | **Keep.** | Complete; the matched-instruction result is the correct Fig. 3 comparison. |
| Dolci corpus audit (`dataset_analysis`, now on T7) | **Keep the SFT descriptive tables; re-run the DPO contrast restricted to `llm_judged` pairs before quoting.** | All 259,922 DPO pairs are strong-vs-weak model pairs; Δ n-gram is 4.7× larger in `delta_learning` pairs. |
| `rl_zero` trials (25% coverage) | Optional — the belief-probe ladder includes `rl_zero` cheaply. | |
| Local Mac belief pilot (SFT, DPO, RLVR, base; 40 items × 25 conditions) | **Keep as the pilot**; the HPC full run supersedes it. | |

## 6. Runtime notes

* 7B in bf16 needs one GPU (≈16 GB); `--constraint=a100|l40s|a40` avoids V100 (no bf16; use `DTYPE=fp16` if you must).
* 32B is sharded across the job's two A100s with `device_map=auto`.
* Missing checkpoints are downloaded into the scratch HF cache (`--allow-download`); the 7B ladder (6 checkpoints) is ~90 GB in bf16, the 32B ladder ~260 GB.
* Everything is resumable: re-submitting with the same `TAG` skips finished (item, condition) rows.
