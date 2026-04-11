# April_analysis/ — 7B OLMo 2-Axis Story + Cross-Family + Ablation

This folder is the single authoritative analysis for the conformity-
susceptibility paper. It contains three concentric layers of analysis,
all recomputed directly from the SQLite databases using the
authoritative LLM-judge labels:

1. **7B OLMo 2-axis decomposition** (base → SFT → DPO → RL × Instruct /
   Think pipelines, all 12 conditions, 6 temperatures) — the
   within-model story of *how* post-training shapes consensus
   susceptibility. Findings 1–5 in `findings_summary.md`.
2. **Cross-family generalization** (12 model families at T ∈ {0, 0.6}
   on the 4 shared conditions, including Claude Sonnet 4, GPT-4o-Mini,
   Llama-3.1-70B, Llama-4-Maverick, Gemini-2.5-Flash-Lite, Grok-4.1,
   GPT-OSS-20B, OLMo-3.1-32B {Instruct, Think, Think-SFT, Think-DPO})
   — tests whether the 7B pattern-completion story generalizes. The
   32B-Think-SFT/DPO entries complete the 32B Think post-training
   trajectory. Findings 6, 8, 9 in `findings_summary.md`.
3. **Ablation probes** (Llama-3.1-70B-Instruct + OLMo-3.1-32B-Instruct
   with the "be truthful" system prompt stripped, plus a pure
   A→B→C `ngram_sequence_baseline` probe) — tests whether any of the
   effect is attributable to the system prompt or to social framing.
   Finding 7 in `findings_summary.md`.

Nothing under `expanded_results/`, `v8_publication/`, `publication_V2/`,
etc. has been modified — this folder is entirely additive. The
cross-family + ablation expansion (2026-04-09, updated with 12 models
including 32B-Think-SFT/DPO) is isolated in sibling manifests and does
not touch the 7B pipeline or its `validation/claim_check.md`
(12 PASS / 1 FAIL). Cross-family scorecard: 8 PASS / 1 FAIL / 1 PARTIAL.

## Goal

Build a single trustworthy numerical foundation that can be used to test
the "autoregressive pattern completion" re-frame of the paper narrative
before any rewrite of the paper itself. Every number that the paper wants
to claim should be traceable to a row in this folder.

## The 7B OLMo 2-axis decomposition

All seven variants share the same OLMo-3-7B pretrained backbone, so
behavioral differences across them reflect post-training choices only.

| Axis / Stage | Base               | SFT               | DPO               | RL (final)  |
| ------------ | ------------------ | ----------------- | ----------------- | ----------- |
| Instruct     | `base` (shared)    | `instruct_sft`    | `instruct_dpo`    | `instruct`  |
| Think        | `base` (shared)    | `think_sft`       | `think_dpo`       | `think`     |

1 base + 2 paths × 3 post-training stages = **7 variants**.

Coverage is intentionally asymmetric:

| Family         | Variants                                             | Temperatures             | Conditions        |
| -------------- | ---------------------------------------------------- | ------------------------ | ----------------- |
| Instruct (+base) | base, instruct_sft, instruct_dpo, instruct         | {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} | all 12            |
| Think SFT/DPO  | think_sft, think_dpo                                 | {0.0, 0.6}               | 4 shared          |
| Think-RL       | think                                                | {0.0}                    | 4 shared          |

The 4 shared conditions are `control`,
`asch_zhu_unbiased_unanimous_confident`, `authoritative_bias`, and
`authority_zhu_unbiased_trust`. Cross-path comparisons are valid only
on those 4 conditions; no Think variant has data at T ∈ {0.2, 0.4,
0.8, 1.0} or on DA / QD / Diverse / tone variants.

Total row count after loading:
`4 × 6 × 4,800 + 2 × 2 × 1,600 + 1 × 1 × 1,600 = 115,200 + 6,400 + 1,600 = 123,200`
across **308 cells** for the 7B layer.

## Cross-family + ablation scope (2026-04-09 expansion)

On top of the 7B story, the expansion layer adds:

| Layer             | Models                                                                                                          | Temperatures | Conditions                                                                                                        | Rows   | Cells |
| ----------------- | --------------------------------------------------------------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------- | -----: | ----: |
| Cross-family main | 12 families: OLMo-32B {Instruct, Think, Think-SFT, Think-DPO}, Llama-{3-8B, 3.1-70B, 4-Maverick}, GPT-{4o-Mini, OSS-20B}, Gemini-2.5-Flash-Lite, Grok-4.1-Fast, Claude-Sonnet-4 | {0.0, 0.6}   | 4 shared                                                                                                          | ~40,000 | 96    |
| Ablation probes   | Llama-3.1-70B-Instruct, OLMo-3.1-32B-Instruct (`system_style:none`)                                             | {0.0}        | `asch_zhu_naked_unanimous_confident`, `ngram_sequence_baseline`                                                   |  1,600 |  8    |

All cross-family and ablation DBs live under `runs/20260327_*`,
`runs/20260329_*`, and `runs-think-hpc/runs-32B/` in the main repo and
are loaded through a separate manifest
(`metadata/cross_family_metadata.json`) and a separate loader entry
point (`load_april_trials(manifest=...,
experiment_group="cross_family")`). The cross-family layer carries
its own post-load assertion bundle (CF1-CF5) that is distinct from
the 7B Think-family bundle (R1-R4). The 32B-Think-SFT/DPO entries
were added 2026-04-09 to complete the full 32B Think post-training
trajectory (SFT → DPO → final).

**Grand total for `April_analysis/` after expansion:**
`123,200 + ~40,000 + 1,600 ≈ 164,800 trials across ~412 cells`.

## Label policy (CRITICAL)

- The authoritative labels are the **top-level fields** of
  `conformity_outputs.parsed_answer_json`:
  - `is_correct` (0/1)
  - `wrong_answer_endorsed` (0/1)
  - `refusal_flag` (0/1)
- `parsed_answer_json._llm_judge` is a metadata sub-object whose presence
  marks a row as LLM-judged. The canonical loader only keeps rows where
  `_llm_judge` is present.
- The column `conformity_outputs.is_correct` is the **heuristic parser
  label** (~87.5 % coverage on Instruct family, ~60 % coverage on Think
  family because it cannot parse post-`<think>` answers reliably). It is
  only used for the heuristic-vs-judge agreement audit under
  `validation/heuristic_vs_judge_agreement.csv`, never for reported metrics.
- Two judge prompt versions are mixed in the dataset:
  `2026-03-03_local_v3` (runs_latest and runs/think/) and
  `2026-03-27_openrouter_v1` (runs-think-hpc). Per-version agreement is
  audited in the agreement CSV.

## Data provenance audit (2026-04-08 remediation)

An earlier revision of this folder loaded the Think family (think_sft,
think_dpo, think) from `runs_latest/runs/`. That turned out to be a
fatal data-provenance error: those runs hard-capped output at ~350
tokens / ~1,400 chars, which is below the length of a finished `<think>`
block. The parsed Think labels were therefore whatever the model
committed to mid-thought, if anything. The fig_temperature_surface.pdf
panel that showed Think curves across 6 temperatures was the forensic
trigger — Think variants physically only have data at T ∈ {0, 0.6}, so
any curve with 6 T points for a Think variant was a truncation artifact.

The correct Think sources live in the main repo and are symlinked into
the worktree:

- `runs-think-hpc/20260330_012727_e5cc991d-...` — think_sft T=0.0
- `runs-think-hpc/20260330_012826_29ac502b-...` — think_sft T=0.6
- `runs-think-hpc/20260330_021503_56c71fda-...` — think_dpo T=0.0
- `runs-think-hpc/20260330_022920_57414a36-...` — think_dpo T=0.6
- `runs/think/20260325_010440_f47fe05e-...`     — think T=0.0

The canonical loader `vivarium.analytics.behavioral.load_april_trials`
is defensive against the regression: after loading, it runs four
post-load assertions that refuse to return a DataFrame if

1. Any row with variant ∈ {think_sft, think_dpo, think} has a
   `db_path` starting with `runs_latest/runs/`.
2. Median `len(raw_text)` per Think variant is ≤ 2,000 chars
   (Think outputs should be several thousand chars).
3. A Think variant appears at a temperature outside its allowed set
   ({0, 0.6} for SFT/DPO, {0} for Think-RL).
4. A Think variant appears on a condition outside the 4 shared
   conditions.

The same four checks are also repeated at the `validate.py` level
(R4.1–R4.4) so the smoke test log shows the current values on every
run. Scope of the original invalidation and the replacement numbers
are documented in the **Data reconciliation** section of
`findings_summary.md`.

## Source of truth

The 7B layer reads [`metadata/runs_metadata.json`](metadata/runs_metadata.json),
the cross-family + ablation layer reads
[`metadata/cross_family_metadata.json`](metadata/cross_family_metadata.json).
Both carry the explicit, auditable list of (db_path, temperature,
variant, conditions) tuples loaded into every downstream table and
the `condition_name_canonicalization` map that translates HPC
short-form condition names (`asch_zhu_unanimous_confident`,
`authority_trust`) into the canonical long-form names used everywhere
else in this folder.

The two manifests are independent: the 7B driver scripts cannot see
cross-family data, and the cross-family driver scripts carry a hard
guard that fails loudly if wired to the 7B manifest. There is no
shared mutable state between the two layers.

## Folder layout

```
April_analysis/
├── README.md                     <-- you are here
├── findings_summary.md           Findings 1-9: 7B (1-5) + cross-family/ablation (6-9)
├── run_all.sh                    Single-command regenerator for the entire folder
├── metadata/
│   ├── runs_metadata.json        7B manifest (schema_version 2)
│   ├── cross_family_metadata.json Cross-family + ablation manifest (schema_version 3)
│   ├── cross_family_db_audit.csv Per-DB audit row (n, judge cov, raw_text len)
│   ├── judge_coverage.csv        Per-(variant, T, condition) judge coverage (7B)
│   └── trial_counts.csv          Per-(variant, T, condition) raw counts (7B)
├── tables/
│   ├── behavioral/               7B error / endorsement / refusal rates
│   ├── stage_decomposition/      7B 2-axis Base -> SFT -> DPO -> RL tables
│   ├── mitigation_taxonomy/      7B DA / QD / diverse / system-prompt / reasoning
│   ├── cross_family/             10-model cross-family tables (per-model metrics,
│   │                             BER ranking with Wilson tie groups, peer/auth
│   │                             delta, scale bridge, knowledge φ, per-dataset)
│   └── ablation_probes/          System-prompt ablation + ngram baseline
├── figures/                      7B replacement paper figures (10 PDFs)
│   └── cross_family/             9 cross-family + ablation figures
├── statistical_tests/            7B McNemar, Cochran-Q, BCa, Wilson, Holm
│   └── cross_family/             Cross-family paired McNemar table
├── pattern_match/                7B 12-condition pattern-match gradient
├── item_level/                   7B per-item endorsement + knowledge correlations
└── validation/
    ├── smoke_tests.log           7B row counts + post-load invariant log
    ├── heuristic_vs_judge_agreement.csv  7B judge vs heuristic parser agreement
    ├── claim_check.{md,json}     7B scorecard (13 claims, 12 PASS / 1 FAIL)
    └── cross_family_claim_check.{md,json}  Cross-family + ablation scorecard
                                    (10 claims C12-C20 including C15b,
                                    9 PASS / 0 FAIL / 1 PARTIAL)
```

## Figure set

### 7B layer (`figures/`, 10 PDFs)

| File                                    | Purpose                                                              |
| --------------------------------------- | -------------------------------------------------------------------- |
| `fig_stage_trajectory`                  | Two panels (T=0, T=0.6): Base → SFT → DPO → RL per path              |
| `fig_2axis_heatmap_instruct`            | Panel A: Instruct family × 12 conditions at T=0                      |
| `fig_2axis_heatmap_think`               | Panel B: Think family × 4 shared conditions at T=0                   |
| `fig_2axis_heatmap_combined`            | Panels A + B column-aligned on the 4 shared conditions               |
| `fig_pattern_match_gradient`            | BER vs target-answer repetition count (asymmetric Think coverage)    |
| `fig_instruct_temperature_sweep`        | Instruct family × 6 T points on unanimous_confident + unanimous_plain |
| `fig_temperature_t0_vs_t06_scatter`     | Cross-path scatter: BER(T=0) vs BER(T=0.6) on 4 shared conditions    |
| `fig_temperature_slope_bars`            | Instruct family polyfit slopes per condition                         |
| `fig_mitigation_delta`                  | DA / QD / Diverse peers Δ from anchor (Instruct family only)         |
| `fig_think_prefix_proxy`                | Cross-path BER on unanimous_confident per post-training stage        |

The old `fig_temperature_surface` (6-T curves for all 7 variants on a
single condition) is no longer produced because the Think family has
no mid-T data — any 6-T Think curve is a truncation artifact. The
three new temperature figures collectively replace it.

### Cross-family + ablation layer (`figures/cross_family/`, 9 PDFs)

| File                                   | Role                     | Purpose                                                                                  |
| -------------------------------------- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `fig_cross_family_headline_ber.pdf`    | **H1 headline**          | 15-bar chart (10 cross-family + 4 7B-Instruct stages + OLMo-7B-Base) ranked by BER at T=0, color-coded by architecture, with Wilson CI whiskers |
| `fig_cross_family_t0_vs_t06.pdf`       | T sweep cross-path       | Scatter BER(T=0) vs BER(T=0.6) per (model × condition), y=x reference, color by architecture |
| `fig_scale_bridge.pdf`                 | **H3 headline**          | Paired bars: 7B Instruct stages + 32B Instruct + 7B Think stages + 32B Think on the 4 shared conditions at T=0 |
| `fig_ablation_ngram_vs_pressure.pdf`   | **H2 headline**          | BER(ngram_baseline) next to BER(unanimous_confident) for Llama-70B + OLMo-32B, with pattern_completion_ratio annotated above each pair |
| `fig_system_prompt_ablation.pdf`       | H2 supplement            | Paired bars with vs without system prompt for both ablation models, McNemar p-value annotated |
| `fig1_stacked_decomposition.pdf`       | paper-style fig1 replacement | 3-state stacked bars at T=0 unanimous_confident (adapted from `paper/generate_paper_figures.py`) |
| `fig3_cross_family_forest.pdf`         | paper-style fig3 replacement | Forest plot ranked by Δ endorsement                                            |
| `fig4_refusal_endorsement.pdf`         | paper-style fig4 replacement | Scatter refusal Δ vs endorsement Δ cross-family                                |
| `fig5_peer_vs_authority.pdf`           | paper-style fig5 replacement | Grouped bars peer Δ vs authority Δ per model                                   |

## Reproducing this folder

Prerequisites. The pipeline reads directly from the following
simulation.db files (all paths are relative to the repo root):

**7B layer (11 DBs)**
- 6 × `runs_latest/runs/*/simulation.db`
- 4 × `runs-think-hpc/20260330_*/simulation.db`
- 1 × `runs/think/20260325_*/simulation.db`

**Cross-family + ablation layer (20 DBs)**
- 16 × `runs/20260327_*/simulation.db` (cross-family main, T ∈ {0, 0.6})
- 2 × `runs/20260329_211*/simulation.db` (Claude Sonnet 4, T ∈ {0, 0.6})
- 2 × `runs/20260329_235*/simulation.db` (Llama-70B + OLMo-32B ablation)

In the `affectionate-chatelet` worktree the `runs-think-hpc/` and
`runs/` directories are symlinks pointing at the main-repo checkout;
the symlinks are intentional and let both manifests stay portable
across checkouts.

### One-command regenerator

```bash
./Comparing_Experiments/April_analysis/run_all.sh
```

`run_all.sh` runs the 8 7B driver scripts (which read
`runs_metadata.json`) followed by the 4 cross-family / ablation driver
scripts (which read `cross_family_metadata.json`), and exits non-zero
on any failure. It is fully idempotent and writes only into
`Comparing_Experiments/April_analysis/`.

### Individual driver scripts

If you prefer to run individual stages, each script is idempotent and
writes only into `Comparing_Experiments/April_analysis/`:

```bash
# --- 7B layer (reads metadata/runs_metadata.json) ---
# Phase B — behavioral tables + stage decomposition
uv run python 'Analysis Scripts/april_analysis/behavioral_tables.py'
uv run python 'Analysis Scripts/april_analysis/stage_decomposition.py'

# Phase C — post-hoc analyses for the pattern-completion re-frame
uv run python 'Analysis Scripts/april_analysis/pattern_match_gradient.py'
uv run python 'Analysis Scripts/april_analysis/temperature_concentration.py'
uv run python 'Analysis Scripts/april_analysis/item_level_correlations.py'
uv run python 'Analysis Scripts/april_analysis/mitigation_taxonomy.py'

# Figures + validation
uv run python 'Analysis Scripts/april_analysis/figures.py'
uv run python 'Analysis Scripts/april_analysis/validate.py'

# --- Cross-family + ablation layer (reads metadata/cross_family_metadata.json) ---
uv run python 'Analysis Scripts/april_analysis/cross_family_tables.py'
uv run python 'Analysis Scripts/april_analysis/ablation_probes.py'
uv run python 'Analysis Scripts/april_analysis/cross_family_figures.py'
uv run python 'Analysis Scripts/april_analysis/cross_family_validate.py'
```

The 7B loader raises `AssertionError` on any post-load invariant
violation (Think-family truncation / contamination). The cross-family
loader runs its own distinct CF1-CF5 assertion bundle after load.
Neither loader can see the other manifest's rows.

## Scope

- **In scope:** 7B OLMo 2-axis decomposition on the 4 shared conditions
  (Instruct + Think) and on the full 12 conditions (Instruct only);
  10-model cross-family corroboration at T ∈ {0, 0.6} on the 4 shared
  conditions; system-prompt ablation + abstract N-gram pattern probe
  on Llama-3.1-70B-Instruct and OLMo-3.1-32B-Instruct at T = 0.
- **Out of scope:** new inference runs, activation steering, probe
  training, paper rewrite (follow-up plan), Think-path mitigation
  battery beyond the 4 shared conditions (future work), cross-family
  pattern-match gradient (only reps=5 data exists cross-family),
  cross-family temperature sweep beyond {0, 0.6} (only 2 points
  exist), partial 32B data under `runs-think-hpc/runs-32B-part*/`, and
  any modification to the SQLite DBs themselves (they are read-only
  input).
