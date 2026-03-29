# Cross-Family Analysis Results

**Generated:** 2026-03-29 01:15
**Data source:** runs/ (cross-family), runs/think/, runs_latest/runs/ (OLMo calibration)
**Labels used:** Judge labels (parsed_answer_json) as authoritative
**Conditions:** control, authoritative_bias, authority_trust, asch_zhu_unanimous_confident
**Total cross-family trials:** 28,808

## Tables

| File | Description |
|------|-------------|
| `tables/conformity_ranking.csv` | Models ranked by peer pressure effect (Δ), with endorsement and refusal rates |
| `tables/pressure_effects.csv` | Per-model, per-condition, per-temperature pressure deltas |
| `tables/per_model_condition_metrics.csv` | Raw error/refusal/endorsement rates per model × condition |
| `tables/per_dataset_metrics.csv` | Per-model × dataset × condition breakdown |
| `statistical_tests/mcnemar_pressure_vs_control.csv` | McNemar tests with Holm correction, odds ratios |

## Figures

| File | Description |
|------|-------------|
| `figures/fig_conformity_ranking.png` | Forest plot: models ranked by peer Δ |
| `figures/fig_refusal_vs_endorsement.png` | Scatter: resistance strategy (refusal vs endorsement) |
| `figures/fig_peer_vs_authority.png` | Grouped bar: peer vs authority effects per model |
| `figures/fig_domain_conformity_heatmap.png` | Heatmap: conformity Δ by model × domain |

## Bridge (OLMo Calibration)

| File | Description |
|------|-------------|
| `bridge/tables/calibrated_ranking.csv` | All models + OLMo variants ranked together |
| `bridge/tables/olmo_training_trajectory.csv` | OLMo base→SFT→DPO→instruct trajectory on same conditions |

## How to Read These Tables

- **error_rate**: 1 - judge_correct_rate (higher = more errors)
- **delta**: pressure_error - control_error (positive = conformity increases errors)
- **endorsed_rate**: fraction of trials where judge says model endorsed the wrong answer
- **refusal_rate**: fraction of trials where judge says model refused to answer
- **odds_ratio**: McNemar OR (>1 = pressure increases errors vs control)
- **p_holm**: Holm-Bonferroni corrected p-value
