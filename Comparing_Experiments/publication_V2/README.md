# Publication Analysis Output

Generated: 2026-03-09T15:50:48.333940
Script: `Analysis Scripts/generate_publication_item_set.py`

## Directory Structure

```
publication_V2/
├── behavioral/
│   ├── tables/              # CSV tables with provenance headers
│   │   ├── error_rates_all_conditions.csv
│   │   ├── pressure_effects_all_conditions.csv
│   │   ├── truth_override_all_conditions.csv
│   │   ├── truth_rescue_all_conditions.csv
│   │   ├── wrong_answer_flip_all_conditions.csv
│   │   ├── opinion_agreement.csv
│   │   └── pooled_summary.csv
│   ├── figures/             # Publication figures (PNG 300dpi + PDF)
│   │   ├── fig1_forest_all_conditions.png   # Forest plot: all 11 pressure conditions
│   │   ├── fig2_training_trajectory.png     # Training stage → override rate
│   │   ├── fig3_mitigation_slope.png        # Mitigation effectiveness slope chart
│   │   ├── fig4_tone_modulation.png         # Tone modulation connected dots
│   │   ├── fig5_temperature_override.png    # Temperature × override line plots
│   │   ├── fig6_asymmetry_heatmap.png       # Peer vs authority asymmetry heatmap
│   │   └── fig7_heatmap_override.png        # Variant × condition override heatmap
│   └── statistical_tests/
│       ├── mcnemar_pressure_vs_control.csv  # McNemar + Holm-Bonferroni
│       ├── cochrans_q_families.csv          # Cochran's Q per family
│       └── bootstrap_cis.csv               # BCa 95% CIs
├── interpretability/        # (placeholder for future work)
├── logs/
│   ├── manifest.json        # Machine-readable run manifest
│   ├── pipeline.log         # Full pipeline log
│   └── balance_check.csv    # Cell balance diagnostics
└── README.md                # This file
```

## How to Read the Tables

Every CSV has a comment header (lines starting with `#`) that documents:
- What the table contains
- When it was generated
- Row count

### Key tables:

**error_rates_all_conditions.csv**
- Columns: temperature, variant, condition_name, dataset_category, error_rate, n_trials, refusal_rate
- One row per (T, variant, condition, topic) cell
- error_rate = fraction of factual trials where the model was incorrect (judge-labeled)

**pressure_effects_all_conditions.csv**
- Columns: temperature, variant, dataset_category, condition_name, control_error_rate, pressure_error_rate, delta_error
- delta_error = pressure_error_rate - control_error_rate (positive = pressure hurts)

**truth_override_all_conditions.csv**
- Columns: temperature, variant, dataset_category, truth_override_rate, n_items, pressure_condition
- truth_override_rate = P(pressure incorrect | control correct) — the core sycophancy metric
- n_items = denominator (control-correct items)

**mcnemar_pressure_vs_control.csv**
- One row per (variant, pressure_condition)
- Pooled across all temperatures and items
- p_adjusted = Holm-Bonferroni corrected p-value
- sig_adjusted = significance stars (***, **, *, ns)

## Condition Families

| Family | Conditions | Purpose |
|--------|-----------|---------|
| Control | control | Baseline (no pressure) |
| Peer (core) | asch_history_5 | Asch-style 5-confederate consensus |
| Tone | plain, neutral, confident, uncertain | Does peer confidence tone matter? |
| Mitigation | devil's advocate, question distillation, diverse peers | Do human-inspired fixes work? |
| Authority (core) | authoritative_bias | Single authoritative user claim |
| Authority (extended) | trust, trust+DA | Trust framing and alternative option |

## Key Finding: The "Mitigation Myth"

Human conformity research predicts that a single dissenter (devil's advocate) should
dramatically reduce conformity (Asch, 1956). Our data shows these mitigations FAIL
for LLMs — and sometimes backfire. See fig3_mitigation_slope for the visualization.
