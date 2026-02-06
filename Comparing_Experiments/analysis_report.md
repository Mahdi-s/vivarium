# Cross-Temperature Conformity Analysis Report

Generated: 2026-02-04 15:04:48

## Experiment Summary

- **Temperature Levels Analyzed**: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- **Number of Experiments**: 6

## Run Information

| Temperature | Run ID | Status | Config |
|-------------|--------|--------|--------|
| 0.0 | 56478e99... | completed | suite_expanded_temp0.0.json |
| 0.2 | 99127619... | completed | suite_expanded_temp0.2.json |
| 0.4 | 271bb5b2... | completed | suite_expanded_temp0.4.json |
| 0.6 | dda9d6b3... | completed | suite_expanded_temp0.6.json |
| 0.8 | eb777acc... | completed | suite_expanded_temp0.8.json |
| 1.0 | fa0b1d4f... | completed | suite_expanded_temp1.0.json |

## Key Findings

### Error Rates by Model

- **T=0.0**: Average error rate = 77.7%
- **T=0.2**: Average error rate = 77.5%
- **T=0.4**: Average error rate = 77.9%
- **T=0.6**: Average error rate = 78.6%
- **T=0.8**: Average error rate = 79.0%
- **T=1.0**: Average error rate = 79.7%

## Output Files

### Figures
- `figures/error_rates_by_temperature.png` - Error rates by condition and temperature
- `figures/temperature_curves.png` - Error rate vs temperature curves
- `figures/social_pressure_effect.png` - Social pressure effect by temperature
- `figures/error_rate_heatmap.png` - Error rate heatmap

### Tables
- `tables/rates_combined.csv` - Combined error rates across all temperatures
- `tables/rates_t{X}.csv` - Per-temperature error rates
