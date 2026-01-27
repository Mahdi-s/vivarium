"""
Analytics infrastructure for Olmo Conformity Experiment.

This package provides comprehensive analytics modules that align with the
Critical Assessment document requirements, producing publication-ready figures
and AI-accessible metrics logs.
"""

from aam.analytics.behavioral import compute_behavioral_metrics, generate_behavioral_graphs, export_behavioral_logs
from aam.analytics.probes import compute_probe_metrics, generate_probe_graphs, export_probe_logs
from aam.analytics.interventions import compute_intervention_metrics, generate_intervention_graphs, export_intervention_logs
from aam.analytics.judgeval import compute_judgeval_metrics, generate_judgeval_graphs, export_judgeval_logs
from aam.analytics.think_tokens import compute_think_metrics, generate_think_graphs, export_think_logs
from aam.analytics.activations import compute_activation_stats, generate_activation_graphs, export_activation_logs
from aam.analytics.statistics import (
    compute_ttest,
    compute_correlation,
    compute_chi2_test,
    compute_effect_size_binary,
    compute_summary_statistics,
)
from aam.analytics.correlations import (
    compute_probe_behavioral_correlations,
    compute_probe_judgeval_correlations,
    compute_logit_lens_probe_correlations,
    compute_intervention_probe_correlations,
    compute_all_correlations,
)
from aam.analytics.tokens import (
    compute_token_metrics,
    generate_token_graphs,
    export_token_logs,
)

__all__ = [
    "compute_behavioral_metrics",
    "generate_behavioral_graphs",
    "export_behavioral_logs",
    "compute_probe_metrics",
    "generate_probe_graphs",
    "export_probe_logs",
    "compute_intervention_metrics",
    "generate_intervention_graphs",
    "export_intervention_logs",
    "compute_judgeval_metrics",
    "generate_judgeval_graphs",
    "export_judgeval_logs",
    "compute_think_metrics",
    "generate_think_graphs",
    "export_think_logs",
    "compute_activation_stats",
    "generate_activation_graphs",
    "export_activation_logs",
    "compute_ttest",
    "compute_correlation",
    "compute_chi2_test",
    "compute_effect_size_binary",
    "compute_summary_statistics",
    "compute_probe_behavioral_correlations",
    "compute_probe_judgeval_correlations",
    "compute_logit_lens_probe_correlations",
    "compute_intervention_probe_correlations",
    "compute_all_correlations",
    "compute_token_metrics",
    "generate_token_graphs",
    "export_token_logs",
]
