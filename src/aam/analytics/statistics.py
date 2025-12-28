"""
Statistical test utilities for Olmo Conformity Experiment.

Provides t-tests, correlation analysis, and effect size calculations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError:
    raise RuntimeError("numpy, pandas, and scipy are required for statistical analysis")


def compute_ttest(
    group1: List[float],
    group2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Compute t-test between two groups.
    
    Args:
        group1: First group of values
        group2: Second group of values
        alternative: 'two-sided', 'less', or 'greater'
        
    Returns:
        Dict with t-statistic, p-value, degrees of freedom, and effect size (Cohen's d)
    """
    if len(group1) == 0 or len(group2) == 0:
        return {
            "t_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "cohens_d": None,
            "mean_diff": None,
            "std_pooled": None,
        }
    
    t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)
    
    # Cohen's d
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    
    # Pooled standard deviation
    std_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / std_pooled if std_pooled > 0 else 0.0
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "degrees_of_freedom": int(n1 + n2 - 2),
        "cohens_d": float(cohens_d),
        "mean_diff": float(mean1 - mean2),
        "std_pooled": float(std_pooled),
        "group1_mean": float(mean1),
        "group2_mean": float(mean2),
        "group1_std": float(std1),
        "group2_std": float(std2),
        "group1_n": int(n1),
        "group2_n": int(n2),
    }


def compute_correlation(
    x: List[float],
    y: List[float],
    method: str = "pearson",
) -> Dict[str, Any]:
    """
    Compute correlation between two variables.
    
    Args:
        x: First variable
        y: Second variable
        method: 'pearson' or 'spearman'
        
    Returns:
        Dict with correlation coefficient, p-value, and sample size
    """
    if len(x) != len(y) or len(x) == 0:
        return {
            "correlation": None,
            "p_value": None,
            "n": 0,
            "method": method,
        }
    
    # Remove NaN pairs
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return {
            "correlation": None,
            "p_value": None,
            "n": len(df),
            "method": method,
        }
    
    if method == "pearson":
        corr, p_value = stats.pearsonr(df["x"], df["y"])
    elif method == "spearman":
        corr, p_value = stats.spearmanr(df["x"], df["y"])
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "correlation": float(corr),
        "p_value": float(p_value),
        "n": int(len(df)),
        "method": method,
    }


def compute_chi2_test(
    observed: List[List[int]],
    expected: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Compute chi-square test for independence.
    
    Args:
        observed: 2D array of observed frequencies
        expected: 2D array of expected frequencies (optional)
        
    Returns:
        Dict with chi-square statistic, p-value, and degrees of freedom
    """
    observed_array = np.array(observed)
    
    if expected is None:
        chi2, p_value, dof, expected_array = stats.chi2_contingency(observed_array)
    else:
        expected_array = np.array(expected)
        chi2 = np.sum((observed_array - expected_array) ** 2 / (expected_array + 1e-10))
        dof = (observed_array.shape[0] - 1) * (observed_array.shape[1] - 1)
        p_value = 1 - stats.chi2.cdf(chi2, dof)
    
    return {
        "chi2_statistic": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(dof),
        "expected": expected_array.tolist() if expected is None else None,
    }


def compute_effect_size_binary(
    success1: int,
    total1: int,
    success2: int,
    total2: int,
) -> Dict[str, Any]:
    """
    Compute effect size for binary outcomes (e.g., flip rates).
    
    Args:
        success1: Number of successes in group 1
        total1: Total trials in group 1
        success2: Number of successes in group 2
        total2: Total trials in group 2
        
    Returns:
        Dict with proportions, difference, and odds ratio
    """
    p1 = success1 / total1 if total1 > 0 else 0.0
    p2 = success2 / total2 if total2 > 0 else 0.0
    
    # Odds ratio
    odds1 = p1 / (1 - p1) if p1 < 1.0 else float("inf")
    odds2 = p2 / (1 - p2) if p2 < 1.0 else float("inf")
    odds_ratio = odds1 / odds2 if odds2 > 0 else float("inf")
    
    return {
        "proportion1": float(p1),
        "proportion2": float(p2),
        "difference": float(p1 - p2),
        "odds_ratio": float(odds_ratio) if not np.isinf(odds_ratio) else None,
        "n1": int(total1),
        "n2": int(total2),
    }


def compute_summary_statistics(
    values: List[float],
    name: str = "values",
) -> Dict[str, Any]:
    """
    Compute comprehensive summary statistics.
    
    Args:
        values: List of numeric values
        name: Name for the variable
        
    Returns:
        Dict with mean, std, min, max, quartiles, etc.
    """
    if len(values) == 0:
        return {
            "name": name,
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    
    values_array = np.array(values)
    values_clean = values_array[~np.isnan(values_array)]
    
    if len(values_clean) == 0:
        return {
            "name": name,
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    
    return {
        "name": name,
        "n": int(len(values_clean)),
        "mean": float(np.mean(values_clean)),
        "std": float(np.std(values_clean, ddof=1)),
        "min": float(np.min(values_clean)),
        "max": float(np.max(values_clean)),
        "median": float(np.median(values_clean)),
        "q25": float(np.percentile(values_clean, 25)),
        "q75": float(np.percentile(values_clean, 75)),
    }
