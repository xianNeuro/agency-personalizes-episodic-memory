#!/usr/bin/env python3
"""
Utility functions for calculating effect sizes and confidence intervals
- 95% confidence intervals for means
- Cohen's d for t-tests (one-sample and two-sample)
- Eta-squared and partial eta-squared for ANOVA
"""

import numpy as np
from scipy import stats

def calculate_ci_mean(data, confidence=0.95):
    """
    Calculate confidence interval for a mean
    
    Parameters:
    -----------
    data : array-like
        Sample data
    confidence : float
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    mean : float
        Sample mean
    """
    data_clean = np.array(data)[~np.isnan(data)]
    
    if len(data_clean) < 2:
        return np.nan, np.nan, np.nan
    
    n = len(data_clean)
    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    sem = std / np.sqrt(n)
    
    # Calculate t-critical value
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Calculate margin of error
    margin = t_critical * sem
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return ci_lower, ci_upper, mean

def cohens_d_one_sample(data, test_value=0):
    """
    Calculate Cohen's d for a one-sample t-test
    
    Parameters:
    -----------
    data : array-like
        Sample data
    test_value : float
        Value being tested against (default: 0)
    
    Returns:
    --------
    d : float
        Cohen's d effect size
    """
    data_clean = np.array(data)[~np.isnan(data)]
    
    if len(data_clean) < 2:
        return np.nan
    
    mean = np.mean(data_clean)
    std = np.std(data_clean, ddof=1)
    
    if std == 0:
        return np.nan
    
    d = (mean - test_value) / std
    
    return d

def cohens_d_two_sample(group1, group2):
    """
    Calculate Cohen's d for a two-sample t-test (independent samples)
    Uses pooled standard deviation
    
    Parameters:
    -----------
    group1 : array-like
        First group data
    group2 : array-like
        Second group data
    
    Returns:
    --------
    d : float
        Cohen's d effect size
    """
    group1_clean = np.array(group1)[~np.isnan(group1)]
    group2_clean = np.array(group2)[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan
    
    mean1 = np.mean(group1_clean)
    mean2 = np.mean(group2_clean)
    std1 = np.std(group1_clean, ddof=1)
    std2 = np.std(group2_clean, ddof=1)
    n1 = len(group1_clean)
    n2 = len(group2_clean)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    d = (mean1 - mean2) / pooled_std
    
    return d

def eta_squared_anova(ss_between, ss_total):
    """
    Calculate eta-squared (η²) for ANOVA
    
    Parameters:
    -----------
    ss_between : float
        Sum of squares between groups
    ss_total : float
        Total sum of squares
    
    Returns:
    --------
    eta_sq : float
        Eta-squared effect size
    """
    if ss_total == 0:
        return np.nan
    
    eta_sq = ss_between / ss_total
    
    return eta_sq

def partial_eta_squared_anova(ss_effect, ss_effect_error):
    """
    Calculate partial eta-squared (ηp²) for ANOVA
    
    Parameters:
    -----------
    ss_effect : float
        Sum of squares for the effect
    ss_effect_error : float
        Sum of squares for the effect + error
    
    Returns:
    --------
    partial_eta_sq : float
        Partial eta-squared effect size
    """
    if ss_effect_error == 0:
        return np.nan
    
    partial_eta_sq = ss_effect / ss_effect_error
    
    return partial_eta_sq

def calculate_anova_effect_sizes(anova_table):
    """
    Calculate effect sizes from an ANOVA table
    
    Parameters:
    -----------
    anova_table : pandas DataFrame
        ANOVA table from statsmodels
    
    Returns:
    --------
    effect_sizes : dict
        Dictionary with eta-squared and partial eta-squared
    """
    effect_sizes = {}
    
    try:
        # Get sum of squares
        if 'sum_sq' in anova_table.columns:
            ss_between = anova_table.loc['C(condition)', 'sum_sq'] if 'C(condition)' in anova_table.index else None
            ss_residual = anova_table.loc['Residual', 'sum_sq'] if 'Residual' in anova_table.index else None
            
            if ss_between is not None and ss_residual is not None:
                ss_total = ss_between + ss_residual
                
                # Eta-squared
                eta_sq = eta_squared_anova(ss_between, ss_total)
                effect_sizes['eta_squared'] = eta_sq
                
                # Partial eta-squared
                partial_eta_sq = partial_eta_squared_anova(ss_between, ss_between + ss_residual)
                effect_sizes['partial_eta_squared'] = partial_eta_sq
    except Exception as e:
        print(f"Warning: Could not calculate ANOVA effect sizes: {e}")
    
    return effect_sizes

def format_ci(ci_lower, ci_upper, decimals=3):
    """
    Format confidence interval for display
    
    Parameters:
    -----------
    ci_lower : float
        Lower bound
    ci_upper : float
        Upper bound
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    ci_str : str
        Formatted CI string, e.g., "[0.123, 0.456]"
    """
    if np.isnan(ci_lower) or np.isnan(ci_upper):
        return "N/A"
    
    return f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"

def format_cohens_d(d, decimals=3):
    """
    Format Cohen's d for display
    
    Parameters:
    -----------
    d : float
        Cohen's d value
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    d_str : str
        Formatted d string, e.g., "d = 0.456"
    """
    if np.isnan(d):
        return "N/A"
    
    return f"d = {d:.{decimals}f}"

def interpret_cohens_d(d):
    """
    Interpret Cohen's d effect size
    
    Parameters:
    -----------
    d : float
        Cohen's d value
    
    Returns:
    --------
    interpretation : str
        Interpretation string (small, medium, large)
    """
    abs_d = abs(d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
