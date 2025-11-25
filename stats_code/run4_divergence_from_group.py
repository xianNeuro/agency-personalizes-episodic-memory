#!/usr/bin/env python3
"""
Divergence from the Group
Analyzes the correlation between memory divergence and choice divergence.

Each Free participant's:
- Memory divergence = 1 - Pearson correlation between their recall performance vector 
  and the group averaged recall performance vector
- Choice divergence = 1 - Pearson correlation between their choice selection vector 
  and the group averaged choice selection vector

Tests the correlation between memory divergence and choice divergence.

Uses precomputed divergence scores from:
- corr_free18 sheet: 18 free subjects
- corr_free100 sheet: 100 free subjects
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from data_structure import RecallDataLoader

def load_divergence_data(sheet_name):
    """Load divergence data from corr_free18 or corr_free100 sheet"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    data_file = os.path.join(data_dir, "romance_data2.xlsx")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading divergence data from: {data_file} (sheet: {sheet_name})")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    # Check for divergence columns
    recall_div_col = 'rcl-1-r_otherm'
    choice_div_col = 'cho-1-r_otherm'
    
    if recall_div_col not in df.columns:
        raise ValueError(f"Column '{recall_div_col}' not found in {sheet_name}")
    if choice_div_col not in df.columns:
        raise ValueError(f"Column '{choice_div_col}' not found in {sheet_name}")
    
    # Extract divergence scores
    recall_div = df[recall_div_col].dropna()
    choice_div = df[choice_div_col].dropna()
    
    # Align indices (only use rows where both are non-null)
    valid_mask = df[recall_div_col].notna() & df[choice_div_col].notna()
    recall_div_valid = df.loc[valid_mask, recall_div_col].values
    choice_div_valid = df.loc[valid_mask, choice_div_col].values
    
    print(f"Loaded {len(recall_div_valid)} subjects with valid divergence scores")
    
    return recall_div_valid, choice_div_valid, df

def analyze_divergence_correlation(sheet_name, n_subjects):
    """Analyze correlation between memory divergence and choice divergence"""
    
    print("\n" + "="*80)
    print(f"ANALYSIS: {n_subjects} Free Subjects")
    print("="*80)
    
    # Load data
    recall_div, choice_div, df = load_divergence_data(sheet_name)
    
    if len(recall_div) < 2:
        print(f"Insufficient data for correlation analysis (N={len(recall_div)})")
        return None
    
    # Compute Pearson correlation
    r, p = pearsonr(recall_div, choice_div)
    
    # Statistics
    n = len(recall_div)
    mean_recall_div = np.mean(recall_div)
    mean_choice_div = np.mean(choice_div)
    std_recall_div = np.std(recall_div, ddof=1)
    std_choice_div = np.std(choice_div, ddof=1)
    
    results = {
        'n': n,
        'r': r,
        'p': p,
        'mean_recall_div': mean_recall_div,
        'mean_choice_div': mean_choice_div,
        'std_recall_div': std_recall_div,
        'std_choice_div': std_choice_div,
        'recall_div': recall_div,
        'choice_div': choice_div,
        'df': df
    }
    
    print(f"\nMemory Divergence:")
    print(f"  N: {n}")
    print(f"  Mean: {mean_recall_div:.4f}")
    print(f"  Std: {std_recall_div:.4f}")
    
    print(f"\nChoice Divergence:")
    print(f"  N: {n}")
    print(f"  Mean: {mean_choice_div:.4f}")
    print(f"  Std: {std_choice_div:.4f}")
    
    print(f"\nCorrelation between Memory and Choice Divergence:")
    print(f"  r({n-2}) = {r:.3f}, p = {p:.3f}")
    if p < 0.001:
        print(f"  Result: Significant correlation (p < 0.001)")
    elif p < 0.01:
        print(f"  Result: Significant correlation (p < 0.01)")
    elif p < 0.05:
        print(f"  Result: Significant correlation (p < 0.05)")
    else:
        print(f"  Result: Not significant (p >= 0.05)")
    
    return results

def analyze_divergence():
    """Main analysis function"""
    
    print("="*80)
    print("DIVERGENCE FROM THE GROUP")
    print("Memory Divergence vs Choice Divergence Correlation")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run4_divergence_from_group")
    
    # Run analyses for both N=18 and N=100
    all_results = {}
    
    # Analysis 1: N=18 free subjects
    results_18 = analyze_divergence_correlation('corr_free18', 18)
    all_results['free18'] = results_18
    
    # Analysis 2: N=100 free subjects
    results_100 = analyze_divergence_correlation('corr_free100', 100)
    all_results['free100'] = results_100
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for analysis_name, results in all_results.items():
        if results is not None:
            summary_data.append({
                'Analysis': analysis_name,
                'N_subjects': results['n'],
                'Mean_recall_divergence': results['mean_recall_div'],
                'Std_recall_divergence': results['std_recall_div'],
                'Mean_choice_divergence': results['mean_choice_div'],
                'Std_choice_divergence': results['std_choice_div'],
                'Correlation_r': results['r'],
                'Correlation_p': results['p'],
                'df': results['n'] - 2
            })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "divergence_correlation_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Save divergence scores
    if results_18 is not None:
        div_df_18 = results_18['df'][['Unnamed: 0', 'rcl-1-r_otherm', 'cho-1-r_otherm']].copy()
        div_df_18.columns = ['subject_id', 'recall_divergence', 'choice_divergence']
        div_file_18 = os.path.join(output_dir, "free18_divergence_scores.xlsx")
        div_df_18.to_excel(div_file_18, index=False)
        print(f"Saved free18 divergence scores to: {div_file_18}")
    
    if results_100 is not None:
        div_df_100 = results_100['df'][['Unnamed: 0', 'rcl-1-r_otherm', 'cho-1-r_otherm']].copy()
        div_df_100.columns = ['subject_id', 'recall_divergence', 'choice_divergence']
        div_file_100 = os.path.join(output_dir, "free100_divergence_scores.xlsx")
        div_df_100.to_excel(div_file_100, index=False)
        print(f"Saved free100 divergence scores to: {div_file_100}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("DIVERGENCE FROM THE GROUP")
    report_lines.append("Memory Divergence vs Choice Divergence Correlation")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source: romance_data2.xlsx")
    report_lines.append("  - corr_free18 sheet: 18 free subjects")
    report_lines.append("  - corr_free100 sheet: 100 free subjects")
    report_lines.append("")
    report_lines.append("Divergence scores:")
    report_lines.append("  - Memory divergence (rcl-1-r_otherm): 1 - correlation with group average recall")
    report_lines.append("  - Choice divergence (cho-1-r_otherm): 1 - correlation with group average choice")
    report_lines.append("")
    
    for analysis_name, results in all_results.items():
        if results is not None:
            n_label = "18" if analysis_name == "free18" else "100"
            report_lines.append("="*80)
            report_lines.append(f"ANALYSIS: {n_label} Free Subjects")
            report_lines.append("="*80)
            report_lines.append("")
            
            report_lines.append("Memory Divergence:")
            report_lines.append(f"  N: {results['n']}")
            report_lines.append(f"  Mean: {results['mean_recall_div']:.4f}")
            report_lines.append(f"  Std: {results['std_recall_div']:.4f}")
            report_lines.append(f"  Min: {np.min(results['recall_div']):.4f}")
            report_lines.append(f"  Max: {np.max(results['recall_div']):.4f}")
            report_lines.append("")
            
            report_lines.append("Choice Divergence:")
            report_lines.append(f"  N: {results['n']}")
            report_lines.append(f"  Mean: {results['mean_choice_div']:.4f}")
            report_lines.append(f"  Std: {results['std_choice_div']:.4f}")
            report_lines.append(f"  Min: {np.min(results['choice_div']):.4f}")
            report_lines.append(f"  Max: {np.max(results['choice_div']):.4f}")
            report_lines.append("")
            
            report_lines.append("Correlation between Memory and Choice Divergence:")
            report_lines.append(f"  r({results['n']-2}) = {results['r']:.3f}, p = {results['p']:.3f}")
            if results['p'] < 0.001:
                report_lines.append(f"  Result: Significant correlation (p < 0.001)")
            elif results['p'] < 0.01:
                report_lines.append(f"  Result: Significant correlation (p < 0.01)")
            elif results['p'] < 0.05:
                report_lines.append(f"  Result: Significant correlation (p < 0.05)")
            else:
                report_lines.append(f"  Result: Not significant (p >= 0.05)")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "divergence_correlation_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_divergence()

