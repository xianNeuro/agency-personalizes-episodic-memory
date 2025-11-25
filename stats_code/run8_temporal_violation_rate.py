#!/usr/bin/env python3
"""
Temporal Violation Rate Analysis
Analyzes temporal violation rate (tv_rate) across conditions.

For both stories (Adventure/BA and Romance/MV), tests:
- One-way ANOVA across conditions
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway
from data_structure import RecallDataLoader

def load_temporal_violation_data(story):
    """Load temporal violation rate data"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    if story == 'BA':
        data_file = os.path.join(data_dir, "adventure_data2.xlsx")
        sheet_name = 'comp_conds'
    else:  # MV
        data_file = os.path.join(data_dir, "romance_data2.xlsx")
        sheet_name = 'comp_conds18'
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading temporal violation rate data from: {data_file} (sheet: {sheet_name})")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    if 'tv_rate' not in df.columns:
        raise ValueError(f"Required column 'tv_rate' not found in {data_file}")
    
    if 'cond' not in df.columns:
        raise ValueError(f"'cond' column not found in {data_file}")
    
    print(f"Loaded {len(df)} subjects")
    print(f"Conditions: {df['cond'].value_counts().to_dict()}")
    
    return df

def run_oneway_anova(values_by_condition, measure_name):
    """Run one-way ANOVA across conditions"""
    
    conditions = ['free', 'yoke', 'pasv']
    condition_values = []
    condition_names = []
    
    for condition in conditions:
        if condition in values_by_condition and len(values_by_condition[condition]) > 0:
            vals = values_by_condition[condition].copy()
            condition_values.append(vals)
            condition_names.append(condition)
    
    if len(condition_values) < 2:
        print(f"Insufficient data for ANOVA")
        return None
    
    # One-way ANOVA
    f_stat, p_val = f_oneway(*condition_values)
    
    # Calculate degrees of freedom
    total_n = sum(len(v) for v in condition_values)
    df_between = len(condition_values) - 1
    df_within = total_n - len(condition_values)
    
    results = {
        'f_stat': f_stat,
        'p_val': p_val,
        'df_between': df_between,
        'df_within': df_within,
        'total_n': total_n,
        'condition_values': condition_values,
        'condition_names': condition_names
    }
    
    print(f"\n{measure_name} - One-way ANOVA across conditions:")
    print(f"  F({df_between},{df_within}) = {f_stat:.2f}, p = {p_val:.6f}")
    if p_val < 0.001:
        print(f"  Result: Significant difference across conditions (p < 0.001)")
    elif p_val < 0.01:
        print(f"  Result: Significant difference across conditions (p < 0.01)")
    elif p_val < 0.05:
        print(f"  Result: Significant difference across conditions (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    # Report means by condition
    print(f"  Means by condition:")
    for i, condition in enumerate(condition_names):
        mean_val = np.mean(condition_values[i])
        std_val = np.std(condition_values[i], ddof=1)
        n = len(condition_values[i])
        print(f"    {condition}: Mean = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
    
    return results

def analyze_temporal_violation():
    """Main analysis function"""
    
    print("="*80)
    print("TEMPORAL VIOLATION RATE ANALYSIS")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run8_temporal_violation_rate")
    
    stories = ['BA', 'MV']
    story_names = {'BA': 'Adventure', 'MV': 'Romance'}
    
    all_results = {}
    
    # Run analyses for both stories
    for story in stories:
        print("\n" + "="*80)
        print(f"{story_names[story]} STORY ({story})")
        print("="*80)
        
        # Load data
        df = load_temporal_violation_data(story)
        
        # Prepare data by condition
        tv_rate_by_condition = {}
        
        for condition in ['free', 'yoke', 'pasv']:
            cond_data = df[df['cond'] == condition]
            tv_rate_by_condition[condition] = cond_data['tv_rate'].dropna().values
        
        # One-way ANOVA across conditions
        print("\n" + "="*80)
        print("TEMPORAL VIOLATION RATE (tv_rate) - One-way ANOVA")
        print("="*80)
        anova_results = run_oneway_anova(tv_rate_by_condition, "Temporal Violation Rate")
        
        all_results[story] = {
            'anova': anova_results,
            'tv_rate_by_condition': tv_rate_by_condition
        }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for story in stories:
        story_name = story_names[story]
        results = all_results[story]
        
        # One-way ANOVA
        if results['anova'] is not None:
            anova = results['anova']
            summary_data.append({
                'Story': story_name,
                'Analysis': 'One-way ANOVA',
                'Measure': 'Temporal Violation Rate',
                'F_statistic': anova['f_stat'],
                'df_between': anova['df_between'],
                'df_within': anova['df_within'],
                'p_value': anova['p_val']
            })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "temporal_violation_rate_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("TEMPORAL VIOLATION RATE ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Adventure (BA): adventure_data2.xlsx (sheet: comp_conds)")
    report_lines.append("  - Romance (MV): romance_data2.xlsx (sheet: comp_conds18)")
    report_lines.append("")
    report_lines.append("Measure:")
    report_lines.append("  - Temporal violation rate (tv_rate): Rate of temporal violations")
    report_lines.append("")
    
    for story in stories:
        story_name = story_names[story]
        results = all_results[story]
        
        report_lines.append("="*80)
        report_lines.append(f"{story_name.upper()} STORY ({story})")
        report_lines.append("="*80)
        report_lines.append("")
        
        # One-way ANOVA
        if results['anova'] is not None:
            anova = results['anova']
            report_lines.append("One-way ANOVA across conditions:")
            report_lines.append(f"  F({int(anova['df_between'])},{int(anova['df_within'])}) = {anova['f_stat']:.2f}, p = {anova['p_val']:.6f}")
            if anova['p_val'] < 0.001:
                report_lines.append(f"  Result: Significant difference across conditions (p < 0.001)")
            elif anova['p_val'] < 0.01:
                report_lines.append(f"  Result: Significant difference across conditions (p < 0.01)")
            elif anova['p_val'] < 0.05:
                report_lines.append(f"  Result: Significant difference across conditions (p < 0.05)")
            else:
                report_lines.append(f"  Result: No significant difference (p >= 0.05)")
            report_lines.append("")
            report_lines.append("  Means by condition:")
            for i, condition in enumerate(anova['condition_names']):
                mean_val = np.mean(anova['condition_values'][i])
                std_val = np.std(anova['condition_values'][i], ddof=1)
                n = len(anova['condition_values'][i])
                report_lines.append(f"    {condition}: Mean = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "temporal_violation_rate_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_temporal_violation()

