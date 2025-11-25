#!/usr/bin/env python3
"""
Overall Recall Comparison Analysis
Performs one-way ANOVA on overall recall across three conditions for each story
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from data_structure import RecallDataLoader

def load_overall_recall_data():
    """Load overall recall data for both stories"""
    
    # Use current directory
    base_path = os.path.abspath('.')
    
    # Load Adventure data
    ba_file = os.path.join(base_path, 'data', 'adventure_data2.xlsx')
    ba_df = pd.read_excel(ba_file, sheet_name=0)
    ba_data = ba_df[['cond', 'overall_rcl']].dropna()
    
    print("="*80)
    print("LOADING OVERALL RECALL DATA")
    print("="*80)
    print(f"\nAdventure Story:")
    print(f"  Total subjects: {len(ba_data)}")
    print(f"  Conditions: {sorted(ba_data['cond'].unique())}")
    for cond in sorted(ba_data['cond'].unique()):
        n = len(ba_data[ba_data['cond'] == cond])
        mean_val = ba_data[ba_data['cond'] == cond]['overall_rcl'].mean()
        std_val = ba_data[ba_data['cond'] == cond]['overall_rcl'].std()
        print(f"    {cond}: N={n}, M={mean_val:.3f}, SD={std_val:.3f}")
    
    # Load Romance data
    mv_file = os.path.join(base_path, 'data', 'romance_data2.xlsx')
    mv_df = pd.read_excel(mv_file, sheet_name=0)
    mv_data = mv_df[['cond', 'overall_rcl']].dropna()
    
    print(f"\nRomance Story:")
    print(f"  Total subjects: {len(mv_data)}")
    print(f"  Conditions: {sorted(mv_data['cond'].unique())}")
    for cond in sorted(mv_data['cond'].unique()):
        n = len(mv_data[mv_data['cond'] == cond])
        mean_val = mv_data[mv_data['cond'] == cond]['overall_rcl'].mean()
        std_val = mv_data[mv_data['cond'] == cond]['overall_rcl'].std()
        print(f"    {cond}: N={n}, M={mean_val:.3f}, SD={std_val:.3f}")
    
    return ba_data, mv_data

def perform_oneway_anova(df_data, story_name):
    """Perform one-way ANOVA on overall recall by condition"""
    
    # Prepare data for ANOVA
    data_list = []
    for _, row in df_data.iterrows():
        data_list.append({
            'condition': row['cond'],
            'overall_rcl': row['overall_rcl']
        })
    
    df_combined = pd.DataFrame(data_list)
    
    # Fit ANOVA model
    model = ols('overall_rcl ~ C(condition)', data=df_combined).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Extract statistics
    f_stat = anova_table.loc['C(condition)', 'F']
    df_between = int(anova_table.loc['C(condition)', 'df'])
    df_within = int(anova_table.loc['Residual', 'df'])
    p_value = anova_table.loc['C(condition)', 'PR(>F)']
    
    # Calculate group means and standard deviations
    group_stats = df_combined.groupby('condition')['overall_rcl'].agg(['mean', 'std', 'count'])
    
    print(f"\n{story_name} Story - One-Way ANOVA Results:")
    print(f"  F({df_between},{df_within}) = {f_stat:.3f}, p = {p_value:.3f}")
    print(f"\nGroup Statistics:")
    for cond in sorted(group_stats.index):
        mean_val = group_stats.loc[cond, 'mean']
        std_val = group_stats.loc[cond, 'std']
        n_val = int(group_stats.loc[cond, 'count'])
        print(f"    {cond}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}")
    
    return {
        'f_stat': f_stat,
        'df_between': df_between,
        'df_within': df_within,
        'p_value': p_value,
        'group_stats': group_stats,
        'anova_table': anova_table
    }

def generate_report(results_ba, results_mv, output_file):
    """Generate text report of ANOVA results"""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OVERALL RECALL COMPARISON: ONE-WAY ANOVA RESULTS\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis compared overall recall across three agency conditions (Free, Yoked, Passive)\n")
        f.write("for both the Adventure and Romance stories using one-way ANOVA.\n\n")
        
        f.write("="*80 + "\n")
        f.write("ADVENTURE STORY\n")
        f.write("="*80 + "\n\n")
        
        # Group statistics
        group_stats_ba = results_ba['group_stats']
        f.write("Group Means and Standard Deviations:\n")
        for condition in ['free', 'pasv', 'yoke']:
            if condition in group_stats_ba.index:
                mean_val = group_stats_ba.loc[condition, 'mean']
                std_val = group_stats_ba.loc[condition, 'std']
                n_val = int(group_stats_ba.loc[condition, 'count'])
                cond_name = {'free': 'Free', 'pasv': 'Passive', 'yoke': 'Yoked'}[condition]
                f.write(f"  {cond_name}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}\n")
        
        # ANOVA results
        f_stat_ba = results_ba['f_stat']
        df_between_ba = results_ba['df_between']
        df_within_ba = results_ba['df_within']
        p_val_ba = results_ba['p_value']
        
        f.write(f"\nOne-Way ANOVA:\n")
        f.write(f"  F({df_between_ba},{df_within_ba}) = {f_stat_ba:.3f}, p = {p_val_ba:.3f}\n")
        
        # Interpret significance
        if p_val_ba < 0.001:
            f.write(f"  Result: Significant difference across conditions (p < 0.001)\n")
        elif p_val_ba < 0.01:
            f.write(f"  Result: Significant difference across conditions (p < 0.01)\n")
        elif p_val_ba < 0.05:
            f.write(f"  Result: Significant difference across conditions (p < 0.05)\n")
        else:
            f.write(f"  Result: No significant difference across conditions (p >= 0.05)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("ROMANCE STORY\n")
        f.write("="*80 + "\n\n")
        
        # Group statistics
        group_stats_mv = results_mv['group_stats']
        f.write("Group Means and Standard Deviations:\n")
        for condition in ['free', 'pasv', 'yoke']:
            if condition in group_stats_mv.index:
                mean_val = group_stats_mv.loc[condition, 'mean']
                std_val = group_stats_mv.loc[condition, 'std']
                n_val = int(group_stats_mv.loc[condition, 'count'])
                cond_name = {'free': 'Free', 'pasv': 'Passive', 'yoke': 'Yoked'}[condition]
                f.write(f"  {cond_name}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}\n")
        
        # ANOVA results
        f_stat_mv = results_mv['f_stat']
        df_between_mv = results_mv['df_between']
        df_within_mv = results_mv['df_within']
        p_val_mv = results_mv['p_value']
        
        f.write(f"\nOne-Way ANOVA:\n")
        f.write(f"  F({df_between_mv},{df_within_mv}) = {f_stat_mv:.3f}, p = {p_val_mv:.3f}\n")
        
        # Interpret significance
        if p_val_mv < 0.001:
            f.write(f"  Result: Significant difference across conditions (p < 0.001)\n")
        elif p_val_mv < 0.01:
            f.write(f"  Result: Significant difference across conditions (p < 0.01)\n")
        elif p_val_mv < 0.05:
            f.write(f"  Result: Significant difference across conditions (p < 0.05)\n")
        else:
            f.write(f"  Result: No significant difference across conditions (p >= 0.05)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Format p-values
        if p_val_ba < 0.001:
            p_text_ba = "p < 0.001"
        else:
            p_text_ba = f"p = {p_val_ba:.3f}"
        
        if p_val_mv < 0.001:
            p_text_mv = "p < 0.001"
        else:
            p_text_mv = f"p = {p_val_mv:.3f}"
        
        f.write(f"Adventure Story: F({df_between_ba},{df_within_ba}) = {f_stat_ba:.3f}, {p_text_ba}\n")
        f.write(f"Romance Story: F({df_between_mv},{df_within_mv}) = {f_stat_mv:.3f}, {p_text_mv}\n")

def main():
    """Main function"""
    
    print("="*80)
    print("OVERALL RECALL COMPARISON: ONE-WAY ANOVA")
    print("="*80)
    
    # Load data
    ba_data, mv_data = load_overall_recall_data()
    
    # Perform ANOVA for each story
    print("\n" + "="*80)
    print("PERFORMING ONE-WAY ANOVA")
    print("="*80)
    
    results_ba = perform_oneway_anova(ba_data, "Adventure")
    results_mv = perform_oneway_anova(mv_data, "Romance")
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    # Use current directory
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run1_overall_recall_comparison")
    
    report_file = os.path.join(output_dir, "overall_recall_anova_results.txt")
    generate_report(results_ba, results_mv, report_file)
    print(f"\nSaved report to: {report_file}")
    
    # Also save detailed statistics to Excel
    stats_file = os.path.join(output_dir, "overall_recall_anova_statistics.xlsx")
    with pd.ExcelWriter(stats_file, engine='openpyxl') as writer:
        # Adventure story
        results_ba['group_stats'].to_excel(writer, sheet_name='Adventure_group_stats')
        results_ba['anova_table'].to_excel(writer, sheet_name='Adventure_anova_table')
        # Romance story
        results_mv['group_stats'].to_excel(writer, sheet_name='Romance_group_stats')
        results_mv['anova_table'].to_excel(writer, sheet_name='Romance_anova_table')
    
    print(f"Saved detailed statistics to: {stats_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return results_ba, results_mv

if __name__ == "__main__":
    main()

