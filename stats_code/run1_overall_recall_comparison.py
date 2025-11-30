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

def generate_report(results_ba, results_mv, engagement_results, output_file):
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
        
        # Add reading time and engagement analysis
        f.write("\n" + "="*80 + "\n")
        f.write("ROMANCE STORY: READING TIME AND ENGAGEMENT\n")
        f.write("="*80 + "\n\n")
        
        f.write("For the Romance story, we additionally recorded the reading time for each subject and\n")
        f.write("measured individual engagement over the course of their reading via a 13-item modified\n")
        f.write("version of the Narrative Transportation scale. The following analyses examine whether\n")
        f.write("there were differences across the three agency conditions on participants' transportation\n")
        f.write("score, average reading time per story sentence, and overall reading time for the entire\n")
        f.write("story-path they experienced.\n\n")
        
        # Transportation score
        trans_result = engagement_results['trans_score']
        f.write("TRANSPORTATION SCORE:\n")
        group_stats = trans_result['group_stats']
        for condition in ['free', 'pasv', 'yoke']:
            if condition in group_stats.index:
                mean_val = group_stats.loc[condition, 'mean']
                std_val = group_stats.loc[condition, 'std']
                n_val = int(group_stats.loc[condition, 'count'])
                cond_name = {'free': 'Free', 'pasv': 'Passive', 'yoke': 'Yoked'}[condition]
                f.write(f"  {cond_name}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}\n")
        f_stat = trans_result['f_stat']
        df_between = trans_result['df_between']
        df_within = trans_result['df_within']
        p_val = trans_result['p_value']
        if p_val < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_val:.3f}"
        f.write(f"  One-Way ANOVA: F({df_between},{df_within}) = {f_stat:.3f}, {p_text}\n\n")
        
        # Average reading time per sentence
        avg_result = engagement_results['avg_sent_readtime']
        f.write("AVERAGE READING TIME PER STORY SENTENCE:\n")
        group_stats = avg_result['group_stats']
        for condition in ['free', 'pasv', 'yoke']:
            if condition in group_stats.index:
                mean_val = group_stats.loc[condition, 'mean']
                std_val = group_stats.loc[condition, 'std']
                n_val = int(group_stats.loc[condition, 'count'])
                cond_name = {'free': 'Free', 'pasv': 'Passive', 'yoke': 'Yoked'}[condition]
                f.write(f"  {cond_name}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}\n")
        f_stat = avg_result['f_stat']
        df_between = avg_result['df_between']
        df_within = avg_result['df_within']
        p_val = avg_result['p_value']
        if p_val < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_val:.3f}"
        f.write(f"  One-Way ANOVA: F({df_between},{df_within}) = {f_stat:.3f}, {p_text}\n\n")
        
        # Total reading time
        sum_result = engagement_results['sum_readtime']
        f.write("OVERALL READING TIME FOR ENTIRE STORY-PATH:\n")
        group_stats = sum_result['group_stats']
        for condition in ['free', 'pasv', 'yoke']:
            if condition in group_stats.index:
                mean_val = group_stats.loc[condition, 'mean']
                std_val = group_stats.loc[condition, 'std']
                n_val = int(group_stats.loc[condition, 'count'])
                cond_name = {'free': 'Free', 'pasv': 'Passive', 'yoke': 'Yoked'}[condition]
                f.write(f"  {cond_name}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}\n")
        f_stat = sum_result['f_stat']
        df_between = sum_result['df_between']
        df_within = sum_result['df_within']
        p_val = sum_result['p_value']
        if p_val < 0.001:
            p_text = "p < 0.001"
        else:
            p_text = f"p = {p_val:.3f}"
        f.write(f"  One-Way ANOVA: F({df_between},{df_within}) = {f_stat:.3f}, {p_text}\n\n")
        
        f.write("These results suggest that the overall engagement for the story remained roughly the\n")
        f.write("same across the three agency conditions.\n")
        
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

def load_romance_reading_engagement_data():
    """Load reading time and transportation score data from romance_data1.xlsx"""
    
    base_path = os.path.abspath('.')
    data_file = os.path.join(base_path, 'data', 'romance_data1.xlsx')
    
    print("\n" + "="*80)
    print("LOADING ROMANCE READING TIME AND ENGAGEMENT DATA")
    print("="*80)
    
    all_data = []
    
    # Load from each condition's sum sheet
    condition_sheets = {
        'free': 'sum_free100',
        'yoke': 'sum_yoke53',
        'pasv': 'sum_pasv55'
    }
    
    for condition, sheet_name in condition_sheets.items():
        df = pd.read_excel(data_file, sheet_name=sheet_name)
        
        # For free condition, filter to only the 18 selected subjects (select_18 == 'y')
        if condition == 'free' and 'select_18' in df.columns:
            df = df[df['select_18'] == 'y']
            print(f"\n{condition.upper()} condition ({sheet_name}): Filtered to 18 selected subjects")
        
        # Extract the last three columns: avg_sent_readtime, sum_readtime, trans_score
        if 'avg_sent_readtime' in df.columns and 'sum_readtime' in df.columns and 'trans_score' in df.columns:
            for _, row in df.iterrows():
                all_data.append({
                    'condition': condition,
                    'avg_sent_readtime': row['avg_sent_readtime'],
                    'sum_readtime': row['sum_readtime'],
                    'trans_score': row['trans_score']
                })
            print(f"  Subjects: {len(df)}")
            print(f"  avg_sent_readtime: M={df['avg_sent_readtime'].mean():.3f}, SD={df['avg_sent_readtime'].std():.3f}")
            print(f"  sum_readtime: M={df['sum_readtime'].mean():.3f}, SD={df['sum_readtime'].std():.3f}")
            print(f"  trans_score: M={df['trans_score'].mean():.3f}, SD={df['trans_score'].std():.3f}")
        else:
            print(f"Warning: Required columns not found in {sheet_name}")
    
    engagement_df = pd.DataFrame(all_data)
    engagement_df = engagement_df.dropna()  # Remove any rows with missing data
    
    print(f"\nTotal subjects with complete data: {len(engagement_df)}")
    
    return engagement_df

def perform_engagement_anova(engagement_df, measure_name):
    """Perform one-way ANOVA on a reading/engagement measure"""
    
    # Fit ANOVA model
    model = ols(f'{measure_name} ~ C(condition)', data=engagement_df).fit()
    anova_table = anova_lm(model, typ=2)
    
    # Extract statistics
    f_stat = anova_table.loc['C(condition)', 'F']
    df_between = int(anova_table.loc['C(condition)', 'df'])
    df_within = int(anova_table.loc['Residual', 'df'])
    p_value = anova_table.loc['C(condition)', 'PR(>F)']
    
    # Calculate group means and standard deviations
    group_stats = engagement_df.groupby('condition')[measure_name].agg(['mean', 'std', 'count'])
    
    print(f"\n{measure_name.upper()} - One-Way ANOVA Results:")
    print(f"  F({df_between},{df_within}) = {f_stat:.3f}, p = {p_value:.3f}")
    print(f"\nGroup Statistics:")
    for cond in sorted(group_stats.index):
        mean_val = group_stats.loc[cond, 'mean']
        std_val = group_stats.loc[cond, 'std']
        n_val = int(group_stats.loc[cond, 'count'])
        print(f"    {cond}: M = {mean_val:.3f}, SD = {std_val:.3f}, N = {n_val}")
    
    return {
        'measure': measure_name,
        'f_stat': f_stat,
        'df_between': df_between,
        'df_within': df_within,
        'p_value': p_value,
        'group_stats': group_stats,
        'anova_table': anova_table
    }

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
    
    # Load and analyze Romance reading time and engagement data
    engagement_df = load_romance_reading_engagement_data()
    
    print("\n" + "="*80)
    print("PERFORMING ANOVAs ON READING TIME AND ENGAGEMENT")
    print("="*80)
    
    engagement_results = {}
    for measure in ['trans_score', 'avg_sent_readtime', 'sum_readtime']:
        engagement_results[measure] = perform_engagement_anova(engagement_df, measure)
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING REPORT")
    print("="*80)
    
    # Use current directory
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run1_overall_recall_comparison")
    
    report_file = os.path.join(output_dir, "overall_recall_anova_results.txt")
    generate_report(results_ba, results_mv, engagement_results, report_file)
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
        # Engagement measures
        for measure, result in engagement_results.items():
            result['group_stats'].to_excel(writer, sheet_name=f'Engagement_{measure}_stats')
            result['anova_table'].to_excel(writer, sheet_name=f'Engagement_{measure}_anova')
    
    print(f"Saved detailed statistics to: {stats_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return results_ba, results_mv, engagement_results

if __name__ == "__main__":
    main()

