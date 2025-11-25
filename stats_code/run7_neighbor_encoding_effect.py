#!/usr/bin/env python3
"""
Neighbor Encoding Effect Analysis
Analyzes the neighbor encoding effect (nghb-ef) across conditions.

For both stories (Adventure/BA and Romance/MV), tests:
1. One-sample t-tests against zero for each condition (to show positive effect)
2. One-way ANOVA across conditions
3. Post-hoc t-tests (Free vs Yoked, Free vs Passive) when ANOVA is significant

Runs analyses on both raw values and Fisher z-transformed values.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway, ttest_1samp, ttest_ind
from data_structure import RecallDataLoader

def load_neighbor_encoding_data(story):
    """Load neighbor encoding effect data from comp_conds sheet"""
    
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
    
    print(f"Loading neighbor encoding data from: {data_file} (sheet: {sheet_name})")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    if 'nghb-ef' not in df.columns:
        raise ValueError(f"Required column 'nghb-ef' not found in {data_file}")
    
    if 'cond' not in df.columns:
        raise ValueError(f"'cond' column not found in {data_file}")
    
    print(f"Loaded {len(df)} subjects")
    print(f"Conditions: {df['cond'].value_counts().to_dict()}")
    
    return df

def run_one_sample_tests(values_by_condition, measure_name, use_z_transform=False):
    """Run one-sample t-tests against zero for each condition"""
    
    conditions = ['free', 'yoke', 'pasv']
    results = {}
    
    val_type = 'z' if use_z_transform else 'r'
    transform_label = ' (Fisher z-transformed)' if use_z_transform else ' (Raw values)'
    
    print(f"\n{measure_name} - One-sample t-tests against zero{transform_label}:")
    
    for condition in conditions:
        if condition in values_by_condition and len(values_by_condition[condition]) > 0:
            vals = values_by_condition[condition].copy()
            
            # Apply Fisher z-transform if requested
            if use_z_transform:
                vals = np.arctanh(np.clip(vals, -0.999, 0.999))
            
            # One-sample t-test against zero
            t_stat, p_val = ttest_1samp(vals, 0)
            n = len(vals)
            mean_val = np.mean(vals)
            std_val = np.std(vals, ddof=1)
            
            results[condition] = {
                'values': vals,
                'mean': mean_val,
                'std': std_val,
                'n': n,
                't_stat': t_stat,
                'p_val': p_val
            }
            
            print(f"  {condition.upper()} Condition:")
            print(f"    N: {n}")
            print(f"    Mean {val_type}: {mean_val:.4f}")
            print(f"    Std: {std_val:.4f}")
            print(f"    One-sample t-test (vs 0):")
            print(f"      t({n-1}) = {t_stat:.4f}, p = {p_val:.6f}")
            if p_val < 0.001:
                print(f"    Result: Significantly positive (p < 0.001)")
            elif p_val < 0.01:
                print(f"    Result: Significantly positive (p < 0.01)")
            elif p_val < 0.05:
                print(f"    Result: Significantly positive (p < 0.05)")
            else:
                print(f"    Result: Not significantly different from zero (p >= 0.05)")
        else:
            print(f"  {condition.upper()} Condition: No data")
            results[condition] = None
    
    return results

def run_oneway_anova(values_by_condition, measure_name, use_z_transform=False):
    """Run one-way ANOVA across conditions"""
    
    conditions = ['free', 'yoke', 'pasv']
    condition_values = []
    condition_names = []
    
    for condition in conditions:
        if condition in values_by_condition and len(values_by_condition[condition]) > 0:
            vals = values_by_condition[condition].copy()
            # Apply Fisher z-transform if requested
            if use_z_transform:
                vals = np.arctanh(np.clip(vals, -0.999, 0.999))
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
    
    val_type = 'z' if use_z_transform else 'r'
    transform_label = ' (Fisher z-transformed)' if use_z_transform else ' (Raw values)'
    
    results = {
        'f_stat': f_stat,
        'p_val': p_val,
        'df_between': df_between,
        'df_within': df_within,
        'total_n': total_n,
        'condition_values': condition_values,
        'condition_names': condition_names,
        'use_z_transform': use_z_transform,
        'val_type': val_type
    }
    
    print(f"\n{measure_name} - One-way ANOVA across conditions{transform_label}:")
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
        print(f"    {condition}: Mean {val_type} = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
    
    # Add post-hoc tests if ANOVA is significant (p < 0.05)
    posthoc_results = None
    if p_val < 0.05 and len(condition_values) == 3:
        posthoc_results = run_posthoc_tests(condition_values, condition_names, measure_name + transform_label)
    
    results['posthoc'] = posthoc_results
    
    return results

def run_posthoc_tests(condition_values, condition_names, measure_name):
    """Run post-hoc t-tests between pairs of conditions"""
    
    if len(condition_values) != 3 or len(condition_names) != 3:
        return None
    
    # Get indices for each condition
    free_idx = condition_names.index('free') if 'free' in condition_names else None
    yoke_idx = condition_names.index('yoke') if 'yoke' in condition_names else None
    pasv_idx = condition_names.index('pasv') if 'pasv' in condition_names else None
    
    posthoc_results = {}
    
    print(f"\n  Post-hoc t-tests ({measure_name}):")
    
    # Free vs Yoke
    if free_idx is not None and yoke_idx is not None:
        t_stat, p_val = ttest_ind(condition_values[free_idx], condition_values[yoke_idx])
        df = len(condition_values[free_idx]) + len(condition_values[yoke_idx]) - 2
        posthoc_results['free_vs_yoke'] = {
            't_stat': t_stat,
            'p_val': p_val,
            'df': df
        }
        print(f"    Free vs Yoke: t({df}) = {t_stat:.4f}, p = {p_val:.6f}")
    
    # Free vs Pasv
    if free_idx is not None and pasv_idx is not None:
        t_stat, p_val = ttest_ind(condition_values[free_idx], condition_values[pasv_idx])
        df = len(condition_values[free_idx]) + len(condition_values[pasv_idx]) - 2
        posthoc_results['free_vs_pasv'] = {
            't_stat': t_stat,
            'p_val': p_val,
            'df': df
        }
        print(f"    Free vs Pasv: t({df}) = {t_stat:.4f}, p = {p_val:.6f}")
    
    return posthoc_results

def analyze_neighbor_encoding():
    """Main analysis function"""
    
    print("="*80)
    print("NEIGHBOR ENCODING EFFECT ANALYSIS")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run7_neighbor_encoding_effect")
    
    stories = ['BA', 'MV']
    story_names = {'BA': 'Adventure', 'MV': 'Romance'}
    
    all_results = {}
    
    # Run analyses for both stories
    for story in stories:
        print("\n" + "="*80)
        print(f"{story_names[story]} STORY ({story})")
        print("="*80)
        
        # Load data
        df = load_neighbor_encoding_data(story)
        
        # Prepare data by condition
        nghb_by_condition = {}
        
        for condition in ['free', 'yoke', 'pasv']:
            cond_data = df[df['cond'] == condition]
            nghb_by_condition[condition] = cond_data['nghb-ef'].dropna().values
        
        story_results = {}
        
        # Run analyses for both raw and z-transformed values
        for use_z in [False, True]:
            analysis_label = "FISHER Z-TRANSFORMED" if use_z else "RAW VALUES"
            
            print("\n" + "="*80)
            print(f"ANALYSIS: {analysis_label}")
            print("="*80)
            
            # One-sample t-tests against zero
            print("\n" + "="*80)
            print(f"NEIGHBOR ENCODING EFFECT (nghb-ef) - One-sample t-tests ({analysis_label})")
            print("="*80)
            one_sample_results = run_one_sample_tests(nghb_by_condition, "Neighbor Encoding Effect", use_z_transform=use_z)
            
            # One-way ANOVA across conditions
            print("\n" + "="*80)
            print(f"NEIGHBOR ENCODING EFFECT (nghb-ef) - One-way ANOVA ({analysis_label})")
            print("="*80)
            anova_results = run_oneway_anova(nghb_by_condition, "Neighbor Encoding Effect", use_z_transform=use_z)
            
            key_suffix = '_z' if use_z else '_raw'
            story_results[f'one_sample{key_suffix}'] = one_sample_results
            story_results[f'anova{key_suffix}'] = anova_results
        
        story_results['nghb_by_condition'] = nghb_by_condition
        all_results[story] = story_results
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for story in stories:
        story_name = story_names[story]
        results = all_results[story]
        
        # Process both raw and z-transformed results
        for suffix, transform_label in [('_raw', 'Raw values'), ('_z', 'Fisher z-transformed')]:
            # One-sample t-tests
            one_sample_key = f'one_sample{suffix}'
            if one_sample_key in results:
                one_sample = results[one_sample_key]
                for condition in ['free', 'yoke', 'pasv']:
                    if condition in one_sample and one_sample[condition] is not None:
                        stats_dict = one_sample[condition]
                        summary_data.append({
                            'Story': story_name,
                            'Analysis': 'One-sample t-test',
                            'Transform': transform_label,
                            'Condition': condition,
                            'Measure': 'Neighbor Encoding Effect',
                            'N': stats_dict['n'],
                            'Mean': stats_dict['mean'],
                            'Std': stats_dict['std'],
                            't_statistic': stats_dict['t_stat'],
                            'p_value': stats_dict['p_val']
                        })
            
            # One-way ANOVA
            anova_key = f'anova{suffix}'
            if anova_key in results and results[anova_key] is not None:
                anova = results[anova_key]
                summary_data.append({
                    'Story': story_name,
                    'Analysis': 'One-way ANOVA',
                    'Transform': transform_label,
                    'Condition': 'All',
                    'Measure': 'Neighbor Encoding Effect',
                    'F_statistic': anova['f_stat'],
                    'df_between': anova['df_between'],
                    'df_within': anova['df_within'],
                    'p_value': anova['p_val']
                })
                
                # Post-hoc tests
                if anova.get('posthoc') is not None:
                    posthoc = anova['posthoc']
                    for comparison, stats_dict in posthoc.items():
                        summary_data.append({
                            'Story': story_name,
                            'Analysis': 'Post-hoc t-test',
                            'Transform': transform_label,
                            'Condition': comparison.replace('_', ' vs '),
                            'Measure': 'Neighbor Encoding Effect',
                            't_statistic': stats_dict['t_stat'],
                            'df_within': stats_dict['df'],
                            'p_value': stats_dict['p_val']
                        })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "neighbor_encoding_effect_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("NEIGHBOR ENCODING EFFECT ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Adventure (BA): adventure_data2.xlsx (sheet: comp_conds)")
    report_lines.append("  - Romance (MV): romance_data2.xlsx (sheet: comp_conds18)")
    report_lines.append("")
    report_lines.append("Measure:")
    report_lines.append("  - Neighbor encoding effect (nghb-ef): Tendency for temporally neighboring")
    report_lines.append("    events at encoding to share the same subsequent memory status")
    report_lines.append("")
    
    for story in stories:
        story_name = story_names[story]
        results = all_results[story]
        
        report_lines.append("="*80)
        report_lines.append(f"{story_name.upper()} STORY ({story})")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Process both raw and z-transformed results
        for suffix, transform_label in [('_raw', 'Raw values'), ('_z', 'Fisher z-transformed')]:
            report_lines.append("-" * 80)
            report_lines.append(f"ANALYSIS: {transform_label.upper()}")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            # One-sample t-tests
            one_sample_key = f'one_sample{suffix}'
            if one_sample_key in results:
                one_sample = results[one_sample_key]
                report_lines.append("One-sample t-tests against zero:")
                for condition in ['free', 'yoke', 'pasv']:
                    if condition in one_sample and one_sample[condition] is not None:
                        stats_dict = one_sample[condition]
                        val_type = stats_dict.get('val_type', 'r') if 'val_type' in stats_dict else ('z' if suffix == '_z' else 'r')
                        report_lines.append(f"  {condition.upper()} Condition:")
                        report_lines.append(f"    N: {stats_dict['n']}")
                        report_lines.append(f"    Mean {val_type}: {stats_dict['mean']:.4f}")
                        report_lines.append(f"    Std: {stats_dict['std']:.4f}")
                        report_lines.append(f"    t({stats_dict['n']-1}) = {stats_dict['t_stat']:.4f}, p = {stats_dict['p_val']:.6f}")
                        if stats_dict['p_val'] < 0.001:
                            report_lines.append(f"    Result: Significantly positive (p < 0.001)")
                        elif stats_dict['p_val'] < 0.01:
                            report_lines.append(f"    Result: Significantly positive (p < 0.01)")
                        elif stats_dict['p_val'] < 0.05:
                            report_lines.append(f"    Result: Significantly positive (p < 0.05)")
                        else:
                            report_lines.append(f"    Result: Not significantly different from zero (p >= 0.05)")
                report_lines.append("")
            
            # One-way ANOVA
            anova_key = f'anova{suffix}'
            if anova_key in results and results[anova_key] is not None:
                anova = results[anova_key]
                val_type = anova.get('val_type', 'r')
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
                    report_lines.append(f"    {condition}: Mean {val_type} = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
                report_lines.append("")
                
                # Post-hoc tests
                if anova.get('posthoc') is not None:
                    posthoc = anova['posthoc']
                    report_lines.append("  Post-hoc t-tests:")
                    for comparison, stats_dict in posthoc.items():
                        comp_name = comparison.replace('_', ' vs ').title()
                        report_lines.append(f"    {comp_name}: t({int(stats_dict['df'])}) = {stats_dict['t_stat']:.4f}, p = {stats_dict['p_val']:.6f}")
                    report_lines.append("")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "neighbor_encoding_effect_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_neighbor_encoding()

