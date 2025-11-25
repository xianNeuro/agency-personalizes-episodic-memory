#!/usr/bin/env python3
"""
Agency Reduced the Influence of Semantic but Not Causal Centrality on Recall
Compares semantic and causal centrality across conditions (Free, Yoked, Passive).

Tests:
1. One-way ANOVA for semantic centrality across conditions
2. One-way ANOVA for causal centrality across conditions
3. Two-way ANOVA (Network Type × Condition) to test for interaction

For both Adventure (BA) and Romance (MV) stories.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway, ttest_ind
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
from data_structure import RecallDataLoader

def load_centrality_data(story):
    """Load semantic and causal centrality data"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    if story == 'BA':
        data_file = os.path.join(data_dir, "adventure_data2.xlsx")
    else:  # MV
        data_file = os.path.join(data_dir, "romance_data2.xlsx")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading centrality data from: {data_file}")
    df = pd.read_excel(data_file)
    
    if 'sem-ef' not in df.columns or 'caus-ef' not in df.columns:
        raise ValueError(f"Required columns (sem-ef, caus-ef) not found")
    
    if 'cond' not in df.columns:
        raise ValueError(f"'cond' column not found")
    
    print(f"Loaded {len(df)} subjects")
    print(f"Conditions: {df['cond'].value_counts().to_dict()}")
    
    return df

def run_oneway_anova(values_by_condition, measure_name, story, use_z_transform=False):
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
    
    # Add post-hoc tests if p < 0.1
    posthoc_results = None
    if p_val < 0.1 and len(condition_values) == 3:
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
    
    # Yoke vs Pasv
    if yoke_idx is not None and pasv_idx is not None:
        t_stat, p_val = ttest_ind(condition_values[yoke_idx], condition_values[pasv_idx])
        df = len(condition_values[yoke_idx]) + len(condition_values[pasv_idx]) - 2
        posthoc_results['yoke_vs_pasv'] = {
            't_stat': t_stat,
            'p_val': p_val,
            'df': df
        }
        print(f"    Yoke vs Pasv: t({df}) = {t_stat:.4f}, p = {p_val:.6f}")
    
    return posthoc_results

def run_repeated_measures_anova(df, story, use_z_transform=False):
    """Run repeated measures ANOVA (Network Type × Condition)
    
    Network Type is within-subjects (each subject has both semantic and causal)
    Condition is between-subjects
    
    Uses difference score approach: calculate semantic - causal for each subject,
    then test if this difference varies across conditions (interaction effect).
    """
    
    # Prepare data: calculate difference score (semantic - causal) for each subject
    diff_data = []
    
    for idx, row in df.iterrows():
        # Only include subjects with both semantic and causal measures
        if pd.notna(row['sem-ef']) and pd.notna(row['caus-ef']):
            sem_val = row['sem-ef']
            caus_val = row['caus-ef']
            
            # Apply Fisher z-transform if requested
            if use_z_transform:
                sem_val = np.arctanh(np.clip(sem_val, -0.999, 0.999))
                caus_val = np.arctanh(np.clip(caus_val, -0.999, 0.999))
            
            diff_data.append({
                'condition': row['cond'],
                'semantic': sem_val,
                'causal': caus_val,
                'difference': sem_val - caus_val
            })
    
    if len(diff_data) == 0:
        print("Insufficient data for repeated measures ANOVA")
        return None
    
    diff_df = pd.DataFrame(diff_data)
    
    try:
        # Test difference scores across conditions (interaction test)
        # If the difference (semantic - causal) varies by condition, that's the interaction
        conditions = ['free', 'yoke', 'pasv']
        diff_by_condition = []
        condition_names = []
        
        for condition in conditions:
            cond_diffs = diff_df[diff_df['condition'] == condition]['difference'].values
            if len(cond_diffs) > 0:
                diff_by_condition.append(cond_diffs)
                condition_names.append(condition)
        
        if len(diff_by_condition) >= 2:
            f_stat, p_val = f_oneway(*diff_by_condition)
            total_n = sum(len(d) for d in diff_by_condition)
            df_between = len(diff_by_condition) - 1
            df_within = total_n - len(diff_by_condition)
            
            val_type = 'z' if use_z_transform else 'r'
            transform_label = ' (Fisher z-transformed)' if use_z_transform else ' (Raw values)'
            
            print(f"\nRepeated Measures ANOVA (Network Type × Condition){transform_label}:")
            print(f"  Testing difference scores (semantic - causal) across conditions")
            print(f"  This tests the interaction effect")
            print(f"  F({int(df_between)},{int(df_within)}) = {f_stat:.2f}, p = {p_val:.6f}")
            
            # Report means of difference scores by condition
            print(f"  Difference score means by condition:")
            for i, condition in enumerate(condition_names):
                mean_diff = np.mean(diff_by_condition[i])
                std_diff = np.std(diff_by_condition[i], ddof=1)
                n = len(diff_by_condition[i])
                print(f"    {condition}: Mean diff ({val_type}) = {mean_diff:.4f}, Std = {std_diff:.4f}, N = {n}")
            
            results = {
                'f_stat': f_stat,
                'p_val': p_val,
                'df_between': df_between,
                'df_within': df_within,
                'method': 'difference_score',
                'use_z_transform': use_z_transform,
                'val_type': val_type
            }
            
            if p_val < 0.001:
                print(f"  Result: Significant interaction (p < 0.001)")
            elif p_val < 0.01:
                print(f"  Result: Significant interaction (p < 0.01)")
            elif p_val < 0.05:
                print(f"  Result: Significant interaction (p < 0.05)")
            else:
                print(f"  Result: No significant interaction (p >= 0.05)")
            
            return results
        else:
            print("Insufficient data for repeated measures ANOVA")
            return None
            
    except Exception as e:
        print(f"Error running repeated measures ANOVA: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_agency_centrality():
    """Main analysis function"""
    
    print("="*80)
    print("AGENCY REDUCED THE INFLUENCE OF SEMANTIC BUT NOT CAUSAL CENTRALITY")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run6_agency_reduced_semantic_centrality")
    
    stories = ['BA', 'MV']
    story_names = {'BA': 'Adventure', 'MV': 'Romance'}
    
    all_results = {}
    
    # Run analyses for both stories
    for story in stories:
        print("\n" + "="*80)
        print(f"{story_names[story]} STORY ({story})")
        print("="*80)
        
        # Load data
        df = load_centrality_data(story)
        
        # Prepare data by condition
        sem_by_condition = {}
        caus_by_condition = {}
        
        for condition in ['free', 'yoke', 'pasv']:
            cond_data = df[df['cond'] == condition]
            sem_by_condition[condition] = cond_data['sem-ef'].dropna().values
            caus_by_condition[condition] = cond_data['caus-ef'].dropna().values
        
        story_results = {}
        
        # Run analyses for both raw and z-transformed values
        for use_z in [False, True]:
            analysis_label = "FISHER Z-TRANSFORMED" if use_z else "RAW VALUES"
            
            print("\n" + "="*80)
            print(f"ANALYSIS: {analysis_label}")
            print("="*80)
            
            # One-way ANOVA for semantic centrality
            print("\n" + "="*80)
            print(f"SEMANTIC CENTRALITY (sem-ef) - One-way ANOVA ({analysis_label})")
            print("="*80)
            sem_anova = run_oneway_anova(sem_by_condition, "Semantic Centrality", story, use_z_transform=use_z)
            
            # One-way ANOVA for causal centrality
            print("\n" + "="*80)
            print(f"CAUSAL CENTRALITY (caus-ef) - One-way ANOVA ({analysis_label})")
            print("="*80)
            caus_anova = run_oneway_anova(caus_by_condition, "Causal Centrality", story, use_z_transform=use_z)
            
            # Repeated Measures ANOVA (Network Type × Condition)
            print("\n" + "="*80)
            print(f"REPEATED MEASURES ANOVA (Network Type × Condition) ({analysis_label})")
            print("="*80)
            print("(Network Type: within-subjects, Condition: between-subjects)")
            rm_anova = run_repeated_measures_anova(df, story, use_z_transform=use_z)
            
            key_suffix = '_z' if use_z else '_raw'
            story_results[f'sem_anova{key_suffix}'] = sem_anova
            story_results[f'caus_anova{key_suffix}'] = caus_anova
            story_results[f'rm_anova{key_suffix}'] = rm_anova
        
        story_results['sem_by_condition'] = sem_by_condition
        story_results['caus_by_condition'] = caus_by_condition
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
            # Semantic ANOVA
            sem_key = f'sem_anova{suffix}'
            if sem_key in results and results[sem_key] is not None:
                sem = results[sem_key]
                summary_data.append({
                    'Story': story_name,
                    'Analysis': 'One-way ANOVA',
                    'Transform': transform_label,
                    'Measure': 'Semantic Centrality',
                    'F_statistic': sem['f_stat'],
                    'df_between': sem['df_between'],
                    'df_within': sem['df_within'],
                    'p_value': sem['p_val']
                })
                
                # Post-hoc tests for semantic centrality
                if sem.get('posthoc') is not None:
                    posthoc = sem['posthoc']
                    for comparison, stats_dict in posthoc.items():
                        summary_data.append({
                            'Story': story_name,
                            'Analysis': 'Post-hoc t-test',
                            'Transform': transform_label,
                            'Measure': f'Semantic Centrality: {comparison}',
                            'F_statistic': np.nan,
                            'df_between': np.nan,
                            'df_within': stats_dict['df'],
                            'p_value': stats_dict['p_val'],
                            't_statistic': stats_dict['t_stat']
                        })
            
            # Causal ANOVA
            caus_key = f'caus_anova{suffix}'
            if caus_key in results and results[caus_key] is not None:
                caus = results[caus_key]
                summary_data.append({
                    'Story': story_name,
                    'Analysis': 'One-way ANOVA',
                    'Transform': transform_label,
                    'Measure': 'Causal Centrality',
                    'F_statistic': caus['f_stat'],
                    'df_between': caus['df_between'],
                    'df_within': caus['df_within'],
                    'p_value': caus['p_val']
                })
            
            # Repeated Measures ANOVA interaction
            rm_key = f'rm_anova{suffix}'
            if rm_key in results and results[rm_key] is not None:
                rm = results[rm_key]
                summary_data.append({
                    'Story': story_name,
                    'Analysis': 'Repeated Measures ANOVA',
                    'Transform': transform_label,
                    'Measure': 'Network Type × Condition Interaction',
                    'F_statistic': rm['f_stat'],
                    'df_between': rm['df_between'],
                    'df_within': rm['df_within'],
                    'p_value': rm['p_val']
                })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "agency_centrality_anova_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("AGENCY REDUCED THE INFLUENCE OF SEMANTIC BUT NOT CAUSAL CENTRALITY")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Adventure (BA): adventure_data2.xlsx")
    report_lines.append("  - Romance (MV): romance_data2.xlsx")
    report_lines.append("")
    report_lines.append("Measures:")
    report_lines.append("  - Semantic centrality (sem-ef): Semantic influence on memory")
    report_lines.append("  - Causal centrality (caus-ef): Causal influence on memory")
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
            
            # Semantic ANOVA
            sem_key = f'sem_anova{suffix}'
            if sem_key in results and results[sem_key] is not None:
                sem = results[sem_key]
                val_type = sem.get('val_type', 'r')
                report_lines.append(f"Semantic Centrality (sem-ef) - One-way ANOVA ({transform_label}):")
                report_lines.append(f"  F({int(sem['df_between'])},{int(sem['df_within'])}) = {sem['f_stat']:.2f}, p = {sem['p_val']:.6f}")
                if sem['p_val'] < 0.001:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.001)")
                elif sem['p_val'] < 0.01:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.01)")
                elif sem['p_val'] < 0.05:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.05)")
                else:
                    report_lines.append(f"  Result: No significant difference (p >= 0.05)")
                report_lines.append("")
                report_lines.append("  Means by condition:")
                for i, condition in enumerate(sem['condition_names']):
                    mean_val = np.mean(sem['condition_values'][i])
                    std_val = np.std(sem['condition_values'][i], ddof=1)
                    n = len(sem['condition_values'][i])
                    report_lines.append(f"    {condition}: Mean {val_type} = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
                report_lines.append("")
                
                # Post-hoc tests
                if sem.get('posthoc') is not None:
                    posthoc = sem['posthoc']
                    report_lines.append("  Post-hoc t-tests:")
                    for comparison, stats_dict in posthoc.items():
                        comp_name = comparison.replace('_', ' vs ').title()
                        report_lines.append(f"    {comp_name}: t({int(stats_dict['df'])}) = {stats_dict['t_stat']:.4f}, p = {stats_dict['p_val']:.6f}")
                    report_lines.append("")
            
            # Causal ANOVA
            caus_key = f'caus_anova{suffix}'
            if caus_key in results and results[caus_key] is not None:
                caus = results[caus_key]
                val_type = caus.get('val_type', 'r')
                report_lines.append(f"Causal Centrality (caus-ef) - One-way ANOVA ({transform_label}):")
                report_lines.append(f"  F({int(caus['df_between'])},{int(caus['df_within'])}) = {caus['f_stat']:.2f}, p = {caus['p_val']:.6f}")
                if caus['p_val'] < 0.001:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.001)")
                elif caus['p_val'] < 0.01:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.01)")
                elif caus['p_val'] < 0.05:
                    report_lines.append(f"  Result: Significant difference across conditions (p < 0.05)")
                else:
                    report_lines.append(f"  Result: No significant difference (p >= 0.05)")
                report_lines.append("")
                report_lines.append("  Means by condition:")
                for i, condition in enumerate(caus['condition_names']):
                    mean_val = np.mean(caus['condition_values'][i])
                    std_val = np.std(caus['condition_values'][i], ddof=1)
                    n = len(caus['condition_values'][i])
                    report_lines.append(f"    {condition}: Mean {val_type} = {mean_val:.4f}, Std = {std_val:.4f}, N = {n}")
                report_lines.append("")
            
            # Repeated Measures ANOVA
            rm_key = f'rm_anova{suffix}'
            if rm_key in results and results[rm_key] is not None:
                rm = results[rm_key]
                val_type = rm.get('val_type', 'r')
                report_lines.append(f"Repeated Measures ANOVA (Network Type × Condition) - Interaction Effect ({transform_label}):")
                report_lines.append("  (Network Type: within-subjects, Condition: between-subjects)")
                report_lines.append(f"  F({int(rm['df_between'])},{int(rm['df_within'])}) = {rm['f_stat']:.2f}, p = {rm['p_val']:.6f}")
                if rm['p_val'] < 0.001:
                    report_lines.append(f"  Result: Significant interaction (p < 0.001)")
                elif rm['p_val'] < 0.01:
                    report_lines.append(f"  Result: Significant interaction (p < 0.01)")
                elif rm['p_val'] < 0.05:
                    report_lines.append(f"  Result: Significant interaction (p < 0.05)")
                else:
                    report_lines.append(f"  Result: No significant interaction (p >= 0.05)")
                report_lines.append("")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "agency_centrality_anova_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_agency_centrality()

