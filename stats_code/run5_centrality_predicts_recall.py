#!/usr/bin/env python3
"""
Event Recall Predicted by Semantic and Causal Centrality
Analyzes how semantic and causal centrality predict recall.

For both stories (Adventure/BA and Romance/MV), tests whether:
- Semantic centrality (sem-ef) significantly predicts recall
- Causal centrality (caus-ef) significantly predicts recall

Runs one-sample t-tests against zero for each condition (free, yoke, pasv).

Runs two analyses:
1. Raw values
2. Fisher z-transformed values
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_1samp
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
        raise ValueError(f"Required columns (sem-ef, caus-ef) not found in {data_file}")
    
    if 'cond' not in df.columns:
        raise ValueError(f"'cond' column not found in {data_file}")
    
    print(f"Loaded {len(df)} subjects")
    print(f"Conditions: {df['cond'].value_counts().to_dict()}")
    
    return df

def analyze_centrality_by_condition(df, story, use_z_transform=False):
    """Analyze semantic and causal centrality by condition"""
    
    conditions = ['free', 'yoke', 'pasv']
    results = {}
    
    for condition in conditions:
        cond_data = df[df['cond'] == condition].copy()
        
        if len(cond_data) == 0:
            print(f"\n{condition.upper()} condition: No data")
            results[condition] = None
            continue
        
        # Extract semantic and causal centrality
        sem_ef = cond_data['sem-ef'].dropna().values
        caus_ef = cond_data['caus-ef'].dropna().values
        
        if len(sem_ef) == 0 or len(caus_ef) == 0:
            print(f"\n{condition.upper()} condition: Insufficient data")
            results[condition] = None
            continue
        
        # Apply Fisher z-transform if requested
        if use_z_transform:
            sem_ef = np.arctanh(np.clip(sem_ef, -0.999, 0.999))
            caus_ef = np.arctanh(np.clip(caus_ef, -0.999, 0.999))
        
        # One-sample t-tests against zero
        sem_t, sem_p = ttest_1samp(sem_ef, 0)
        caus_t, caus_p = ttest_1samp(caus_ef, 0)
        
        results[condition] = {
            'sem_ef': {
                'values': sem_ef,
                'mean': np.mean(sem_ef),
                'std': np.std(sem_ef, ddof=1),
                'n': len(sem_ef),
                't_stat': sem_t,
                'p_val': sem_p
            },
            'caus_ef': {
                'values': caus_ef,
                'mean': np.mean(caus_ef),
                'std': np.std(caus_ef, ddof=1),
                'n': len(caus_ef),
                't_stat': caus_t,
                'p_val': caus_p
            }
        }
        
        val_type = 'z' if use_z_transform else 'r'
        print(f"\n{condition.upper()} Condition:")
        print(f"  Semantic centrality (sem-ef):")
        print(f"    N: {len(sem_ef)}")
        print(f"    Mean {val_type}: {np.mean(sem_ef):.4f}")
        print(f"    Std: {np.std(sem_ef, ddof=1):.4f}")
        print(f"    One-sample t-test (vs 0):")
        print(f"      t({len(sem_ef)-1}) = {sem_t:.4f}, p = {sem_p:.6f}")
        
        print(f"  Causal centrality (caus-ef):")
        print(f"    N: {len(caus_ef)}")
        print(f"    Mean {val_type}: {np.mean(caus_ef):.4f}")
        print(f"    Std: {np.std(caus_ef, ddof=1):.4f}")
        print(f"    One-sample t-test (vs 0):")
        print(f"      t({len(caus_ef)-1}) = {caus_t:.4f}, p = {caus_p:.6f}")
    
    return results

def analyze_centrality():
    """Main analysis function"""
    
    print("="*80)
    print("EVENT RECALL PREDICTED BY SEMANTIC AND CAUSAL CENTRALITY")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run5_centrality_predicts_recall")
    
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
        
        # Analysis 1: Raw values
        print("\n" + "="*80)
        print(f"ANALYSIS 1: RAW VALUES ({story_names[story]})")
        print("="*80)
        results_raw = analyze_centrality_by_condition(df, story, use_z_transform=False)
        all_results[f'{story}_raw'] = results_raw
        
        # Analysis 2: Fisher z-transformed
        print("\n" + "="*80)
        print(f"ANALYSIS 2: FISHER Z-TRANSFORMED ({story_names[story]})")
        print("="*80)
        results_z = analyze_centrality_by_condition(df, story, use_z_transform=True)
        all_results[f'{story}_z'] = results_z
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for analysis_key, results in all_results.items():
        story = analysis_key.split('_')[0]
        analysis_type = analysis_key.split('_')[1]
        story_name = story_names[story]
        
        for condition in ['free', 'yoke', 'pasv']:
            if results.get(condition) is not None:
                # Semantic centrality
                sem = results[condition]['sem_ef']
                summary_data.append({
                    'Story': story_name,
                    'Analysis': analysis_type,
                    'Condition': condition,
                    'Centrality_Type': 'Semantic',
                    'N': sem['n'],
                    'Mean': sem['mean'],
                    'Std': sem['std'],
                    't_statistic': sem['t_stat'],
                    'p_value': sem['p_val']
                })
                
                # Causal centrality
                caus = results[condition]['caus_ef']
                summary_data.append({
                    'Story': story_name,
                    'Analysis': analysis_type,
                    'Condition': condition,
                    'Centrality_Type': 'Causal',
                    'N': caus['n'],
                    'Mean': caus['mean'],
                    'Std': caus['std'],
                    't_statistic': caus['t_stat'],
                    'p_value': caus['p_val']
                })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "centrality_predicts_recall_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EVENT RECALL PREDICTED BY SEMANTIC AND CAUSAL CENTRALITY")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Adventure (BA): adventure_data2.xlsx")
    report_lines.append("  - Romance (MV): romance_data2.xlsx")
    report_lines.append("")
    report_lines.append("Centrality measures:")
    report_lines.append("  - Semantic centrality (sem-ef): Semantic influence on memory")
    report_lines.append("  - Causal centrality (caus-ef): Causal influence on memory")
    report_lines.append("")
    report_lines.append("Analysis: One-sample t-tests against zero for each condition")
    report_lines.append("")
    
    for story in stories:
        story_name = story_names[story]
        report_lines.append("="*80)
        report_lines.append(f"{story_name.upper()} STORY ({story})")
        report_lines.append("="*80)
        report_lines.append("")
        
        for analysis_type in ['raw', 'z']:
            analysis_key = f'{story}_{analysis_type}'
            results = all_results.get(analysis_key, {})
            
            report_lines.append("-" * 80)
            report_lines.append(f"ANALYSIS: {analysis_type.upper().replace('_', ' ')}")
            if analysis_type == 'z':
                report_lines.append("(Fisher z-transformed)")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            for condition in ['free', 'yoke', 'pasv']:
                if results.get(condition) is not None:
                    sem = results[condition]['sem_ef']
                    caus = results[condition]['caus_ef']
                    val_type = 'z' if analysis_type == 'z' else 'r'
                    
                    report_lines.append(f"{condition.upper()} Condition:")
                    report_lines.append("")
                    
                    report_lines.append("  Semantic Centrality (sem-ef):")
                    report_lines.append(f"    N: {sem['n']}")
                    report_lines.append(f"    Mean {val_type}: {sem['mean']:.4f}")
                    report_lines.append(f"    Std: {sem['std']:.4f}")
                    report_lines.append(f"    One-sample t-test (vs 0):")
                    report_lines.append(f"      t({sem['n']-1}) = {sem['t_stat']:.4f}, p = {sem['p_val']:.6f}")
                    if sem['p_val'] < 0.001:
                        report_lines.append(f"    Result: Significant semantic influence on memory (p < 0.001)")
                    elif sem['p_val'] < 0.01:
                        report_lines.append(f"    Result: Significant semantic influence on memory (p < 0.01)")
                    elif sem['p_val'] < 0.05:
                        report_lines.append(f"    Result: Significant semantic influence on memory (p < 0.05)")
                    else:
                        report_lines.append(f"    Result: No significant semantic influence (p >= 0.05)")
                    report_lines.append("")
                    
                    report_lines.append("  Causal Centrality (caus-ef):")
                    report_lines.append(f"    N: {caus['n']}")
                    report_lines.append(f"    Mean {val_type}: {caus['mean']:.4f}")
                    report_lines.append(f"    Std: {caus['std']:.4f}")
                    report_lines.append(f"    One-sample t-test (vs 0):")
                    report_lines.append(f"      t({caus['n']-1}) = {caus['t_stat']:.4f}, p = {caus['p_val']:.6f}")
                    if caus['p_val'] < 0.001:
                        report_lines.append(f"    Result: Significant causal influence on memory (p < 0.001)")
                    elif caus['p_val'] < 0.01:
                        report_lines.append(f"    Result: Significant causal influence on memory (p < 0.01)")
                    elif caus['p_val'] < 0.05:
                        report_lines.append(f"    Result: Significant causal influence on memory (p < 0.05)")
                    else:
                        report_lines.append(f"    Result: No significant causal influence (p >= 0.05)")
                    report_lines.append("")
    
    report_file = os.path.join(output_dir, "centrality_predicts_recall_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_centrality()

