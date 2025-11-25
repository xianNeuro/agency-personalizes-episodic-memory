#!/usr/bin/env python3
"""
Memory Divergence and Semantic Influence Correlation Analysis
Analyzes correlations between memory divergence and semantic influence scores in Free participants.

For Romance story:
- Correlation between memory divergence (rcl-1-r_otherm) and semantic influence (sem-ef)
- Analysis for both 18 Free participants (corr_free18) and 100 Free participants (corr_free100)
- Also tests correlation between choice divergence (cho-1-r_otherm) and semantic influence
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from data_structure import RecallDataLoader

def load_correlation_data(story, n_subjects=None):
    """Load correlation data from corr_free sheets"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    if story == 'BA':
        data_file = os.path.join(data_dir, "adventure_data2.xlsx")
        # Check available corr_free sheets
        sheet_name = 'corr_free22'  # Based on previous checks
    else:  # MV
        data_file = os.path.join(data_dir, "romance_data2.xlsx")
        if n_subjects == 18:
            sheet_name = 'corr_free18'
        elif n_subjects == 100:
            sheet_name = 'corr_free100'
        else:
            raise ValueError(f"For Romance story, n_subjects must be 18 or 100, got {n_subjects}")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading correlation data from: {data_file} (sheet: {sheet_name})")
    df = pd.read_excel(data_file, sheet_name=sheet_name)
    
    required_cols = ['rcl-1-r_otherm', 'sem-ef']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns not found: {missing_cols}")
    
    print(f"Loaded {len(df)} subjects")
    
    return df

def analyze_correlations(df, story, n_subjects=None):
    """Analyze correlations between divergence and semantic influence"""
    
    results = {}
    
    # Memory divergence vs Semantic influence
    mem_div = df['rcl-1-r_otherm'].dropna().values
    sem_ef = df['sem-ef'].dropna().values
    
    # Align arrays (only use subjects with both measures)
    valid_mask = df['rcl-1-r_otherm'].notna() & df['sem-ef'].notna()
    mem_div_aligned = df.loc[valid_mask, 'rcl-1-r_otherm'].values
    sem_ef_aligned = df.loc[valid_mask, 'sem-ef'].values
    
    if len(mem_div_aligned) > 1:
        r_mem_sem, p_mem_sem = pearsonr(mem_div_aligned, sem_ef_aligned)
        n_mem_sem = len(mem_div_aligned)
        
        results['memory_divergence_vs_semantic'] = {
            'r': r_mem_sem,
            'p': p_mem_sem,
            'n': n_mem_sem
        }
        
        print(f"\nMemory Divergence vs Semantic Influence:")
        print(f"  N: {n_mem_sem}")
        print(f"  Correlation: r({n_mem_sem-2}) = {r_mem_sem:.3f}, p = {p_mem_sem:.6f}")
        if p_mem_sem < 0.001:
            print(f"  Result: Significant negative correlation (p < 0.001)")
        elif p_mem_sem < 0.01:
            print(f"  Result: Significant negative correlation (p < 0.01)")
        elif p_mem_sem < 0.05:
            print(f"  Result: Significant negative correlation (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        print(f"  Memory Divergence: Mean = {np.mean(mem_div_aligned):.4f}, Std = {np.std(mem_div_aligned, ddof=1):.4f}")
        print(f"  Semantic Influence: Mean = {np.mean(sem_ef_aligned):.4f}, Std = {np.std(sem_ef_aligned, ddof=1):.4f}")
    
    # Choice divergence vs Semantic influence (if available)
    if 'cho-1-r_otherm' in df.columns:
        choice_div = df['cho-1-r_otherm'].dropna().values
        
        # Align arrays
        valid_mask_choice = df['cho-1-r_otherm'].notna() & df['sem-ef'].notna()
        choice_div_aligned = df.loc[valid_mask_choice, 'cho-1-r_otherm'].values
        sem_ef_choice_aligned = df.loc[valid_mask_choice, 'sem-ef'].values
        
        if len(choice_div_aligned) > 1:
            r_choice_sem, p_choice_sem = pearsonr(choice_div_aligned, sem_ef_choice_aligned)
            n_choice_sem = len(choice_div_aligned)
            
            results['choice_divergence_vs_semantic'] = {
                'r': r_choice_sem,
                'p': p_choice_sem,
                'n': n_choice_sem
            }
            
            print(f"\nChoice Divergence vs Semantic Influence:")
            print(f"  N: {n_choice_sem}")
            print(f"  Correlation: r({n_choice_sem-2}) = {r_choice_sem:.3f}, p = {p_choice_sem:.6f}")
            if p_choice_sem < 0.001:
                print(f"  Result: Significant correlation (p < 0.001)")
            elif p_choice_sem < 0.01:
                print(f"  Result: Significant correlation (p < 0.01)")
            elif p_choice_sem < 0.05:
                print(f"  Result: Significant correlation (p < 0.05)")
            else:
                print(f"  Result: Not significant (p >= 0.05)")
            print(f"  Choice Divergence: Mean = {np.mean(choice_div_aligned):.4f}, Std = {np.std(choice_div_aligned, ddof=1):.4f}")
            print(f"  Semantic Influence: Mean = {np.mean(sem_ef_choice_aligned):.4f}, Std = {np.std(sem_ef_choice_aligned, ddof=1):.4f}")
    
    return results

def analyze_memory_divergence_semantic():
    """Main analysis function"""
    
    print("="*80)
    print("MEMORY DIVERGENCE AND SEMANTIC INFLUENCE CORRELATION ANALYSIS")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run9_memory_divergence_semantic_correlation")
    
    all_results = {}
    
    # For Romance story, analyze both 18 and 100 subjects
    print("\n" + "="*80)
    print("ROMANCE STORY (MV)")
    print("="*80)
    
    for n_subjects in [18, 100]:
        print("\n" + "="*80)
        print(f"ANALYSIS: {n_subjects} Free Participants")
        print("="*80)
        
        # Load data
        df = load_correlation_data('MV', n_subjects=n_subjects)
        
        # Analyze correlations
        results = analyze_correlations(df, 'MV', n_subjects=n_subjects)
        
        all_results[f'MV_{n_subjects}'] = {
            'data': df,
            'results': results
        }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for key, data_dict in all_results.items():
        story_n = key.split('_')
        story = story_n[0]
        n = int(story_n[1])
        results = data_dict['results']
        
        # Memory divergence vs Semantic
        if 'memory_divergence_vs_semantic' in results:
            mem_sem = results['memory_divergence_vs_semantic']
            summary_data.append({
                'Story': 'Romance',
                'N_subjects': n,
                'Correlation': 'Memory Divergence vs Semantic Influence',
                'r': mem_sem['r'],
                'n': mem_sem['n'],
                'p_value': mem_sem['p']
            })
        
        # Choice divergence vs Semantic
        if 'choice_divergence_vs_semantic' in results:
            choice_sem = results['choice_divergence_vs_semantic']
            summary_data.append({
                'Story': 'Romance',
                'N_subjects': n,
                'Correlation': 'Choice Divergence vs Semantic Influence',
                'r': choice_sem['r'],
                'n': choice_sem['n'],
                'p_value': choice_sem['p']
            })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "memory_divergence_semantic_correlation_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MEMORY DIVERGENCE AND SEMANTIC INFLUENCE CORRELATION ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Romance (MV): romance_data2.xlsx")
    report_lines.append("    - 18 Free participants: sheet corr_free18")
    report_lines.append("    - 100 Free participants: sheet corr_free100")
    report_lines.append("")
    report_lines.append("Measures:")
    report_lines.append("  - Memory divergence: rcl-1-r_otherm (1 - correlation with group average recall)")
    report_lines.append("  - Choice divergence: cho-1-r_otherm (1 - correlation with group average choice)")
    report_lines.append("  - Semantic influence: sem-ef (semantic centrality effect)")
    report_lines.append("")
    
    for n_subjects in [18, 100]:
        key = f'MV_{n_subjects}'
        if key in all_results:
            results = all_results[key]['results']
            
            report_lines.append("="*80)
            report_lines.append(f"ROMANCE STORY: {n_subjects} Free Participants")
            report_lines.append("="*80)
            report_lines.append("")
            
            # Memory divergence vs Semantic
            if 'memory_divergence_vs_semantic' in results:
                mem_sem = results['memory_divergence_vs_semantic']
                report_lines.append("Memory Divergence vs Semantic Influence:")
                report_lines.append(f"  N: {mem_sem['n']}")
                report_lines.append(f"  Correlation: r({mem_sem['n']-2}) = {mem_sem['r']:.3f}, p = {mem_sem['p']:.6f}")
                if mem_sem['p'] < 0.001:
                    report_lines.append(f"  Result: Significant negative correlation (p < 0.001)")
                elif mem_sem['p'] < 0.01:
                    report_lines.append(f"  Result: Significant negative correlation (p < 0.01)")
                elif mem_sem['p'] < 0.05:
                    report_lines.append(f"  Result: Significant negative correlation (p < 0.05)")
                else:
                    report_lines.append(f"  Result: Not significant (p >= 0.05)")
                report_lines.append("")
            
            # Choice divergence vs Semantic
            if 'choice_divergence_vs_semantic' in results:
                choice_sem = results['choice_divergence_vs_semantic']
                report_lines.append("Choice Divergence vs Semantic Influence:")
                report_lines.append(f"  N: {choice_sem['n']}")
                report_lines.append(f"  Correlation: r({choice_sem['n']-2}) = {choice_sem['r']:.3f}, p = {choice_sem['p']:.6f}")
                if choice_sem['p'] < 0.001:
                    report_lines.append(f"  Result: Significant correlation (p < 0.001)")
                elif choice_sem['p'] < 0.01:
                    report_lines.append(f"  Result: Significant correlation (p < 0.01)")
                elif choice_sem['p'] < 0.05:
                    report_lines.append(f"  Result: Significant correlation (p < 0.05)")
                else:
                    report_lines.append(f"  Result: Not significant (p >= 0.05)")
                report_lines.append("")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "memory_divergence_semantic_correlation_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_memory_divergence_semantic()

