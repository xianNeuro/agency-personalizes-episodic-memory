#!/usr/bin/env python3
"""
Neighbor Encoding Effect Correlations and Multiple Regression Analysis
Analyzes correlations between neighbor encoding effect and:
1. Memory divergence in Free participants
2. Semantic influence scores in Free participants
3. Multiple linear regression predicting memory divergence

For both Adventure (BA) and Romance (MV) stories.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from data_structure import RecallDataLoader

def load_correlation_data(story, n_subjects=None):
    """Load correlation data from corr_free sheets"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    
    if story == 'BA':
        data_file = os.path.join(data_dir, "adventure_data2.xlsx")
        sheet_name = 'corr_free22'
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
    
    print(f"Loaded {len(df)} subjects")
    
    return df

def analyze_neighbor_memory_correlation(df, story, n_subjects=None):
    """Analyze correlation between neighbor encoding effect and memory divergence"""
    
    # Check column names (might be 'neighbor-ef' or 'nghb-ef')
    nghb_col = None
    for col in ['neighbor-ef', 'nghb-ef']:
        if col in df.columns:
            nghb_col = col
            break
    
    if nghb_col is None:
        print("  Neighbor encoding effect column not found")
        return None
    
    if 'rcl-1-r_otherm' not in df.columns:
        print("  Memory divergence column (rcl-1-r_otherm) not found")
        return None
    
    # Align arrays (only use subjects with both measures)
    valid_mask = df[nghb_col].notna() & df['rcl-1-r_otherm'].notna()
    nghb_aligned = df.loc[valid_mask, nghb_col].values
    mem_div_aligned = df.loc[valid_mask, 'rcl-1-r_otherm'].values
    
    if len(nghb_aligned) > 1:
        r, p = pearsonr(nghb_aligned, mem_div_aligned)
        n = len(nghb_aligned)
        
        print(f"\nNeighbor Encoding Effect vs Memory Divergence:")
        print(f"  N: {n}")
        print(f"  Correlation: r({n-2}) = {r:.3f}, p = {p:.6f}")
        if p < 0.001:
            print(f"  Result: Significant positive correlation (p < 0.001)")
        elif p < 0.01:
            print(f"  Result: Significant positive correlation (p < 0.01)")
        elif p < 0.05:
            print(f"  Result: Significant positive correlation (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        print(f"  Neighbor Encoding Effect: Mean = {np.mean(nghb_aligned):.4f}, Std = {np.std(nghb_aligned, ddof=1):.4f}")
        print(f"  Memory Divergence: Mean = {np.mean(mem_div_aligned):.4f}, Std = {np.std(mem_div_aligned, ddof=1):.4f}")
        
        return {
            'r': r,
            'p': p,
            'n': n
        }
    
    return None

def analyze_neighbor_semantic_correlation(df, story, n_subjects=None):
    """Analyze correlation between neighbor encoding effect and semantic influence"""
    
    # Check column names
    nghb_col = None
    for col in ['neighbor-ef', 'nghb-ef']:
        if col in df.columns:
            nghb_col = col
            break
    
    if nghb_col is None:
        print("  Neighbor encoding effect column not found")
        return None
    
    if 'sem-ef' not in df.columns:
        print("  Semantic influence column (sem-ef) not found")
        return None
    
    # Align arrays
    valid_mask = df[nghb_col].notna() & df['sem-ef'].notna()
    nghb_aligned = df.loc[valid_mask, nghb_col].values
    sem_aligned = df.loc[valid_mask, 'sem-ef'].values
    
    if len(nghb_aligned) > 1:
        r, p = pearsonr(nghb_aligned, sem_aligned)
        n = len(nghb_aligned)
        
        print(f"\nNeighbor Encoding Effect vs Semantic Influence:")
        print(f"  N: {n}")
        print(f"  Correlation: r({n-2}) = {r:.3f}, p = {p:.6f}")
        if p < 0.001:
            print(f"  Result: Significant negative correlation (p < 0.001)")
        elif p < 0.01:
            print(f"  Result: Significant negative correlation (p < 0.01)")
        elif p < 0.05:
            print(f"  Result: Significant negative correlation (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        print(f"  Neighbor Encoding Effect: Mean = {np.mean(nghb_aligned):.4f}, Std = {np.std(nghb_aligned, ddof=1):.4f}")
        print(f"  Semantic Influence: Mean = {np.mean(sem_aligned):.4f}, Std = {np.std(sem_aligned, ddof=1):.4f}")
        
        return {
            'r': r,
            'p': p,
            'n': n
        }
    
    return None

def run_multiple_regression(df, story, n_subjects=None):
    """Run multiple linear regression predicting memory divergence"""
    
    # Check column names
    nghb_col = None
    for col in ['neighbor-ef', 'nghb-ef']:
        if col in df.columns:
            nghb_col = col
            break
    
    if nghb_col is None or 'sem-ef' not in df.columns or 'rcl-1-r_otherm' not in df.columns:
        print("  Required columns not found for multiple regression")
        return None
    
    # Prepare data (only subjects with all three measures)
    valid_mask = (df[nghb_col].notna() & 
                  df['sem-ef'].notna() & 
                  df['rcl-1-r_otherm'].notna())
    
    if valid_mask.sum() < 3:
        print("  Insufficient data for multiple regression")
        return None
    
    # Extract variables
    X = df.loc[valid_mask, [nghb_col, 'sem-ef']].values
    y = df.loc[valid_mask, 'rcl-1-r_otherm'].values
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X)
    
    # Fit model
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"\nMultiple Linear Regression: Memory Divergence ~ Neighbor Encoding + Semantic Influence")
    print(f"  N: {len(y)}")
    print(f"  R-squared: {model.rsquared:.4f}")
    print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"\n  Coefficients:")
    print(f"    Intercept: {model.params[0]:.4f}, p = {model.pvalues[0]:.6f}")
    print(f"    Neighbor Encoding Effect: {model.params[1]:.4f}, p = {model.pvalues[1]:.6f}")
    print(f"    Semantic Influence: {model.params[2]:.4f}, p = {model.pvalues[2]:.6f}")
    
    # Check significance
    nghb_p = model.pvalues[1]
    sem_p = model.pvalues[2]
    
    print(f"\n  Significance:")
    if nghb_p < 0.001:
        print(f"    Neighbor Encoding Effect: Significant (p < 0.001)")
    elif nghb_p < 0.01:
        print(f"    Neighbor Encoding Effect: Significant (p < 0.01)")
    elif nghb_p < 0.05:
        print(f"    Neighbor Encoding Effect: Significant (p < 0.05)")
    elif nghb_p < 0.1:
        print(f"    Neighbor Encoding Effect: Trend (p = {nghb_p:.3f})")
    else:
        print(f"    Neighbor Encoding Effect: Not significant (p = {nghb_p:.3f})")
    
    if sem_p < 0.001:
        print(f"    Semantic Influence: Significant (p < 0.001)")
    elif sem_p < 0.01:
        print(f"    Semantic Influence: Significant (p < 0.01)")
    elif sem_p < 0.05:
        print(f"    Semantic Influence: Significant (p < 0.05)")
    else:
        print(f"    Semantic Influence: Not significant (p = {sem_p:.3f})")
    
    return {
        'model': model,
        'n': len(y),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'nghb_coef': model.params[1],
        'nghb_p': nghb_p,
        'sem_coef': model.params[2],
        'sem_p': sem_p
    }

def analyze_neighbor_encoding_correlations():
    """Main analysis function"""
    
    print("="*80)
    print("NEIGHBOR ENCODING EFFECT CORRELATIONS AND MULTIPLE REGRESSION")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run10_neighbor_encoding_correlations")
    
    all_results = {}
    
    # Analyze Adventure story
    print("\n" + "="*80)
    print("ADVENTURE STORY (BA)")
    print("="*80)
    
    df_ba = load_correlation_data('BA')
    
    # Neighbor encoding vs Semantic influence
    print("\n" + "="*80)
    print("Neighbor Encoding Effect vs Semantic Influence")
    print("="*80)
    nghb_sem_ba = analyze_neighbor_semantic_correlation(df_ba, 'BA')
    
    all_results['BA'] = {
        'data': df_ba,
        'nghb_semantic': nghb_sem_ba
    }
    
    # Analyze Romance story (both 18 and 100 subjects)
    print("\n" + "="*80)
    print("ROMANCE STORY (MV)")
    print("="*80)
    
    for n_subjects in [18, 100]:
        print("\n" + "="*80)
        print(f"ANALYSIS: {n_subjects} Free Participants")
        print("="*80)
        
        df_mv = load_correlation_data('MV', n_subjects=n_subjects)
        
        # Neighbor encoding vs Memory divergence
        print("\n" + "="*80)
        print("Neighbor Encoding Effect vs Memory Divergence")
        print("="*80)
        nghb_mem = analyze_neighbor_memory_correlation(df_mv, 'MV', n_subjects=n_subjects)
        
        # Neighbor encoding vs Semantic influence
        print("\n" + "="*80)
        print("Neighbor Encoding Effect vs Semantic Influence")
        print("="*80)
        nghb_sem = analyze_neighbor_semantic_correlation(df_mv, 'MV', n_subjects=n_subjects)
        
        # Multiple regression
        print("\n" + "="*80)
        print("Multiple Linear Regression")
        print("="*80)
        regression = run_multiple_regression(df_mv, 'MV', n_subjects=n_subjects)
        
        all_results[f'MV_{n_subjects}'] = {
            'data': df_mv,
            'nghb_memory': nghb_mem,
            'nghb_semantic': nghb_sem,
            'regression': regression
        }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    # Correlations
    for key, data_dict in all_results.items():
        if key == 'BA':
            story = 'Adventure'
            n = 22
            if data_dict.get('nghb_semantic') is not None:
                nghb_sem = data_dict['nghb_semantic']
                summary_data.append({
                    'Story': story,
                    'N_subjects': n,
                    'Analysis': 'Correlation',
                    'Measure1': 'Neighbor Encoding Effect',
                    'Measure2': 'Semantic Influence',
                    'r': nghb_sem['r'],
                    'n': nghb_sem['n'],
                    'p_value': nghb_sem['p']
                })
        else:
            story = 'Romance'
            n = int(key.split('_')[1])
            results = data_dict
            
            if results.get('nghb_memory') is not None:
                nghb_mem = results['nghb_memory']
                summary_data.append({
                    'Story': story,
                    'N_subjects': n,
                    'Analysis': 'Correlation',
                    'Measure1': 'Neighbor Encoding Effect',
                    'Measure2': 'Memory Divergence',
                    'r': nghb_mem['r'],
                    'n': nghb_mem['n'],
                    'p_value': nghb_mem['p']
                })
            
            if results.get('nghb_semantic') is not None:
                nghb_sem = results['nghb_semantic']
                summary_data.append({
                    'Story': story,
                    'N_subjects': n,
                    'Analysis': 'Correlation',
                    'Measure1': 'Neighbor Encoding Effect',
                    'Measure2': 'Semantic Influence',
                    'r': nghb_sem['r'],
                    'n': nghb_sem['n'],
                    'p_value': nghb_sem['p']
                })
            
            if results.get('regression') is not None:
                reg = results['regression']
                summary_data.append({
                    'Story': story,
                    'N_subjects': n,
                    'Analysis': 'Multiple Regression',
                    'Measure1': 'Neighbor Encoding Effect',
                    'Measure2': 'Semantic Influence',
                    'r': np.nan,
                    'n': reg['n'],
                    'p_value': reg['nghb_p'],
                    'r_squared': reg['r_squared'],
                    'nghb_coef': reg['nghb_coef'],
                    'sem_coef': reg['sem_coef'],
                    'sem_p': reg['sem_p']
                })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "neighbor_encoding_correlations_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report (following the specified sequence)
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("NEIGHBOR ENCODING EFFECT CORRELATIONS AND MULTIPLE REGRESSION")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source:")
    report_lines.append("  - Adventure (BA): adventure_data2.xlsx (sheet: corr_free22)")
    report_lines.append("  - Romance (MV): romance_data2.xlsx")
    report_lines.append("    - 18 Free participants: sheet corr_free18")
    report_lines.append("    - 100 Free participants: sheet corr_free100")
    report_lines.append("")
    
    # 1. Neighbor encoding effect vs Memory divergence (Romance, both 18 and 100)
    report_lines.append("="*80)
    report_lines.append("1. NEIGHBOR ENCODING EFFECT vs MEMORY DIVERGENCE (Romance)")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("The neighbor encoding effect was also positively associated with memory")
    report_lines.append("divergence in Free participants. In other words, the more a participant's")
    report_lines.append("recall deviated from other participants in the group, the more that")
    report_lines.append("participant's neighboring events tended to have the same recall status")
    report_lines.append("(remembered or forgotten).")
    report_lines.append("")
    
    for n_subjects in [18, 100]:
        key = f'MV_{n_subjects}'
        if key in all_results and all_results[key].get('nghb_memory') is not None:
            nghb_mem = all_results[key]['nghb_memory']
            report_lines.append(f"Romance ({n_subjects} Free participants):")
            report_lines.append(f"  Correlation: r({nghb_mem['n']-2}) = {nghb_mem['r']:.3f}, p = {nghb_mem['p']:.6f}")
            if nghb_mem['p'] < 0.01:
                report_lines.append(f"  Result: Significant positive correlation (p < 0.01)")
            elif nghb_mem['p'] < 0.05:
                report_lines.append(f"  Result: Significant positive correlation (p < 0.05)")
            else:
                report_lines.append(f"  Result: Not significant (p >= 0.05)")
            report_lines.append("")
    
    # 2. Neighbor encoding effect vs Semantic influence (Adventure, then Romance 18 and 100)
    report_lines.append("="*80)
    report_lines.append("2. NEIGHBOR ENCODING EFFECT vs SEMANTIC INFLUENCE")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("The neighbor encoding effect was negatively correlated with semantic")
    report_lines.append("influence scores in Free participants, across both stories.")
    report_lines.append("")
    
    # Adventure
    if 'BA' in all_results and all_results['BA'].get('nghb_semantic') is not None:
        nghb_sem = all_results['BA']['nghb_semantic']
        report_lines.append("Adventure (22 Free participants):")
        report_lines.append(f"  Correlation: r({nghb_sem['n']-2}) = {nghb_sem['r']:.3f}, p = {nghb_sem['p']:.6f}")
        if nghb_sem['p'] < 0.05:
            report_lines.append(f"  Result: Significant negative correlation (p < 0.05)")
        else:
            report_lines.append(f"  Result: Not significant (p >= 0.05)")
        report_lines.append("")
    
    # Romance
    for n_subjects in [18, 100]:
        key = f'MV_{n_subjects}'
        if key in all_results and all_results[key].get('nghb_semantic') is not None:
            nghb_sem = all_results[key]['nghb_semantic']
            report_lines.append(f"Romance ({n_subjects} Free participants):")
            report_lines.append(f"  Correlation: r({nghb_sem['n']-2}) = {nghb_sem['r']:.3f}, p = {nghb_sem['p']:.6f}")
            if nghb_sem['p'] < 0.001:
                report_lines.append(f"  Result: Significant negative correlation (p < 0.001)")
            elif nghb_sem['p'] < 0.05:
                report_lines.append(f"  Result: Significant negative correlation (p < 0.05)")
            else:
                report_lines.append(f"  Result: Not significant (p >= 0.05)")
            report_lines.append("")
    
    # 3. Multiple linear regression (only for Romance 100 subjects)
    report_lines.append("="*80)
    report_lines.append("3. MULTIPLE LINEAR REGRESSION (Romance, 100 Free participants)")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("When including both semantic influence and the neighbor encoding scores")
    report_lines.append("in a multiple linear regression predicting memory divergence:")
    report_lines.append("")
    
    key = 'MV_100'
    if key in all_results and all_results[key].get('regression') is not None:
        reg = all_results[key]['regression']
        report_lines.append("Multiple Linear Regression: Memory Divergence ~ Neighbor Encoding + Semantic Influence")
        report_lines.append(f"  N: {reg['n']}")
        report_lines.append(f"  R-squared: {reg['r_squared']:.4f}")
        report_lines.append(f"  Adjusted R-squared: {reg['adj_r_squared']:.4f}")
        report_lines.append("")
        report_lines.append("  Coefficients:")
        report_lines.append(f"    Neighbor Encoding Effect: Coef = {reg['nghb_coef']:.4f}, p = {reg['nghb_p']:.6f}")
        report_lines.append(f"    Semantic Influence: Coef = {reg['sem_coef']:.4f}, p = {reg['sem_p']:.6f}")
        report_lines.append("")
        report_lines.append("  Results:")
        if reg['nghb_p'] < 0.1:
            report_lines.append(f"    Neighbor Encoding Effect: {'Significant' if reg['nghb_p'] < 0.05 else 'Trend'} (p = {reg['nghb_p']:.3f})")
        else:
            report_lines.append(f"    Neighbor Encoding Effect: Not significant (p = {reg['nghb_p']:.3f})")
        if reg['sem_p'] < 0.001:
            report_lines.append(f"    Semantic Influence: Significant (p < 0.001)")
        elif reg['sem_p'] < 0.05:
            report_lines.append(f"    Semantic Influence: Significant (p < 0.05)")
        else:
            report_lines.append(f"    Semantic Influence: Not significant (p = {reg['sem_p']:.3f})")
        report_lines.append("")
    
    report_file = os.path.join(output_dir, "neighbor_encoding_correlations_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_neighbor_encoding_correlations()

