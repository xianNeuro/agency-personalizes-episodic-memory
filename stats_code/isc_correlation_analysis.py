#!/usr/bin/env python3
"""
ISC Correlation Analysis
Performs 1-sample t-tests on raw Pearson correlation r-values against zero,
and z-tests on Fisher's Z-transformed correlations
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
from data_structure import RecallDataLoader

def fisher_z_transform(r):
    """Apply Fisher's Z transformation to correlation coefficient"""
    # Clip r values to valid range [-1, 1] to avoid numerical issues
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inverse_fisher_z(z):
    """Inverse Fisher's Z transformation back to correlation"""
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def perform_one_sample_ttest(data, test_value=0):
    """Perform 1-sample t-test against test_value"""
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return None, None, None, len(data_clean)
    
    t_stat, p_val = stats.ttest_1samp(data_clean, test_value)
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=1)
    
    return t_stat, p_val, mean_val, len(data_clean)

def perform_one_sample_ztest(data, test_value=0):
    """Perform 1-sample z-test against test_value (for Fisher's Z-transformed data)"""
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return None, None, None, len(data_clean)
    
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=1)  # Sample standard deviation
    n = len(data_clean)
    
    # Standard error of the mean
    se = std_val / np.sqrt(n)
    
    # Z-score
    z_score = (mean_val - test_value) / se
    
    # Two-tailed p-value from standard normal distribution
    p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_val, mean_val, n

def analyze_isc_correlations():
    """Analyze ISC correlations with and without Fisher's Z transformation"""
    
    # Use current directory
    base_path = os.path.abspath('.')
    file_path = os.path.join(base_path, 'data', 'romance_data2.xlsx')
    
    # Load merge_isc sheet, columns 1 (cond) and 3 (recall_pw-isc)
    df = pd.read_excel(file_path, sheet_name='merge_isc')
    
    # Extract columns: column 1 (index 0) is cond, column 3 (index 2) is recall_pw-isc
    df_clean = df.iloc[:, [0, 2]].copy()
    df_clean.columns = ['cond', 'r_value']
    df_clean = df_clean.dropna()
    
    print("="*80)
    print("ISC CORRELATION ANALYSIS: t-tests (raw) and z-tests (Fisher's Z) against zero")
    print("="*80)
    print(f"\nTotal observations: {len(df_clean)}")
    print(f"Conditions: {sorted(df_clean['cond'].unique())}\n")
    
    # Apply Fisher's Z transformation
    df_clean['z_value'] = df_clean['r_value'].apply(fisher_z_transform)
    
    results = []
    
    print("="*80)
    print("ANALYSIS WITH RAW CORRELATIONS")
    print("="*80 + "\n")
    
    for cond in sorted(df_clean['cond'].unique()):
        cond_data = df_clean[df_clean['cond'] == cond]
        r_values = cond_data['r_value']
        
        # Perform 1-sample t-test on raw correlations
        t_stat, p_val, mean_r, n = perform_one_sample_ttest(r_values, test_value=0)
        
        if t_stat is not None:
            std_r = np.std(r_values.dropna(), ddof=1)
            
            print(f"Condition: {cond}")
            print(f"  N = {n}")
            print(f"  Mean r = {mean_r:.4f}, SD = {std_r:.4f}")
            print(f"  t({n-1}) = {t_stat:.4f}, p = {p_val:.4f}")
            
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            print(f"  Significance: {sig_text}\n")
            
            results.append({
                'condition': cond,
                'analysis': 'raw',
                'N': n,
                'mean': mean_r,
                'std': std_r,
                't_stat': t_stat,
                'df': n - 1,
                'p_value': p_val
            })
    
    print("="*80)
    print("ANALYSIS WITH FISHER'S Z-TRANSFORMED CORRELATIONS (Z-TEST)")
    print("="*80 + "\n")
    
    for cond in sorted(df_clean['cond'].unique()):
        cond_data = df_clean[df_clean['cond'] == cond]
        z_values = cond_data['z_value']
        
        # Perform 1-sample z-test on Z-transformed correlations
        z_score, p_val, mean_z, n = perform_one_sample_ztest(z_values, test_value=0)
        
        if z_score is not None:
            std_z = np.std(z_values.dropna(), ddof=1)
            
            # Convert mean Z back to r for comparison
            mean_r_from_z = inverse_fisher_z(mean_z)
            
            print(f"Condition: {cond}")
            print(f"  N = {n}")
            print(f"  Mean Z = {mean_z:.4f}, SD = {std_z:.4f}")
            print(f"  Mean r (from Z) = {mean_r_from_z:.4f}")
            print(f"  z = {z_score:.4f}, p = {p_val:.4f}")
            
            if p_val < 0.001:
                sig_text = "***"
            elif p_val < 0.01:
                sig_text = "**"
            elif p_val < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            print(f"  Significance: {sig_text}\n")
            
            results.append({
                'condition': cond,
                'analysis': 'fisher_z',
                'N': n,
                'mean_z': mean_z,
                'mean_r_from_z': mean_r_from_z,
                'std_z': std_z,
                'z_score': z_score,
                'p_value': p_val
            })
    
    # Save results
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("isc_correlation_analysis")
    
    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, "isc_correlation_ttest_results.xlsx")
    results_df.to_excel(output_file, index=False)
    print(f"Saved results to: {output_file}")
    
    # Generate text report
    report_file = os.path.join(output_dir, "isc_correlation_ttest_report.txt")
    generate_report(df_clean, results, report_file)
    print(f"Saved report to: {report_file}")
    
    return results_df

def generate_report(df_clean, results, output_file):
    """Generate text report comparing raw and Z-transformed analyses"""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ISC CORRELATION ANALYSIS: t-tests (raw) and z-tests (Fisher's Z) against zero\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis performed 1-sample t-tests on raw Pearson correlation r-values\n")
        f.write("and z-tests on Fisher's Z-transformed correlations against zero.\n")
        f.write("Fisher's Z transformation is typically applied when averaging correlations\n")
        f.write("because correlations tend to be skewed. After transformation, z-tests are\n")
        f.write("appropriate as the distribution is approximately normal.\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS WITH RAW CORRELATIONS\n")
        f.write("="*80 + "\n\n")
        
        raw_results = [r for r in results if r['analysis'] == 'raw']
        for result in raw_results:
            cond = result['condition']
            n = result['N']
            mean_r = result['mean']
            t_stat = result['t_stat']
            df = result['df']
            p_val = result['p_value']
            
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
            
            f.write(f"{cond.capitalize()} condition:\n")
            f.write(f"  N = {n}\n")
            f.write(f"  Mean r = {mean_r:.4f}\n")
            f.write(f"  t({df}) = {t_stat:.4f}, {p_text}\n\n")
        
        f.write("="*80 + "\n")
        f.write("RESULTS WITH FISHER'S Z-TRANSFORMED CORRELATIONS (Z-TEST)\n")
        f.write("="*80 + "\n\n")
        
        z_results = [r for r in results if r['analysis'] == 'fisher_z']
        for result in z_results:
            cond = result['condition']
            n = result['N']
            mean_z = result['mean_z']
            mean_r_from_z = result['mean_r_from_z']
            z_score = result['z_score']
            p_val = result['p_value']
            
            if p_val < 0.001:
                p_text = "p < 0.001"
            else:
                p_text = f"p = {p_val:.3f}"
            
            f.write(f"{cond.capitalize()} condition:\n")
            f.write(f"  N = {n}\n")
            f.write(f"  Mean Z = {mean_z:.4f}\n")
            f.write(f"  Mean r (from Z) = {mean_r_from_z:.4f}\n")
            f.write(f"  z = {z_score:.4f}, {p_text}\n\n")
        
        f.write("="*80 + "\n")
        f.write("ROBUSTNESS COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write("Comparison of significance results between raw and Z-transformed analyses:\n\n")
        
        for cond in sorted(set([r['condition'] for r in results])):
            raw_r = next((r for r in raw_results if r['condition'] == cond), None)
            z_r = next((r for r in z_results if r['condition'] == cond), None)
            
            if raw_r and z_r:
                raw_sig = raw_r['p_value'] < 0.05
                z_sig = z_r['p_value'] < 0.05
                
                f.write(f"{cond.capitalize()} condition:\n")
                f.write(f"  Raw correlation: p = {raw_r['p_value']:.4f} ({'significant' if raw_sig else 'not significant'})\n")
                f.write(f"  Z-transformed: p = {z_r['p_value']:.4f} ({'significant' if z_sig else 'not significant'})\n")
                
                if raw_sig == z_sig:
                    f.write(f"  Result: ROBUST - Both analyses yield the same conclusion\n\n")
                else:
                    f.write(f"  Result: NOT ROBUST - Analyses yield different conclusions\n\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        # Check if all results are robust
        all_robust = True
        for cond in sorted(set([r['condition'] for r in results])):
            raw_r = next((r for r in raw_results if r['condition'] == cond), None)
            z_r = next((r for r in z_results if r['condition'] == cond), None)
            if raw_r and z_r:
                raw_sig = raw_r['p_value'] < 0.05
                z_sig = z_r['p_value'] < 0.05
                if raw_sig != z_sig:
                    all_robust = False
        
        if all_robust:
            f.write("The reported results are ROBUST when rerunning analyses based on Z-transformed\n")
            f.write("correlation coefficients. Both raw and Z-transformed analyses yield the same\n")
            f.write("conclusions regarding statistical significance.\n")
        else:
            f.write("The reported results are NOT fully robust when rerunning analyses based on\n")
            f.write("Z-transformed correlation coefficients. Some conditions show different conclusions\n")
            f.write("between raw and Z-transformed analyses.\n")

def main():
    """Main function"""
    results_df = analyze_isc_correlations()
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    return results_df

if __name__ == "__main__":
    main()

