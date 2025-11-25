#!/usr/bin/env python3
"""
Individual Variability in Choices Made
Computes inter-participant correlation (ISC) for choice selection vectors
at choice-points in shared story sections.

Extracts choice vectors from individual event files for Free and Yoked participants,
then computes Choice ISC (pairwise Pearson correlations).

Runs two analyses:
1. Raw correlation r-values
2. Fisher z-transformed r-values

For each analysis, performs:
- One-sample t-tests per condition (Free, Yoked) testing if Choice ISC > 0
- Two-sample t-test comparing Free vs Yoked
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, ttest_1samp, ttest_ind
from scipy import stats
from data_structure import RecallDataLoader

def get_shared_choice_base_scenes():
    """Get the 15 unique base scene numbers for shared choice events"""
    
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    map_file = os.path.join(data_dir, "romance_data1.xlsx")
    
    map_df = pd.read_excel(map_file, sheet_name='story_map')
    shared_choice = map_df[(map_df['Converge'] == 'Y') & (map_df['Choice'] == 'Z')]
    
    base_scenes = set()
    for scene_lab in shared_choice['Scene_lab'].dropna():
        if '_' in str(scene_lab):
            base_scene = int(str(scene_lab).split('_')[0])
            base_scenes.add(base_scene)
    
    shared_choice_base_scenes = sorted(list(base_scenes))
    print(f"Identified {len(shared_choice_base_scenes)} shared choice events (base scenes)")
    print(f"  Base scene numbers: {shared_choice_base_scenes}")
    
    return shared_choice_base_scenes

def extract_choice_from_event_file(subject_id, condition, shared_choice_base_scenes, base_path):
    """Extract choice vector from individual subject event file"""
    
    condition_num = {'free': '1', 'yoke': '2', 'pasv': '3'}[condition]
    event_file = os.path.join(base_path, "data", "individual_data", "romance", 
                              f"{condition_num}_{condition}", f"{subject_id}_events.xlsx")
    
    if not os.path.exists(event_file):
        return None
    
    event_df = pd.read_excel(event_file)
    
    if 'scenes' not in event_df.columns:
        return None
    
    # For yoked subjects, use subj_choice column (actual choice made)
    # For free subjects, extract choice from scenes column (choice that led to that scene)
    use_subj_choice = (condition == 'yoke' and 'subj_choice' in event_df.columns)
    
    # Extract choices for each base scene
    choice_vector = []
    for base_scene in shared_choice_base_scenes:
        # Look for scenes values that start with base_scene and have underscore
        found_choice = None
        for idx, row in event_df.iterrows():
            scenes_val = row['scenes']
            if pd.notna(scenes_val):
                scenes_str = str(scenes_val).strip()
                if scenes_str.startswith(f"{base_scene}_"):
                    if use_subj_choice:
                        # For yoked: use subj_choice column (actual choice made)
                        subj_choice = row.get('subj_choice', np.nan)
                        if pd.notna(subj_choice):
                            try:
                                found_choice = int(subj_choice)
                                break
                            except (ValueError, TypeError):
                                pass
                    else:
                        # For free: extract choice from scenes value (e.g., "4_2" -> 2)
                        parts = scenes_str.split('_')
                        if len(parts) >= 2:
                            try:
                                found_choice = int(parts[1])
                                break
                            except ValueError:
                                pass
        
        if found_choice is not None:
            choice_vector.append(found_choice)
        else:
            choice_vector.append(np.nan)
    
    return np.array(choice_vector)

def identify_story_paths(loader):
    """Identify story paths by matching free subjects to yoked/pasv subjects"""
    
    free_ids = loader.get_subject_ids_from_events('MV', 'free')
    yoke_ids = loader.get_subject_ids_from_events('MV', 'yoke')
    pasv_ids = loader.get_subject_ids_from_events('MV', 'pasv')
    
    free_ids = [sid for sid in free_ids if not sid.startswith('~$')]
    yoke_ids = [sid for sid in yoke_ids if not sid.startswith('~$')]
    pasv_ids = [sid for sid in pasv_ids if not sid.startswith('~$')]
    
    free_subject_map = {}
    for free_id in free_ids:
        try:
            num = int(free_id.split('_')[0].replace('sub', ''))
            free_subject_map[num] = free_id
        except:
            continue
    
    story_paths = {}
    for yoke_id in yoke_ids:
        try:
            to_num = int(yoke_id.split('_')[0].replace('to', ''))
            if to_num not in story_paths:
                story_paths[to_num] = {'free': None, 'yoke': [], 'pasv': []}
            story_paths[to_num]['yoke'].append(yoke_id)
        except:
            continue
    
    for pasv_id in pasv_ids:
        try:
            to_num = int(pasv_id.split('_')[0].replace('to', ''))
            if to_num not in story_paths:
                story_paths[to_num] = {'free': None, 'yoke': [], 'pasv': []}
            story_paths[to_num]['pasv'].append(pasv_id)
        except:
            continue
    
    for to_num, free_id in free_subject_map.items():
        if to_num in story_paths:
            story_paths[to_num]['free'] = free_id
    
    return story_paths

def load_choice_vectors_from_individual_files(condition, shared_choice_base_scenes, base_path):
    """Load choice vectors from individual event files"""
    
    loader = RecallDataLoader(base_path)
    
    # For free condition, get only the 18 subjects corresponding to story paths
    if condition == 'free':
        story_paths = identify_story_paths(loader)
        free_ids_from_paths = []
        for to_num, path_info in story_paths.items():
            free_id = path_info['free']
            if free_id is not None:
                free_ids_from_paths.append(free_id)
        subject_ids = sorted(set(free_ids_from_paths))
        print(f"Filtered to {len(subject_ids)} free subjects corresponding to story paths")
    else:
        subject_ids = loader.get_subject_ids_from_events('MV', condition)
        subject_ids = [sid for sid in subject_ids if not sid.startswith('~$')]
    
    # Extract choice vectors
    choice_vectors = {}
    for subject_id in subject_ids:
        choice_vector = extract_choice_from_event_file(subject_id, condition, 
                                                      shared_choice_base_scenes, base_path)
        if choice_vector is not None and not np.all(np.isnan(choice_vector)):
            choice_vectors[subject_id] = choice_vector
    
    print(f"Loaded choice vectors for {len(choice_vectors)} {condition} subjects")
    valid_counts = [np.sum(~np.isnan(choice_vectors[sid])) for sid in list(choice_vectors.keys())[:5]]
    print(f"  Valid choices per subject (sample): {valid_counts}...")
    
    return choice_vectors

def compute_pairwise_correlations(choice_vectors):
    """Compute pairwise correlations from choice vectors"""
    
    correlations = []
    subject_pairs = []
    subject_list = list(choice_vectors.keys())
    
    for i in range(len(subject_list)):
        for j in range(i + 1, len(subject_list)):
            subj1 = subject_list[i]
            subj2 = subject_list[j]
            
            vec1 = choice_vectors[subj1]
            vec2 = choice_vectors[subj2]
            
            # Find valid indices
            valid_mask = ~(np.isnan(vec1) | np.isnan(vec2))
            
            if np.sum(valid_mask) < 2:
                continue
            
            vec1_valid = vec1[valid_mask]
            vec2_valid = vec2[valid_mask]
            
            # Check if constant
            if np.std(vec1_valid) == 0 or np.std(vec2_valid) == 0:
                continue
            
            r, p = pearsonr(vec1_valid, vec2_valid)
            
            if not np.isnan(r):
                correlations.append(r)
                subject_pairs.append((subj1, subj2))
    
    print(f"Computed {len(correlations)} pairwise correlations")
    return np.array(correlations), subject_pairs

def run_analysis(condition, shared_choice_base_scenes, base_path, use_z_transform=False):
    """Run Choice ISC analysis for a condition"""
    
    print("\n" + "="*80)
    print(f"Computing Choice ISC for {condition} condition")
    if use_z_transform:
        print("(Fisher z-transformed)")
    print("="*80)
    
    # Load choice vectors
    choice_vectors = load_choice_vectors_from_individual_files(condition, shared_choice_base_scenes, base_path)
    
    if len(choice_vectors) == 0:
        print(f"No choice vectors loaded for {condition} condition")
        return None
    
    # Compute pairwise correlations
    correlations, subject_pairs = compute_pairwise_correlations(choice_vectors)
    
    if len(correlations) == 0:
        print(f"No correlations computed for {condition} condition")
        return None
    
    # Apply Fisher z-transform if requested
    if use_z_transform:
        correlations = np.arctanh(np.clip(correlations, -0.999, 0.999))
    
    # Statistics
    t_stat, p_val = ttest_1samp(correlations, 0)
    mean_val = np.mean(correlations)
    std_val = np.std(correlations, ddof=1)
    n = len(correlations)
    
    results = {
        'values': correlations,
        'mean': mean_val,
        'std': std_val,
        'n': n,
        't_stat': t_stat,
        'p_val': p_val,
        'choice_vectors': choice_vectors,
        'subject_pairs': subject_pairs
    }
    
    val_type = 'z' if use_z_transform else 'r'
    print(f"\n{condition.upper()} Condition:")
    print(f"  N pairs: {n}")
    print(f"  Mean {val_type}: {mean_val:.4f}")
    print(f"  Std: {std_val:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_val:.6f}")
    
    return results

def analyze_choice_isc():
    """Main analysis function"""
    
    print("="*80)
    print("INDIVIDUAL VARIABILITY IN CHOICES MADE")
    print("Choice ISC Analysis")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run3_individual_variability_choices")
    
    # Get shared choice base scenes
    shared_choice_base_scenes = get_shared_choice_base_scenes()
    
    # Run analyses
    all_results = {}
    
    # Analysis 1: Raw r-values
    print("\n" + "="*80)
    print("ANALYSIS 1: RAW CORRELATION R-VALUES")
    print("="*80)
    results_raw = {}
    for condition in ['free', 'yoke']:
        results_raw[condition] = run_analysis(condition, shared_choice_base_scenes, base_path, use_z_transform=False)
    all_results['raw'] = results_raw
    
    # Analysis 2: Fisher z-transformed
    print("\n" + "="*80)
    print("ANALYSIS 2: FISHER Z-TRANSFORMED R-VALUES")
    print("="*80)
    results_z = {}
    for condition in ['free', 'yoke']:
        results_z[condition] = run_analysis(condition, shared_choice_base_scenes, base_path, use_z_transform=True)
    all_results['z_transformed'] = results_z
    
    # Two-sample t-test: Free vs Yoked
    for analysis_type, results in all_results.items():
        if results.get('free') is not None and results.get('yoke') is not None:
            free_vals = results['free']['values']
            yoke_vals = results['yoke']['values']
            
            t_stat, p_val = ttest_ind(free_vals, yoke_vals)
            
            results['free_vs_yoke'] = {
                't_stat': t_stat,
                'p_val': p_val,
                'free_mean': results['free']['mean'],
                'yoke_mean': results['yoke']['mean'],
                'free_n': results['free']['n'],
                'yoke_n': results['yoke']['n']
            }
            
            print("\n" + "="*80)
            print(f"FREE vs YOKED Comparison ({analysis_type.upper()})")
            print("="*80)
            val_type = 'z' if analysis_type == 'z_transformed' else 'r'
            print(f"Free mean {val_type}: {results['free']['mean']:.4f} (N={results['free']['n']})")
            print(f"Yoke mean {val_type}: {results['yoke']['mean']:.4f} (N={results['yoke']['n']})")
            print(f"t-statistic: {t_stat:.4f}")
            print(f"p-value: {p_val:.6f}")
    
    # Save choice vectors
    print("\n" + "="*80)
    print("SAVING CHOICE VECTORS")
    print("="*80)
    
    # Save free choice vectors
    if all_results['raw']['free'] is not None:
        free_choice_vectors = all_results['raw']['free']['choice_vectors']
        choice_df_data = {'subject_id': list(free_choice_vectors.keys())}
        for i, base_scene in enumerate(shared_choice_base_scenes):
            choice_df_data[f'choice_event_{base_scene}'] = [vec[i] if i < len(vec) and not np.isnan(vec[i]) else np.nan 
                                                             for vec in free_choice_vectors.values()]
        choice_df = pd.DataFrame(choice_df_data)
        choice_file = os.path.join(output_dir, "free_choice_vectors.xlsx")
        choice_df.to_excel(choice_file, index=False)
        print(f"Saved free choice vectors to: {choice_file}")
    
    # Save yoke choice vectors
    if all_results['raw']['yoke'] is not None:
        yoke_choice_vectors = all_results['raw']['yoke']['choice_vectors']
        choice_df_data = {'subject_id': list(yoke_choice_vectors.keys())}
        for i, base_scene in enumerate(shared_choice_base_scenes):
            choice_df_data[f'choice_event_{base_scene}'] = [vec[i] if i < len(vec) and not np.isnan(vec[i]) else np.nan 
                                                             for vec in yoke_choice_vectors.values()]
        choice_df = pd.DataFrame(choice_df_data)
        choice_file = os.path.join(output_dir, "yoke_choice_vectors.xlsx")
        choice_df.to_excel(choice_file, index=False)
        print(f"Saved yoke choice vectors to: {choice_file}")
    
    # Save pairwise correlations
    print("\n" + "="*80)
    print("SAVING PAIRWISE CORRELATIONS")
    print("="*80)
    
    for analysis_type, results in all_results.items():
        for condition in ['free', 'yoke']:
            if results.get(condition) is not None:
                pairs_data = {
                    'subject_1': [pair[0] for pair in results[condition]['subject_pairs']],
                    'subject_2': [pair[1] for pair in results[condition]['subject_pairs']],
                    'correlation_r': results[condition]['values']
                }
                pairs_df = pd.DataFrame(pairs_data)
                pairs_file = os.path.join(output_dir, f"{condition}_pairwise_choice_isc_{analysis_type}.xlsx")
                pairs_df.to_excel(pairs_file, index=False)
                print(f"Saved {condition} pairwise correlations ({analysis_type}) to: {pairs_file}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING STATISTICAL RESULTS")
    print("="*80)
    
    # Create summary DataFrame
    summary_data = []
    
    for analysis_type, results in all_results.items():
        for condition in ['free', 'yoke']:
            if results.get(condition) is not None:
                summary_data.append({
                    'Analysis': analysis_type,
                    'Condition': condition,
                    'N_pairs': results[condition]['n'],
                    'Mean': results[condition]['mean'],
                    'Std': results[condition]['std'],
                    't_statistic': results[condition]['t_stat'],
                    'p_value': results[condition]['p_val']
                })
        
        # Add comparison
        if results.get('free_vs_yoke') is not None:
            summary_data.append({
                'Analysis': analysis_type,
                'Condition': 'Free_vs_Yoke',
                'N_pairs': f"{results['free_vs_yoke']['free_n']} vs {results['free_vs_yoke']['yoke_n']}",
                'Mean': f"{results['free_vs_yoke']['free_mean']:.4f} vs {results['free_vs_yoke']['yoke_mean']:.4f}",
                'Std': '',
                't_statistic': results['free_vs_yoke']['t_stat'],
                'p_value': results['free_vs_yoke']['p_val']
            })
    
    summary_df = pd.DataFrame(summary_data)
    stats_file = os.path.join(output_dir, "choice_isc_statistical_results.xlsx")
    summary_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    # Create text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("INDIVIDUAL VARIABILITY IN CHOICES MADE")
    report_lines.append("Choice ISC Analysis Results")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Data source: Individual event files from data/individual_data/romance/")
    report_lines.append(f"Shared choice events analyzed: {shared_choice_base_scenes}")
    report_lines.append(f"Total: {len(shared_choice_base_scenes)} choice events")
    report_lines.append("")
    report_lines.append("Note: For yoked subjects, using subj_choice column (actual choice made)")
    report_lines.append("      For free subjects, extracting choice from scenes column")
    report_lines.append("")
    
    for analysis_type, results in all_results.items():
        report_lines.append("="*80)
        report_lines.append(f"ANALYSIS: {analysis_type.upper().replace('_', ' ')}")
        report_lines.append("="*80)
        report_lines.append("")
        
        for condition in ['free', 'yoke']:
            if results.get(condition) is not None:
                r = results[condition]
                val_type = 'z' if analysis_type == 'z_transformed' else 'r'
                report_lines.append(f"{condition.upper()} Condition:")
                report_lines.append(f"  N pairs: {r['n']}")
                report_lines.append(f"  Mean {val_type}: {r['mean']:.4f}")
                report_lines.append(f"  Std: {r['std']:.4f}")
                report_lines.append(f"  Min {val_type}: {np.min(r['values']):.4f}")
                report_lines.append(f"  Max {val_type}: {np.max(r['values']):.4f}")
                report_lines.append(f"  One-sample t-test (vs 0):")
                report_lines.append(f"    t({r['n']-1}) = {r['t_stat']:.4f}, p = {r['p_val']:.6f}")
                if r['p_val'] < 0.001:
                    report_lines.append(f"    Result: Choice ISC is significantly above zero (p < 0.001)")
                elif r['p_val'] < 0.01:
                    report_lines.append(f"    Result: Choice ISC is significantly above zero (p < 0.01)")
                elif r['p_val'] < 0.05:
                    report_lines.append(f"    Result: Choice ISC is significantly above zero (p < 0.05)")
                else:
                    report_lines.append(f"    Result: Choice ISC is not significantly different from zero (p >= 0.05)")
                report_lines.append("")
        
        if results.get('free_vs_yoke') is not None:
            comp = results['free_vs_yoke']
            val_type = 'z' if analysis_type == 'z_transformed' else 'r'
            report_lines.append("FREE vs YOKED Comparison:")
            report_lines.append(f"  Free mean {val_type}: {comp['free_mean']:.4f} (N={comp['free_n']})")
            report_lines.append(f"  Yoke mean {val_type}: {comp['yoke_mean']:.4f} (N={comp['yoke_n']})")
            report_lines.append(f"  Two-sample t-test:")
            report_lines.append(f"    t = {comp['t_stat']:.4f}, p = {comp['p_val']:.6f}")
            if comp['p_val'] < 0.001:
                report_lines.append(f"    Result: Significant difference (p < 0.001)")
            elif comp['p_val'] < 0.01:
                report_lines.append(f"    Result: Significant difference (p < 0.01)")
            elif comp['p_val'] < 0.05:
                report_lines.append(f"    Result: Significant difference (p < 0.05)")
            else:
                report_lines.append(f"    Result: No significant difference (p >= 0.05)")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "choice_isc_statistical_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_choice_isc()
