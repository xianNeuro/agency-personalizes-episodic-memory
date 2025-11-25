#!/usr/bin/env python3
"""
Permutation Test for Recall ISC
Non-parametric permutation test to ensure higher Recall ISC in Yoked and Passive
conditions is not due to participants sharing the same story-path.

Randomly samples 1 Yoked and 1 Passive participant from each of 18 story-paths,
repeats 10,000 times to generate distributions, and compares Free condition's
Recall ISC to these distributions.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, ttest_ind
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from data_structure import RecallDataLoader
from run2_individual_variability_recalled_events import (
    identify_story_paths, get_subject_shared_not_vector,
    extract_shared_recall_vector, extract_nonchoice_shared_recall_vector,
    identify_choice_events_in_shared, load_monthy_map
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERMUTATIONS = 10000

def load_shared_recall_vectors(loader, condition, converge_dict, shared_choice_mask=None, 
                               use_nonchoice_only=False, base_path=None):
    """Load shared recall vectors for all subjects in a condition"""
    
    # Get all subject IDs
    all_subject_ids = loader.get_subject_ids_from_events('MV', condition)
    subject_ids = [sid for sid in all_subject_ids if not sid.startswith('~$')]
    
    # For free condition, filter to only subjects that correspond to story paths (N=18)
    if condition == 'free':
        story_paths = identify_story_paths(loader)
        free_ids_from_paths = []
        for to_num, path_info in story_paths.items():
            free_id = path_info['free']
            if free_id is not None:
                free_ids_from_paths.append(free_id)
        subject_ids = [sid for sid in subject_ids if sid in free_ids_from_paths]
    
    # Load recall data
    recall_df = loader.load_recall_data('MV', condition)
    
    # Extract shared recall vectors
    shared_recall_vectors = {}
    expected_length = 49 if use_nonchoice_only else 64
    
    for subject_id in subject_ids:
        try:
            shared_not_vector = get_subject_shared_not_vector(loader, condition, subject_id, 
                                                             converge_dict, base_path)
            
            if subject_id not in recall_df.columns:
                continue
            
            recall_vector = recall_df[subject_id].values
            
            if use_nonchoice_only:
                shared_recall = extract_nonchoice_shared_recall_vector(recall_vector, 
                                                                      shared_not_vector, 
                                                                      shared_choice_mask)
            else:
                shared_recall = extract_shared_recall_vector(recall_vector, shared_not_vector)
            
            if len(shared_recall) == expected_length:
                shared_recall_vectors[subject_id] = shared_recall
        except Exception as e:
            continue
    
    return shared_recall_vectors

def compute_mean_isc_broadcast(recall_vectors):
    """Compute mean ISC using broadcasting for efficiency"""
    if len(recall_vectors) < 2:
        return np.nan
    
    # Convert to numpy array (subjects x events)
    vectors_array = np.array([recall_vectors[sid] for sid in sorted(recall_vectors.keys())])
    n_subjects = vectors_array.shape[0]
    
    # Compute all pairwise correlations using broadcasting
    # Center each vector (subtract mean)
    vectors_centered = vectors_array - vectors_array.mean(axis=1, keepdims=True)
    
    # Compute norms
    norms = np.linalg.norm(vectors_centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    
    # Normalize
    vectors_normalized = vectors_centered / norms
    
    # Compute all pairwise correlations (matrix multiplication)
    correlation_matrix = np.dot(vectors_normalized, vectors_normalized.T)
    
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n_subjects, n_subjects)), k=1).astype(bool)
    correlations = correlation_matrix[mask]
    
    return np.mean(correlations)

def sample_and_compute_isc(args):
    """Sample subjects and compute mean ISC for one permutation"""
    story_paths, yoke_vectors, pasv_vectors, use_nonchoice_only = args
    
    # Sample 1 Yoked and 1 Passive from each story path
    sampled_yoke = []
    sampled_pasv = []
    
    for to_num, path_info in story_paths.items():
        yoke_ids = path_info.get('yoke', [])
        pasv_ids = path_info.get('pasv', [])
        
        if len(yoke_ids) > 0:
            # Randomly sample one yoked subject
            available_yoke = [sid for sid in yoke_ids if sid in yoke_vectors]
            if len(available_yoke) > 0:
                sampled_yoke.append(np.random.choice(available_yoke))
        
        if len(pasv_ids) > 0:
            # Randomly sample one passive subject
            available_pasv = [sid for sid in pasv_ids if sid in pasv_vectors]
            if len(available_pasv) > 0:
                sampled_pasv.append(np.random.choice(available_pasv))
    
    # Compute mean ISC for sampled subjects
    yoke_isc = np.nan
    if len(sampled_yoke) >= 2:
        yoke_sample_vectors = {sid: yoke_vectors[sid] for sid in sampled_yoke}
        yoke_isc = compute_mean_isc_broadcast(yoke_sample_vectors)
    
    pasv_isc = np.nan
    if len(sampled_pasv) >= 2:
        pasv_sample_vectors = {sid: pasv_vectors[sid] for sid in sampled_pasv}
        pasv_isc = compute_mean_isc_broadcast(pasv_sample_vectors)
    
    return yoke_isc, pasv_isc

def run_permutation_test(loader, converge_dict, shared_choice_mask, use_nonchoice_only, base_path):
    """Run permutation test"""
    
    event_type = "49 non-choice events" if use_nonchoice_only else "64 shared events"
    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST: {event_type}")
    print(f"{'='*80}")
    
    # Load recall vectors for all conditions
    print("\nLoading recall vectors...")
    free_vectors = load_shared_recall_vectors(loader, 'free', converge_dict, shared_choice_mask,
                                             use_nonchoice_only, base_path)
    yoke_vectors = load_shared_recall_vectors(loader, 'yoke', converge_dict, shared_choice_mask,
                                             use_nonchoice_only, base_path)
    pasv_vectors = load_shared_recall_vectors(loader, 'pasv', converge_dict, shared_choice_mask,
                                             use_nonchoice_only, base_path)
    
    print(f"Loaded vectors: Free={len(free_vectors)}, Yoke={len(yoke_vectors)}, Pasv={len(pasv_vectors)}")
    
    # Compute Free condition mean ISC
    print("\nComputing Free condition mean ISC...")
    free_mean_isc = compute_mean_isc_broadcast(free_vectors)
    print(f"Free mean ISC: {free_mean_isc:.4f}")
    
    # Identify story paths
    print("\nIdentifying story paths...")
    story_paths = identify_story_paths(loader)
    print(f"Found {len(story_paths)} story paths")
    
    # Prepare arguments for parallel processing
    print(f"\nRunning {N_PERMUTATIONS} permutations in parallel...")
    args_list = [(story_paths, yoke_vectors, pasv_vectors, use_nonchoice_only)] * N_PERMUTATIONS
    
    # Run permutations in parallel
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")
    
    with Pool(n_cores) as pool:
        results = pool.map(sample_and_compute_isc, args_list)
    
    # Extract results
    yoke_isc_values = np.array([r[0] for r in results])
    pasv_isc_values = np.array([r[1] for r in results])
    
    # Remove NaN values
    yoke_isc_values = yoke_isc_values[~np.isnan(yoke_isc_values)]
    pasv_isc_values = pasv_isc_values[~np.isnan(pasv_isc_values)]
    
    print(f"Valid permutations: Yoke={len(yoke_isc_values)}, Pasv={len(pasv_isc_values)}")
    
    # Compute p-values (proportion of samples below Free mean)
    p_yoke = np.mean(yoke_isc_values < free_mean_isc)
    p_pasv = np.mean(pasv_isc_values < free_mean_isc)
    
    # Compare Yoke vs Pasv distributions
    t_stat_yoke_pasv, p_yoke_pasv = ttest_ind(yoke_isc_values, pasv_isc_values)
    
    return {
        'free_mean_isc': free_mean_isc,
        'yoke_isc_values': yoke_isc_values,
        'pasv_isc_values': pasv_isc_values,
        'p_yoke': p_yoke,
        'p_pasv': p_pasv,
        'yoke_mean': np.mean(yoke_isc_values),
        'yoke_std': np.std(yoke_isc_values, ddof=1),
        'pasv_mean': np.mean(pasv_isc_values),
        'pasv_std': np.std(pasv_isc_values, ddof=1),
        't_yoke_pasv': t_stat_yoke_pasv,
        'p_yoke_pasv': p_yoke_pasv,
        'event_type': event_type
    }

def plot_distributions(results, output_dir, suffix=""):
    """Plot distributions with Free mean as vertical line"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(results['yoke_isc_values'], bins=50, alpha=0.7, color='orange', 
            label=f"Yoked (mean={results['yoke_mean']:.4f})", density=True)
    ax.hist(results['pasv_isc_values'], bins=50, alpha=0.7, color='green',
            label=f"Passive (mean={results['pasv_mean']:.4f})", density=True)
    
    # Plot Free mean as vertical line
    ax.axvline(results['free_mean_isc'], color='blue', linestyle='--', linewidth=2,
              label=f"Free (mean={results['free_mean_isc']:.4f})")
    
    ax.set_xlabel('Recall ISC (Mean Pairwise Correlation)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Permutation Test: {results["event_type"]}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"permutation_test_distribution{suffix}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_file}")

def analyze_permutation_test():
    """Main analysis function"""
    
    print("="*80)
    print("PERMUTATION TEST FOR RECALL ISC")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run12_permutation_test_recall_isc")
    
    # Load converge dict and identify choice events
    print("\nLoading story map and identifying shared events...")
    converge_dict = load_monthy_map()
    shared_choice_mask, _ = identify_choice_events_in_shared(loader, 'free', converge_dict, base_path)
    
    # Run permutation test for 64 events
    results_64 = run_permutation_test(loader, converge_dict, shared_choice_mask, 
                                     use_nonchoice_only=False, base_path=base_path)
    
    # Run permutation test for 49 non-choice events
    results_49 = run_permutation_test(loader, converge_dict, shared_choice_mask,
                                     use_nonchoice_only=True, base_path=base_path)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save 10k values to CSV
    csv_data = pd.DataFrame({
        'yoke_isc_64': results_64['yoke_isc_values'],
        'pasv_isc_64': results_64['pasv_isc_values'],
        'yoke_isc_49': results_49['yoke_isc_values'],
        'pasv_isc_49': results_49['pasv_isc_values']
    })
    csv_file = os.path.join(output_dir, "permutation_isc_values.csv")
    csv_data.to_csv(csv_file, index=False)
    print(f"Saved 10k permutation values to: {csv_file}")
    
    # Create plots
    plot_distributions(results_64, output_dir, suffix="_64events")
    plot_distributions(results_49, output_dir, suffix="_49events")
    
    # Create report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PERMUTATION TEST FOR RECALL ISC")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Method: Non-parametric permutation test")
    report_lines.append("  - Randomly sampled 1 Yoked and 1 Passive participant from each of 18 story-paths")
    report_lines.append("  - Repeated 10,000 times to generate distributions")
    report_lines.append("  - Compared Free condition's Recall ISC to these distributions")
    report_lines.append("")
    report_lines.append(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    report_lines.append("")
    
    for results, event_label in [(results_64, "64 Shared Events"), (results_49, "49 Non-Choice Events")]:
        report_lines.append("="*80)
        report_lines.append(f"ANALYSIS: {event_label}")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"Free condition mean ISC: {results['free_mean_isc']:.4f}")
        report_lines.append("")
        report_lines.append("Permutation distributions:")
        report_lines.append(f"  Yoked: Mean = {results['yoke_mean']:.4f}, Std = {results['yoke_std']:.4f}, N = {len(results['yoke_isc_values'])}")
        report_lines.append(f"  Passive: Mean = {results['pasv_mean']:.4f}, Std = {results['pasv_std']:.4f}, N = {len(results['pasv_isc_values'])}")
        report_lines.append("")
        report_lines.append("Statistical tests:")
        report_lines.append(f"  Free vs Yoked: p = {results['p_yoke']:.6f} (proportion of Yoked samples below Free mean)")
        if results['p_yoke'] < 0.001:
            report_lines.append(f"    Result: Free ISC significantly lower than Yoked (p < 0.001)")
        elif results['p_yoke'] < 0.01:
            report_lines.append(f"    Result: Free ISC significantly lower than Yoked (p < 0.01)")
        elif results['p_yoke'] < 0.05:
            report_lines.append(f"    Result: Free ISC significantly lower than Yoked (p < 0.05)")
        else:
            report_lines.append(f"    Result: Not significant (p >= 0.05)")
        report_lines.append("")
        report_lines.append(f"  Free vs Passive: p = {results['p_pasv']:.6f} (proportion of Passive samples below Free mean)")
        if results['p_pasv'] < 0.001:
            report_lines.append(f"    Result: Free ISC significantly lower than Passive (p < 0.001)")
        elif results['p_pasv'] < 0.01:
            report_lines.append(f"    Result: Free ISC significantly lower than Passive (p < 0.01)")
        elif results['p_pasv'] < 0.05:
            report_lines.append(f"    Result: Free ISC significantly lower than Passive (p < 0.05)")
        else:
            report_lines.append(f"    Result: Not significant (p >= 0.05)")
        report_lines.append("")
        report_lines.append(f"  Yoked vs Passive: t = {results['t_yoke_pasv']:.4f}, p = {results['p_yoke_pasv']:.6f}")
        if results['p_yoke_pasv'] < 0.001:
            report_lines.append(f"    Result: Significant difference (p < 0.001)")
        elif results['p_yoke_pasv'] < 0.01:
            report_lines.append(f"    Result: Significant difference (p < 0.01)")
        elif results['p_yoke_pasv'] < 0.05:
            report_lines.append(f"    Result: Significant difference (p < 0.05)")
        else:
            report_lines.append(f"    Result: Not significant (p >= 0.05)")
        report_lines.append("")
    
    report_file = os.path.join(output_dir, "permutation_test_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_permutation_test()

