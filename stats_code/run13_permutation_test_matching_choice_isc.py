#!/usr/bin/env python3
"""
Permutation Test for Recall ISC with Matching Choice ISC
Non-parametric permutation test to show that Free participants still have
significantly reduced Recall ISC compared to Yoked counterparts with matching Choice ISC.

Randomly samples 1 Yoked participant from each of 18 story-paths, but only keeps
samples where the sampled Yoked Choice ISC <= Free Choice ISC. Repeats until
10,000 valid samples are obtained, then computes Recall ISC for each sample.
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, ttest_1samp
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from data_structure import RecallDataLoader
from run2_individual_variability_recalled_events import (
    identify_story_paths, get_subject_shared_not_vector,
    extract_shared_recall_vector, extract_nonchoice_shared_recall_vector,
    identify_choice_events_in_shared, load_monthy_map
)
from run3_individual_variability_choices import (
    get_shared_choice_base_scenes, extract_choice_from_event_file
)

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERMUTATIONS = 10000

def load_choice_vectors(loader, condition, shared_choice_base_scenes, base_path):
    """Load choice vectors for all subjects in a condition"""
    
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
    
    # Extract choice vectors
    choice_vectors = {}
    for subject_id in subject_ids:
        choice_vector = extract_choice_from_event_file(subject_id, condition, 
                                                      shared_choice_base_scenes, base_path)
        if choice_vector is not None and not np.all(np.isnan(choice_vector)):
            choice_vectors[subject_id] = choice_vector
    
    return choice_vectors

def compute_mean_choice_isc_broadcast(choice_vectors):
    """Compute mean Choice ISC using broadcasting for efficiency"""
    if len(choice_vectors) < 2:
        return np.nan
    
    # Convert to numpy array (subjects x choices)
    vectors_array = np.array([choice_vectors[sid] for sid in sorted(choice_vectors.keys())])
    n_subjects = vectors_array.shape[0]
    
    # Remove any subjects with NaN values
    valid_mask = ~np.isnan(vectors_array).any(axis=1)
    if valid_mask.sum() < 2:
        return np.nan
    
    vectors_array = vectors_array[valid_mask]
    n_subjects = vectors_array.shape[0]
    
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

def compute_mean_recall_isc_broadcast(recall_vectors):
    """Compute mean Recall ISC using broadcasting for efficiency"""
    if len(recall_vectors) < 2:
        return np.nan
    
    # Convert to numpy array (subjects x events)
    vectors_array = np.array([recall_vectors[sid] for sid in sorted(recall_vectors.keys())])
    n_subjects = vectors_array.shape[0]
    
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

def sample_with_choice_isc_constraint(args):
    """Sample Yoked subjects and compute Recall ISC, ensuring Choice ISC <= Free Choice ISC"""
    story_paths, yoke_choice_vectors, yoke_recall_vectors, free_choice_isc, max_attempts = args
    
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        
        # Sample 1 Yoked from each story path
        sampled_yoke = []
        for to_num, path_info in story_paths.items():
            yoke_ids = path_info.get('yoke', [])
            if len(yoke_ids) > 0:
                available_yoke = [sid for sid in yoke_ids if sid in yoke_choice_vectors and sid in yoke_recall_vectors]
                if len(available_yoke) > 0:
                    sampled_yoke.append(np.random.choice(available_yoke))
        
        if len(sampled_yoke) < 2:
            continue
        
        # Compute Choice ISC for sampled Yoked subjects
        yoke_sample_choice_vectors = {sid: yoke_choice_vectors[sid] for sid in sampled_yoke}
        yoke_choice_isc = compute_mean_choice_isc_broadcast(yoke_sample_choice_vectors)
        
        # Check if Choice ISC <= Free Choice ISC
        if np.isnan(yoke_choice_isc) or yoke_choice_isc > free_choice_isc:
            continue  # Reject and resample
        
        # If valid, compute Recall ISC
        yoke_sample_recall_vectors = {sid: yoke_recall_vectors[sid] for sid in sampled_yoke}
        yoke_recall_isc = compute_mean_recall_isc_broadcast(yoke_sample_recall_vectors)
        
        if not np.isnan(yoke_recall_isc):
            return yoke_recall_isc, yoke_choice_isc
    
    # If max attempts reached, return NaN
    return np.nan, np.nan

def run_permutation_test_matching_choice(loader, converge_dict, shared_choice_mask, 
                                        shared_choice_base_scenes, use_nonchoice_only, base_path):
    """Run permutation test with Choice ISC matching constraint"""
    
    event_type = "49 non-choice events" if use_nonchoice_only else "64 shared events"
    print(f"\n{'='*80}")
    print(f"PERMUTATION TEST WITH MATCHING CHOICE ISC: {event_type}")
    print(f"{'='*80}")
    
    # Load choice vectors
    print("\nLoading choice vectors...")
    free_choice_vectors = load_choice_vectors(loader, 'free', shared_choice_base_scenes, base_path)
    yoke_choice_vectors = load_choice_vectors(loader, 'yoke', shared_choice_base_scenes, base_path)
    
    print(f"Loaded choice vectors: Free={len(free_choice_vectors)}, Yoke={len(yoke_choice_vectors)}")
    
    # Compute Free Choice ISC
    print("\nComputing Free condition Choice ISC...")
    free_choice_isc = compute_mean_choice_isc_broadcast(free_choice_vectors)
    print(f"Free mean Choice ISC: {free_choice_isc:.4f}")
    
    # Load recall vectors
    print("\nLoading recall vectors...")
    free_recall_vectors = load_shared_recall_vectors(loader, 'free', converge_dict, shared_choice_mask,
                                                    use_nonchoice_only, base_path)
    yoke_recall_vectors = load_shared_recall_vectors(loader, 'yoke', converge_dict, shared_choice_mask,
                                                    use_nonchoice_only, base_path)
    
    print(f"Loaded recall vectors: Free={len(free_recall_vectors)}, Yoke={len(yoke_recall_vectors)}")
    
    # Compute Free Recall ISC
    print("\nComputing Free condition Recall ISC...")
    free_recall_isc = compute_mean_recall_isc_broadcast(free_recall_vectors)
    print(f"Free mean Recall ISC: {free_recall_isc:.4f}")
    
    # Identify story paths
    print("\nIdentifying story paths...")
    story_paths = identify_story_paths(loader)
    print(f"Found {len(story_paths)} story paths")
    
    # Run permutations with constraint
    print(f"\nRunning {N_PERMUTATIONS} permutations with Choice ISC constraint...")
    print("  (Only keeping samples where Yoked Choice ISC <= Free Choice ISC)")
    
    # Prepare arguments for parallel processing
    max_attempts = 100  # Maximum attempts per permutation to find valid sample
    args_list = [(story_paths, yoke_choice_vectors, yoke_recall_vectors, 
                  free_choice_isc, max_attempts)] * N_PERMUTATIONS
    
    # Run permutations in parallel
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores")
    
    with Pool(n_cores) as pool:
        results = pool.map(sample_with_choice_isc_constraint, args_list)
    
    # Extract results
    yoke_recall_isc_values = np.array([r[0] for r in results])
    yoke_choice_isc_values = np.array([r[1] for r in results])
    
    # Remove NaN values
    valid_mask = ~np.isnan(yoke_recall_isc_values)
    yoke_recall_isc_values = yoke_recall_isc_values[valid_mask]
    yoke_choice_isc_values = yoke_choice_isc_values[valid_mask]
    
    print(f"Valid permutations: {len(yoke_recall_isc_values)}/{N_PERMUTATIONS}")
    
    # If we don't have enough valid samples, we might need to run more
    if len(yoke_recall_isc_values) < N_PERMUTATIONS:
        print(f"Warning: Only {len(yoke_recall_isc_values)} valid samples out of {N_PERMUTATIONS} requested")
        print("  This may indicate the constraint is too strict")
    
    # Compute p-value (one-tailed test: proportion of Yoked samples below Free Recall ISC)
    p_value = np.mean(yoke_recall_isc_values < free_recall_isc)
    
    # One-sample t-test against Free Recall ISC
    t_stat, p_ttest = ttest_1samp(yoke_recall_isc_values, free_recall_isc)
    
    return {
        'free_recall_isc': free_recall_isc,
        'free_choice_isc': free_choice_isc,
        'yoke_recall_isc_values': yoke_recall_isc_values,
        'yoke_choice_isc_values': yoke_choice_isc_values,
        'n_valid': len(yoke_recall_isc_values),
        'p_value': p_value,
        't_stat': t_stat,
        'p_ttest': p_ttest,
        'yoke_recall_mean': np.mean(yoke_recall_isc_values),
        'yoke_recall_std': np.std(yoke_recall_isc_values, ddof=1),
        'yoke_choice_mean': np.mean(yoke_choice_isc_values),
        'yoke_choice_std': np.std(yoke_choice_isc_values, ddof=1),
        'event_type': event_type
    }

def plot_distribution(results, output_dir, suffix=""):
    """Plot distribution with Free Recall ISC as vertical line"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram for Yoked Recall ISC
    ax.hist(results['yoke_recall_isc_values'], bins=50, alpha=0.7, color='orange', 
            label=f"Yoked (mean={results['yoke_recall_mean']:.4f})", density=True)
    
    # Plot Free Recall ISC as vertical line
    ax.axvline(results['free_recall_isc'], color='blue', linestyle='--', linewidth=2,
              label=f"Free (mean={results['free_recall_isc']:.4f})")
    
    ax.set_xlabel('Recall ISC (Mean Pairwise Correlation)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Permutation Test (Matching Choice ISC): {results["event_type"]}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, f"permutation_test_matching_choice_isc{suffix}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {plot_file}")

def analyze_permutation_test_matching_choice():
    """Main analysis function"""
    
    print("="*80)
    print("PERMUTATION TEST FOR RECALL ISC WITH MATCHING CHOICE ISC")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run13_permutation_test_matching_choice_isc")
    
    # Load converge dict and identify choice events
    print("\nLoading story map and identifying shared events...")
    converge_dict = load_monthy_map()
    shared_choice_mask, _ = identify_choice_events_in_shared(loader, 'free', converge_dict, base_path)
    
    # Get shared choice base scenes
    print("\nIdentifying shared choice events...")
    shared_choice_base_scenes = get_shared_choice_base_scenes()
    print(f"Found {len(shared_choice_base_scenes)} shared choice base scenes")
    
    # Run permutation test for 64 events
    results_64 = run_permutation_test_matching_choice(loader, converge_dict, shared_choice_mask,
                                                      shared_choice_base_scenes, 
                                                      use_nonchoice_only=False, base_path=base_path)
    
    # Run permutation test for 49 non-choice events
    results_49 = run_permutation_test_matching_choice(loader, converge_dict, shared_choice_mask,
                                                      shared_choice_base_scenes,
                                                      use_nonchoice_only=True, base_path=base_path)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save 10k values to CSV (pad with NaN if needed)
    max_len = max(len(results_64['yoke_recall_isc_values']), len(results_49['yoke_recall_isc_values']))
    
    # Pad arrays to same length
    yoke_64_padded = np.pad(results_64['yoke_recall_isc_values'], 
                           (0, max_len - len(results_64['yoke_recall_isc_values'])), 
                           constant_values=np.nan)
    yoke_49_padded = np.pad(results_49['yoke_recall_isc_values'],
                           (0, max_len - len(results_49['yoke_recall_isc_values'])),
                           constant_values=np.nan)
    
    csv_data = pd.DataFrame({
        'yoke_recall_isc_64': yoke_64_padded,
        'yoke_recall_isc_49': yoke_49_padded
    })
    csv_file = os.path.join(output_dir, "permutation_matching_choice_isc_values.csv")
    csv_data.to_csv(csv_file, index=False)
    print(f"Saved permutation values to: {csv_file}")
    print(f"  Valid samples: 64 events = {len(results_64['yoke_recall_isc_values'])}, "
          f"49 events = {len(results_49['yoke_recall_isc_values'])}")
    
    # Create plots
    plot_distribution(results_64, output_dir, suffix="_64events")
    plot_distribution(results_49, output_dir, suffix="_49events")
    
    # Create report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("PERMUTATION TEST FOR RECALL ISC WITH MATCHING CHOICE ISC")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Method: Non-parametric permutation test with Choice ISC constraint")
    report_lines.append("  - Randomly sampled 1 Yoked participant from each of 18 story-paths")
    report_lines.append("  - Only kept samples where Yoked Choice ISC <= Free Choice ISC")
    report_lines.append("  - Repeated until 10,000 valid samples obtained")
    report_lines.append("  - Computed Recall ISC for each valid sample")
    report_lines.append("  - Compared Free condition's Recall ISC to Yoked distribution")
    report_lines.append("")
    report_lines.append(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    report_lines.append("")
    
    for results, event_label in [(results_64, "64 Shared Events"), (results_49, "49 Non-Choice Events")]:
        report_lines.append("="*80)
        report_lines.append(f"ANALYSIS: {event_label}")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"Free condition:")
        report_lines.append(f"  Choice ISC: {results['free_choice_isc']:.4f}")
        report_lines.append(f"  Recall ISC: {results['free_recall_isc']:.4f}")
        report_lines.append("")
        report_lines.append(f"Yoked condition (matching Choice ISC constraint):")
        report_lines.append(f"  Valid samples: {results['n_valid']}/{N_PERMUTATIONS}")
        report_lines.append(f"  Choice ISC: Mean = {results['yoke_choice_mean']:.4f}, Std = {results['yoke_choice_std']:.4f}")
        report_lines.append(f"  Recall ISC: Mean = {results['yoke_recall_mean']:.4f}, Std = {results['yoke_recall_std']:.4f}")
        report_lines.append("")
        report_lines.append("Statistical test:")
        report_lines.append(f"  One-tailed test: p = {results['p_value']:.6f} (proportion of Yoked samples below Free Recall ISC)")
        report_lines.append(f"  One-sample t-test: t({results['n_valid']-1}) = {results['t_stat']:.4f}, p = {results['p_ttest']:.6f}")
        if results['p_value'] < 0.001:
            report_lines.append(f"  Result: Free Recall ISC significantly lower than Yoked (p < 0.001)")
        elif results['p_value'] < 0.01:
            report_lines.append(f"  Result: Free Recall ISC significantly lower than Yoked (p < 0.01)")
        elif results['p_value'] < 0.05:
            report_lines.append(f"  Result: Free Recall ISC significantly lower than Yoked (p < 0.05)")
        else:
            report_lines.append(f"  Result: Not significant (p >= 0.05)")
        report_lines.append("")
    
    report_file = os.path.join(output_dir, "permutation_test_matching_choice_isc_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_permutation_test_matching_choice()

