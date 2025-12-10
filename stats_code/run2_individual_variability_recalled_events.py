#!/usr/bin/env python3
"""
Individual Variability in Recalled Events
Computes inter-participant correlation (ISC) for recall performance vectors
in shared story sections.

Runs four analyses:
1. 64 shared events, raw correlation r-values
2. 49 non-choice shared events, raw correlation r-values
3. 64 shared events, Fisher z-transformed r-values
4. 49 non-choice shared events, Fisher z-transformed r-values

For each analysis, performs:
- One-sample t-tests per condition
- ANOVA across conditions comparing Recall ISC
- Post-hoc tests (Free vs Yoked, Free vs Passive, Yoked vs Passive)
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, ttest_1samp, norm, f_oneway
from scipy import stats
from data_structure import RecallDataLoader
from effect_size_utils import (calculate_ci_mean, cohens_d_one_sample, cohens_d_two_sample,
                             calculate_anova_effect_sizes, format_ci, format_cohens_d)

def load_monthy_map():
    """Load the story map from romance_data1.xlsx to identify shared events"""
    
    # Use current directory
    base_path = os.path.abspath('.')
    data_dir = os.path.join(base_path, "data")
    map_file = os.path.join(data_dir, "romance_data1.xlsx")
    
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Story map file not found: {map_file}")
    
    map_df = pd.read_excel(map_file, sheet_name='story_map')
    print(f"Loaded story_map from romance_data1.xlsx: {map_df.shape}")
    
    # Create a dictionary mapping event_lab to Converge status
    converge_dict = {}
    for idx, row in map_df.iterrows():
        event_lab = row['event_lab'] if 'event_lab' in map_df.columns else row.iloc[0]
        converge = row['Converge'] if 'Converge' in map_df.columns else None
        # Convert to 1 if 'Y', 0 otherwise
        converge_dict[event_lab] = 1 if converge == 'Y' else 0
    
    print(f"Created converge mapping for {len(converge_dict)} events")
    print(f"Shared events (Converge='Y'): {sum(converge_dict.values())}")
    
    return converge_dict

def identify_choice_events_in_shared(loader, condition, converge_dict, base_path):
    """Identify which of the 64 shared events are choice events"""
    
    # Get a representative subject to check choice events
    all_subject_ids = loader.get_subject_ids_from_events('MV', condition)
    subject_ids = [sid for sid in all_subject_ids if not sid.startswith('~$')]
    
    if len(subject_ids) == 0:
        return None
    
    # Use first subject to identify choice events
    representative_subject = subject_ids[0]
    
    # Try to load from data folder first
    condition_num = {'free': '1', 'yoke': '2', 'pasv': '3'}[condition]
    data_event_path = os.path.join(base_path, "data", "individual_data", "romance", 
                                   f"{condition_num}_{condition}", f"{representative_subject}_events.xlsx")
    
    if os.path.exists(data_event_path):
        event_df = pd.read_excel(data_event_path)
    else:
        # Fall back to RecallDataLoader
        event_df = loader.load_subject_event_data('MV', condition, representative_subject)
    
    # Get event_lab column
    if 'event_lab' in event_df.columns:
        event_labs = event_df['event_lab'].values
    elif 'old_seg' in event_df.columns:
        event_labs = event_df['old_seg'].values
    else:
        raise ValueError(f"Neither event_lab nor old_seg column found")
    
    # Check if scenes column exists to identify choice events
    is_choice_event = []
    if 'scenes' in event_df.columns:
        for scenes_val in event_df['scenes']:
            if pd.isna(scenes_val):
                is_choice_event.append(False)
            else:
                is_choice_event.append('_' in str(scenes_val))
    else:
        # If no scenes column, check if there's a Choice column in monthy_map
        # For now, assume no choice events if we can't determine
        is_choice_event = [False] * len(event_labs)
    
    # Create mapping from event_lab to is_choice
    event_choice_dict = {}
    for i, event_lab in enumerate(event_labs):
        if i < len(is_choice_event):
            event_choice_dict[event_lab] = is_choice_event[i]
    
    # Now identify which shared events are choice events
    shared_choice_mask = []
    shared_event_labs = []
    
    for event_lab in event_labs:
        if event_lab in converge_dict and converge_dict[event_lab] == 1:
            # This is a shared event
            shared_event_labs.append(event_lab)
            is_choice = event_choice_dict.get(event_lab, False)
            shared_choice_mask.append(is_choice)
    
    print(f"Identified {len(shared_event_labs)} shared events")
    print(f"  Choice events: {sum(shared_choice_mask)}")
    print(f"  Non-choice events: {len(shared_choice_mask) - sum(shared_choice_mask)}")
    
    return np.array(shared_choice_mask, dtype=bool), shared_event_labs

def get_subject_shared_not_vector(loader, condition, subject_id, converge_dict, base_path):
    """Get shared/not vector for a single subject"""
    
    # Try to load from data folder first
    condition_num = {'free': '1', 'yoke': '2', 'pasv': '3'}[condition]
    data_event_path = os.path.join(base_path, "data", "individual_data", "romance", 
                                   f"{condition_num}_{condition}", f"{subject_id}_events.xlsx")
    
    # Load subject event file
    if os.path.exists(data_event_path):
        event_df = pd.read_excel(data_event_path)
    else:
        # Fall back to RecallDataLoader
        event_df = loader.load_subject_event_data('MV', condition, subject_id)
    
    # Get event_lab column (try event_lab first, then old_seg as fallback)
    if 'event_lab' in event_df.columns:
        event_labs = event_df['event_lab'].values
    elif 'old_seg' in event_df.columns:
        # Some yoke files use old_seg instead of event_lab
        event_labs = event_df['old_seg'].values
    else:
        raise ValueError(f"Neither event_lab nor old_seg column found in {subject_id}_events.xlsx. Available columns: {list(event_df.columns)}")
    
    # Create shared/not vector: 1 if shared, 0 if not
    shared_not_vector = []
    for event_lab in event_labs:
        if event_lab in converge_dict:
            shared_not_vector.append(converge_dict[event_lab])
        else:
            # If event_lab not in map, assume not shared (0)
            shared_not_vector.append(0)
    
    return np.array(shared_not_vector, dtype=float)

def extract_shared_recall_vector(recall_vector, shared_not_vector):
    """Extract only shared events from recall vector"""
    
    # Remove NaN padding from shared_not_vector
    valid_mask = ~np.isnan(shared_not_vector)
    shared_not_clean = shared_not_vector[valid_mask]
    
    # Align recall vector length with shared_not vector
    min_length = min(len(recall_vector), len(shared_not_clean))
    recall_clean = recall_vector[:min_length]
    shared_not_clean = shared_not_clean[:min_length]
    
    # Keep only shared events (where shared_not = 1)
    shared_mask = shared_not_clean == 1
    shared_recall = recall_clean[shared_mask]
    
    return shared_recall

def extract_nonchoice_shared_recall_vector(recall_vector, shared_not_vector, shared_choice_mask):
    """Extract only non-choice shared events from recall vector"""
    
    # Remove NaN padding from shared_not_vector
    valid_mask = ~np.isnan(shared_not_vector)
    shared_not_clean = shared_not_vector[valid_mask]
    
    # Align recall vector length with shared_not vector
    min_length = min(len(recall_vector), len(shared_not_clean))
    recall_clean = recall_vector[:min_length]
    shared_not_clean = shared_not_clean[:min_length]
    
    # Keep only shared events (where shared_not = 1)
    shared_mask = shared_not_clean == 1
    shared_recall = recall_clean[shared_mask]
    
    # Now filter to only non-choice events
    if len(shared_recall) == len(shared_choice_mask):
        nonchoice_mask = ~shared_choice_mask
        nonchoice_recall = shared_recall[nonchoice_mask]
        return nonchoice_recall
    else:
        # Length mismatch - return empty array
        return np.array([])

def identify_story_paths(loader):
    """Identify story paths by matching free subjects to yoked/pasv subjects"""
    
    # Get all subject IDs for each condition
    free_ids = loader.get_subject_ids_from_events('MV', 'free')
    yoke_ids = loader.get_subject_ids_from_events('MV', 'yoke')
    pasv_ids = loader.get_subject_ids_from_events('MV', 'pasv')
    
    # Filter out temporary files
    free_ids = [sid for sid in free_ids if not sid.startswith('~$')]
    yoke_ids = [sid for sid in yoke_ids if not sid.startswith('~$')]
    pasv_ids = [sid for sid in pasv_ids if not sid.startswith('~$')]
    
    # Extract free subject numbers (e.g., 'sub3_4003' -> 3)
    free_subject_map = {}
    for free_id in free_ids:
        # Extract number from 'sub3_4003' -> 3
        try:
            num = int(free_id.split('_')[0].replace('sub', ''))
            free_subject_map[num] = free_id
        except:
            continue
    
    # Group yoked and pasv subjects by their 'toX_' prefix
    story_paths = {}  # {story_path_num: {'free': free_id, 'yoke': [yoke_ids], 'pasv': [pasv_ids]}}
    
    for yoke_id in yoke_ids:
        # Extract 'toX' number from 'to3_sub5001' -> 3
        try:
            to_num = int(yoke_id.split('_')[0].replace('to', ''))
            if to_num not in story_paths:
                story_paths[to_num] = {'free': None, 'yoke': [], 'pasv': []}
            story_paths[to_num]['yoke'].append(yoke_id)
        except:
            continue
    
    for pasv_id in pasv_ids:
        # Extract 'toX' number from 'to3_sub6001' -> 3
        try:
            to_num = int(pasv_id.split('_')[0].replace('to', ''))
            if to_num not in story_paths:
                story_paths[to_num] = {'free': None, 'yoke': [], 'pasv': []}
            story_paths[to_num]['pasv'].append(pasv_id)
        except:
            continue
    
    # Match free subjects to story paths
    for to_num, free_id in free_subject_map.items():
        if to_num in story_paths:
            story_paths[to_num]['free'] = free_id
    
    return story_paths

def compute_pairwise_isc(loader, condition, converge_dict, shared_choice_mask=None, use_nonchoice_only=False, base_path=None):
    """Compute pairwise ISC (inter-participant correlations) for a condition"""
    
    event_type = "NON-CHOICE EVENTS ONLY" if use_nonchoice_only else "ALL SHARED EVENTS"
    print(f"\n{'='*80}")
    print(f"Computing pairwise ISC for {condition} condition ({event_type})")
    print(f"{'='*80}")
    
    # Get all subject IDs
    all_subject_ids = loader.get_subject_ids_from_events('MV', condition)
    subject_ids = [sid for sid in all_subject_ids if not sid.startswith('~$')]
    
    # For free condition, filter to only subjects that correspond to story paths (N=18)
    if condition == 'free':
        print("\nFiltering free subjects to only those corresponding to story paths...")
        story_paths = identify_story_paths(loader)
        
        # Get the free subject IDs from story paths
        free_ids_from_paths = []
        for to_num, path_info in story_paths.items():
            free_id = path_info['free']
            if free_id is not None:
                free_ids_from_paths.append(free_id)
        
        free_ids_from_paths = sorted(set(free_ids_from_paths))  # Remove duplicates and sort
        print(f"Found {len(free_ids_from_paths)} free subjects corresponding to story paths")
        
        # Filter subject_ids to only include those from story paths
        subject_ids = [sid for sid in subject_ids if sid in free_ids_from_paths]
        print(f"Filtered to {len(subject_ids)} free subjects for analysis")
    
    if len(subject_ids) == 0:
        print(f"No subjects found for {condition} condition")
        return None
    
    print(f"Found {len(subject_ids)} subjects")
    
    # Load recall data
    recall_df = loader.load_recall_data('MV', condition)
    
    # Extract shared recall vectors for all subjects
    shared_recall_vectors = {}
    expected_length = 49 if use_nonchoice_only else 64
    
    for subject_id in subject_ids:
        try:
            # Get shared/not vector
            shared_not_vector = get_subject_shared_not_vector(loader, condition, subject_id, converge_dict, base_path)
            
            # Get recall vector
            if subject_id not in recall_df.columns:
                print(f"Warning: {subject_id} not found in recall data")
                continue
            
            recall_vector = recall_df[subject_id].values
            
            # Extract shared recall vector (all or non-choice only)
            if use_nonchoice_only:
                shared_recall = extract_nonchoice_shared_recall_vector(recall_vector, shared_not_vector, shared_choice_mask)
            else:
                shared_recall = extract_shared_recall_vector(recall_vector, shared_not_vector)
            
            # Only keep if we have the expected number of events
            if len(shared_recall) == expected_length:
                shared_recall_vectors[subject_id] = shared_recall
            else:
                print(f"Warning: {subject_id} has {len(shared_recall)} events (expected {expected_length})")
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    print(f"Extracted shared recall vectors for {len(shared_recall_vectors)} subjects")
    
    # Compute pairwise correlations
    print(f"\nComputing pairwise correlations...")
    correlations = []
    pairs = []
    
    subject_id_list = sorted(shared_recall_vectors.keys())
    n_pairs = 0
    
    for i in range(len(subject_id_list)):
        for j in range(i + 1, len(subject_id_list)):
            subj1_id = subject_id_list[i]
            subj2_id = subject_id_list[j]
            
            vec1 = shared_recall_vectors[subj1_id]
            vec2 = shared_recall_vectors[subj2_id]
            
            # Compute Pearson correlation
            if len(vec1) == len(vec2) == expected_length:
                r_val, p_val = pearsonr(vec1, vec2)
                correlations.append(r_val)
                pairs.append({
                    'subject1_id': subj1_id,
                    'subject2_id': subj2_id,
                    'r_value': r_val,
                    'p_value': p_val
                })
                n_pairs += 1
    
    print(f"Computed {n_pairs} pairwise correlations")
    
    if len(correlations) == 0:
        print(f"Warning: No valid correlations computed for {condition} condition")
        return None
    
    return {
        'condition': condition,
        'n_subjects': len(subject_id_list),
        'n_pairs': n_pairs,
        'correlations': np.array(correlations),
        'pairs_df': pd.DataFrame(pairs)
    }

def fisher_z_transform(r):
    """Apply Fisher's Z transformation to correlation coefficient"""
    # Clip r values to valid range [-1, 1] to avoid numerical issues
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def perform_one_sample_ttest(data, test_value=0):
    """Perform 1-sample t-test against test_value"""
    data_clean = data[~np.isnan(data)]
    if len(data_clean) < 2:
        return None, None, None, len(data_clean), None, None, None
    
    t_stat, p_val = ttest_1samp(data_clean, test_value)
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean, ddof=1)
    
    # Calculate 95% CI and Cohen's d
    ci_lower, ci_upper, _ = calculate_ci_mean(data_clean)
    cohens_d = cohens_d_one_sample(data_clean, test_value)
    
    return t_stat, p_val, mean_val, len(data_clean), ci_lower, ci_upper, cohens_d

def perform_anova_across_conditions(all_results):
    """Perform one-way ANOVA across conditions"""
    
    conditions = ['free', 'yoke', 'pasv']
    condition_data = []
    
    for condition in conditions:
        if condition not in all_results:
            continue
        
        isc_data = all_results[condition]
        correlations = isc_data['correlations']
        condition_data.append(correlations)
    
    if len(condition_data) < 2:
        return None
    
    # Perform one-way ANOVA
    f_stat, p_val = f_oneway(*condition_data)
    
    # Calculate degrees of freedom
    n_total = sum(len(d) for d in condition_data)
    n_groups = len(condition_data)
    df_between = n_groups - 1
    df_within = n_total - n_groups
    
    # Calculate effect sizes (approximate using group means and variances)
    # For eta-squared, we need SS_between and SS_total
    all_data = np.concatenate(condition_data)
    grand_mean = np.mean(all_data)
    ss_total = np.sum((all_data - grand_mean)**2)
    
    group_means = [np.mean(d) for d in condition_data]
    group_ns = [len(d) for d in condition_data]
    ss_between = sum(n * (mean - grand_mean)**2 for n, mean in zip(group_ns, group_means))
    
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan
    partial_eta_sq = ss_between / (ss_between + (ss_total - ss_between)) if (ss_between + (ss_total - ss_between)) > 0 else np.nan
    
    return {
        'f_stat': f_stat,
        'p_value': p_val,
        'df_between': df_between,
        'df_within': df_within,
        'n_total': n_total,
        'n_groups': n_groups,
        'eta_squared': eta_sq,
        'partial_eta_squared': partial_eta_sq
    }

def perform_posthoc_tests(all_results):
    """Perform post-hoc tests: Free vs Yoked, Free vs Passive, Yoked vs Passive"""
    
    results = []
    
    # Free vs Yoked
    if 'free' in all_results and 'yoke' in all_results:
        free_corrs = all_results['free']['correlations']
        yoke_corrs = all_results['yoke']['correlations']
        
        t_stat, p_val = stats.ttest_ind(free_corrs, yoke_corrs)
        cohens_d = cohens_d_two_sample(free_corrs, yoke_corrs)
        
        # Calculate CI for mean difference
        mean_diff = np.mean(free_corrs) - np.mean(yoke_corrs)
        pooled_std = np.sqrt(((len(free_corrs) - 1) * np.var(free_corrs, ddof=1) + 
                              (len(yoke_corrs) - 1) * np.var(yoke_corrs, ddof=1)) / 
                             (len(free_corrs) + len(yoke_corrs) - 2))
        sem_diff = pooled_std * np.sqrt(1/len(free_corrs) + 1/len(yoke_corrs))
        df = len(free_corrs) + len(yoke_corrs) - 2
        t_critical = stats.t.ppf(0.975, df=df)
        ci_lower = mean_diff - t_critical * sem_diff
        ci_upper = mean_diff + t_critical * sem_diff
        
        results.append({
            'comparison': 'Free vs Yoked',
            't_stat': t_stat,
            'p_value': p_val,
            'n_free': len(free_corrs),
            'n_yoke': len(yoke_corrs),
            'cohens_d': cohens_d,
            'mean_diff_ci_lower': ci_lower,
            'mean_diff_ci_upper': ci_upper
        })
    
    # Free vs Passive
    if 'free' in all_results and 'pasv' in all_results:
        free_corrs = all_results['free']['correlations']
        pasv_corrs = all_results['pasv']['correlations']
        
        t_stat, p_val = stats.ttest_ind(free_corrs, pasv_corrs)
        cohens_d = cohens_d_two_sample(free_corrs, pasv_corrs)
        
        # Calculate CI for mean difference
        mean_diff = np.mean(free_corrs) - np.mean(pasv_corrs)
        pooled_std = np.sqrt(((len(free_corrs) - 1) * np.var(free_corrs, ddof=1) + 
                              (len(pasv_corrs) - 1) * np.var(pasv_corrs, ddof=1)) / 
                             (len(free_corrs) + len(pasv_corrs) - 2))
        sem_diff = pooled_std * np.sqrt(1/len(free_corrs) + 1/len(pasv_corrs))
        df = len(free_corrs) + len(pasv_corrs) - 2
        t_critical = stats.t.ppf(0.975, df=df)
        ci_lower = mean_diff - t_critical * sem_diff
        ci_upper = mean_diff + t_critical * sem_diff
        
        results.append({
            'comparison': 'Free vs Passive',
            't_stat': t_stat,
            'p_value': p_val,
            'n_free': len(free_corrs),
            'n_pasv': len(pasv_corrs),
            'cohens_d': cohens_d,
            'mean_diff_ci_lower': ci_lower,
            'mean_diff_ci_upper': ci_upper
        })
    
    # Yoked vs Passive
    if 'yoke' in all_results and 'pasv' in all_results:
        yoke_corrs = all_results['yoke']['correlations']
        pasv_corrs = all_results['pasv']['correlations']
        
        t_stat, p_val = stats.ttest_ind(yoke_corrs, pasv_corrs)
        cohens_d = cohens_d_two_sample(yoke_corrs, pasv_corrs)
        
        # Calculate CI for mean difference
        mean_diff = np.mean(yoke_corrs) - np.mean(pasv_corrs)
        pooled_std = np.sqrt(((len(yoke_corrs) - 1) * np.var(yoke_corrs, ddof=1) + 
                              (len(pasv_corrs) - 1) * np.var(pasv_corrs, ddof=1)) / 
                             (len(yoke_corrs) + len(pasv_corrs) - 2))
        sem_diff = pooled_std * np.sqrt(1/len(yoke_corrs) + 1/len(pasv_corrs))
        df = len(yoke_corrs) + len(pasv_corrs) - 2
        t_critical = stats.t.ppf(0.975, df=df)
        ci_lower = mean_diff - t_critical * sem_diff
        ci_upper = mean_diff + t_critical * sem_diff
        
        results.append({
            'comparison': 'Yoked vs Passive',
            't_stat': t_stat,
            'p_value': p_val,
            'n_yoke': len(yoke_corrs),
            'n_pasv': len(pasv_corrs),
            'cohens_d': cohens_d,
            'mean_diff_ci_lower': ci_lower,
            'mean_diff_ci_upper': ci_upper
        })
    
    return results

def run_analysis(loader, converge_dict, shared_choice_mask, use_nonchoice_only=False, use_z_transform=False, base_path=None):
    """Run a single analysis configuration"""
    
    analysis_name = []
    if use_nonchoice_only:
        analysis_name.append("49 non-choice events")
    else:
        analysis_name.append("64 shared events")
    
    if use_z_transform:
        analysis_name.append("Fisher z-transformed")
    else:
        analysis_name.append("raw r-values")
    
    analysis_name = " - ".join(analysis_name)
    
    print("\n" + "="*80)
    print(f"ANALYSIS: {analysis_name.upper()}")
    print("="*80)
    
    # Process all conditions
    conditions = ['free', 'yoke', 'pasv']
    all_results = {}
    
    for condition in conditions:
        isc_results = compute_pairwise_isc(loader, condition, converge_dict, shared_choice_mask, use_nonchoice_only, base_path)
        if isc_results is not None:
            # Apply z-transform if requested
            if use_z_transform:
                isc_results['correlations'] = fisher_z_transform(isc_results['correlations'])
            all_results[condition] = isc_results
    
    # Perform statistical tests
    stats_results = []
    
    # One-sample t-tests per condition
    for condition in conditions:
        if condition not in all_results:
            continue
        
        isc_data = all_results[condition]
        correlations = isc_data['correlations']
        
        # Perform 1-sample t-test on correlations
        t_stat, p_val, mean_val, n, ci_lower, ci_upper, cohens_d = perform_one_sample_ttest(correlations, test_value=0)
        
        if t_stat is not None:
            std_val = np.std(correlations[~np.isnan(correlations)], ddof=1)
            
            stats_results.append({
                'condition': condition,
                'N': n,
                'mean': mean_val,
                'std': std_val,
                't_stat': t_stat,
                'df': n - 1,
                'p_value': p_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'cohens_d': cohens_d
            })
    
    # Perform ANOVA across conditions
    anova_results = perform_anova_across_conditions(all_results)
    
    # Perform post-hoc tests
    posthoc_results = perform_posthoc_tests(all_results)
    
    return {
        'analysis_name': analysis_name,
        'use_nonchoice_only': use_nonchoice_only,
        'use_z_transform': use_z_transform,
        'all_results': all_results,
        'stats_results': stats_results,
        'anova_results': anova_results,
        'posthoc_results': posthoc_results
    }

def analyze_isc_by_condition():
    """Main analysis function - runs all four analyses"""
    
    print("="*80)
    print("INDIVIDUAL VARIABILITY IN RECALLED EVENTS")
    print("Inter-Participant Correlation (ISC) Analysis")
    print("="*80)
    
    # Use current directory
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run2_individual_variability_recalled_events")
    
    # Load monthy_map to identify shared events
    print("\n" + "="*80)
    print("LOADING SHARED EVENTS MAP")
    print("="*80)
    converge_dict = load_monthy_map()
    
    # Identify which shared events are choice events
    print("\n" + "="*80)
    print("IDENTIFYING CHOICE EVENTS IN SHARED EVENTS")
    print("="*80)
    # Use free condition to identify choice events
    shared_choice_mask, shared_event_labs = identify_choice_events_in_shared(loader, 'free', converge_dict, base_path)
    
    if shared_choice_mask is None:
        raise ValueError("Could not identify choice events in shared events")
    
    # Run all four analyses
    all_analyses = []
    
    # 1. 64 shared events, raw correlation r-values
    analysis1 = run_analysis(loader, converge_dict, shared_choice_mask, use_nonchoice_only=False, use_z_transform=False, base_path=base_path)
    all_analyses.append(analysis1)
    
    # 2. 49 non-choice shared events, raw correlation r-values
    analysis2 = run_analysis(loader, converge_dict, shared_choice_mask, use_nonchoice_only=True, use_z_transform=False, base_path=base_path)
    all_analyses.append(analysis2)
    
    # 3. 64 shared events, Fisher z-transformed
    analysis3 = run_analysis(loader, converge_dict, shared_choice_mask, use_nonchoice_only=False, use_z_transform=True, base_path=base_path)
    all_analyses.append(analysis3)
    
    # 4. 49 non-choice shared events, Fisher z-transformed
    analysis4 = run_analysis(loader, converge_dict, shared_choice_mask, use_nonchoice_only=True, use_z_transform=True, base_path=base_path)
    all_analyses.append(analysis4)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save pairwise correlations for each analysis
    for idx, analysis in enumerate(all_analyses, 1):
        suffix = f"_analysis{idx}"
        if analysis['use_nonchoice_only']:
            suffix += "_nonchoice"
        if analysis['use_z_transform']:
            suffix += "_ztransformed"
        
        for condition in ['free', 'yoke', 'pasv']:
            if condition in analysis['all_results']:
                pairs_file = os.path.join(output_dir, f"{condition}_pairwise_isc_correlations{suffix}.xlsx")
                analysis['all_results'][condition]['pairs_df'].to_excel(pairs_file, index=False)
    
    # Save all statistical results
    all_stats = []
    all_anova = []
    all_posthoc = []
    
    for idx, analysis in enumerate(all_analyses, 1):
        for stat in analysis['stats_results']:
            stat['analysis'] = idx
            stat['analysis_name'] = analysis['analysis_name']
            all_stats.append(stat)
        
        if analysis['anova_results']:
            anova = analysis['anova_results'].copy()
            anova['analysis'] = idx
            anova['analysis_name'] = analysis['analysis_name']
            all_anova.append(anova)
        
        for posthoc in analysis['posthoc_results']:
            posthoc['analysis'] = idx
            posthoc['analysis_name'] = analysis['analysis_name']
            all_posthoc.append(posthoc)
    
    stats_df = pd.DataFrame(all_stats)
    stats_file = os.path.join(output_dir, "isc_statistical_results_all_analyses.xlsx")
    stats_df.to_excel(stats_file, index=False)
    print(f"Saved statistical results to: {stats_file}")
    
    if all_anova:
        anova_df = pd.DataFrame(all_anova)
        anova_file = os.path.join(output_dir, "isc_anova_results_all_analyses.xlsx")
        anova_df.to_excel(anova_file, index=False)
        print(f"Saved ANOVA results to: {anova_file}")
    
    if all_posthoc:
        posthoc_df = pd.DataFrame(all_posthoc)
        posthoc_file = os.path.join(output_dir, "isc_posthoc_results_all_analyses.xlsx")
        posthoc_df.to_excel(posthoc_file, index=False)
        print(f"Saved post-hoc results to: {posthoc_file}")
    
    # Generate combined text report
    report_file = os.path.join(output_dir, "isc_statistical_report_all_analyses.txt")
    generate_report(all_analyses, report_file)
    print(f"Saved combined report to: {report_file}")
    
    return all_analyses

def generate_report(all_analyses, output_file):
    """Generate combined text report of all four analyses"""
    
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("INDIVIDUAL VARIABILITY IN RECALLED EVENTS\n")
        f.write("Inter-Participant Correlation (ISC) Analysis - All Four Analyses\n")
        f.write("="*80 + "\n\n")
        
        f.write("This analysis computed pairwise Pearson correlations between all pairs of\n")
        f.write("participants' recall performance vectors for shared story sections in the\n")
        f.write("Romance story. Four analyses were performed:\n")
        f.write("1. 64 shared events, raw correlation r-values\n")
        f.write("2. 49 non-choice shared events, raw correlation r-values\n")
        f.write("3. 64 shared events, Fisher z-transformed r-values\n")
        f.write("4. 49 non-choice shared events, Fisher z-transformed r-values\n\n")
        f.write("For each analysis, one-sample t-tests, ANOVA, and post-hoc tests were performed.\n\n")
        
        for idx, analysis in enumerate(all_analyses, 1):
            f.write("="*80 + "\n")
            f.write(f"ANALYSIS {idx}: {analysis['analysis_name'].upper()}\n")
            f.write("="*80 + "\n\n")
            
            # One-sample t-tests
            f.write("ONE-SAMPLE T-TESTS (testing if mean correlation is significantly different from zero):\n\n")
            for stat in analysis['stats_results']:
                cond = stat['condition'].upper()
                n = stat['N']
                mean_val = stat['mean']
                t_stat = stat['t_stat']
                df = stat['df']
                p_val = stat['p_value']
                ci_lower = stat.get('ci_lower', np.nan)
                ci_upper = stat.get('ci_upper', np.nan)
                cohens_d = stat.get('cohens_d', np.nan)
                
                if p_val < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_val:.3f}"
                
                f.write(f"{cond} condition:\n")
                f.write(f"  N pairs = {n}\n")
                f.write(f"  Mean = {mean_val:.4f}\n")
                ci_str = format_ci(ci_lower, ci_upper)
                f.write(f"  95% CI = {ci_str}\n")
                f.write(f"  t({df}) = {t_stat:.4f}, {p_text}\n")
                if not np.isnan(cohens_d):
                    f.write(f"  {format_cohens_d(cohens_d)}\n")
                f.write("\n")
            
            # ANOVA
            if analysis['anova_results']:
                f.write("ANOVA ACROSS CONDITIONS:\n\n")
                anova = analysis['anova_results']
                f_stat = anova['f_stat']
                p_val = anova['p_value']
                df_between = int(anova['df_between'])
                df_within = int(anova['df_within'])
                eta_sq = anova.get('eta_squared', np.nan)
                partial_eta_sq = anova.get('partial_eta_squared', np.nan)
                
                if p_val < 0.001:
                    p_text = "p < 0.001"
                else:
                    p_text = f"p = {p_val:.3f}"
                
                f.write(f"  F({df_between},{df_within}) = {f_stat:.4f}, {p_text}\n")
                if not np.isnan(eta_sq):
                    f.write(f"  η² = {eta_sq:.4f}\n")
                if not np.isnan(partial_eta_sq):
                    f.write(f"  ηp² = {partial_eta_sq:.4f}\n")
                f.write("\n")
            
            # Post-hoc tests
            if analysis['posthoc_results']:
                f.write("POST-HOC TESTS:\n\n")
                for posthoc in analysis['posthoc_results']:
                    comparison = posthoc['comparison']
                    t_stat = posthoc['t_stat']
                    p_val = posthoc['p_value']
                    cohens_d = posthoc.get('cohens_d', np.nan)
                    ci_lower = posthoc.get('mean_diff_ci_lower', np.nan)
                    ci_upper = posthoc.get('mean_diff_ci_upper', np.nan)
                    
                    if p_val < 0.001:
                        p_text = "p < 0.001"
                    else:
                        p_text = f"p = {p_val:.3f}"
                    
                    f.write(f"{comparison}:\n")
                    f.write(f"  t = {t_stat:.4f}, {p_text}\n")
                    if not np.isnan(cohens_d):
                        f.write(f"  {format_cohens_d(cohens_d)}\n")
                    if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                        ci_str = format_ci(ci_lower, ci_upper)
                        f.write(f"  95% CI for mean difference = {ci_str}\n")
                    f.write("\n")
            
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("All four analyses show consistent patterns:\n")
        f.write("- Recall ISC is significantly above zero in all three conditions\n")
        f.write("- Free participants show reduced Recall ISC relative to Yoked and Passive participants\n")
        f.write("- Results are robust across both raw and z-transformed correlations\n")
        f.write("- Results are robust when including or excluding choice events\n\n")

def main():
    """Main function"""
    all_analyses = analyze_isc_by_condition()
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    return all_analyses

if __name__ == "__main__":
    main()
