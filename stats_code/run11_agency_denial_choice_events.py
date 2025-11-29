#!/usr/bin/env python3
"""
Agency Denial Effect on Choice Events Memory Analysis
Analyzes how having one's agency denied affects memory for choice events.

For each yoked subject:
1. Load recall vector for 29 choice events
2. Load want/not-want vector indicating whether each choice event was wanted
3. Split recall by want vs not-want events
4. Compute mean recall for want events and not-want events

For each corresponding free subject:
1. Load recall vector for the same 29 choice events
2. Use the yoked subject's want/not-want vector to split the free subject's recall
3. Compute mean recall for want events and not-want events

Compare: Yoked not-want (denied) events vs Free not-want (granted) events
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import ttest_ind, pearsonr
from data_structure import RecallDataLoader

def identify_story_paths(loader, story):
    """Identify story paths by matching free subjects to yoked subjects"""
    
    # Get all subject IDs for each condition
    free_ids = loader.get_subject_ids_from_events(story, 'free')
    yoke_ids = loader.get_subject_ids_from_events(story, 'yoke')
    
    # Filter out temporary files
    free_ids = [sid for sid in free_ids if not sid.startswith('~$')]
    yoke_ids = [sid for sid in yoke_ids if not sid.startswith('~$')]
    
    # Extract free subject numbers (e.g., 'sub3_4003' -> 3)
    free_subject_map = {}
    for free_id in free_ids:
        try:
            num = int(free_id.split('_')[0].replace('sub', ''))
            free_subject_map[num] = free_id
        except:
            continue
    
    # Group yoked subjects by their 'toX_' prefix
    story_paths = {}  # {story_path_num: {'free': free_id, 'yoke': [yoke_ids]}}
    
    for yoke_id in yoke_ids:
        try:
            to_num = int(yoke_id.split('_')[0].replace('to', ''))
            if to_num not in story_paths:
                story_paths[to_num] = {'free': None, 'yoke': []}
            story_paths[to_num]['yoke'].append(yoke_id)
        except:
            continue
    
    # Match free subjects to story paths
    for to_num, free_id in free_subject_map.items():
        if to_num in story_paths:
            story_paths[to_num]['free'] = free_id
    
    return story_paths

def get_choice_events_recall_and_want(loader, story, condition, subject_id):
    """Get recall vector and want/not-want vector for choice events only"""
    
    # Get choice event information and want_not values
    is_choice_event, want_not_values = loader.get_choice_and_want_not_info(story, condition, subject_id)
    
    # Get full recall vector
    recall_vector = loader.get_subject_recall_vector(story, condition, subject_id)
    
    # Convert to numpy arrays
    recall_array = np.array(recall_vector)
    is_choice_array = np.array(is_choice_event, dtype=bool)
    want_not_array = np.array(want_not_values)
    
    # Ensure arrays are same length
    min_len = min(len(recall_array), len(is_choice_array), len(want_not_array))
    recall_array = recall_array[:min_len]
    is_choice_array = is_choice_array[:min_len]
    want_not_array = want_not_array[:min_len]
    
    # Extract only choice events
    choice_mask = is_choice_array
    choice_recall = recall_array[choice_mask]
    choice_want_not = want_not_array[choice_mask]
    
    return choice_recall, choice_want_not

def compute_want_not_recall_means(recall_vector, want_not_vector):
    """Compute mean recall for want (1) and not-want (0) events"""
    
    # Convert to numpy arrays
    recall_array = np.array(recall_vector)
    want_not_array = np.array(want_not_vector)
    
    # Filter to events with valid want_not values
    valid_mask = pd.notna(want_not_array)
    valid_recall = recall_array[valid_mask]
    valid_want_not = want_not_array[valid_mask]
    
    # Split by want (1) vs not-want (0)
    want_mask = valid_want_not == 1
    not_want_mask = valid_want_not == 0
    
    want_recall = valid_recall[want_mask]
    not_want_recall = valid_recall[not_want_mask]
    
    # Compute means
    want_mean = np.mean(want_recall) if len(want_recall) > 0 else np.nan
    not_want_mean = np.mean(not_want_recall) if len(not_want_recall) > 0 else np.nan
    
    return want_mean, not_want_mean, want_recall, not_want_recall

def analyze_agency_denial():
    """Main analysis function"""
    
    print("="*80)
    print("AGENCY DENIAL EFFECT ON CHOICE EVENTS MEMORY")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("run11_agency_denial_choice_events")
    
    stories = ['BA', 'MV']
    story_names = {'BA': 'Adventure', 'MV': 'Romance'}
    
    all_results = {}
    
    for story in stories:
        print("\n" + "="*80)
        print(f"{story_names[story]} STORY ({story})")
        print("="*80)
        
        # Identify story paths (matching yoked to free subjects)
        story_paths = identify_story_paths(loader, story)
        print(f"Identified {len(story_paths)} story paths")
        
        # Collect data for all matched pairs
        yoke_want_recall = []
        yoke_not_want_recall = []
        free_want_recall = []
        free_not_want_recall = []
        
        matched_pairs = []
        
        for to_num, path_info in story_paths.items():
            free_id = path_info['free']
            yoke_ids = path_info['yoke']
            
            if free_id is None:
                continue
            
            # Process each yoked subject matched to this free subject
            for yoke_id in yoke_ids:
                try:
                    # Get choice events recall and want/not for yoked subject
                    yoke_choice_recall, yoke_choice_want_not = get_choice_events_recall_and_want(
                        loader, story, 'yoke', yoke_id)
                    
                    if len(yoke_choice_recall) == 0 or np.sum(pd.notna(yoke_choice_want_not)) == 0:
                        print(f"  Skipping {yoke_id}: insufficient choice/want data")
                        continue
                    
                    # Compute means for yoked subject
                    yoke_want_mean, yoke_not_want_mean, yoke_want_vec, yoke_not_want_vec = \
                        compute_want_not_recall_means(yoke_choice_recall, yoke_choice_want_not)
                    
                    if np.isnan(yoke_want_mean) or np.isnan(yoke_not_want_mean):
                        print(f"  Skipping {yoke_id}: missing want/not-want data")
                        continue
                    
                    # Get choice events recall for free subject
                    free_choice_recall, _ = get_choice_events_recall_and_want(
                        loader, story, 'free', free_id)
                    
                    if len(free_choice_recall) != len(yoke_choice_want_not):
                        print(f"  Warning: {yoke_id}/{free_id}: length mismatch ({len(free_choice_recall)} vs {len(yoke_choice_want_not)})")
                        # Try to align by taking first N elements
                        min_len = min(len(free_choice_recall), len(yoke_choice_want_not))
                        free_choice_recall = free_choice_recall[:min_len]
                        yoke_choice_want_not = yoke_choice_want_not[:min_len]
                    
                    # Use yoked subject's want/not vector to split free subject's recall
                    free_want_mean, free_not_want_mean, free_want_vec, free_not_want_vec = \
                        compute_want_not_recall_means(free_choice_recall, yoke_choice_want_not)
                    
                    if np.isnan(free_want_mean) or np.isnan(free_not_want_mean):
                        print(f"  Skipping {yoke_id}/{free_id}: missing free want/not-want data")
                        continue
                    
                    # Store results
                    yoke_want_recall.append(yoke_want_mean)
                    yoke_not_want_recall.append(yoke_not_want_mean)
                    free_want_recall.append(free_want_mean)
                    free_not_want_recall.append(free_not_want_mean)
                    
                    matched_pairs.append({
                        'yoke_id': yoke_id,
                        'free_id': free_id,
                        'yoke_want_mean': yoke_want_mean,
                        'yoke_not_want_mean': yoke_not_want_mean,
                        'free_want_mean': free_want_mean,
                        'free_not_want_mean': free_not_want_mean,
                        'yoke_want_vec': yoke_want_vec,
                        'yoke_not_want_vec': yoke_not_want_vec,
                        'free_want_vec': free_want_vec,
                        'free_not_want_vec': free_not_want_vec
                    })
                    
                except Exception as e:
                    print(f"  Error processing {yoke_id}/{free_id}: {e}")
                    continue
        
        print(f"\nSuccessfully processed {len(matched_pairs)} matched pairs")
        
        if len(matched_pairs) == 0:
            print("  No valid matched pairs found")
            all_results[story] = None
            continue
        
        # Convert to arrays
        yoke_want_recall = np.array(yoke_want_recall)
        yoke_not_want_recall = np.array(yoke_not_want_recall)
        free_want_recall = np.array(free_want_recall)
        free_not_want_recall = np.array(free_not_want_recall)
        
        # Statistical test: Yoked not-want (denied) vs Free not-want (granted)
        print("\n" + "="*80)
        print("STATISTICAL TEST: Denied Choice Events (Yoked not-want) vs Granted Choice Events (Free not-want)")
        print("="*80)
        
        t_stat, p_val = ttest_ind(yoke_not_want_recall, free_not_want_recall)
        n = len(yoke_not_want_recall)
        df = 2 * n - 2
        
        print(f"\nTwo-sample t-test:")
        print(f"  Yoked not-want (denied) events: N = {n}, Mean = {np.mean(yoke_not_want_recall):.4f}, Std = {np.std(yoke_not_want_recall, ddof=1):.4f}")
        print(f"  Free not-want (granted) events: N = {n}, Mean = {np.mean(free_not_want_recall):.4f}, Std = {np.std(free_not_want_recall, ddof=1):.4f}")
        print(f"  t({df}) = {t_stat:.4f}, p = {p_val:.6f}")
        
        if p_val < 0.001:
            print(f"  Result: Significant difference (p < 0.001)")
        elif p_val < 0.01:
            print(f"  Result: Significant difference (p < 0.01)")
        elif p_val < 0.05:
            print(f"  Result: Significant difference (p < 0.05)")
        else:
            print(f"  Result: Not significant (p >= 0.05)")
        
        # Also report want events comparison for completeness
        print(f"\nFor comparison - Want events:")
        print(f"  Yoked want events: N = {n}, Mean = {np.mean(yoke_want_recall):.4f}, Std = {np.std(yoke_want_recall, ddof=1):.4f}")
        print(f"  Free want events: N = {n}, Mean = {np.mean(free_want_recall):.4f}, Std = {np.std(free_want_recall, ddof=1):.4f}")
        
        # NEW ANALYSIS 1: PE-boost - Correlation between want-not vector and recall vector per subject
        print("\n" + "="*80)
        print("PE-BOOST: Per-Subject Correlation between Want-Not Vector and Recall Vector")
        print("="*80)
        
        # For each yoked subject, compute correlation between wn vector and recall vector for choice events
        pe_boost_data = []  # List of dicts with subject_id, pe_boost, percentage_wanted
        
        for pair in matched_pairs:
            yoke_id = pair['yoke_id']
            try:
                # Get choice events recall and want/not for yoked subject
                yoke_choice_recall, yoke_choice_want_not = get_choice_events_recall_and_want(
                    loader, story, 'yoke', yoke_id)
                
                if len(yoke_choice_recall) == 0 or len(yoke_choice_want_not) == 0:
                    continue
                
                # Ensure same length
                min_len = min(len(yoke_choice_recall), len(yoke_choice_want_not))
                yoke_choice_recall = np.array(yoke_choice_recall[:min_len])
                yoke_choice_want_not = np.array(yoke_choice_want_not[:min_len])
                
                # Filter to valid data (non-NaN want_not values)
                valid_mask = pd.notna(yoke_choice_want_not)
                if np.sum(valid_mask) < 3:  # Need at least 3 data points for correlation
                    continue
                
                valid_recall = yoke_choice_recall[valid_mask]
                valid_want_not = yoke_choice_want_not[valid_mask]
                
                # Compute correlation between want-not vector and recall vector
                # This is the PE-boost for this subject
                r_pe_boost, p_pe_boost = pearsonr(valid_want_not, valid_recall)
                
                # Compute percentage wanted (average of wn vector)
                percentage_wanted = np.mean(valid_want_not == 1)  # Proportion where want_not = 1
                
                if not np.isnan(r_pe_boost) and not np.isnan(percentage_wanted):
                    pe_boost_data.append({
                        'yoke_id': yoke_id,
                        'pe_boost': r_pe_boost,
                        'percentage_wanted': percentage_wanted,
                        'n_choice_events': len(valid_want_not)
                    })
                    
            except Exception as e:
                print(f"  Error processing {yoke_id} for PE-boost: {e}")
                continue
        
        print(f"\nComputed PE-boost for {len(pe_boost_data)} yoked subjects")
        if len(pe_boost_data) > 0:
            pe_boosts = [d['pe_boost'] for d in pe_boost_data]
            print(f"  PE-boost: Mean = {np.mean(pe_boosts):.4f}, Std = {np.std(pe_boosts, ddof=1):.4f}, Range = [{np.min(pe_boosts):.4f}, {np.max(pe_boosts):.4f}]")
        
        # NEW ANALYSIS 2: Correlation between PE-boost and percentage wanted
        print("\n" + "="*80)
        print("CORRELATION: PE-Boost vs Percentage of Choices Wanted")
        print("="*80)
        
        if len(pe_boost_data) >= 3:
            pe_boosts = np.array([d['pe_boost'] for d in pe_boost_data])
            percentage_wanted = np.array([d['percentage_wanted'] for d in pe_boost_data])
            
            # Compute Pearson correlation
            r, p = pearsonr(pe_boosts, percentage_wanted)
            n_corr = len(pe_boost_data)
            df_corr = n_corr - 2
            
            print(f"\nCorrelation analysis:")
            print(f"  N = {n_corr}")
            print(f"  PE-boost: Mean = {np.mean(pe_boosts):.4f}, Std = {np.std(pe_boosts, ddof=1):.4f}")
            print(f"  Percentage wanted: Mean = {np.mean(percentage_wanted):.4f}, Std = {np.std(percentage_wanted, ddof=1):.4f}")
            print(f"  r({df_corr}) = {r:.3f}, p = {p:.3f}")
            
            if p < 0.001:
                print(f"  Result: Significant correlation (p < 0.001)")
            elif p < 0.01:
                print(f"  Result: Significant correlation (p < 0.01)")
            elif p < 0.05:
                print(f"  Result: Significant correlation (p < 0.05)")
            else:
                print(f"  Result: Not significant (p >= 0.05)")
            
            pe_boost_correlation = {
                'r': r,
                'p': p,
                'n': n_corr,
                'df': df_corr
            }
        else:
            print(f"\nInsufficient data for correlation analysis (N = {len(pe_boost_data)})")
            pe_boost_correlation = None
        
        all_results[story] = {
            'matched_pairs': matched_pairs,
            'yoke_want_recall': yoke_want_recall,
            'yoke_not_want_recall': yoke_not_want_recall,
            'free_want_recall': free_want_recall,
            'free_not_want_recall': free_not_want_recall,
            't_stat': t_stat,
            'p_val': p_val,
            'n': n,
            'pe_boost_data': pe_boost_data,
            'pe_boost_correlation': pe_boost_correlation
        }
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    for story in stories:
        if all_results[story] is None:
            continue
        
        story_name = story_names[story]
        results = all_results[story]
        
        # Save the 4 final vectors
        vectors_df = pd.DataFrame({
            'yoke_want_recall': results['yoke_want_recall'],
            'yoke_not_want_recall': results['yoke_not_want_recall'],
            'free_want_recall': results['free_want_recall'],
            'free_not_want_recall': results['free_not_want_recall']
        })
        
        vectors_file = os.path.join(output_dir, f"{story_name.lower()}_agency_denial_vectors.xlsx")
        vectors_df.to_excel(vectors_file, index=False)
        print(f"Saved {story_name} vectors to: {vectors_file}")
        
        # Save detailed results
        detailed_data = []
        for pair in results['matched_pairs']:
            detailed_data.append({
                'yoke_id': pair['yoke_id'],
                'free_id': pair['free_id'],
                'yoke_want_mean': pair['yoke_want_mean'],
                'yoke_not_want_mean': pair['yoke_not_want_mean'],
                'free_want_mean': pair['free_want_mean'],
                'free_not_want_mean': pair['free_not_want_mean']
            })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = os.path.join(output_dir, f"{story_name.lower()}_agency_denial_detailed.xlsx")
        detailed_df.to_excel(detailed_file, index=False)
        print(f"Saved {story_name} detailed results to: {detailed_file}")
        
        # Save PE-boost data
        if results.get('pe_boost_data') is not None and len(results['pe_boost_data']) > 0:
            pe_boost_df = pd.DataFrame(results['pe_boost_data'])
            pe_boost_file = os.path.join(output_dir, f"{story_name.lower()}_pe_boost_per_subject.xlsx")
            pe_boost_df.to_excel(pe_boost_file, index=False)
            print(f"Saved {story_name} PE-boost data to: {pe_boost_file}")
        
        # Save PE-boost correlation results
        if results.get('pe_boost_correlation') is not None:
            pe_boost_corr_df = pd.DataFrame([results['pe_boost_correlation']])
            pe_boost_corr_file = os.path.join(output_dir, f"{story_name.lower()}_correlation_pe_boost_percentage_wanted.xlsx")
            pe_boost_corr_df.to_excel(pe_boost_corr_file, index=False)
            print(f"Saved {story_name} PE-boost correlation results to: {pe_boost_corr_file}")
    
    # Create summary report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("AGENCY DENIAL EFFECT ON CHOICE EVENTS MEMORY")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("Analysis: Having one's agency denied can have unique memory effects")
    report_lines.append("at local choice events: one's memory for the denied choice events")
    report_lines.append("is selectively reduced compared to its choice-granted counterparts")
    report_lines.append("in the Free condition.")
    report_lines.append("")
    
    for story in stories:
        if all_results[story] is None:
            continue
        
        story_name = story_names[story]
        results = all_results[story]
        
        report_lines.append("="*80)
        report_lines.append(f"{story_name.upper()} STORY ({story})")
        report_lines.append("="*80)
        report_lines.append("")
        report_lines.append(f"Matched pairs analyzed: {results['n']}")
        report_lines.append("")
        report_lines.append("Two-sample t-test: Denied Choice Events (Yoked not-want) vs Granted Choice Events (Free not-want)")
        report_lines.append(f"  Yoked not-want (denied) events: Mean = {np.mean(results['yoke_not_want_recall']):.4f}, Std = {np.std(results['yoke_not_want_recall'], ddof=1):.4f}")
        report_lines.append(f"  Free not-want (granted) events: Mean = {np.mean(results['free_not_want_recall']):.4f}, Std = {np.std(results['free_not_want_recall'], ddof=1):.4f}")
        report_lines.append(f"  t({2*results['n']-2}) = {results['t_stat']:.4f}, p = {results['p_val']:.6f}")
        if results['p_val'] < 0.001:
            report_lines.append(f"  Result: Significant difference (p < 0.001)")
        elif results['p_val'] < 0.01:
            report_lines.append(f"  Result: Significant difference (p < 0.01)")
        elif results['p_val'] < 0.05:
            report_lines.append(f"  Result: Significant difference (p < 0.05)")
        else:
            report_lines.append(f"  Result: Not significant (p >= 0.05)")
        report_lines.append("")
        report_lines.append("For comparison - Want events:")
        report_lines.append(f"  Yoked want events: Mean = {np.mean(results['yoke_want_recall']):.4f}, Std = {np.std(results['yoke_want_recall'], ddof=1):.4f}")
        report_lines.append(f"  Free want events: Mean = {np.mean(results['free_want_recall']):.4f}, Std = {np.std(results['free_want_recall'], ddof=1):.4f}")
        report_lines.append("")
        
        # Add PE-boost correlation results
        if results.get('pe_boost_correlation') is not None:
            pe_corr = results['pe_boost_correlation']
            report_lines.append("PE-Boost Correlation: PE-Boost vs Percentage of Choices Wanted")
            report_lines.append(f"  PE-boost = correlation between want-not vector and recall vector for choice events (per subject)")
            report_lines.append(f"  r({pe_corr['df']}) = {pe_corr['r']:.3f}, p = {pe_corr['p']:.3f}")
            if pe_corr['p'] < 0.001:
                report_lines.append(f"  Result: Significant correlation (p < 0.001)")
            elif pe_corr['p'] < 0.01:
                report_lines.append(f"  Result: Significant correlation (p < 0.01)")
            elif pe_corr['p'] < 0.05:
                report_lines.append(f"  Result: Significant correlation (p < 0.05)")
            else:
                report_lines.append(f"  Result: Not significant (p >= 0.05)")
            report_lines.append("")
        
        # Add correlation results
        if results.get('correlation') is not None:
            corr = results['correlation']
            report_lines.append("Correlation: Percentage of Choices Wanted vs Recall/Forget Tendency for Choice-Denied Events")
            report_lines.append(f"  r({corr['df']}) = {corr['r']:.3f}, p = {corr['p']:.3f}")
            if corr['p'] < 0.001:
                report_lines.append(f"  Result: Significant correlation (p < 0.001)")
            elif corr['p'] < 0.01:
                report_lines.append(f"  Result: Significant correlation (p < 0.01)")
            elif corr['p'] < 0.05:
                report_lines.append(f"  Result: Significant correlation (p < 0.05)")
            else:
                report_lines.append(f"  Result: Not significant (p >= 0.05)")
            report_lines.append("")
    
    report_file = os.path.join(output_dir, "agency_denial_choice_events_report.txt")
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report to: {report_file}")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    analyze_agency_denial()

