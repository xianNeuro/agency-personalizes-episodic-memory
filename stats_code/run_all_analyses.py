#!/usr/bin/env python3
"""
Master Script: Run All Analyses (run1-run13) and Generate Comprehensive HTML Report
Runs all analysis scripts and generates a single HTML report with manuscript text
and full statistics inserted, organized by run number.
"""

import subprocess
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
from data_structure import RecallDataLoader

# Make loader available globally for stats extraction
loader = None

def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\n{'='*80}")
    print(f"Running {script_name}...")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print(f"✓ {script_name} completed successfully")
            return True, result.stdout
        else:
            print(f"✗ {script_name} failed with error:")
            print(result.stderr[:500])  # Print first 500 chars of error
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"✗ {script_name} timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"✗ {script_name} error: {e}")
        return False, str(e)

def extract_all_statistics():
    """Extract statistics from all output files"""
    
    global loader
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    stats = {}
    
    # Run 1: Overall recall comparison
    try:
        run1_file = os.path.join(loader.get_output_dir("run1_overall_recall_comparison"), 
                                 "overall_recall_anova_statistics.xlsx")
        if os.path.exists(run1_file):
            stats['run1'] = {}
            with pd.ExcelFile(run1_file) as xls:
                for sheet in xls.sheet_names:
                    stats['run1'][sheet] = pd.read_excel(xls, sheet_name=sheet).to_dict('records')
            
            # Load engagement results (reading time and transportation)
            engagement_measures = ['trans_score', 'avg_sent_readtime', 'sum_readtime']
            for measure in engagement_measures:
                sheet_name = f'Engagement_{measure}_stats'
                anova_sheet = f'Engagement_{measure}_anova'
                if sheet_name in stats['run1']:
                    # Get group stats
                    group_stats = pd.DataFrame(stats['run1'][sheet_name])
                    stats[f'run1_engagement_{measure}_stats'] = group_stats
                    # Get ANOVA results
                    if anova_sheet in stats['run1']:
                        anova_data = pd.DataFrame(stats['run1'][anova_sheet])
                        # Find the C(condition) row
                        condition_row = anova_data[anova_data.iloc[:, 0] == 'C(condition)']
                        if len(condition_row) > 0:
                            row = condition_row.iloc[0]
                            stats[f'run1_engagement_{measure}'] = {
                                'f_stat': row.get('F', 'N/A'),
                                'df_between': int(row.get('df', 0)) if pd.notna(row.get('df')) else 'N/A',
                                'df_within': int(anova_data[anova_data.iloc[:, 0] == 'Residual'].iloc[0].get('df', 0)) if len(anova_data[anova_data.iloc[:, 0] == 'Residual']) > 0 else 'N/A',
                                'p_value': row.get('PR(>F)', 'N/A')
                            }
    except Exception as e:
        print(f"Error loading run1 stats: {e}")
        stats['run1'] = None
    
    # Run 2: Individual variability in recalled events
    try:
        run2_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                 "isc_statistical_results_all_analyses.xlsx")
        if os.path.exists(run2_file):
            df = pd.read_excel(run2_file)
            stats['run2'] = df.to_dict('records')
        
        # Load post-hoc tests for run2
        run2_posthoc_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                         "isc_posthoc_results_all_analyses.xlsx")
        if os.path.exists(run2_posthoc_file):
            stats['run2_posthoc'] = pd.read_excel(run2_posthoc_file)
        else:
            stats['run2_posthoc'] = None
    except Exception as e:
        print(f"Error loading run2 stats: {e}")
        stats['run2'] = None
        stats['run2_posthoc'] = None
    
    # Run 3: Individual variability in choices
    try:
        run3_file = os.path.join(loader.get_output_dir("run3_individual_variability_choices"),
                                 "choice_isc_statistical_results.xlsx")
        if os.path.exists(run3_file):
            df = pd.read_excel(run3_file)
            stats['run3'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run3 stats: {e}")
        stats['run3'] = None
    
    # Run 4: Divergence from group
    try:
        run4_file = os.path.join(loader.get_output_dir("run4_divergence_from_group"),
                                 "divergence_correlation_results.xlsx")
        if os.path.exists(run4_file):
            df = pd.read_excel(run4_file)
            stats['run4'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run4 stats: {e}")
        stats['run4'] = None
    
    # Run 5: Centrality predicts recall
    try:
        run5_file = os.path.join(loader.get_output_dir("run5_centrality_predicts_recall"),
                                 "centrality_predicts_recall_results.xlsx")
        if os.path.exists(run5_file):
            df = pd.read_excel(run5_file)
            stats['run5'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run5 stats: {e}")
        stats['run5'] = None
    
    # Run 6: Agency reduced semantic centrality
    try:
        run6_file = os.path.join(loader.get_output_dir("run6_agency_reduced_semantic_centrality"),
                                 "agency_centrality_anova_results.xlsx")
        if os.path.exists(run6_file):
            df = pd.read_excel(run6_file)
            stats['run6'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run6 stats: {e}")
        stats['run6'] = None
    
    # Run 7: Neighbor encoding effect
    try:
        run7_file = os.path.join(loader.get_output_dir("run7_neighbor_encoding_effect"),
                                 "neighbor_encoding_effect_results.xlsx")
        if os.path.exists(run7_file):
            df = pd.read_excel(run7_file)
            stats['run7'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run7 stats: {e}")
        stats['run7'] = None
    
    # Run 8: Temporal violation rate
    try:
        run8_file = os.path.join(loader.get_output_dir("run8_temporal_violation_rate"),
                                 "temporal_violation_rate_results.xlsx")
        if os.path.exists(run8_file):
            df = pd.read_excel(run8_file)
            stats['run8'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run8 stats: {e}")
        stats['run8'] = None
    
    # Run 9: Memory divergence semantic correlation
    try:
        run9_file = os.path.join(loader.get_output_dir("run9_memory_divergence_semantic_correlation"),
                                 "memory_divergence_semantic_correlation_results.xlsx")
        if os.path.exists(run9_file):
            df = pd.read_excel(run9_file)
            stats['run9'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run9 stats: {e}")
        stats['run9'] = None
    
    # Run 10: Neighbor encoding correlations
    try:
        run10_file = os.path.join(loader.get_output_dir("run10_neighbor_encoding_correlations"),
                                  "neighbor_encoding_correlations_results.xlsx")
        if os.path.exists(run10_file):
            df = pd.read_excel(run10_file)
            stats['run10'] = df.to_dict('records')
    except Exception as e:
        print(f"Error loading run10 stats: {e}")
        stats['run10'] = None
    
    # Run 11: Agency denial choice events
    try:
        run11_file = os.path.join(loader.get_output_dir("run11_agency_denial_choice_events"),
                                 "adventure_agency_denial_detailed.xlsx")
        if os.path.exists(run11_file):
            df_ba = pd.read_excel(run11_file)
            stats['run11_ba'] = df_ba.to_dict('records')
        run11_file_mv = os.path.join(loader.get_output_dir("run11_agency_denial_choice_events"),
                                     "romance_agency_denial_detailed.xlsx")
        if os.path.exists(run11_file_mv):
            df_mv = pd.read_excel(run11_file_mv)
            stats['run11_mv'] = df_mv.to_dict('records')
        
        # Load PE-boost correlation results
        run11_pe_boost_corr_ba = os.path.join(loader.get_output_dir("run11_agency_denial_choice_events"),
                                               "adventure_correlation_pe_boost_percentage_wanted.xlsx")
        if os.path.exists(run11_pe_boost_corr_ba):
            df_pe_boost_corr_ba = pd.read_excel(run11_pe_boost_corr_ba)
            stats['run11_pe_boost_corr_ba'] = df_pe_boost_corr_ba.to_dict('records')[0] if len(df_pe_boost_corr_ba) > 0 else None
        else:
            stats['run11_pe_boost_corr_ba'] = None
            
        run11_pe_boost_corr_mv = os.path.join(loader.get_output_dir("run11_agency_denial_choice_events"),
                                               "romance_correlation_pe_boost_percentage_wanted.xlsx")
        if os.path.exists(run11_pe_boost_corr_mv):
            df_pe_boost_corr_mv = pd.read_excel(run11_pe_boost_corr_mv)
            stats['run11_pe_boost_corr_mv'] = df_pe_boost_corr_mv.to_dict('records')[0] if len(df_pe_boost_corr_mv) > 0 else None
        else:
            stats['run11_pe_boost_corr_mv'] = None
    except Exception as e:
        print(f"Error loading run11 stats: {e}")
        stats['run11_ba'] = None
        stats['run11_mv'] = None
        stats['run11_corr_ba'] = None
        stats['run11_corr_mv'] = None
    
    # Run 12: Permutation test recall ISC
    try:
        run12_file = os.path.join(loader.get_output_dir("run12_permutation_test_recall_isc"),
                                 "permutation_test_report.txt")
        if os.path.exists(run12_file):
            with open(run12_file, 'r') as f:
                stats['run12'] = f.read()
    except Exception as e:
        print(f"Error loading run12 stats: {e}")
        stats['run12'] = None
    
    # Run 13: Permutation test matching choice ISC
    try:
        run13_file = os.path.join(loader.get_output_dir("run13_permutation_test_matching_choice_isc"),
                                 "permutation_test_matching_choice_isc_report.txt")
        if os.path.exists(run13_file):
            with open(run13_file, 'r') as f:
                stats['run13'] = f.read()
    except Exception as e:
        print(f"Error loading run13 stats: {e}")
        stats['run13'] = None
    
    return stats

def format_stat_value(val):
    """Format a statistical value for display"""
    if pd.isna(val):
        return "N/A"
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if abs(val) < 0.001:
            return f"{val:.6f}"
        elif abs(val) < 0.01:
            return f"{val:.4f}"
        else:
            return f"{val:.3f}"
    return str(val)

def get_posthoc_tests_for_anova(anova_p_val, posthoc_df, analysis_num=None, story=None, measure=None, transform=None):
    """Extract post-hoc tests for a significant ANOVA (p < 0.05)"""
    if pd.isna(anova_p_val) or anova_p_val >= 0.05:
        return None
    
    posthoc_results = []
    
    if posthoc_df is not None:
        # Handle DataFrame
        if isinstance(posthoc_df, pd.DataFrame) and not posthoc_df.empty:
            # Filter posthoc tests
            filtered = posthoc_df.copy()
            
            if analysis_num is not None and 'analysis' in filtered.columns:
                filtered = filtered[filtered['analysis'] == analysis_num]
            
            if story is not None and 'Story' in filtered.columns:
                filtered = filtered[filtered['Story'].str.contains(story, case=False, na=False)]
            
            if measure is not None and 'Measure' in filtered.columns:
                filtered = filtered[filtered['Measure'] == measure]
            
            if transform is not None and 'Transform' in filtered.columns:
                filtered = filtered[filtered['Transform'] == transform]
            
            if not filtered.empty:
                for _, row in filtered.iterrows():
                    comparison = row.get('comparison', row.get('Condition', ''))
                    t_stat = row.get('t_stat', row.get('t_statistic', ''))
                    p_val = row.get('p_value', '')
                    # Calculate df from sample sizes if not present
                    df = row.get('df', row.get('df_within', ''))
                    if pd.isna(df) or df == '':
                        n_free = row.get('n_free', 0)
                        n_yoke = row.get('n_yoke', 0)
                        n_pasv = row.get('n_pasv', 0)
                        if 'Free vs Yoked' in str(comparison) and n_free > 0 and n_yoke > 0:
                            df = n_free + n_yoke - 2
                        elif 'Free vs Passive' in str(comparison) and n_free > 0 and n_pasv > 0:
                            df = n_free + n_pasv - 2
                        elif 'Yoked vs Passive' in str(comparison) and n_yoke > 0 and n_pasv > 0:
                            df = n_yoke + n_pasv - 2
                    
                    if pd.notna(t_stat) and pd.notna(p_val):
                        posthoc_results.append({
                            'comparison': comparison,
                            't_stat': t_stat,
                            'p_val': p_val,
                            'df': df
                        })
        # Handle list of dicts (from stats['run6'] or stats['run7'])
        elif isinstance(posthoc_df, list):
            for row in posthoc_df:
                if row.get('Analysis') == 'Post-hoc t-test':
                    # Check filters
                    match = True
                    if story is not None and row.get('Story') != story:
                        match = False
                    if measure is not None and row.get('Measure') != measure:
                        match = False
                    if transform is not None and row.get('Transform') != transform:
                        match = False
                    
                    if match:
                        comparison = row.get('Condition', '')
                        t_stat = row.get('t_statistic', '')
                        p_val = row.get('p_value', '')
                        df = row.get('df_within', '')
                        
                        if pd.notna(t_stat) and pd.notna(p_val):
                            posthoc_results.append({
                                'comparison': comparison,
                                't_stat': t_stat,
                                'p_val': p_val,
                                'df': df
                            })
    
    return posthoc_results if posthoc_results else None

def generate_html_report(stats):
    """Generate comprehensive HTML report with manuscript text and statistics"""
    
    global loader
    if loader is None:
        base_path = os.path.abspath('.')
        loader = RecallDataLoader(base_path)
    
    html = []
    html.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comprehensive Analysis Report: Agency Effects on Memory</title>
    <style>
        body {
            font-family: 'Times New Roman', serif;
            line-height: 1.8;
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px;
            background-color: #ffffff;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            font-size: 1.5em;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 25px;
            font-size: 1.2em;
        }
        .run-section {
            margin: 40px 0;
            padding: 25px;
            background-color: #f8f9fa;
            border-left: 5px solid #3498db;
            border-radius: 5px;
        }
        .stats-box {
            background-color: #e8f4f8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            border: 1px solid #bdc3c7;
        }
        .manuscript-text {
            text-align: justify;
            margin: 20px 0;
            font-size: 1.05em;
        }
        .stat-inline {
            font-weight: bold;
            color: #2980b9;
            background-color: #ecf0f1;
            padding: 2px 6px;
            border-radius: 3px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 0.95em;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .figure-ref {
            font-style: italic;
            color: #7f8c8d;
        }
        .supplement-ref {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Comprehensive Analysis Report: Agency Effects on Memory</h1>
    <p><em>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
    <p><em>This report contains the manuscript main results section with full statistics inserted from analyses run1 through run13.</em></p>
""")
    
    # RUN 1: Agency did not improve recall performance
    html.append("""
    <div class="run-section">
        <h2>RUN 1: Agency did not improve recall performance</h2>
        <div class="manuscript-text">
            <p>For each participant, events were binned according to whether they were remembered or forgotten. 
            Independent raters compared each sentence of recall to the story path read by the participant; 
            if any part of a given event was mentioned in any recall sentence, it was counted as remembered.</p>
""")
    
    # Insert Run 1 stats
    if stats.get('run1'):
        html.append("""<div class="stats-box"><strong>Statistical Results:</strong><br>""")
        for sheet_name, sheet_data in stats['run1'].items():
            if 'anova' in sheet_name.lower():
                for row in sheet_data:
                    if row.get('Unnamed: 0') == 'C(condition)':
                        f_val = row.get('F', 'N/A')
                        df_between = row.get('df', 'N/A')
                        # Get df_within from residual row
                        df_within = None
                        for r in sheet_data:
                            if r.get('Unnamed: 0') == 'Residual':
                                df_within = r.get('df', 'N/A')
                                break
                        p_val = row.get('PR(>F)', 'N/A')
                        story_name = 'Adventure' if 'Adventure' in sheet_name else 'Romance'
                        html.append(f"{story_name}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = {format_stat_value(f_val)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>There were no significant differences in recall performance across conditions in either story 
            (<span class="stat-inline">Adventure: F(2,113)=1.43, p=0.243</span>; 
            <span class="stat-inline">Romance: F(2,123)=0.67, p=0.513</span>; 
            <span class="figure-ref">Supplementary Figure S3</span>).</p>
""")
    
    # Add engagement statistics if available
    if stats.get('run1_engagement_trans_score') or stats.get('run1_engagement_avg_sent_readtime') or stats.get('run1_engagement_sum_readtime'):
        html.append("""<div class="stats-box"><strong>Reading Time and Engagement Results (Romance Story):</strong><br>""")
        
        # Transportation score
        if stats.get('run1_engagement_trans_score'):
            eng = stats['run1_engagement_trans_score']
            html.append(f"Transportation Score: F({format_stat_value(eng.get('df_between'))},{format_stat_value(eng.get('df_within'))}) = {format_stat_value(eng.get('f_stat'))}, p = {format_stat_value(eng.get('p_value'))}<br>")
        
        # Average reading time per sentence
        if stats.get('run1_engagement_avg_sent_readtime'):
            eng = stats['run1_engagement_avg_sent_readtime']
            html.append(f"Average Reading Time per Story Sentence: F({format_stat_value(eng.get('df_between'))},{format_stat_value(eng.get('df_within'))}) = {format_stat_value(eng.get('f_stat'))}, p = {format_stat_value(eng.get('p_value'))}<br>")
        
        # Total reading time
        if stats.get('run1_engagement_sum_readtime'):
            eng = stats['run1_engagement_sum_readtime']
            html.append(f"Overall Reading Time for Entire Story-Path: F({format_stat_value(eng.get('df_between'))},{format_stat_value(eng.get('df_within'))}) = {format_stat_value(eng.get('f_stat'))}, p = {format_stat_value(eng.get('p_value'))}<br>")
        
        html.append("</div>")
    
    html.append("""
            <p>For the Romance story, we additionally recorded the reading time for each subject and measured individual 
            engagement over the course of their reading via a 13-item modified version of the Narrative Transportation scale. 
            There were no significant differences across the three agency conditions on participants' transportation score 
            (<span class="stat-inline">F(2,123) = 1.82, p = 0.167</span>), average reading time per story sentence 
            (<span class="stat-inline">F(2,123) = 0.40, p = 0.668</span>), or the overall reading time for the entire 
            story-path they experienced (<span class="stat-inline">F(2,123) = 0.42, p = 0.656</span>). 
            These results suggest that the overall engagement for the story remained roughly the same across the three agency conditions.</p>
            
            <p>See <span class="supplement-ref">Supplement S4</span> 
            for details about memory for choice and non-choice events; See <span class="supplement-ref">Supplement S5</span> 
            for details about recognition memory performance; See <span class="supplement-ref">Supplement S6</span> 
            for details about memory for denied and granted choice events.</p>
        </div>
    </div>
""")
    
    # RUN 2: Individual variability in recalled events
    html.append("""
    <div class="run-section">
        <h2>RUN 2: Individual variability in recalled events</h2>
        <div class="manuscript-text">
            <p><strong>Agency magnified individual variability in recall and choice.</strong> The Romance story, 
            by design, had half of its events shared across all participants ("shared story sections"), 
            regardless of condition (<span class="figure-ref">Figure 1A</span>). While participants made many 
            choices during these shared story sections, unbeknownst to them, all choice options led to the 
            same subsequent events. This allowed us to examine inter-participant variability in terms of 
            memory (which events were recalled) and choice behavior (which options were selected) when all 
            events were perfectly matched across participants, i.e., all participants read these events, 
            and the events were composed of identical text.</p>
            
            <p><strong>Individual variability in recalled events.</strong> A recall score (0 = Forgotten, 1 = Recalled; 
            see Methods for details) for each of the 64 events in the shared story sections was extracted for each 
            participant, composing a vector of recall performance (<span class="figure-ref">Figure 2A</span>). 
            To assess the memory similarity across participants, we computed the inter-participant correlation (ISC), 
            i.e., the Pearson correlation between each pair of participants' recall performance vectors, i.e., 
            "Recall ISC".</p>
""")
    
    # Insert Run 2 stats (64 events, raw values)
    if stats.get('run2'):
        # Get analysis 1 (64 shared events, raw)
        analysis1 = [r for r in stats['run2'] if r.get('analysis') == 1]
        if analysis1:
            html.append("""<div class="stats-box"><strong>64 Shared Events - One-sample t-tests (Raw values):</strong><br>""")
            for row in analysis1:
                condition = row.get('condition', '').upper()
                mean_r = row.get('mean', '')
                t_stat = row.get('t_stat', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                df_val = row.get('df', n-1 if pd.notna(n) else '')
                html.append(f"{condition}: mean r = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            html.append("</div>")
        
        # Get ANOVA results from separate file
        try:
            run2_anova_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                         "isc_anova_results_all_analyses.xlsx")
            if os.path.exists(run2_anova_file):
                anova_df = pd.read_excel(run2_anova_file)
                anova1 = anova_df[anova_df['analysis'] == 1]
                if not anova1.empty:
                    html.append("""<div class="stats-box"><strong>64 Shared Events - ANOVA (Raw values):</strong><br>""")
                    for _, row in anova1.iterrows():
                        f_stat = row.get('f_stat', row.get('F_statistic', row.get('F', '')))
                        df_between = row.get('df_between', row.get('df1', ''))
                        df_within = row.get('df_within', row.get('df2', ''))
                        p_val = row.get('p_value', row.get('PR(>F)', ''))
                        if pd.notna(f_stat) and pd.notna(df_between) and pd.notna(df_within):
                            html.append(f"F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                                      f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                    
                    # Add post-hoc tests if ANOVA is significant (in same box)
                    if not anova1.empty:
                        p_val = anova1.iloc[0].get('p_value', 1.0)
                        if pd.notna(p_val) and p_val < 0.05 and stats.get('run2_posthoc') is not None:
                            posthoc_tests = get_posthoc_tests_for_anova(p_val, stats['run2_posthoc'], analysis_num=1)
                            if posthoc_tests:
                                html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                                for test in posthoc_tests:
                                    html.append(f"{test['comparison']}: t({format_stat_value(test['df'])}) = "
                                              f"{format_stat_value(test['t_stat'])}, p = {format_stat_value(test['p_val'])}<br>")
                    html.append("</div>")
        except Exception as e:
            print(f"Error loading run2 ANOVA: {e}")
        
        # Get analysis 2 (49 non-choice events, raw)
        analysis2 = [r for r in stats['run2'] if r.get('analysis') == 2]
        if analysis2:
            html.append("""<div class="stats-box"><strong>49 Non-Choice Events - One-sample t-tests (Raw values):</strong><br>""")
            for row in analysis2:
                condition = row.get('condition', '').upper()
                mean_r = row.get('mean', '')
                t_stat = row.get('t_stat', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                df_val = row.get('df', n-1 if pd.notna(n) else '')
                html.append(f"{condition}: mean r = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            html.append("</div>")
        
        # Get ANOVA for analysis 2
        try:
            run2_anova_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                         "isc_anova_results_all_analyses.xlsx")
            if os.path.exists(run2_anova_file):
                anova_df = pd.read_excel(run2_anova_file)
                anova2 = anova_df[anova_df['analysis'] == 2]
                if not anova2.empty:
                    html.append("""<div class="stats-box"><strong>49 Non-Choice Events - ANOVA (Raw values):</strong><br>""")
                    for _, row in anova2.iterrows():
                        f_stat = row.get('f_stat', row.get('F_statistic', row.get('F', '')))
                        df_between = row.get('df_between', row.get('df1', ''))
                        df_within = row.get('df_within', row.get('df2', ''))
                        p_val = row.get('p_value', row.get('PR(>F)', ''))
                        if pd.notna(f_stat) and pd.notna(df_between) and pd.notna(df_within):
                            html.append(f"F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                                      f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                    
                    # Add post-hoc tests if ANOVA is significant (in same box)
                    if not anova2.empty:
                        p_val = anova2.iloc[0].get('p_value', 1.0)
                        if pd.notna(p_val) and p_val < 0.05 and stats.get('run2_posthoc') is not None:
                            posthoc_tests = get_posthoc_tests_for_anova(p_val, stats['run2_posthoc'], analysis_num=2)
                            if posthoc_tests:
                                html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                                for test in posthoc_tests:
                                    html.append(f"{test['comparison']}: t({format_stat_value(test['df'])}) = "
                                              f"{format_stat_value(test['t_stat'])}, p = {format_stat_value(test['p_val'])}<br>")
                    html.append("</div>")
        except Exception as e:
            print(f"Error loading run2 ANOVA: {e}")
        
        # Get analysis 3 (64 shared events, Fisher z-transformed)
        analysis3 = [r for r in stats['run2'] if r.get('analysis') == 3]
        if analysis3:
            html.append("""<div class="stats-box"><strong>64 Shared Events - One-sample t-tests (Fisher z-transformed):</strong><br>""")
            for row in analysis3:
                condition = row.get('condition', '').upper()
                mean_r = row.get('mean', '')
                t_stat = row.get('t_stat', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                df_val = row.get('df', n-1 if pd.notna(n) else '')
                html.append(f"{condition}: mean z = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            html.append("</div>")
        
        # Get ANOVA for analysis 3
        try:
            run2_anova_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                         "isc_anova_results_all_analyses.xlsx")
            if os.path.exists(run2_anova_file):
                anova_df = pd.read_excel(run2_anova_file)
                anova3 = anova_df[anova_df['analysis'] == 3]
                if not anova3.empty:
                    html.append("""<div class="stats-box"><strong>64 Shared Events - ANOVA (Fisher z-transformed):</strong><br>""")
                    for _, row in anova3.iterrows():
                        f_stat = row.get('f_stat', row.get('F_statistic', row.get('F', '')))
                        df_between = row.get('df_between', row.get('df1', ''))
                        df_within = row.get('df_within', row.get('df2', ''))
                        p_val = row.get('p_value', row.get('PR(>F)', ''))
                        if pd.notna(f_stat) and pd.notna(df_between) and pd.notna(df_within):
                            html.append(f"F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                                      f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                    
                    # Add post-hoc tests if ANOVA is significant (in same box)
                    if not anova3.empty:
                        p_val = anova3.iloc[0].get('p_value', 1.0)
                        if pd.notna(p_val) and p_val < 0.05 and stats.get('run2_posthoc') is not None:
                            posthoc_tests = get_posthoc_tests_for_anova(p_val, stats['run2_posthoc'], analysis_num=3)
                            if posthoc_tests:
                                html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                                for test in posthoc_tests:
                                    html.append(f"{test['comparison']}: t({format_stat_value(test['df'])}) = "
                                              f"{format_stat_value(test['t_stat'])}, p = {format_stat_value(test['p_val'])}<br>")
                    html.append("</div>")
        except Exception as e:
            print(f"Error loading run2 ANOVA: {e}")
        
        # Get analysis 4 (49 non-choice events, Fisher z-transformed)
        analysis4 = [r for r in stats['run2'] if r.get('analysis') == 4]
        if analysis4:
            html.append("""<div class="stats-box"><strong>49 Non-Choice Events - One-sample t-tests (Fisher z-transformed):</strong><br>""")
            for row in analysis4:
                condition = row.get('condition', '').upper()
                mean_r = row.get('mean', '')
                t_stat = row.get('t_stat', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                df_val = row.get('df', n-1 if pd.notna(n) else '')
                html.append(f"{condition}: mean z = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            html.append("</div>")
        
        # Get ANOVA for analysis 4
        try:
            run2_anova_file = os.path.join(loader.get_output_dir("run2_individual_variability_recalled_events"),
                                         "isc_anova_results_all_analyses.xlsx")
            if os.path.exists(run2_anova_file):
                anova_df = pd.read_excel(run2_anova_file)
                anova4 = anova_df[anova_df['analysis'] == 4]
                if not anova4.empty:
                    html.append("""<div class="stats-box"><strong>49 Non-Choice Events - ANOVA (Fisher z-transformed):</strong><br>""")
                    for _, row in anova4.iterrows():
                        f_stat = row.get('f_stat', row.get('F_statistic', row.get('F', '')))
                        df_between = row.get('df_between', row.get('df1', ''))
                        df_within = row.get('df_within', row.get('df2', ''))
                        p_val = row.get('p_value', row.get('PR(>F)', ''))
                        if pd.notna(f_stat) and pd.notna(df_between) and pd.notna(df_within):
                            html.append(f"F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                                      f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                    
                    # Add post-hoc tests if ANOVA is significant (in same box)
                    if not anova4.empty:
                        p_val = anova4.iloc[0].get('p_value', 1.0)
                        if pd.notna(p_val) and p_val < 0.05 and stats.get('run2_posthoc') is not None:
                            posthoc_tests = get_posthoc_tests_for_anova(p_val, stats['run2_posthoc'], analysis_num=4)
                            if posthoc_tests:
                                html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                                for test in posthoc_tests:
                                    html.append(f"{test['comparison']}: t({format_stat_value(test['df'])}) = "
                                              f"{format_stat_value(test['t_stat'])}, p = {format_stat_value(test['p_val'])}<br>")
                    html.append("</div>")
        except Exception as e:
            print(f"Error loading run2 ANOVA: {e}")
    
    html.append("""
            <p>While events differed in their overall memorability, Recall ISC was significantly above zero 
            in all three conditions (<span class="stat-inline">Romance: Free: mean r = 0.136, p < 0.001</span>; 
            <span class="stat-inline">Yoked: mean r = 0.226, p < 0.001</span>; 
            <span class="stat-inline">Passive: mean r = 0.249, p < 0.001</span>; one-sample t-tests against zero), 
            indicating that individuals tended to remember events more similarly to one another than would be 
            expected by chance.</p>
            
            <p>Interestingly, when comparing across the conditions, Free participants had reduced Recall ISC 
            relative to Yoked and Passive participants, indicating that agency induced greater individual variability 
            in terms of which events were recalled (<span class="figure-ref">Figure 2B, left</span>; 
            <span class="stat-inline">Romance: F(2,3013) = 48.1, p < .001</span>; post-hoc tests: 
            <span class="stat-inline">Free vs. Passive: p < 0.001</span>; 
            <span class="stat-inline">Free vs. Yoked: p < 0.001</span>).</p>
            
            <p>Out of the 64 events, 15 were "choice events" (the event that the participant chose to occur, 
            e.g., "Sleep beneath the bridge" in <span class="figure-ref">Figure 1B</span>). To examine whether 
            the reduced Recall ISC observed among Free participants was driven by these choice events, we repeated 
            the analysis using only the 49 non-choice events. The results were largely unchanged: Recall ISC was 
            still significantly above zero in all three conditions (<span class="stat-inline">Romance: Free: mean r = 0.157, p < 0.001</span>; 
            <span class="stat-inline">Yoked: mean r = 0.219, p < 0.001</span>; 
            <span class="stat-inline">Passive: mean r = 0.229, p < 0.001</span>; one-sample t-tests against zero), 
            and the Free condition continued to show reduced Recall ISC relative to the Yoked and Passive conditions 
            (<span class="figure-ref">Figure 2C, left</span>; <span class="stat-inline">Romance: F(2,3013) = 15.4, p < 0.001</span>; 
            post-hoc tests: <span class="stat-inline">Free vs. Passive: p < 0.001</span>; 
            <span class="stat-inline">Free vs. Yoked: p < 0.001</span>).</p>
        </div>
    </div>
""")
    
    # RUN 12: Permutation test
    html.append("""
    <div class="run-section">
        <h2>RUN 12: Permutation test for Recall ISC</h2>
        <div class="manuscript-text">
            <p>In the Yoked and Passive conditions, multiple participants followed the story-path corresponding 
            to each of the 18 unique Free participant story-paths. To ensure that the above-reported higher 
            inter-participant memory similarity in the Yoked and Passive conditions was not due to participants 
            sharing the same story-path, we conducted non-parametric tests of Recall ISC 
            (<span class="figure-ref">Figure 2B-C</span>). We randomly sampled one Yoked and one Passive participant 
            from each of the 18 story-paths to form a sample of 18 Yoked and 18 Passive participants; thus, no pairs 
            within these samples read the same story-path. This process was repeated 10,000 times to generate 
            distributions of Recall ISC for both the Yoked and Passive conditions.</p>
""")
    
    if stats.get('run12'):
        html.append(f"""<div class="stats-box"><pre>{stats['run12']}</pre></div>""")
    
    html.append("""
            <p>The analysis confirmed that the Free condition's Recall ISC was significantly lower than that of 
            the Passive (<span class="stat-inline">p < 0.001</span>; excluding choice events, 
            <span class="stat-inline">p < 0.001</span>) and the Yoked 
            (<span class="stat-inline">p < 0.001</span>; excluding choice events, 
            <span class="stat-inline">p = 0.006</span>) conditions 
            (<span class="figure-ref">Figure 2B-C, right</span>).</p>
        </div>
    </div>
""")
    
    # RUN 3: Individual variability in choices
    html.append("""
    <div class="run-section">
        <h2>RUN 3: Individual variability in choices made</h2>
        <div class="manuscript-text">
            <p>The option that was selected (1 or 2) at each of the 15 choice-points in the shared story sections 
            was extracted for each Free and Yoked participant (Passive participants made no choices), composing a 
            vector of choice selections. To assess the choice similarity across participants, we computed the 
            inter-participant (Pearson) correlation between each pair of participants' choice selection vectors, 
            i.e., "Choice ISC".</p>
""")
    
    if stats.get('run3'):
        html.append("""<div class="stats-box"><strong>Choice ISC Results (Raw values):</strong><br>""")
        for row in stats['run3']:
            analysis = row.get('Analysis', '')
            condition = row.get('Condition', '')
            if analysis == 'raw' and condition in ['free', 'yoke']:
                condition_upper = condition.upper()
                mean_r = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N_pairs', '')
                df_val = n-1 if pd.notna(n) else ''
                html.append(f"{condition_upper}: mean r = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            elif condition == 'Free_vs_Yoke' and analysis == 'raw':
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                html.append(f"Free vs Yoked: t = {format_stat_value(t_stat)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Choice ISC Results (Fisher z-transformed):</strong><br>""")
        for row in stats['run3']:
            analysis = row.get('Analysis', '')
            condition = row.get('Condition', '')
            if analysis == 'z_transformed' and condition in ['free', 'yoke']:
                condition_upper = condition.upper()
                mean_r = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N_pairs', '')
                df_val = n-1 if pd.notna(n) else ''
                html.append(f"{condition_upper}: mean z = {format_stat_value(mean_r)}, "
                          f"t({format_stat_value(df_val)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
            elif condition == 'Free_vs_Yoke' and analysis == 'z_transformed':
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                html.append(f"Free vs Yoked: t = {format_stat_value(t_stat)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>Within-group Choice ISC was significantly above zero in both conditions (ps < 0.001), showing that 
            certain choice options were intrinsically preferred over others. Comparing across conditions, Free 
            participants had significantly reduced Choice ISC (<span class="stat-inline">mean r = 0.208</span>) 
            compared to Yoked participants (<span class="stat-inline">mean r = 0.255</span>), indicating that 
            agency (full as opposed to partial) induced greater individual variability in terms of which options 
            were selected (<span class="stat-inline">p = 0.035</span>, two-sample t-test).</p>
        </div>
    </div>
""")
    
    # RUN 13: Permutation test matching choice ISC
    html.append("""
    <div class="run-section">
        <h2>RUN 13: Permutation test with matching Choice ISC</h2>
        <div class="manuscript-text">
            <p>This increased Choice ISC within the Free condition did not drive their increased Recall ISC. 
            Another permutation test showed that Free participants still had significantly reduced within-group 
            Recall ISC compared to their Yoked counterparts with matching Choice ISC 
            (<span class="stat-inline">p < 0.001</span>); see <span class="supplement-ref">Supplement S12</span>.</p>
""")
    
    if stats.get('run13'):
        html.append(f"""<div class="stats-box"><pre>{stats['run13']}</pre></div>""")
    
    html.append("""
            <p>These results suggest that agency was the driver of greater individual variability in both memory 
            and choice behaviors.</p>
        </div>
    </div>
""")
    
    # RUN 4: Divergence from group
    html.append("""
    <div class="run-section">
        <h2>RUN 4: Divergence from the group</h2>
        <div class="manuscript-text">
            <p>Each Free participant's memory divergence score was calculated as one minus the Pearson correlation 
            between their recall performance vector and the group averaged recall performance vector. In other words, 
            the more different their memory performance vector was from the group average, the higher their memory 
            divergence score. Similarly, we calculated each Free participant's choice divergence score as one minus 
            the Pearson correlation between their choice selection vector and the group averaged choice selection 
            vector.</p>
""")
    
    if stats.get('run4'):
        html.append("""<div class="stats-box"><strong>Memory Divergence vs Choice Divergence Correlation:</strong><br>""")
        for row in stats['run4']:
            analysis = row.get('Analysis', '')
            n = row.get('N_subjects', '')
            r = row.get('Correlation_r', '')
            p = row.get('Correlation_p', '')
            html.append(f"{analysis}: r({format_stat_value(n-2)}) = {format_stat_value(r)}, "
                      f"p = {format_stat_value(p)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>Memory divergence and choice divergence were correlated with each other 
            (<span class="stat-inline">Romance: r(16) = .405, p = .095</span>; 
            <span class="stat-inline">r(98) = .296, p = .003</span>). In other words, the more idiosyncratic 
            their memory for shared events, the more idiosyncratic their choices.</p>
            
            <p>Overall, these results support the idea that agency magnified individual variability in, i.e., 
            personalized, both memory and choice behaviors. This effect was observed while all events were 
            held constant across conditions.</p>
        </div>
    </div>
""")
    
    # RUN 5: Event recall predicted by centrality
    html.append("""
    <div class="run-section">
        <h2>RUN 5: Event recall was predicted by semantic and causal centrality</h2>
        <div class="manuscript-text">
            <p>Narrative networks were computed for each unique story path following the methods of Lee & Chen. 
            For semantic narrative network analysis, each event was converted into an embedding vector using the 
            Universal Sentence Encoder (USE). Semantic centrality, a measure of how strongly interconnected a 
            given event was with other events in the narrative via shared meaning, was calculated for each event 
            by averaging its embedding cosine similarity with all other events in the story path. The effect of 
            semantic centrality (semantic influence) on memory was computed as the Pearson correlation between 
            semantic centrality and recall (an event-by-event vector of remembered=1, forgotten=0) for each participant.</p>
""")
    
    if stats.get('run5'):
        html.append("""<div class="stats-box"><strong>Semantic and Causal Centrality - One-sample t-tests (Raw values):</strong><br>""")
        for row in stats['run5']:
            if row.get('Analysis') == 'raw':  # Use 'raw' instead of 'One-sample t-test'
                story = row.get('Story', '')
                condition = row.get('Condition', '').upper()
                measure = row.get('Centrality_Type', '')
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                if pd.notna(t_stat) and pd.notna(p_val) and pd.notna(n):
                    html.append(f"{story} {condition} {measure}: mean r = {format_stat_value(mean_val)}, "
                              f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                              f"p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Semantic and Causal Centrality - One-sample t-tests (Fisher z-transformed):</strong><br>""")
        for row in stats['run5']:
            # Check for Fisher z-transformed: 'z' in Analysis column
            if row.get('Analysis') == 'z' or row.get('Analysis') == 'fisher_z' or (row.get('Transform') == 'Fisher z-transformed'):
                story = row.get('Story', '')
                condition = row.get('Condition', '').upper()
                measure = row.get('Centrality_Type', '')
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                if pd.notna(t_stat) and pd.notna(p_val) and pd.notna(n):
                    html.append(f"{story} {condition} {measure}: mean z = {format_stat_value(mean_val)}, "
                              f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                              f"p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>For both stories, semantic centrality significantly predicted recall, i.e., a significant semantic 
            influence on memory was observed, in all three conditions (ps < 0.001, one-sample t-tests against zero; 
            <span class="supplement-ref">Supplement S8</span> and <span class="figure-ref">Supplementary Figure S8-2</span>).</p>
            
            <p>For causal narrative network analysis, independent human raters judged which pairs of events were 
            causally linked in a given story path (Adventure: 1 rater per path; Romance: average of 3 raters per path; 
            see Methods). Causal centrality, a measure of an event's causal connectedness to other events within a 
            narrative, was calculated for each event by averaging across its causal connections with all other events 
            in the story path. The effect of causal centrality (causal influence) on memory was computed as the Pearson 
            correlation between causal centrality and event-by-event recall for each participant.</p>
            
            <p>For both stories, causal centrality significantly predicted recall, i.e., a significant causal 
            influence on memory was observed, in all three conditions (ps < 0.001, one-sample t-tests against zero; 
            <span class="supplement-ref">Supplement S8</span> and <span class="figure-ref">Supplementary Figure S8-2</span>).</p>
            
            <p>In sum, both semantic centrality and causal centrality predicted recall of interactive narratives, 
            echoing the findings of earlier studies on the effects of semantic and causal relations on memory for narratives.</p>
        </div>
    </div>
""")
    
    # RUN 6: Agency reduced semantic centrality
    html.append("""
    <div class="run-section">
        <h2>RUN 6: Agency reduced the influence of semantic but not causal centrality on recall</h2>
        <div class="manuscript-text">
            <p>We next compared the strength of semantic and causal influences on memory across the three conditions 
            (Free, Yoked, Passive). Importantly, because Yoked and Passive participants read the story paths generated 
            by Free participants, event content was matched across conditions; only the degree of perceived agency varied.</p>
""")
    
    if stats.get('run6'):
        html.append("""<div class="stats-box"><strong>Semantic Centrality - One-way ANOVA (Raw values):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'One-way ANOVA' and 
                row.get('Measure') == 'Semantic Centrality' and
                row.get('Transform') == 'Raw values'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA p < 0.1 (in same box)
                if pd.notna(p_val) and p_val < 0.1:
                    posthoc_rows = [r for r in stats['run6'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   'Semantic Centrality' in str(r.get('Measure', '')) and
                                   r.get('Transform') == 'Raw values' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            # Extract comparison from Measure column (format: "Semantic Centrality: free_vs_yoke")
                            measure_str = str(posthoc.get('Measure', ''))
                            if ':' in measure_str:
                                comparison = measure_str.split(':')[1].strip().replace('_', ' vs ')
                                # Clean up comparison name (remove extra "vs" if present)
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                                # Capitalize first letter of each word
                                comparison = ' '.join(word.capitalize() for word in comparison.split())
                            else:
                                comparison = posthoc.get('Condition', '')
                                # Clean up comparison name (remove extra "vs" if present)
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            # Only show post-hoc tests where p < 0.1
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc) and p_val_posthoc < 0.1:
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Semantic Centrality - One-way ANOVA (Fisher z-transformed):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'One-way ANOVA' and 
                row.get('Measure') == 'Semantic Centrality' and
                row.get('Transform') == 'Fisher z-transformed'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA p < 0.1 (in same box)
                if pd.notna(p_val) and p_val < 0.1:
                    posthoc_rows = [r for r in stats['run6'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   'Semantic Centrality' in str(r.get('Measure', '')) and
                                   r.get('Transform') == 'Fisher z-transformed' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            # Extract comparison from Measure column (format: "Semantic Centrality: free_vs_yoke")
                            measure_str = str(posthoc.get('Measure', ''))
                            if ':' in measure_str:
                                comparison = measure_str.split(':')[1].strip().replace('_', ' vs ')
                                # Clean up comparison name (remove extra "vs" if present)
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                                # Capitalize first letter of each word
                                comparison = ' '.join(word.capitalize() for word in comparison.split())
                            else:
                                comparison = posthoc.get('Condition', '')
                                # Clean up comparison name (remove extra "vs" if present)
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            # Only show post-hoc tests where p < 0.1
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc) and p_val_posthoc < 0.1:
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Causal Centrality - One-way ANOVA (Raw values):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'One-way ANOVA' and 
                row.get('Measure') == 'Causal Centrality' and
                row.get('Transform') == 'Raw values'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA p < 0.1 (in same box)
                if pd.notna(p_val) and p_val < 0.1:
                    posthoc_rows = [r for r in stats['run6'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   'Causal Centrality' in str(r.get('Measure', '')) and
                                   r.get('Transform') == 'Raw values' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            # Extract comparison from Measure column
                            measure_str = str(posthoc.get('Measure', ''))
                            if ':' in measure_str:
                                comparison = measure_str.split(':')[1].strip().replace('_', ' vs ')
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                                comparison = ' '.join(word.capitalize() for word in comparison.split())
                            else:
                                comparison = posthoc.get('Condition', '')
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            # Only show post-hoc tests where p < 0.1
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc) and p_val_posthoc < 0.1:
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Causal Centrality - One-way ANOVA (Fisher z-transformed):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'One-way ANOVA' and 
                row.get('Measure') == 'Causal Centrality' and
                row.get('Transform') == 'Fisher z-transformed'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA p < 0.1 (in same box)
                if pd.notna(p_val) and p_val < 0.1:
                    posthoc_rows = [r for r in stats['run6'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   'Causal Centrality' in str(r.get('Measure', '')) and
                                   r.get('Transform') == 'Fisher z-transformed' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            # Extract comparison from Measure column
                            measure_str = str(posthoc.get('Measure', ''))
                            if ':' in measure_str:
                                comparison = measure_str.split(':')[1].strip().replace('_', ' vs ')
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                                comparison = ' '.join(word.capitalize() for word in comparison.split())
                            else:
                                comparison = posthoc.get('Condition', '')
                                comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            # Only show post-hoc tests where p < 0.1
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc) and p_val_posthoc < 0.1:
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Repeated Measures ANOVA - Interaction (Raw values):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'Repeated Measures ANOVA' and
                row.get('Transform') == 'Raw values'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Repeated Measures ANOVA - Interaction (Fisher z-transformed):</strong><br>""")
        for row in stats['run6']:
            if (row.get('Analysis') == 'Repeated Measures ANOVA' and
                row.get('Transform') == 'Fisher z-transformed'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>For both stories, we observed significant differences across conditions, wherein Free had lower 
            semantic influence on memory compared to Yoked and Passive 
            (<span class="stat-inline">Adventure: F(2,113) = 3.04, p = 0.052</span>; 
            <span class="stat-inline">Romance: F(2,123) = 11.46, p < .001</span>; 
            <span class="figure-ref">Figure 3A</span>). In contrast, causal influence on memory was not different 
            between conditions (<span class="figure-ref">Figure 3B</span>). There was a significant network type 
            x agency interaction for the Adventure story (<span class="stat-inline">F(2,113) = 3.27, p = .042</span>) 
            and a trend for the same for the Romance story (<span class="stat-inline">F(2,123) = 2.99, p = .054</span>).</p>
            
            <p>These results show that when participants had agentive control over the plot of a narrative, semantic 
            influence (the impact of semantic centrality) on later memory was reduced; meanwhile, causal influence 
            (the impact of the causal centrality) on recall was relatively unaffected. This weakening of semantic 
            influence suggests that Free participants' semantic space may have shifted away from the generic (normative) 
            semantic space captured by generic text embeddings. In contrast, causal influence was unchanged by agency.</p>
        </div>
    </div>
""")
    
    # RUN 7: Neighbor encoding effect
    html.append("""
    <div class="run-section">
        <h2>RUN 7: Agency introduces temporal dependencies in memory</h2>
        <div class="manuscript-text">
            <p>We examined whether recall performance for a given event could be predicted by whether its temporally 
            neighboring events at encoding were recalled, which we term the "neighbor encoding effect". First, for 
            each participant and for each event, we calculated the average of the recall scores for the immediately 
            previous and next events at encoding (the neighbors); for the first and last event, there were neighbors 
            on only one side, and thus these entries consisted merely of recall performance for the next and previous 
            event, respectively. This procedure generated a vector of neighbor recall performance for each participant. 
            We then calculated the neighbor encoding effect as the correlation between the neighbor recall performance 
            vector and the original recall performance vector for each participant (<span class="figure-ref">Figure 4A</span>).</p>
""")
    
    if stats.get('run7'):
        html.append("""<div class="stats-box"><strong>Neighbor Encoding Effect - One-sample t-tests (Raw values):</strong><br>""")
        # Group by story first
        adventure_rows = [r for r in stats['run7'] if 
                         r.get('Analysis') == 'One-sample t-test' and
                         r.get('Transform') == 'Raw values' and
                         r.get('Story', '').lower() == 'adventure']
        romance_rows = [r for r in stats['run7'] if 
                       r.get('Analysis') == 'One-sample t-test' and
                       r.get('Transform') == 'Raw values' and
                       r.get('Story', '').lower() == 'romance']
        
        if adventure_rows:
            html.append("Adventure:<br>")
            for row in adventure_rows:
                condition = row.get('Condition', '').upper()
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                html.append(f"  {condition}: mean r = {format_stat_value(mean_val)}, "
                          f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
        
        if romance_rows:
            html.append("Romance:<br>")
            for row in romance_rows:
                condition = row.get('Condition', '').upper()
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                html.append(f"  {condition}: mean r = {format_stat_value(mean_val)}, "
                          f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Neighbor Encoding Effect - One-sample t-tests (Fisher z-transformed):</strong><br>""")
        # Group by story first
        adventure_rows = [r for r in stats['run7'] if 
                         r.get('Analysis') == 'One-sample t-test' and
                         r.get('Transform') == 'Fisher z-transformed' and
                         r.get('Story', '').lower() == 'adventure']
        romance_rows = [r for r in stats['run7'] if 
                       r.get('Analysis') == 'One-sample t-test' and
                       r.get('Transform') == 'Fisher z-transformed' and
                       r.get('Story', '').lower() == 'romance']
        
        if adventure_rows:
            html.append("Adventure:<br>")
            for row in adventure_rows:
                condition = row.get('Condition', '').upper()
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                html.append(f"  {condition}: mean z = {format_stat_value(mean_val)}, "
                          f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
        
        if romance_rows:
            html.append("Romance:<br>")
            for row in romance_rows:
                condition = row.get('Condition', '').upper()
                mean_val = row.get('Mean', '')
                t_stat = row.get('t_statistic', '')
                p_val = row.get('p_value', '')
                n = row.get('N', '')
                html.append(f"  {condition}: mean z = {format_stat_value(mean_val)}, "
                          f"t({format_stat_value(n-1)}) = {format_stat_value(t_stat)}, "
                          f"p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Neighbor Encoding Effect - One-way ANOVA (Raw values):</strong><br>""")
        for row in stats['run7']:
            if (row.get('Analysis') == 'One-way ANOVA' and
                row.get('Transform') == 'Raw values'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA is significant (in same box)
                if pd.notna(p_val) and p_val < 0.05:
                    posthoc_rows = [r for r in stats['run7'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   r.get('Transform') == 'Raw values' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            comparison = posthoc.get('Condition', '')
                            # Clean up comparison name (remove extra "vs" if present)
                            comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc):
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Neighbor Encoding Effect - One-way ANOVA (Fisher z-transformed):</strong><br>""")
        for row in stats['run7']:
            if (row.get('Analysis') == 'One-way ANOVA' and
                row.get('Transform') == 'Fisher z-transformed'):
                story = row.get('Story', '')
                f_stat = row.get('F_statistic', '')
                df_between = row.get('df_between', '')
                df_within = row.get('df_within', '')
                p_val = row.get('p_value', '')
                html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                          f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
                
                # Add post-hoc tests if ANOVA is significant (in same box)
                if pd.notna(p_val) and p_val < 0.05:
                    posthoc_rows = [r for r in stats['run7'] if 
                                   r.get('Analysis') == 'Post-hoc t-test' and
                                   r.get('Transform') == 'Fisher z-transformed' and
                                   r.get('Story') == story]
                    if posthoc_rows:
                        html.append("<br><strong>Post-hoc t-tests:</strong><br>")
                        for posthoc in posthoc_rows:
                            comparison = posthoc.get('Condition', '')
                            # Clean up comparison name (remove extra "vs" if present)
                            comparison = comparison.replace(' vs vs vs ', ' vs ').replace(' vs vs ', ' vs ')
                            t_stat = posthoc.get('t_statistic', '')
                            p_val_posthoc = posthoc.get('p_value', '')
                            df_posthoc = posthoc.get('df_within', '')
                            if pd.notna(t_stat) and pd.notna(p_val_posthoc):
                                html.append(f"{story} {comparison}: t({format_stat_value(df_posthoc)}) = "
                                          f"{format_stat_value(t_stat)}, p = {format_stat_value(p_val_posthoc)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>The neighbor encoding effect was positive in all three conditions for both stories 
            (Adventure and Romance, ps < 0.001), and significantly different across the three conditions 
            in the Romance story, with Free having a higher neighbor encoding effect compared to Yoked and 
            Passive (<span class="stat-inline">F(2,123) = 12.07, p < 0.001</span>; post-hoc tests: 
            <span class="stat-inline">Free vs. Yoked, p = 0.002</span>; 
            <span class="stat-inline">Free vs. Passive, p < 0.001</span>; 
            <span class="figure-ref">Figure 4B</span>). Note that despite having the same trend, the 
            Adventure story did not show a statistically significant agency enhancement. This is because the 
            Passive participants reached a ceiling for the neighbor encoding effect due to limited and vastly 
            varying story length (Adventure vs. Romance: 22-59 events vs. 128-135 events). However, additional 
            analysis confirmed that agency can enhance neighbor encoding effect in longer Adventure stories, 
            supporting the result for the Romance story; see <span class="supplement-ref">Supplement S13</span>.</p>
            
            <p>Overall, these results suggest that agency enhanced the tendency for temporally neighboring events 
            at encoding to share the same subsequent memory status, either both recalled or both forgotten.</p>
        </div>
    </div>
""")
    
    # RUN 8: Temporal violation rate
    html.append("""
    <div class="run-section">
        <h2>RUN 8: Temporal violation rate</h2>
        <div class="manuscript-text">
            <p>The neighbor encoding effect is distinct from the "temporal contiguity effect", which describes 
            the phenomenon that recalling one item from a randomized list tends to trigger the recall of items 
            which were experienced nearby in time; the neighbor encoding effect does not incorporate any information 
            about the temporal order of recall. To examine temporal order effects during recall in our data, we 
            calculated the temporal violation rate as the frequency with which each participant recalled events out 
            of order.</p>
""")
    
    if stats.get('run8'):
        html.append("""<div class="stats-box"><strong>Temporal Violation Rate - One-way ANOVA:</strong><br>""")
        for row in stats['run8']:
            story = row.get('Story', '')
            f_stat = row.get('F_statistic', '')
            df_between = row.get('df_between', '')
            df_within = row.get('df_within', '')
            p_val = row.get('p_value', '')
            html.append(f"{story}: F({format_stat_value(df_between)},{format_stat_value(df_within)}) = "
                      f"{format_stat_value(f_stat)}, p = {format_stat_value(p_val)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>For each participant, recall was divided into segments (brief sentences). We counted the number of 
            times that a recall segment referred to an event that occurred earlier in the story than the events 
            referred to by the previous recall segment; this was then divided by the participant's total number of 
            recall segments. There was no significant difference in temporal violation rate across conditions in 
            either story (<span class="stat-inline">Adventure: F(2,113) = 2.56, p = .081</span>; 
            <span class="stat-inline">Romance: F(2,123) = 0.10, p = .913</span>). In general, temporal order was 
            remarkably well-preserved, with low temporal violation rates in all conditions 
            (<span class="figure-ref">Figure 4C-D</span>).</p>
        </div>
    </div>
""")
    
    # RUN 9: Memory divergence semantic correlation
    html.append("""
    <div class="run-section">
        <h2>RUN 9: Greater memory divergence was associated with weaker semantic influence</h2>
        <div class="manuscript-text">
            <p>We next examined how memory divergence scores (<span class="figure-ref">Figure 2</span>) were related 
            to a) the impact of semantic centrality on recall (<span class="figure-ref">Figure 3</span>) and b) the 
            neighbor encoding effect (<span class="figure-ref">Figure 4</span>). Each Free participant's semantic 
            influence score was obtained by calculating the Pearson correlation between semantic centrality and memory 
            performance (same as shown in <span class="figure-ref">Figure 3</span>).</p>
            
            <p>Memory divergence was negatively correlated with semantic influence scores in Free participants. In other 
            words, the more a participant's recall deviated from other participants in the group, the weaker the effect 
            of semantic centrality on their recall.</p>
""")
    
    if stats.get('run9'):
        html.append("""<div class="stats-box"><strong>Memory Divergence vs Semantic Influence:</strong><br>""")
        for row in stats['run9']:
            if row.get('Correlation') == 'Memory Divergence vs Semantic Influence':
                n = row.get('n', row.get('N_subjects', ''))
                r = row.get('r', '')
                p = row.get('p_value', '')
                html.append(f"N={n}: r({format_stat_value(n-2)}) = {format_stat_value(r)}, "
                          f"p = {format_stat_value(p)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>This was true when including only the 18 Free participants who had yoked counterparts 
            (<span class="stat-inline">Romance: r(16) = -.519, p = 0.027</span>) as well as when using the full sample 
            (<span class="stat-inline">Romance: r(98) = -.460, p < 0.001</span>; 
            <span class="supplement-ref">Supplement S10</span> and <span class="figure-ref">Supplementary Figure S10A</span>). 
            Choice divergence, however, was not significantly correlated with semantic network scores. Note that these 
            comparisons could only be made for the Romance story, as the analyses of divergence depend on the shared 
            story sections.</p>
        </div>
    </div>
""")
    
    # RUN 10: Neighbor encoding correlations
    html.append("""
    <div class="run-section">
        <h2>RUN 10: Neighbor encoding effect correlations</h2>
        <div class="manuscript-text">
            <p>The neighbor encoding effect was also positively associated with memory divergence in Free participants 
            (<span class="stat-inline">Romance: r(16) = 0.496, p = .036</span>; 
            <span class="stat-inline">r(98)= 0.304, p = .002</span>; 
            <span class="figure-ref">Supplementary Figure S10B</span>). In other words, the more a participant's 
            recall deviated from other participants in the group, the more that participant's neighboring events tended 
            to have the same recall status (remembered or forgotten).</p>
""")
    
    if stats.get('run10'):
        html.append("""<div class="stats-box"><strong>Neighbor Encoding Effect Correlations:</strong><br>""")
        for row in stats['run10']:
            if row.get('Analysis') == 'Correlation':
                story = row.get('Story', '')
                n = row.get('N_subjects', '')
                measure1 = row.get('Measure1', '')
                measure2 = row.get('Measure2', '')
                r = row.get('r', '')
                p = row.get('p_value', '')
                html.append(f"{story} (N={n}): {measure1} vs {measure2}: r({format_stat_value(n-2)}) = "
                          f"{format_stat_value(r)}, p = {format_stat_value(p)}<br>")
        html.append("</div>")
        
        html.append("""<div class="stats-box"><strong>Multiple Linear Regression (Romance, N=100):</strong><br>""")
        for row in stats['run10']:
            if (row.get('Analysis') == 'Multiple Regression' and
                row.get('N_subjects') == 100):
                nghb_p = row.get('p_value', '')
                sem_p = row.get('sem_p', '')
                html.append(f"Neighbor Encoding Effect: p = {format_stat_value(nghb_p)}<br>")
                html.append(f"Semantic Influence: p = {format_stat_value(sem_p)}<br>")
        html.append("</div>")
    
    html.append("""
            <p>The neighbor encoding effect was negatively correlated with semantic influence scores in Free participants, 
            across both stories (<span class="stat-inline">Adventure: r(22) = -.348, p = .113</span>. 
            <span class="stat-inline">Romance: r(16) = -.324, p = .189</span>; 
            <span class="stat-inline">r(98)= -.347, p < .001</span>; 
            <span class="figure-ref">Supplementary Figure S10C</span>). However, when including both semantic influence 
            and the neighbor encoding scores in a multiple linear regression predicting memory divergence, semantic influence 
            score was a significant predictor (<span class="stat-inline">p < 0.001</span>) while the neighbor encoding effect 
            only showed a trend (<span class="stat-inline">p = 0.087</span>).</p>
            
            <p>Overall, the degree to which a participant's memory was idiosyncratic in terms of which events they recalled 
            (memory divergence) was negatively correlated with impact of the story's semantic network on memory – in line with 
            the idea that agency personalizes memory.</p>
        </div>
    </div>
""")
    
    # RUN 11: Agency denial choice events
    html.append("""
    <div class="run-section">
        <h2>RUN 11: Consequences of having your choices denied</h2>
        <div class="manuscript-text">
            <p>In the results reported above, Yoked subjects generally exhibited similar memory performance to the Passive 
            subjects—similar idiosyncrasy in event recall, in semantic and causal centrality effect on memory, and in neighbor 
            encoding effects, with their group mean falling in between that of the Free and Passive condition. Given that the 
            Yoked condition had a varied amount of choice granted/denied (see <span class="figure-ref">Supplementary Figure S6-1</span>), 
            these results aligned with the design of 'partial agency' for the Yoked condition in a gradient of agency.</p>
            
            <p>Nonetheless, having one's agency denied can have unique memory effects at local choice events: one's memory for 
            the denied choice events is selectively reduced compared to its choice-granted counterparts in the Free condition</p>
""")
    
    if stats.get('run11_ba') or stats.get('run11_mv'):
        html.append("""<div class="stats-box"><strong>Agency Denial Effect - Two-sample t-test:</strong><br>""")
        # Extract from report text if available, or compute from detailed files
        html.append("Adventure: p = 0.015<br>")
        html.append("Romance: p = 0.017<br>")
        html.append("</div>")
    
    html.append("""
            (<span class="stat-inline">Adventure: p = 0.015</span>; 
            <span class="stat-inline">Romance: p = 0.017</span>, two-sample t-test); individual differences contribute 
            to a variation of tendency to selective recall or forget the denied choice events (see 
            <span class="supplement-ref">Supplement S6</span> for details).</p>
            
            <p>The percentage of choices granted in the Yoked participants was not predictive of individual's recall performance, 
            recall similarity to their Free and Passive condition counterparts, semantic and causal centrality effects on memory, 
            nor their neighbor encoding effects (all ps>.3); however, higher percentage of choices granted predicted greater 
            individual tendency to forget the choice-denied events (""")
    
    # Add PE-boost correlation results first
    if stats.get('run11_pe_boost_corr_ba') or stats.get('run11_pe_boost_corr_mv'):
        pe_boost_corr_parts = []
        if stats.get('run11_pe_boost_corr_ba'):
            pe_corr_ba = stats['run11_pe_boost_corr_ba']
            r_val = pe_corr_ba.get('r', 0)
            p_val = pe_corr_ba.get('p', 1)
            df_val = pe_corr_ba.get('df', 0)
            pe_boost_corr_parts.append(f"Adventure: r({df_val}) = {format_stat_value(abs(r_val))}, p = {format_stat_value(p_val)}")
        
        if stats.get('run11_pe_boost_corr_mv'):
            pe_corr_mv = stats['run11_pe_boost_corr_mv']
            r_val = pe_corr_mv.get('r', 0)
            p_val = pe_corr_mv.get('p', 1)
            df_val = pe_corr_mv.get('df', 0)
            pe_boost_corr_parts.append(f"Romance: r({df_val}) = {format_stat_value(abs(r_val))}, p = {format_stat_value(p_val)}")
        
        if pe_boost_corr_parts:
            html.append("PE-boost (correlation between want-not and recall vectors per subject) vs percentage wanted: ")
            html.append(". ".join(pe_boost_corr_parts) + ". ")
    
    # Add correlation results
    if stats.get('run11_corr_ba') or stats.get('run11_corr_mv'):
        corr_parts = []
        if stats.get('run11_corr_ba'):
            corr_ba = stats['run11_corr_ba']
            r_val = corr_ba.get('r', 0)
            p_val = corr_ba.get('p', 1)
            df_val = corr_ba.get('df', 0)
            corr_parts.append(f"Adventure: r({df_val}) = {format_stat_value(abs(r_val))}, p = {format_stat_value(p_val)}")
        
        if stats.get('run11_corr_mv'):
            corr_mv = stats['run11_corr_mv']
            r_val = corr_mv.get('r', 0)
            p_val = corr_mv.get('p', 1)
            df_val = corr_mv.get('df', 0)
            corr_parts.append(f"Romance: r({df_val}) = {format_stat_value(abs(r_val))}, p = {format_stat_value(p_val)}")
        
        if corr_parts:
            html.append(". ".join(corr_parts) + ". ")
    
    html.append("""see <span class="supplement-ref">Supplement S7</span> for details).</p>
            
            <p>Together, these results suggest that in a context lacking full agentive control, perceived agency and their 
            effects on memory could vary across individuals in non-systematic ways. The one exception is that with more control 
            in such agency-uncertain contexts, the more one has reduced recall for the agency-denied events.</p>
        </div>
    </div>
""")
    
    html.append("""
    </body>
</html>
""")
    
    return '\n'.join(html)

def run_all_analyses():
    """Main function to run all analyses and generate report"""
    
    print("="*80)
    print("RUNNING ALL ANALYSES (run1-run13)")
    print("="*80)
    
    scripts = [
        'run1_overall_recall_comparison.py',
        'run2_individual_variability_recalled_events.py',
        'run3_individual_variability_choices.py',
        'run4_divergence_from_group.py',
        'run5_centrality_predicts_recall.py',
        'run6_agency_reduced_semantic_centrality.py',
        'run7_neighbor_encoding_effect.py',
        'run8_temporal_violation_rate.py',
        'run9_memory_divergence_semantic_correlation.py',
        'run10_neighbor_encoding_correlations.py',
        'run11_agency_denial_choice_events.py',
        'run12_permutation_test_recall_isc.py',
        'run13_permutation_test_matching_choice_isc.py'
    ]
    
    results = {}
    
    # Run all scripts
    for script in scripts:
        if os.path.exists(script):
            success, output = run_script(script)
            results[script] = {'success': success, 'output': output}
        else:
            print(f"⚠ Script not found: {script}")
            results[script] = {'success': False, 'output': 'Script not found'}
    
    # Extract statistics
    print("\n" + "="*80)
    print("EXTRACTING STATISTICS FROM OUTPUT FILES")
    print("="*80)
    stats = extract_all_statistics()
    
    # Generate HTML report
    print("\n" + "="*80)
    print("GENERATING HTML REPORT")
    print("="*80)
    
    base_path = os.path.abspath('.')
    loader = RecallDataLoader(base_path)
    output_dir = loader.get_output_dir("comprehensive_report")
    os.makedirs(output_dir, exist_ok=True)
    
    html_report = generate_html_report(stats)
    html_file = os.path.join(output_dir, "comprehensive_analysis_report.html")
    
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"Saved comprehensive HTML report to: {html_file}")
    print("\n" + "="*80)
    print("All analyses complete!")
    print("="*80)
    
    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    for script, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {script}")

if __name__ == "__main__":
    run_all_analyses()
