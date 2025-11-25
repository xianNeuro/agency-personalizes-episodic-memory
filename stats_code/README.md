# Agency Effects on Memory: Analysis Scripts

This repository contains the complete analysis pipeline for the study examining how agency (the ability to make choices) affects memory for interactive narratives. The manuscript is currently **in revision**, and these scripts generate the updated results section.

**Preprint**: [https://osf.io/preprints/psyarxiv/7evwj_v1](https://osf.io/preprints/psyarxiv/7evwj_v1)

## Overview

This repository includes:
- **13 main analysis scripts** (run1 through run13) that perform all statistical analyses
- **Supporting scripts** for data loading and structure management
- **Comprehensive HTML report** that combines all results with full statistics
- **Data files** organized in the `data/` directory

## Repository Structure

```
_scripts_github/
├── README.md                          # This file
├── data_structure.py                  # Data loading utilities
├── data_summary_read.py               # Data exploration utilities
├── isc_correlation_analysis.py       # ISC correlation utilities
├── run_all_analyses.py                # Master script to run all analyses
│
├── run1_overall_recall_comparison.py              # Analysis 1: Overall recall performance
├── run2_individual_variability_recalled_events.py # Analysis 2: Recall ISC
├── run3_individual_variability_choices.py         # Analysis 3: Choice ISC
├── run4_divergence_from_group.py                  # Analysis 4: Memory/choice divergence
├── run5_centrality_predicts_recall.py             # Analysis 5: Semantic/causal centrality
├── run6_agency_reduced_semantic_centrality.py     # Analysis 6: Agency effects on centrality
├── run7_neighbor_encoding_effect.py               # Analysis 7: Neighbor encoding effect
├── run8_temporal_violation_rate.py                # Analysis 8: Temporal violation rate
├── run9_memory_divergence_semantic_correlation.py # Analysis 9: Divergence-semantic correlation
├── run10_neighbor_encoding_correlations.py         # Analysis 10: Neighbor encoding correlations
├── run11_agency_denial_choice_events.py           # Analysis 11: Agency denial effects
├── run12_permutation_test_recall_isc.py          # Analysis 12: Permutation test (Recall ISC)
├── run13_permutation_test_matching_choice_isc.py  # Analysis 13: Permutation test (Choice ISC)
│
├── data/                              # Data directory
│   ├── adventure_data1.xlsx          # Adventure story data (part 1)
│   ├── adventure_data2.xlsx          # Adventure story data (part 2)
│   ├── romance_data1.xlsx             # Romance story data (part 1)
│   ├── romance_data2.xlsx             # Romance story data (part 2)
│   └── individual_data/              # Individual subject data
│       ├── adventure/                 # Individual Adventure subject files
│       └── romance/                   # Individual Romance subject files
│
└── output/                            # Analysis outputs
    ├── run1_overall_recall_comparison/
    ├── run2_individual_variability_recalled_events/
    ├── run3_individual_variability_choices/
    ├── run4_divergence_from_group/
    ├── run5_centrality_predicts_recall/
    ├── run6_agency_reduced_semantic_centrality/
    ├── run7_neighbor_encoding_effect/
    ├── run8_temporal_violation_rate/
    ├── run9_memory_divergence_semantic_correlation/
    ├── run10_neighbor_encoding_correlations/
    ├── run11_agency_denial_choice_events/
    ├── run12_permutation_test_recall_isc/
    ├── run13_permutation_test_matching_choice_isc/
    └── comprehensive_report/          # Comprehensive HTML report
        └── comprehensive_analysis_report.html
```

## Quick Start

### Prerequisites

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scipy
  - openpyxl (for Excel file reading)
  - matplotlib
  - seaborn
  - statsmodels
  - scikit-learn
  - joblib (for parallel processing)

Install dependencies:
```bash
pip install pandas numpy scipy openpyxl matplotlib seaborn statsmodels scikit-learn joblib
```

### Running All Analyses

To run all 13 analyses and generate the comprehensive HTML report:

```bash
python run_all_analyses.py
```

This will:
1. Execute all 13 analysis scripts sequentially
2. Extract statistics from output files
3. Generate a comprehensive HTML report with manuscript text and full statistics

### Running Individual Analyses

Each analysis can also be run independently:

```bash
python run1_overall_recall_comparison.py
python run2_individual_variability_recalled_events.py
# ... etc
```

## Analysis Scripts

### Core Supporting Scripts

- **`data_structure.py`**: Defines the `RecallDataLoader` class for managing file paths and loading data from Excel files
- **`data_summary_read.py`**: Utilities for exploring and summarizing data structure
- **`isc_correlation_analysis.py`**: Utilities for inter-subject correlation (ISC) analysis and Fisher z-transformation

### Main Analysis Scripts

#### RUN 1: Overall Recall Comparison
**Script**: `run1_overall_recall_comparison.py`  
**Analysis**: One-way ANOVA comparing overall recall performance across Free, Yoked, and Passive conditions  
**Output**: ANOVA statistics and group means for both Adventure and Romance stories

#### RUN 2: Individual Variability in Recalled Events
**Script**: `run2_individual_variability_recalled_events.py`  
**Analysis**: 
- Computes Recall ISC (inter-subject correlation) for recall performance vectors
- One-sample t-tests to test if ISC > 0
- One-way ANOVA comparing ISC across conditions
- Post-hoc tests
- Performed for: 64 shared events and 49 non-choice events (excluding choice events)
- Both raw r-values and Fisher z-transformed values

**Output**: Statistical results, pairwise correlation matrices, ANOVA and post-hoc test results

#### RUN 3: Individual Variability in Choices Made
**Script**: `run3_individual_variability_choices.py`  
**Analysis**: 
- Computes Choice ISC for choice selection vectors (15 shared choice events)
- One-sample t-tests and two-sample t-test comparing Free vs Yoked
- Both raw r-values and Fisher z-transformed values

**Output**: Choice ISC statistics and comparison results

#### RUN 4: Divergence from the Group
**Script**: `run4_divergence_from_group.py`  
**Analysis**: 
- Calculates memory divergence and choice divergence scores for Free participants
- Correlates memory divergence with choice divergence
- Analyzes both N=18 and N=100 Free participants (Romance story)

**Output**: Correlation results between memory and choice divergence

#### RUN 5: Event Recall Predicted by Semantic and Causal Centrality
**Script**: `run5_centrality_predicts_recall.py`  
**Analysis**: 
- One-sample t-tests for semantic and causal centrality effects on recall
- Tests if semantic influence (sem-ef) and causal influence (caus-ef) are significantly > 0
- Performed for all three conditions (Free, Yoked, Passive) in both stories
- Both raw values and Fisher z-transformed values

**Output**: t-test results for semantic and causal centrality effects

#### RUN 6: Agency Reduced Semantic but Not Causal Centrality
**Script**: `run6_agency_reduced_semantic_centrality.py`  
**Analysis**: 
- One-way ANOVA comparing semantic and causal influence across conditions
- Post-hoc t-tests (if ANOVA p < 0.1)
- Repeated measures ANOVA (mixed design) testing network type × condition interaction
- Both raw values and Fisher z-transformed values

**Output**: ANOVA results, post-hoc tests, and interaction effects

#### RUN 7: Neighbor Encoding Effect
**Script**: `run7_neighbor_encoding_effect.py`  
**Analysis**: 
- One-sample t-tests to test if neighbor encoding effect (nghb-ef) > 0
- One-way ANOVA comparing neighbor encoding effect across conditions
- Post-hoc tests (if ANOVA p < 0.1)
- Both raw values and Fisher z-transformed values

**Output**: Neighbor encoding effect statistics and condition comparisons

#### RUN 8: Temporal Violation Rate
**Script**: `run8_temporal_violation_rate.py`  
**Analysis**: 
- One-way ANOVA comparing temporal violation rate (tv_rate) across conditions
- Tests if recall order preservation differs by condition

**Output**: ANOVA results for temporal violation rate

#### RUN 9: Memory Divergence and Semantic Influence Correlation
**Script**: `run9_memory_divergence_semantic_correlation.py`  
**Analysis**: 
- Correlates memory divergence with semantic influence scores
- Correlates choice divergence with semantic influence scores
- Analyzes both N=18 and N=100 Free participants (Romance story)

**Output**: Correlation results between divergence and semantic influence

#### RUN 10: Neighbor Encoding Effect Correlations
**Script**: `run10_neighbor_encoding_correlations.py`  
**Analysis**: 
- Correlates neighbor encoding effect with memory divergence (Romance: N=18, N=100)
- Correlates neighbor encoding effect with semantic influence (Adventure: N=22; Romance: N=18, N=100)
- Multiple linear regression predicting memory divergence from neighbor encoding effect and semantic influence (Romance, N=100 only)

**Output**: Correlation and regression results

#### RUN 11: Agency Denial Choice Events
**Script**: `run11_agency_denial_choice_events.py`  
**Analysis**: 
- Extracts recall for "want" vs "not-want" choice events for Yoked participants
- Compares Yoked "not-want" recall to Free "not-want" recall (matched by Yoked's want/not pattern)
- Two-sample t-test comparing denied choice events memory

**Output**: Mean recall vectors and t-test results for both stories

#### RUN 12: Permutation Test for Recall ISC
**Script**: `run12_permutation_test_recall_isc.py`  
**Analysis**: 
- Non-parametric permutation test for Recall ISC
- Randomly samples 18 Yoked and 18 Passive participants (one per story-path) 10,000 times
- Compares Free condition ISC to permutation distributions
- Performed for both 64 shared events and 49 non-choice events

**Output**: Permutation distributions (CSV), distribution plots, and p-values

#### RUN 13: Permutation Test with Matching Choice ISC
**Script**: `run13_permutation_test_matching_choice_isc.py`  
**Analysis**: 
- Permutation test matching Yoked participants to Free participants by Choice ISC
- Tests if reduced Recall ISC in Free condition persists when Choice ISC is matched
- 10,000 permutations

**Output**: Permutation distributions, plots, and statistical results

## Data Structure

### Data Files

All data files are located in the `data/` directory:

- **`adventure_data1.xlsx`**: Adventure story data (part 1)
  - Contains recall vectors (`rcl_free*`, `rcl_yoke*`, `rcl_pasv*` sheets)
  - Contains story maps (`story_map` sheet)
  - Contains summary data for individual subjects

- **`adventure_data2.xlsx`**: Adventure story data (part 2)
  - Contains centrality measures (`sem-ef`, `caus-ef`)
  - Contains neighbor encoding effects (`nghb-ef`)
  - Contains temporal violation rates (`tv_rate`)
  - Contains correlation data (`corr_free22` sheet)

- **`romance_data1.xlsx`**: Romance story data (part 1)
  - Contains recall vectors (`rcl_free*`, `rcl_yoke*`, `rcl_pasv*` sheets)
  - Contains story maps (`story_map` sheet)
  - Contains summary data for individual subjects

- **`romance_data2.xlsx`**: Romance story data (part 2)
  - Contains centrality measures (`sem-ef`, `caus-ef`)
  - Contains neighbor encoding effects (`nghb-ef`)
  - Contains temporal violation rates (`tv_rate`)
  - Contains correlation data (`corr_free18`, `corr_free100` sheets)
  - Contains ISC data (`merge_isc` sheet)

### Individual Subject Data

Individual subject event and recall files are located in:
- `data/individual_data/adventure/` - Adventure story individual files
- `data/individual_data/romance/` - Romance story individual files

## Output Files

Each analysis script generates output in its respective folder under `output/`:

- **Excel files** (`.xlsx`): Statistical results, pairwise correlations, detailed statistics
- **Text files** (`.txt`): Detailed reports with full statistical output
- **Image files** (`.png`): Plots and visualizations (where applicable)
- **CSV files** (`.csv`): Permutation test distributions (run12, run13)

### Comprehensive Report

The master script (`run_all_analyses.py`) generates a comprehensive HTML report:

**Location**: `output/comprehensive_report/comprehensive_analysis_report.html`

This report contains:
- Complete manuscript results section text
- Full statistics inserted at appropriate locations
- All 13 analyses organized by RUN number
- Formatted with proper styling for easy reading

## Key Statistical Methods

### Inter-Subject Correlation (ISC)
Pearson correlation between pairs of participants' vectors (recall performance or choice selections). Used to measure similarity across participants.

### Fisher's Z-Transformation
Applied to correlation coefficients to normalize the distribution for statistical testing:
```
z = 0.5 * ln((1 + r) / (1 - r))
```

### Permutation Tests
Non-parametric tests that generate null distributions by randomly sampling participants while maintaining story-path matching constraints.

### Repeated Measures ANOVA
Mixed-design ANOVA testing interactions between within-subjects (network type: semantic vs causal) and between-subjects (condition: Free vs Yoked vs Passive) factors.

## Citation

If you use this code, please cite:

> [Preprint Title]  
> Preprint: https://osf.io/preprints/psyarxiv/7evwj_v1  
> (Manuscript in revision)

## Notes

- **Manuscript Status**: The manuscript is currently in revision. These scripts generate the updated results section.
- **Data Availability**: Individual subject data files are required to run the analyses. Contact the authors for data access.
- **Reproducibility**: All scripts use fixed random seeds where applicable to ensure reproducibility.
- **Portability**: All scripts use relative paths and should work when the repository is downloaded and run from the root directory.

## Contact

For questions about the code or analyses, please contact the corresponding authors or open an issue on GitHub.

## License

[Specify license if applicable]

