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

---

## Full Statistical Analysis Report

Below is the complete comprehensive statistical analysis report containing all results from analyses run1 through run13, including both raw and Fisher z-transformed statistics. This report reproduces the manuscript results section with full statistical details.

**Note**: The report is displayed below for easy viewing. You can also find the standalone HTML file at `output/comprehensive_report/comprehensive_analysis_report.html`.

<details open>
<summary><strong>Comprehensive Analysis Report (Click to collapse/expand)</strong></summary>

# Comprehensive Analysis Report: Agency Effects on Memory


*Generated: 2025-11-26 15:51:32*


*This report contains the manuscript main results section with full statistics inserted from analyses run1 through run13.*



---


## RUN 1: Agency did not improve recall performance



For each participant, events were binned according to whether they were remembered or forgotten.
Independent raters compared each sentence of recall to the story path read by the participant;
if any part of a given event was mentioned in any recall sentence, it was counted as remembered.

**Statistical Results:**

Adventure: F(2,113) = 1.433, p = 0.243

Romance: F(2,123) = 0.671, p = 0.513


There were no significant differences in recall performance across conditions in either story
(**Adventure: F(2,113)=1.43, p=0.243**;
**Romance: F(2,123)=0.67, p=0.513**;
*Supplementary Figure S3*). The Romance story additionally recorded
individual reading speed and showed no difference across the three agency conditions (ps> .3;
see **Supplement S3** for details). See **Supplement S4**
for details about memory for choice and non-choice events; See **Supplement S5**
for details about recognition memory performance; See **Supplement S6**
for details about memory for denied and granted choice events.






---


## RUN 2: Individual variability in recalled events



**Agency magnified individual variability in recall and choice.** The Romance story,
by design, had half of its events shared across all participants ("shared story sections"),
regardless of condition (*Figure 1A*). While participants made many
choices during these shared story sections, unbeknownst to them, all choice options led to the
same subsequent events. This allowed us to examine inter-participant variability in terms of
memory (which events were recalled) and choice behavior (which options were selected) when all
events were perfectly matched across participants, i.e., all participants read these events,
and the events were composed of identical text.



**Individual variability in recalled events.** A recall score (0 = Forgotten, 1 = Recalled;
see Methods for details) for each of the 64 events in the shared story sections was extracted for each
participant, composing a vector of recall performance (*Figure 2A*).
To assess the memory similarity across participants, we computed the inter-participant correlation (ISC),
i.e., the Pearson correlation between each pair of participants' recall performance vectors, i.e.,
"Recall ISC".

**64 Shared Events - One-sample t-tests (Raw values):**

FREE: mean r = 0.136, t(152) = 11.746, p = 0.000000

YOKE: mean r = 0.226, t(1377) = 58.288, p = 0.000000

PASV: mean r = 0.249, t(1484) = 70.871, p = 0.000000

**64 Shared Events - ANOVA (Raw values):**

F(2,3013) = 48.127, p = 0.000000

**49 Non-Choice Events - One-sample t-tests (Raw values):**

FREE: mean r = 0.157, t(152) = 13.212, p = 0.000000

YOKE: mean r = 0.219, t(1377) = 51.429, p = 0.000000

PASV: mean r = 0.229, t(1484) = 59.028, p = 0.000000

**49 Non-Choice Events - ANOVA (Raw values):**

F(2,3013) = 15.445, p = 0.000000

**64 Shared Events - One-sample t-tests (Fisher z-transformed):**

FREE: mean z = 0.140, t(152) = 11.672, p = 0.000000

YOKE: mean z = 0.235, t(1377) = 55.827, p = 0.000000

PASV: mean z = 0.260, t(1484) = 67.258, p = 0.000000

**64 Shared Events - ANOVA (Fisher z-transformed):**

F(2,3013) = 45.955, p = 0.000000

**49 Non-Choice Events - One-sample t-tests (Fisher z-transformed):**

FREE: mean z = 0.161, t(152) = 13.077, p = 0.000000

YOKE: mean z = 0.229, t(1377) = 49.029, p = 0.000000

PASV: mean z = 0.239, t(1484) = 55.878, p = 0.000000

**49 Non-Choice Events - ANOVA (Fisher z-transformed):**

F(2,3013) = 14.975, p = 0.000000


While events differed in their overall memorability, Recall ISC was significantly above zero
in all three conditions (**Romance: Free: mean r = 0.136, p
```
================================================================================
PERMUTATION TEST FOR RECALL ISC
================================================================================

Method: Non-parametric permutation test
- Randomly sampled 1 Yoked and 1 Passive participant from each of 18 story-paths
- Repeated 10,000 times to generate distributions
- Compared Free condition's Recall ISC to these distributions

Random seed: 42 (for reproducibility)

================================================================================
ANALYSIS: 64 Shared Events
================================================================================

Free condition mean ISC: 0.1364

Permutation distributions:
Yoked: Mean = 0.2183, Std = 0.0201, N = 10000
Passive: Mean = 0.2530, Std = 0.0180, N = 10000

Statistical tests:
Free vs Yoked: p = 0.000000 (proportion of Yoked samples below Free mean)
Result: Free ISC significantly lower than Yoked (p **Choice ISC Results (Raw values):**

FREE: mean r = 0.208, t(152) = 9.548, p = 0.000000

YOKE: mean r = 0.255, t(1377) = 36.193, p = 0.000000

Free vs Yoked: t = -2.105, p = 0.035

**Choice ISC Results (Fisher z-transformed):**

FREE: mean z = 0.233, t(152) = 9.133, p = 0.000000

YOKE: mean z = 0.289, t(1377) = 31.175, p = 0.000000

Free vs Yoked: t = -1.934, p = 0.053


Within-group Choice ISC was significantly above zero in both conditions (ps
```
================================================================================
PERMUTATION TEST FOR RECALL ISC WITH MATCHING CHOICE ISC
================================================================================

Method: Non-parametric permutation test with Choice ISC constraint
- Randomly sampled 1 Yoked participant from each of 18 story-paths
- Only kept samples where Yoked Choice ISC **Memory Divergence vs Choice Divergence Correlation:**

free18: r(16) = 0.405, p = 0.095

free100: r(98) = 0.296, p = 0.0028


Memory divergence and choice divergence were correlated with each other
(**Romance: r(16) = .405, p = .095**;
**r(98) = .296, p = .003**). In other words, the more idiosyncratic
their memory for shared events, the more idiosyncratic their choices.



Overall, these results support the idea that agency magnified individual variability in, i.e.,
personalized, both memory and choice behaviors. This effect was observed while all events were
held constant across conditions.






---


## RUN 5: Event recall was predicted by semantic and causal centrality



Narrative networks were computed for each unique story path following the methods of Lee & Chen.
For semantic narrative network analysis, each event was converted into an embedding vector using the
Universal Sentence Encoder (USE). Semantic centrality, a measure of how strongly interconnected a
given event was with other events in the narrative via shared meaning, was calculated for each event
by averaging its embedding cosine similarity with all other events in the story path. The effect of
semantic centrality (semantic influence) on memory was computed as the Pearson correlation between
semantic centrality and recall (an event-by-event vector of remembered=1, forgotten=0) for each participant.

**Semantic and Causal Centrality - One-sample t-tests (Raw values):**

Adventure FREE Semantic: t(21) = 5.582, p = 0.000015

Adventure FREE Causal: t(21) = 4.604, p = 0.000153

Adventure YOKE Semantic: t(44) = 16.071, p = 0.000000

Adventure YOKE Causal: t(44) = 5.453, p = 0.000002

Adventure PASV Semantic: t(48) = 12.681, p = 0.000000

Adventure PASV Causal: t(48) = 4.922, p = 0.000011

Romance FREE Semantic: t(17) = 6.109, p = 0.000012

Romance FREE Causal: t(17) = 8.613, p = 0.000000

Romance YOKE Semantic: t(52) = 19.021, p = 0.000000

Romance YOKE Causal: t(52) = 14.398, p = 0.000000

Romance PASV Semantic: t(54) = 20.003, p = 0.000000

Romance PASV Causal: t(54) = 23.988, p = 0.000000

**Semantic and Causal Centrality - One-sample t-tests (Fisher z-transformed):**


For both stories, semantic centrality significantly predicted recall, i.e., a significant semantic
influence on memory was observed, in all three conditions (ps **Semantic Centrality - One-way ANOVA (Raw values):**

Adventure: F(2.000,113) = 3.036, p = 0.052

Romance: F(2.000,123) = 11.455, p = 0.000027

**Semantic Centrality - One-way ANOVA (Fisher z-transformed):**

Adventure: F(2.000,113) = 2.870, p = 0.061

Romance: F(2.000,123) = 11.343, p = 0.000030

**Causal Centrality - One-way ANOVA (Raw values):**

Adventure: F(2.000,113) = 0.864, p = 0.424

Romance: F(2.000,123) = 0.638, p = 0.530

**Causal Centrality - One-way ANOVA (Fisher z-transformed):**

Adventure: F(2.000,113) = 0.809, p = 0.448

Romance: F(2.000,123) = 0.562, p = 0.572

**Repeated Measures ANOVA - Interaction (Raw values):**

Adventure: F(2.000,113) = 3.272, p = 0.042

Romance: F(2.000,123) = 2.990, p = 0.054

**Repeated Measures ANOVA - Interaction (Fisher z-transformed):**

Adventure: F(2.000,113) = 3.175, p = 0.046

Romance: F(2.000,123) = 3.079, p = 0.050


For both stories, we observed significant differences across conditions, wherein Free had lower
semantic influence on memory compared to Yoked and Passive
(**Adventure: F(2,113) = 3.04, p = 0.052**;
**Romance: F(2,123) = 11.46, p **Neighbor Encoding Effect - One-sample t-tests (Raw values):**

FREE: t(21.000) = 6.103, p = 0.000005

YOKE: t(44.000) = 6.193, p = 0.000000

PASV: t(48.000) = 8.714, p = 0.000000

FREE: t(17.000) = 7.589, p = 0.000001

YOKE: t(52.000) = 7.409, p = 0.000000

PASV: t(54.000) = 6.426, p = 0.000000

**Neighbor Encoding Effect - One-sample t-tests (Fisher z-transformed):**

FREE: t(21.000) = 5.625, p = 0.000014

YOKE: t(44.000) = 6.030, p = 0.000000

PASV: t(48.000) = 8.421, p = 0.000000

FREE: t(17.000) = 7.216, p = 0.000001

YOKE: t(52.000) = 7.095, p = 0.000000

PASV: t(54.000) = 6.306, p = 0.000000

**Neighbor Encoding Effect - One-way ANOVA (Raw values):**

Adventure: F(2.000,113.000) = 0.368, p = 0.693

Romance: F(2.000,123.000) = 12.073, p = 0.000016

**Neighbor Encoding Effect - One-way ANOVA (Fisher z-transformed):**

Adventure: F(2.000,113.000) = 0.400, p = 0.671

Romance: F(2.000,123.000) = 12.556, p = 0.000011


The neighbor encoding effect was positive in all three conditions for both stories
(Adventure and Romance, ps **Temporal Violation Rate - One-way ANOVA:**

Adventure: F(2,113) = 2.564, p = 0.081

Romance: F(2,123) = 0.096, p = 0.908


For each participant, recall was divided into segments (brief sentences). We counted the number of
times that a recall segment referred to an event that occurred earlier in the story than the events
referred to by the previous recall segment; this was then divided by the participant's total number of
recall segments. There was no significant difference in temporal violation rate across conditions in
either story (**Adventure: F(2,113) = 2.56, p = .081**;
**Romance: F(2,123) = 0.10, p = .913**). In general, temporal order was
remarkably well-preserved, with low temporal violation rates in all conditions
(*Figure 4C-D*).






---


## RUN 9: Greater memory divergence was associated with weaker semantic influence



We next examined how memory divergence scores (*Figure 2*) were related
to a) the impact of semantic centrality on recall (*Figure 3*) and b) the
neighbor encoding effect (*Figure 4*). Each Free participant's semantic
influence score was obtained by calculating the Pearson correlation between semantic centrality and memory
performance (same as shown in *Figure 3*).



Memory divergence was negatively correlated with semantic influence scores in Free participants. In other
words, the more a participant's recall deviated from other participants in the group, the weaker the effect
of semantic centrality on their recall.

**Memory Divergence vs Semantic Influence:**

N=18: r(16) = -0.519, p = 0.027

N=100: r(98) = -0.460, p = 0.000001


This was true when including only the 18 Free participants who had yoked counterparts
(**Romance: r(16) = -.519, p = 0.027**) as well as when using the full sample
(**Romance: r(98) = -.460, p **Neighbor Encoding Effect Correlations:**

Adventure (N=22): Neighbor Encoding Effect vs Semantic Influence: r(20) = -0.348, p = 0.113

Romance (N=18): Neighbor Encoding Effect vs Memory Divergence: r(16) = 0.496, p = 0.036

Romance (N=18): Neighbor Encoding Effect vs Semantic Influence: r(16) = -0.324, p = 0.189

Romance (N=100): Neighbor Encoding Effect vs Memory Divergence: r(98) = 0.304, p = 0.0021

Romance (N=100): Neighbor Encoding Effect vs Semantic Influence: r(98) = -0.347, p = 0.000402

**Multiple Linear Regression (Romance, N=100):**

Neighbor Encoding Effect: p = 0.087

Semantic Influence: p = 0.000047


The neighbor encoding effect was negatively correlated with semantic influence scores in Free participants,
across both stories (**Adventure: r(22) = -.348, p = .113**.
**Romance: r(16) = -.324, p = .189**;
**r(98)= -.347, p **Agency Denial Effect - Two-sample t-test:**

Adventure: p = 0.015

Romance: p = 0.017

(**Adventure: p = 0.015**;
**Romance: p = 0.017**, two-sample t-test); individual differences contribute
to a variation of tendency to selective recall or forget the denied choice events (see
**Supplement S6** for details).



The percentage of choices granted in the Yoked participants was not predictive of individual's recall performance,
recall similarity to their Free and Passive condition counterparts, semantic and causal centrality effects on memory,
nor their neighbor encoding effects (all ps>.3); however, higher percentage of choices granted predicted greater
individual tendency to forget the choice-denied events (Adventure: r(44) = .335, p = .026. Romance: r(51) = .137,
p = .337; see **Supplement S7** for details).



Together, these results suggest that in a context lacking full agentive control, perceived agency and their
effects on memory could vary across individuals in non-systematic ways. The one exception is that with more control
in such agency-uncertain contexts, the more one has reduced recall for the agency-denied events.
</details>
