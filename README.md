# choose-your-own-adventure (cyoa)

Data associated with the manuscript **"Agency personalizes episodic memories"**

**Authors**: Xian Li, Nicole Kim Ni, Savannah Born, Ria Gualano, Iris Lee, Buddhika Bellana, and Janice Chen

**Preprint**: [https://osf.io/preprints/psyarxiv/7evwj_v1](https://osf.io/preprints/psyarxiv/7evwj_v1)

**Status**: Manuscript currently under revision at Nature Communications

---

## Overview

This repository contains the complete codebase, data, and supplementary materials for the study examining how agency (the ability to make choices) affects memory for interactive narratives. The study used "Choose Your Own Adventure" (CYOA) interactive narratives to examine how having control over story choices affects memory encoding and recall, compared to passive viewing or yoked control conditions.

This repository provides comprehensive resources for understanding, reproducing, and extending the research on agency effects on memory.

## Repository Structure

```
cyoa/
├── README.md                    # This file - comprehensive repository guide
│
├── stats_code/                  # Statistical analysis scripts (manuscript results reproduction)
│   ├── README.md               # Detailed analysis documentation
│   ├── run_all_analyses.py     # Master script to run all 13 analyses
│   ├── run1_*.py ... run13_*.py # Individual analysis scripts
│   ├── data/                   # Analysis-ready data files
│   └── output/                 # Analysis results and reports
│
├── data/                        # Cleaned, tidied individual subject data
│   ├── README.md               # Data structure documentation
│   ├── adv_free.xlsx           # Adventure story, Free condition (N=22)
│   ├── adv_yoke.xlsx           # Adventure story, Yoked condition (N=45)
│   ├── adv_pasv.xlsx           # Adventure story, Passive condition (N=49)
│   ├── rom_free.xlsx           # Romance story, Free condition (N=100, selected N=18)
│   ├── rom_yoke.xlsx           # Romance story, Yoked condition (N=53)
│   ├── rom_pasv.xlsx           # Romance story, Passive condition (N=55)
│   ├── subjects_demographics.xlsx # Participant demographics
│   └── ratings/                # Raw rating files per subject
│       ├── adv_free/           # Individual subject ratings (Adventure, Free)
│       ├── adv_yoke/           # Individual subject ratings (Adventure, Yoked)
│       ├── adv_pasv/           # Individual subject ratings (Adventure, Passive)
│       ├── rom_free/           # Individual subject ratings (Romance, Free)
│       ├── rom_yoke/           # Individual subject ratings (Romance, Yoked)
│       └── rom_pasv/           # Individual subject ratings (Romance, Passive)
│
├── cyoa_story/                  # Stimuli: Story maps and structure
│   ├── README.md               # Story structure documentation
│   ├── adv_story_map.xlsx      # Adventure story event map
│   └── rom_story_map.xlsx      # Romance story event map
│
├── cyoa_demo/                   # Study presentation: Interactive demo files
│   ├── README.md               # Demo documentation
│   ├── adv_free_demo.html      # Adventure story, Free condition demo
│   ├── adv_pasv_demo.html      # Adventure story, Passive condition demo
│   ├── adv_yoke_demo.html      # Adventure story, Yoked condition demo
│   ├── rom_free_demo.html      # Romance story, Free condition demo
│   ├── rom_pasv_demo.html      # Romance story, Passive condition demo
│   └── rom_yoke_demo.html      # Romance story, Yoked condition demo
│
├── figures/                     # Figure generation code and data
│   ├── README.md               # Figure documentation
│   ├── 1_plot_bars.ipynb       # Bar plot generation
│   ├── 2_plot_scatters.ipynb   # Scatter plot generation
│   ├── 3_plot_violin.ipynb     # Violin plot generation
│   ├── 4_plot_line_raster_dist.ipynb # Line/raster/distribution plots
│   ├── 5_plot_mat.ipynb        # Matrix/heatmap plots
│   ├── 6_plot_chatgpt4agent.ipynb # ChatGPT-4 agent analysis plots
│   ├── 7_plot_network_graphs.ipynb # Network graph visualizations
│   └── xianfunc.py             # Custom plotting utilities
│
├── instruct-raters/             # Instructions for human raters
│   ├── README.md               # Rater instructions documentation
│   ├── event-segment.docx      # Event segmentation instructions
│   ├── rate-recall.docx        # Recall rating instructions
│   ├── rate-causal.docx        # Causal rating instructions
│   └── chatGPT4_prompt_rate-causal.docx # ChatGPT-4 prompt for causal ratings
│
└── supplement/                  # Supplementary materials
    ├── README.md               # Supplementary materials documentation
    ├── chatGPT/                # ChatGPT-4 agent analysis materials
    │   ├── chatgpt4agent_result.xlsx # ChatGPT-4 agent results
    │   ├── gpt4_causal-rating/ # GPT-4 causal ratings
    │   ├── human_causal-rating/ # Human causal ratings
    │   └── narratives/         # Narrative event files
    ├── rom_test-recognition.xlsx # Recognition test data (Romance)
    └── sem-caus_dis-corr.xlsx  # Semantic-causal distance correlation data
```

## Key Resources

### 1. `stats_code/` - Manuscript Results Reproduction

**Purpose**: Complete statistical analysis pipeline that reproduces all results reported in the manuscript's Results section.

**Contents**:
- **13 main analysis scripts** (`run1_*.py` through `run13_*.py`) covering:
  - Overall recall comparison across conditions
  - Individual variability in recalled events (Recall ISC)
  - Individual variability in choices made (Choice ISC)
  - Divergence from group patterns
  - Semantic and causal centrality effects
  - Agency effects on centrality
  - Neighbor encoding effects
  - Temporal violation rates
  - Memory divergence correlations
  - Agency denial effects
  - Permutation tests
- **Supporting utilities**: Data loading, ISC correlation analysis, data structure management
- **Master script**: `run_all_analyses.py` runs all analyses and generates a comprehensive HTML report
- **Output**: Statistical results, pairwise correlations, permutation distributions, and a complete HTML report

**See**: `stats_code/README.md` for detailed documentation of each analysis.

### 2. `data/` - Cleaned Individual Subject Data

**Purpose**: Cleaned, tidied data files containing individual subjects' story paths and recall performance.

**Contents**:
- **Main data files** (`.xlsx`): Aggregated data for each story (Adventure/Romance) and condition (Free/Yoked/Passive)
  - Story paths: Which events each subject experienced
  - Recall performance: Which events each subject recalled
  - Centrality measures: Semantic and causal influence scores
  - Neighbor encoding effects
  - Temporal violation rates
- **Individual subject files** (`ratings/` subdirectories):
  - Event files: Individual subject's experienced events
  - Rate-recall files: Individual subject's recall ratings
  - Rate-causal files: Individual subject's causal ratings
- **Demographics**: `subjects_demographics.xlsx` contains participant demographic information

**Sample sizes**:
- Adventure: Free (N=22), Yoked (N=45), Passive (N=49)
- Romance: Free (N=100, with selected N=18 for matching), Yoked (N=53), Passive (N=55)

**See**: `data/README.md` for data structure details.

### 3. `cyoa_story/` - Experimental Stimuli

**Purpose**: The story maps and structure files that define the interactive narratives used as experimental stimuli.

**Contents**:
- **Story maps** (`.xlsx`): Complete event structures for both stories
  - `adv_story_map.xlsx`: Adventure story event map
  - `rom_story_map.xlsx`: Romance story event map
- These files define the branching narrative structure, including:
  - All possible events
  - Choice points and their options
  - Event sequences and dependencies
  - Story paths through the narrative

**See**: `cyoa_story/README.md` for story structure documentation.

### 4. `cyoa_demo/` - Study Presentation

**Purpose**: Interactive HTML demo files that demonstrate how the Choose Your Own Adventure stories were presented to participants during the study.

**Contents**:
- **Demo files** (`.html`): Interactive presentations for each story and condition
  - Free condition: Participants make choices
  - Yoked condition: Participants receive choices made by Free participants
  - Passive condition: Participants view a predetermined path
- These demos allow users to experience the experimental paradigm and understand how choices were presented and how story paths unfolded.

**See**: `cyoa_demo/README.md` for demo documentation.

### 5. `figures/` - Figure Generation Code

**Purpose**: Jupyter notebooks and data files for generating all figures in the manuscript and supplementary materials.

**Contents**:
- **Plotting notebooks**:
  - Bar plots, scatter plots, violin plots
  - Line plots, raster plots, distribution plots
  - Matrix/heatmap visualizations
  - Network graph visualizations
  - ChatGPT-4 agent analysis visualizations
- **Data files**: Preprocessed data for each figure type
- **Utilities**: Custom plotting functions (`xianfunc.py`)

**See**: `figures/README.md` for figure documentation.

### 6. `instruct-raters/` - Rater Instructions

**Purpose**: Documentation and instructions provided to human raters who performed event segmentation, recall ratings, and causal ratings.

**Contents**:
- **Event segmentation**: Instructions for identifying and segmenting story events
- **Recall rating**: Instructions for rating recall performance
- **Causal rating**: Instructions for rating causal relationships between events
- **ChatGPT-4 prompts**: Prompts used for automated causal ratings with GPT-4

These materials enable replication of the rating procedures and understanding of how events and relationships were coded.

**See**: `instruct-raters/README.md` for detailed instructions.

### 7. `supplement/` - Supplementary Materials

**Purpose**: Additional analyses and materials not included in the main manuscript but relevant for understanding the research.

**Contents**:
- **ChatGPT-4 agent analysis**: 
  - Results from using ChatGPT-4 as an agent to experience the narratives
  - Comparison of GPT-4 and human causal ratings
  - Narrative event files for GPT-4 analysis
- **Recognition test data**: Recognition memory data for Romance story
- **Semantic-causal correlations**: Distance correlation analyses

**See**: `supplement/README.md` for supplementary materials documentation.

## Quick Start Guide

### Reproducing Manuscript Results

1. **Navigate to the stats_code directory**:
   ```bash
   cd stats_code
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy scipy openpyxl matplotlib seaborn statsmodels scikit-learn joblib
   ```

3. **Run all analyses**:
   ```bash
   python run_all_analyses.py
   ```

   This will:
   - Execute all 13 analysis scripts
   - Generate statistical outputs
   - Create a comprehensive HTML report with all results

4. **View results**: Check the `stats_code/output/` directory for:
   - Individual analysis outputs
   - Comprehensive HTML report: `output/comprehensive_report/comprehensive_analysis_report.html`

### Exploring the Data

1. **Main data files**: Start with the `.xlsx` files in `data/` for aggregated data
2. **Individual data**: Explore `data/ratings/` for subject-level files
3. **Story structure**: Examine `cyoa_story/*.xlsx` to understand the narrative structure

### Trying the Demos

Open any `.html` file in `cyoa_demo/` in a web browser to experience the interactive narratives as participants did.

### Generating Figures

1. **Navigate to figures directory**:
   ```bash
   cd figures
   ```

2. **Open Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```

3. **Run notebooks** to generate figures (data files are included)

## Data Availability

All data files are provided in this repository:
- **Cleaned data**: `data/*.xlsx` files contain the main aggregated datasets
- **Individual data**: `data/ratings/` contains subject-level files
- **Analysis-ready data**: `stats_code/data/` contains data formatted for analyses

## Code Availability

All analysis code is provided:
- **Statistical analyses**: `stats_code/` contains all 13 analysis scripts
- **Figure generation**: `figures/` contains all plotting code
- **Utilities**: Supporting functions for data loading and analysis

Additional Information
- **Reproducibility check**: An independent research assistant has successfully reproduced the results reported in this manuscript by following the methods and analysis procedures described herein. The reproduction repository is available at [SissiLai361/Reproduce](https://github.com/SissiLai361/Reproduce)

## Citation

If you use this code or data, please cite:

> Li, X., Ni, N. K., Born, S., Gualano, R., Lee, I., Bellana, B., & Chen, J. (preprint). Agency personalizes episodic memories. *PsyArXiv*. https://osf.io/preprints/psyarxiv/7evwj_v1  
> (Manuscript under revision at Nature Communications)

## Contact

For questions about the code, data, or analyses, please:
- Open an issue on GitHub
- Contact via email: xianl.cogneuro@gmail.com

## License

This project is licensed under the MIT License.

## Notes

- **Manuscript Status**: The manuscript is currently under revision at Nature Communications
- **Reproducibility**: All scripts use fixed random seeds where applicable
- **Data Organization**: Data files are organized by story (Adventure/Romance) and condition (Free/Yoked/Passive)
- **Documentation**: Each major directory contains its own README with specific details
