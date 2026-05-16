# nteprsm

`nteprsm` is a research codebase for Bayesian analysis of turfgrass visual rating data from the National Turfgrass Evaluation Program (NTEP). The repository combines:

- Stan models for latent-scale rating analysis
- Python utilities for preprocessing and post-processing
- configuration-driven local and HPC workflows
- notebooks for manuscript figures and parameter recovery studies
- manuscript assets in a separate Git submodule under `reports/manuscript2`

The scientific goal is to estimate latent turf quality while accounting for rater behavior, seasonality, and spatial heterogeneity in field trials.

## Scientific context

This project builds on item response theory and Gaussian process modeling to address a practical problem in NTEP data: turf quality is commonly recorded on an ordinal 1 to 9 visual scale, and those ratings are affected by both environmental variation and subjective rater behavior.

The current repository supports work on:

- latent-scale modeling of visual rating data
- annual seasonality estimation
- spatial effects in field plots
- parameter recovery experiments
- manuscript figure generation

## Publications

**A latent scale model to minimize subjectivity in the analysis of visual rating data for the National Turfgrass Evaluation Program**  
Yuanshuo Qu, Len Kne, Steve Graham, Eric Watkins, and Kevin Morris  
Frontiers in Plant Science, 2023, 14:1135918

Publication: https://www.frontiersin.org/articles/10.3389/fpls.2023.1135918/full

```bibtex
@article{qu2023latent,
   title     = {A latent scale model to minimize subjectivity in the analysis of visual rating data for the National Turfgrass Evaluation Program},
   author    = {Qu, Yuanshuo and Kne, Len and Graham, Steve and Watkins, Eric and Morris, Kevin},
   journal   = {Frontiers in Plant Science},
   volume    = {14},
   year      = {2023},
   publisher = {Frontiers Media SA}
}
```

## Repository layout

Key directories:

- `nteprsm/`: Python package with CLI entry points and shared utilities
- `models/`: Stan model files and older experimental model variants
- `config/`: YAML configuration files for model runs
- `data/`: local data, generated Stan output, and parameter recovery artifacts
- `notebooks/`: analysis and visualization notebooks
- `reports/manuscript1/`: manuscript 1 figures and tables
- `reports/manuscript2/`: manuscript 2 LaTeX source tracked as a Git submodule
- `tools/`: setup scripts

Important files:

- `nteprsm/model.py`: local config-driven Stan runner
- `nteprsm/model_umn.py`: HPC-oriented runner with explicit data and output arguments
- `nteprsm/utils.py`: data preprocessing, Stan data assembly, and posterior analysis helpers
- `settings.py`: repository-wide paths for data, logs, configs, models, and reports

## Data expectations

The main preprocessing pipeline expects CSV data with columns equivalent to:

- `rater`
- `date`
- `entry_code`
- `plot_code`
- `row`
- `col`
- `value`

`DataHandler.preprocess_data()` converts dates, derives rating event codes, normalizes the response to start at zero, and prepares plot-level and time-based structures for Stan.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/QhenryQ/nteprsm.git
cd nteprsm
git submodule update --init --recursive
```

The submodule step is required if you want the LaTeX manuscript in `reports/manuscript2`.

### 2. Install Python dependencies

The project targets Python 3.12 and uses Poetry.

Recommended:

```bash
poetry install
```

The repository also includes `tools/repo_setup.sh`, which attempts to:

- prompt for a shared data directory
- rewrite `settings.py` to point `DATA_DIR` at that directory
- verify Python 3.12
- install or adjust Poetry
- install project dependencies

Run it only if you are comfortable with a setup script modifying `settings.py` directly:

```bash
source tools/repo_setup.sh
```

### 3. Install CmdStan if needed

This repository uses `cmdstanpy`, so CmdStan must be available. If it is not already installed on your machine, follow the CmdStanPy installation guidance and install CmdStan before running the Stan workflows.

## Running the models

There are two main execution paths.

### Local config-driven workflow

`nteprsm/model.py` expects a YAML file that already contains:

- `data_path`
- `stan_file`
- `stan_additional_data`
- `sampling`

Example invocation:

```bash
python nteprsm/model.py path/to/config.yml
```

What happens in that workflow:

1. load YAML config
2. load and preprocess the input CSV
3. generate the Stan data dictionary
4. compile the Stan model
5. run MCMC and write CSV output to the configured output directory

### HPC workflow

`nteprsm/model_umn.py` is designed for cluster runs and takes the data path, working directory, and pickle output path as command-line arguments:

```bash
python nteprsm/model_umn.py \
  config/annual_seasonality_model.yml \
  data/raw/quality_nj2.csv \
  model_runs/seasonality_nj2_quality \
  fit_seasonality_nj2_quality.pkl
```

This wrapper is what the Slurm scripts use.

## Batch execution

Two Slurm job-array scripts are included:

- `submit_job_array_seasonality.sh`
- `submit_job_array_spatial.sh`

These scripts reflect the Minnesota Supercomputing Institute workflow and currently assume:

- Slurm is available
- `source activate nteprsm_env` works on the target system
- data paths like `kb2017/<location>/<trait>.csv` exist
- MSI-specific scheduler options are valid for your account

If you are not running on the original cluster environment, expect to edit these scripts.

## Notebooks and reports

The notebooks directory contains analysis workflows for:

- manuscript visualizations
- seasonality figures
- parameter recovery training and analysis
- exploratory plotting utilities

The most publication-oriented content lives in:

- `notebooks/manuscript1_visualizations.ipynb`
- `notebooks/manuscript2_visualizations_nj2.ipynb`
- `reports/manuscript1/`
- `reports/manuscript2/`

`reports/manuscript2/` is a separate Git repository included as a submodule. Update it with standard submodule commands when collaborating on the manuscript.

## Configuration files

The `config/` directory contains two kinds of YAML files:

- generic model configs such as `annual_seasonality_model.yml` and `spatial_model.yml`
- historical per-location configs such as `nteprsm_nj2kbg07.yml`

Be careful when reusing old configs. Some of the per-location configs still point to model filenames that are not present in the current `models/` directory.

## Current limitations

This is an active research repository rather than a polished distribution package. Known limitations include:

- `settings.py` still contains a machine-specific default `DATA_DIR`
- several workflows depend on external shared data that is not fully packaged in the repo
- some Slurm scripts are tied to a specific cluster environment
- parts of the posterior plotting code in `nteprsm/utils.py` appear to be older than the current data pipeline
- there is no automated test suite or CI workflow yet

## Practical recommendations for collaborators

- treat the YAML config files as the single source of truth for a model run
- verify data paths and Stan filenames before starting long sampling jobs
- use notebooks for analysis and figure generation, not as the only record of model assumptions
- keep manuscript editing isolated in `reports/manuscript2`
- expect to adapt local paths when moving between machines

## Citation and provenance

If you use this repository for a paper, presentation, or derivative project, cite the corresponding publication and record the exact Git commit or release tag used to generate your results.
