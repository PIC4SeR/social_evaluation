# Social evaluation

## Installation

```bash
git clone ...
cd social_evaluation
pip install -e . # to install the package in editable mode
```

## Usage

The package is divided into different parts:

- The extraction of the data from the rosbags and the conversion of the data into pandas dataframes stored in pickle files
- The filtering of the data to remove the outliers
- The plotting of animations of the trajectories of the agents
- The computation of the social evaluation metrics
- The plotting of the social evaluation metrics

### Bag extraction and data conversion

- [Conversion of the data](docs/bag_extraction.md)

### Filtering of the data

- [Filtering of the data](docs/data_filtering.md)

### Plotting of the animations of the trajectories

- [Plotting of the animations of the trajectories](docs/trajectory_plots.md)

### Computation of the social evaluation metrics

- [Computation of the social evaluation metrics](docs/metrics_evaluation.md)

### Plotting and elaboration of the social evaluation metrics

- [Plotting and elaboration of the social evaluation metrics](docs/metrics_visualization.md)

### Benchmark report configuration

The report generator in `social_evaluation/benchmark_report_.py` now exposes a flag to control whether failed runs contribute to the metric summaries. By default these runs are discarded (matching the previous behaviour), but you can override this globally or per-scenario in your YAML configuration:

```yaml
benchmark:
  discard_failed_metrics: true  # default; set to false to keep failed runs
  scenario:
    - name: my_scenario
      episodes: [...]
      discard_failed_metrics: false  # optional per-scenario override
```

When `discard_failed_metrics` is `false`, all recorded runs are used when computing metric tables, plots, ANOVA analyses, and rliable summaries.
