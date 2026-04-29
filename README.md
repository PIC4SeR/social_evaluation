[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
# Social Evaluation

Utilities for extracting social-navigation experiment data from ROS 2 bags,
filtering trajectories, computing social metrics, plotting trajectories, and
generating benchmark reports.

## Installation

```bash
git clone ...
cd social_evaluation_graphs
pip install -e .
```

The editable install registers these console scripts:

- `bag_extraction`
- `data_filtering`
- `plot_trajectories`
- `metrics_evaluation`
- `benchmark_report`

The package pins the plotting stack used by the report generator
(`pandas>=1.5,<2.2`, `matplotlib>=3.6.3,<3.9`, `seaborn>=0.13,<0.14`). If you
already installed the package before a dependency change, reinstall it with
`pip install -e .`.

## Usage

The package is divided into different parts:

- Extract ROS 2 bag topics into pandas dataframes stored in pickle files.
- Filter robot and agent trajectories with EMA smoothing and Hampel outlier removal.
- Plot single-run and aggregated trajectory images or animations.
- Compute social-navigation metrics from extracted data.
- Generate benchmark tables, plots, statistical analyses, and LaTeX report drivers.

### Bag extraction and data conversion

- [Bag extraction](docs/bag_extraction.md)

### Filtering of the data

- [Data filtering](docs/data_filtering.md)

### Plotting of the animations of the trajectories

- [Trajectory plots](docs/trajectory_plots.md)

### Computation of the social evaluation metrics

- [Metrics evaluation](docs/metrics_evaluation.md)

### Benchmark report generation

- [Benchmark reports](docs/benchmark_report.md)

## Typical Pipeline

```bash
bag_extraction \
  --config-file config/config.yaml \
  --exp-config-file config/experiments_config.yaml \
  --base-output-path extracted_bags

data_filtering \
  --base-path extracted_bags \
  --base-output-dir filtered_bags

metrics_evaluation \
  --config-file config/config.yaml \
  --metrics-file config/metrics.yaml \
  --data-file filtered_bags/my_experiment.pickle \
  --output-file results/metrics.pickle

plot_trajectories \
  --config-file config/trajectory_plots.yaml \
  --pickle-base-path filtered_bags \
  --output-dir results/trajectories

benchmark_report \
  --config config/report_config.yaml \
  --base-output-dir results/report
```

## Failed-run Metrics

The report generator controls whether failed runs contribute to metric
summaries. By default these runs are discarded, but you can override this
globally or per scenario in the benchmark YAML:

```yaml
benchmark:
  discard_failed_metrics: true  # default; set to false to keep failed runs
  scenario:
    - name: my_scenario
      episodes: [...]
      discard_failed_metrics: false  # optional per-scenario override
```

When `discard_failed_metrics` is `false`, all recorded runs are used when computing metric tables, plots, ANOVA analyses, and rliable summaries.
