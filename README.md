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

- [Conversion of the data](social_evaluation/bag_extraction/README.md)

### Filtering of the data

- [Filtering of the data](social_evaluation/data_filtering/README.md)

### Plotting of the animations of the trajectories

- [Plotting of the animations of the trajectories](social_evaluation/trajectory_plots/README.md)

### Computation of the social evaluation metrics

- [Computation of the social evaluation metrics](social_evaluation/metrics_evaluation/README.md)

### Plotting and elaboration of the social evaluation metrics

- [Plotting and elaboration of the social evaluation metrics](social_evaluation/metrics_visualization/README.md)