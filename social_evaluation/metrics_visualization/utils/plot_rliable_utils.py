# %%
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

import collections
import numpy as np
import matplotlib.patches as patches
import sys
import pickle
import pandas as pd
import copy
import functools
import json
import os

import itertools as it
import random
import inspect
import scipy.stats

from pathlib import Path
import getpass
import os.path as osp

# See warnings only once
import warnings

warnings.filterwarnings("default")
import inflection
from functools import partial

from typing import List, Dict, Any, Tuple, Union

# The answer to life, universe and everything
RAND_STATE = np.random.RandomState(42)

EXPERIMENTS = ["DWA", "DWB", "MPPI"]  # List of experiments
# TODO Add the path to the metrics file
# TODO Retrieve experiment names from the metrics file

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("white")

# Matplotlib params
from matplotlib import rcParams
from matplotlib import rc

rcParams["legend.loc"] = "best"
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42

rc("text", usetex=False)
rc("font", family="roman")


def set_axes(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=14)


def set_ticks(ax, xticks, xticklabels, yticks, yticklabels):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


def decorate_axis(ax, wrect=10, hrect=10, labelsize="large"):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


def load_metrics(file: Path):
    with open(file, "rb") as f:
        metrics = pickle.load(f)
    return metrics


def separate_metrics(results: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame]]:
    metrics_list = results.columns
    metrics_list = [
        x for x in metrics_list if x not in ["experiment", "algorithm", "run"]
    ]
    metrics_dict = {}
    for metric in metrics_list:
        metrics_dict[metric] = results[[metric, "algorithm", "experiment", "run"]]
    return metrics_dict


def pgs(f):
    print(inspect.getsource(f))


def score_normalization(res_dict, min_scores, max_scores):
    experiments = res_dict.keys()
    norm_scores = {}
    for experiment, scores in res_dict.items():
        norm_scores[experiment] = (scores - min_scores[experiment]) / (
            max_scores[experiment] - min_scores[experiment]
        )
    return norm_scores


def convert_to_matrix(experiment_dict):
    keys = sorted(list(experiment_dict.keys()))
    return np.stack([experiment_dict[k] for k in keys], axis=1)


def plot_score_hist(
    score_matrix,
    bins=20,
    figsize=(28, 14),
    fontsize="xx-large",
    N=6,
    extra_row=1,
    names=None,
):
    num_tasks = score_matrix.shape[1]
    if names is None:
        names = EXPERIMENTS
    N1 = (num_tasks // N) + extra_row
    fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
    for i in range(N):
        for j in range(N1):
            idx = j * N + i
            if idx < num_tasks:
                ax[j, i].set_title(names[idx], fontsize=fontsize)
                sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j, i], kde=True)
            else:
                ax[j, i].axis("off")
            decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize="xx-large")
            ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
            if idx % N == 0:
                ax[j, i].set_ylabel("Count", size=fontsize)
            else:
                ax[j, i].yaxis.label.set_visible(False)
            ax[j, i].grid(axis="y", alpha=0.1)
    return fig


StratifiedBootstrap = rly.StratifiedBootstrap

IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)


def load_metrics(file: Path):
    with open(file, "rb") as f:
        metrics = pickle.load(f)
    return metrics


def separate_metrics(results: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame]]:
    metrics_list = results.columns
    metrics_list = [
        x for x in metrics_list if x not in ["experiment", "algorithm", "run"]
    ]
    metrics_dict = {}
    for metric in metrics_list:
        metrics_dict[metric] = results[[metric, "algorithm", "experiment", "run"]]
    return metrics_dict


def load_normalization_scores(random_scores_file: Path, human_scores_file: Path):
    with open(random_scores_file, "rb") as f:
        random_scores = json.load(f)
    with open(human_scores_file, "rb") as f:
        human_scores = json.load(f)
    return random_scores, human_scores


def create_metrics_dict_social_env(
    main_df,
    normalization=True,
    evaluation_key="eval_average_return",
    normalization_scores=load_normalization_scores(
        Path("random_scores.json"), Path("human_scores.json")
    ),
):
    """Creates a dictionary of scores."""
    metrics_dict = {}
    for key, df in main_df.items():
        metrics_dict[key] = df[evaluation_key].values
    if normalization:
        metrics_dict = score_normalization(metrics_dict, *normalization_scores)
    return metrics_dict


def load_metrics_normalized(algorithm_name, metric, metrics_dict, normalization_scores):
    df = metrics_dict[metric]
    normalized_metrics = {}
    random_scores, human_scores = normalization_scores
    for experiment in df["experiment"].unique():
        normalized_metrics[experiment] = df[
            (df["algorithm"] == algorithm_name) & (df["experiment"] == experiment)
        ][metric].values
    normalized_metrics = score_normalization(
        normalized_metrics, random_scores, human_scores
    )
    normalized_metrics_matrix = convert_to_matrix(normalized_metrics)
    median, mean = MEDIAN(normalized_metrics_matrix), MEAN(normalized_metrics_matrix)
    print("{}: Median: {}, Mean: {}".format(metric, median, mean))
    return normalized_metrics, normalized_metrics_matrix


def subsample_scores(metrics_dict, n=5, replace=False):
    subsampled_dict = {}
    total_samples = len(metrics_dict[list(metrics_dict.keys())[0]])
    for experiment, scores in metrics_dict.items():
        indices = np.random.choice(range(total_samples), size=n, replace=replace)
        subsampled_dict[experiment] = scores[indices]
    return subsampled_dict


def subsample_scores_mat(score_mat, num_samples=5, replace=False):
    subsampled_dict = []
    total_samples, num_experiments = score_mat.shape
    subsampled_scores = np.empty((num_samples, num_experiments))
    for i in range(num_experiments):
        indices = np.random.choice(total_samples, size=num_samples, replace=replace)
        subsampled_scores[:, i] = score_mat[indices, i]
    return subsampled_scores


def subsample_seeds(score_mat, num_samples=5, replace=False):
    indices = np.random.choice(score_mat.shape[0], size=num_samples, replace=replace)
    return score_mat[indices]


def batch_subsample_seeds(score_mat, num_samples=5, batch_size=100, replace=False):
    indices = [
        np.random.choice(score_mat.shape[0], size=num_samples, replace=replace)
        for _ in range(batch_size)
    ]
    return (score_mat[idx] for idx in indices)


def subsample_scores_mat_with_replacement(score_mat, num_samples=5):
    subsampled_dict = []
    total_samples, num_experiments = score_mat.shape
    indices = np.random.choice(
        total_samples, size=(num_samples, num_experiments), replace=True
    )
    col_indices = np.expand_dims(np.arange(num_experiments), axis=0)
    col_indices = np.repeat(col_indices, num_samples, axis=0)
    subsampled_scores = score_mat[indices, col_indices]
    return subsampled_scores


SIZES = [3, 5, 10, 25, 50, 100]


def calc_aggregate_fn(
    score_data, num_samples=5, total_n=20000, aggregate_fn=MEDIAN, replace=False
):
    subsampled_scores = batch_subsample_seeds(
        score_data, num_samples, batch_size=total_n, replace=replace
    )
    aggregates = [aggregate_fn(scores) for scores in subsampled_scores]
    return np.array(aggregates)


def calculate_aggregate_varying_sizes(
    score_matrix, aggregate_fn, total_n=20000, sizes=None, replace=False
):
    agg_dict = {}
    if sizes is None:
        sizes = SIZES
    for size in sizes:
        agg_dict[n] = calc_aggregate_fn(
            score_matrix,
            num_samples=size,
            aggregate_fn=aggregate_fn,
            total_n=total_n,
            replace=replace,
        )
        print("Mean Aggregate: {}".format(np.mean(agg_dict[n])))
    return agg_dict


def CI(bootstrap_dist, stat_val=None, alpha=0.05, is_pivotal=False):
    """
    Get the bootstrap confidence interval for a given distribution.
    Args:
      bootstrap_distribution: numpy array of bootstrap results.
      stat_val: The overall statistic that this method is attempting to
        calculate error bars for. Default is None.
      alpha: The alpha value for the confidence intervals.
      is_pivotal: if true, use the pivotal (reverse percentile) method.
        If false, use the percentile method.
    Returns:
      (low, high): The lower and upper limit for `alpha` x 100% CIs.
      val: The median value of the bootstrap distribution if `stat_val` is None
        else `stat_val`.
    """
    # Adapted from https://pypi.org/project/bootstrapped
    if is_pivotal:
        assert stat_val is not None, "Please pass the statistic for a pivotal"
        "confidence interval"
        low = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.0))
        val = stat_val
        high = 2 * stat_val - np.percentile(bootstrap_dist, 100 * (alpha / 2.0))
    else:
        low = np.percentile(bootstrap_dist, 100 * (alpha / 2.0))
        val = np.percentile(bootstrap_dist, 50)
        high = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2.0))
    return (low, high), val


# %%
# @title Aggregates on Atari 100K (with 10 runs)


aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    experiments_path_length_dict, aggregate_func, reps=50000
)

# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ', 'DrQ(ε)', 'SPR']
algorithms = ["DWA", "DWB", "MPPI"]
fig, axes = plot_utils.plot_interval_estimates(
    aggregate_scores,
    aggregate_interval_estimates,
    metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
    algorithms=algorithms,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel_y_coordinate=-0.5,
    xlabel="Human Normalized Score",
)
plt.show()
# save_fig(fig, 'atari_100k_aggregates')

# %%
# @title DER Scores Distribution plots

# fig = plot_score_hist(path_length_dwa, bins=20, N=2, figsize=(26, 11))
# fig.subplots_adjust(hspace=0.85, wspace=0.17)
# plt.show()

# %%
from scipy.stats.stats import find_repeats

# @title Calculate score distributions and average score distributions for for Atari 100k

# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR']
algorithms = ["DWA", "DWB", "MPPI"]
# score_dict = {key: score_data_dict[key][:10] for key in algorithms}
score_dict = {key: experiments_path_length_dict[key] for key in algorithms}
EXPERIMENTS_TAU = np.linspace(0.0, 2.0, 201)
# Higher value of reps corresponds to more accurate estimates but are slower
# to computed. `reps` corresponds to number of bootstrap resamples.
reps = 2000

score_distributions, score_distributions_cis = rly.create_performance_profile(
    score_dict, EXPERIMENTS_TAU, reps=reps
)
avg_score_distributions, avg_score_distributions_cis = rly.create_performance_profile(
    score_dict, EXPERIMENTS_TAU, use_score_distribution=False, reps=reps
)

# %%
# @title Plot performance profiles (score distributions)

fig, ax = plt.subplots(ncols=1, figsize=(7.25, 4.7))

plot_utils.plot_performance_profiles(
    score_distributions,
    EXPERIMENTS_TAU,
    performance_profile_cis=score_distributions_cis,
    # colors=EXPERIMENTS_COLOR_DICT,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel=r"Human Normalized Score $(\tau)$",
    labelsize="xx-large",
    ax=ax,
)

ax.axhline(0.5, ls="--", color="k", alpha=0.4)
# fake_patches = [mpatches.Patch(color=EXPERIMENTS_COLOR_DICT[alg],
#  alpha=0.75) for alg in algorithms]
fake_patches = [
    mpatches.Patch(color=EXPERIMENTS_COLOR_DICT[alg], label=alg) for alg in algorithms
]
legend = fig.legend(
    fake_patches,
    algorithms,
    loc="upper center",
    fancybox=True,
    ncol=len(algorithms),
    fontsize="x-large",
    bbox_to_anchor=(0.6, 1.1),
)

# %%
# @title Plot performance profiles (score distributions) and contrast with average score distribution
fig, axes = plt.subplots(ncols=2, figsize=(14.5, 4.7))

plot_utils.plot_performance_profiles(
    score_distributions,
    EXPERIMENTS_TAU,
    performance_profile_cis=score_distributions_cis,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel=r"Human Normalized Score $(\tau)$",
    labelsize="xx-large",
    ax=axes[0],
)


plot_utils.plot_performance_profiles(
    avg_score_distributions,
    EXPERIMENTS_TAU,
    performance_profile_cis=avg_score_distributions_cis,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel=r"Human Normalized Score $(\tau)$",
    ylabel=r"Fraction of tasks with score > $\tau$",
    labelsize="xx-large",
    ax=axes[1],
)

axes[0].axhline(0.5, ls="--", color="k", alpha=0.4)
axes[1].axhline(0.5, ls="--", color="k", alpha=0.4)

fake_patches = [
    mpatches.Patch(color=EXPERIMENTS_COLOR_DICT[alg], alpha=0.75) for alg in algorithms
]
legend = fig.legend(
    fake_patches,
    algorithms,
    loc="upper center",
    fancybox=True,
    ncol=len(algorithms),
    fontsize="x-large",
    bbox_to_anchor=(0.45, 1.1),
)
fig.subplots_adjust(top=0.92, wspace=0.24)

# %%
# @title Performance profiles for Atari 100K

fig, ax = plt.subplots(ncols=2, figsize=(7 * 2, 4.5))
# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR']
algorithms = ["DWA", "DWB", "MPPI"]
plot_utils.plot_performance_profiles(
    score_distributions,
    EXPERIMENTS_TAU,
    performance_profile_cis=score_distributions_cis,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel=r"Human Normalized Score $(\tau)$",
    labelsize="xx-large",
    ax=ax[0],
)
ax[0].set_title("Score Distributions ", size="x-large")

xticks = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
plot_utils.plot_performance_profiles(
    score_distributions,
    EXPERIMENTS_TAU,
    performance_profile_cis=score_distributions_cis,
    colors=EXPERIMENTS_COLOR_DICT,
    xlabel=r"Human Normalized Score $(\tau)$",
    ylabel=r"Fraction of runs with score > $\tau$",
    labelsize="xx-large",
    use_non_linear_scaling=True,
    xticks=xticks,
    ax=ax[1],
)
ax[1].set_title("Score Distributions with Non Linear Scaling", size="x-large")

fake_patches = [
    mpatches.Patch(color=EXPERIMENTS_COLOR_DICT[alg], alpha=0.75) for alg in algorithms
]
legend = fig.legend(
    fake_patches,
    algorithms,
    loc="upper center",
    fancybox=True,
    ncol=len(algorithms),
    fontsize="x-large",
    bbox_to_anchor=(0.45, 1.15),
)
fig.subplots_adjust(wspace=0.24)

# %%
# @title Compute Probability of Improvement for all comparisons on Atari 100k

# algorithms = ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR']
algorithms = ["DWA", "DWB", "MPPI"]
# our_algorithm = 'SPR' #@param ['SimPLe', 'DER', 'OTR', 'CURL', 'DrQ(ε)', 'SPR']
our_algorithm = "MPPI"
all_pairs = {}
for alg in algorithms:
    if alg == our_algorithm:
        continue
    pair_name = f"{our_algorithm}_{alg}"
    all_pairs[pair_name] = (
        experiments_path_length_dict[our_algorithm],
        experiments_path_length_dict[alg],
    )

probabilities, probability_cis = {}, {}
reps = 1000
probabilities, probability_cis = rly.get_interval_estimates(
    all_pairs, metrics.probability_of_improvement, reps=reps
)

# %%
# @title Plot probabilities of improvement P(X > Y) with 95% CIs

fig, ax = plt.subplots(figsize=(4, 3))
h = 0.6
algorithm_labels = []

for i, (alg_pair, prob) in enumerate(probabilities.items()):
    _, alg1 = alg_pair.split("_")
    algorithm_labels.append(alg1)
    (l, u) = probability_cis[alg_pair]
    ax.barh(
        y=i,
        width=u - l,
        height=h,
        left=l,
        color=EXPERIMENTS_COLOR_DICT[alg1],
        alpha=0.75,
    )
    ax.vlines(
        x=prob, ymin=i - 7.5 * h / 16, ymax=i + (6 * h / 16), color="k", alpha=0.85
    )
ax.set_yticks(range(len(algorithm_labels)))
ax.set_yticklabels(algorithm_labels)


ax.set_title(rf"P({our_algorithm} > $Y$)", size="xx-large")
plot_utils._annotate_and_decorate_axis(
    ax, labelsize="xx-large", ticklabelsize="xx-large"
)
ax.set_ylabel(r"Algorithm $Y$", size="xx-large")
ax.xaxis.set_major_locator(MaxNLocator(4))
fig.subplots_adjust(wspace=0.25, hspace=0.45)
