#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation import utils
from social_evaluation.bag_extraction import *
import pandas as pd
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import os
import numpy as np
from tqdm import tqdm


def evaluate_algorithm(
    paths: List[PosixPath],
    experiment: str,
    algorithm: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluate the algorithm on the data
    :param paths: List of paths with the data
    :param data: Dictionary with the data
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return: Dataframe with the results of the evaluation for the algorithm
    """
    series = []
    for i, path in enumerate(
        tqdm(
            paths,
            desc=f"Evaluating scenarios for {experiment} with {algorithm}",
            leave=False,
        )
    ):
        results, _ = evaluate_selected_metrics(data[path], config, metrics)
        results["experiment"] = experiment
        results["algorithm"] = algorithm
        results["run"] = i + 1
        series.append(results)
    df = pd.DataFrame(series)
    return df


def evaluate_selected_metrics(
    data: Dict[str, Any], config: Dict[str, Any], metrics: Dict[str, Any]
) -> Tuple[pd.Series, pd.DataFrame]:
    """Evaluate the selected metrics on the data
    :param data: Dictionary with the data
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return: Dictionary with the results
    """
    # Extract the data from the dictionary
    # print(data.keys())
    df_odometry = data["odometry"]
    df_people = data["people"]
    df_agents_list = data["agents"]
    df_obstacles = data["obstacles"]
    transforms = data["transforms"]
    goal = data["goal"]

    metrics_list = metrics["metrics"]

    # Evaluate the metrics
    results, results_df = evaluate_metrics(
        df_odometry,
        df_people,
        df_agents_list,
        df_obstacles,
        transforms,
        goal,
        config,
        metrics_list,
    )

    return results, results_df


def evaluate_metrics(
    df_odometry: pd.DataFrame,
    df_people: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
    transforms: Dict[str, pd.DataFrame],
    goal: Union[np.ndarray, Tuple[float, float]],
    config: Dict[str, Any],
    metrics: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """Evaluate the metrics on the data
    :param df_odometry: Dataframe with the odometry data
    :param df_people: Dataframe with the people data
    :param df_agents_list: List of dataframes with the agents data
    :param df_obstacles: Dataframe with the obstacles data
    :param transforms: Dictionary with the transforms
    :param goal: Goal position
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return:
    """

    # Compute the metrics for the experiment
    metrics_results = {}
    metrics_df = {}
    for metric in metrics:

        result = utils.metrics[metric](
            df_agents_list,
            transforms["map_to_base_link"],
            df_obstacles=df_obstacles,
            goal=goal,
            **config,
        )
        if isinstance(result, Tuple):
            metrics_results[metric] = result[0]
            metrics_df[metric] = result[1]
        else:
            metrics_results[metric] = result
            metrics_df[metric] = pd.DataFrame()

    metrics_results = pd.Series(metrics_results)
    return metrics_results, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate social metrics")
    parser.add_argument(
        "--config-file", type=Path, help="Path to the configuration file"
    )
    parser.add_argument("--data-file", type=Path, help="Path to the data file")
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to the output file",
        default="metrics.pickle",
    )
    parser.add_argument("--metrics-file", type=Path, help="Path to the metrics file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    with open(args.metrics_file, "r") as file:
        metrics = yaml.safe_load(file)

    with open(args.data_file, "rb") as file:
        data = pickle.load(file)

    experiment_names = list(config["experiment_names"])
    algorithms = list(config["algorithms"])
    metrics_config = metrics["metrics_config"]

    # find all the paths that contain the experiment names and the algorithms
    paths_list = {}
    for experiment_name in experiment_names:
        for algorithm in algorithms:
            paths_list[experiment_name + "_" + algorithm] = [
                k
                for k in data.keys()
                if ((experiment_name in k.split("/")) and (algorithm in k.split("/")))
            ]
    results_list = []
    for key, value in paths_list.items():
        print(f"experiment and algo: {key} with {len(value)} scenarios")
        for i, scenario in enumerate(value):
            print(f"\tscenario_{i+1}: {scenario}")

    for path in tqdm(paths_list, desc="Evaluating experiments", leave=False):
        experiment, algorithm = path.split("_")
        results_list.append(
            evaluate_algorithm(
                paths_list[path], experiment, algorithm, data, metrics_config, metrics
            )
        )
    # results = pd.DataFrame(results_list)
    results = pd.concat(results_list, ignore_index=True)

    with open(args.output_file, "wb") as file:
        pickle.dump(results, file)


if __name__ == "__main__":
    main()
