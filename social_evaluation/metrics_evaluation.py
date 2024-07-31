#!/usr/bin/env python3
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


def evaluate_selected_metrics(
    data: Dict[str, Any], config: Dict[str, Any], metrics: List[str]
) -> Dict[str, Any]:
    """Evaluate the selected metrics on the data
    :param data: Dictionary with the data
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return: Dictionary with the results
    """
    pass


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

    # Compute the desired metrics for the experiment
    desired_metrics = {}
    for metric in metrics:
        utils.metrics[metric](
            df_agents_list,
            transforms["map_to_base_link"],
            df_obstacles=df_obstacles,
            goal=goal,
        )

    # Compute the metrics for the experiment


def main():
    parser = argparse.ArgumentParser(description="Evaluate social metrics")
    parser.add_argument("config_file", type=Path, help="Path to the configuration file")
    parser.add_argument("data_file", type=Path, help="Path to the data file")
    parser.add_argument("output_file", type=Path, help="Path to the output file")
    parser.add_argument("metrics_file", type=Path, help="Path to the metrics file")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    with open(args.metrics_file, "r") as file:
        metrics = yaml.safe_load(file)

    with open(args.data_file, "rb") as file:
        data = pickle.load(file)

    results = evaluate_selected_metrics(data, config, metrics)

    with open(args.output_file, "wb") as file:
        pickle.dump(results, file)
