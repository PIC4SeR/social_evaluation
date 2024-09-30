#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# %%
import pandas as pd
from pathlib import Path, PosixPath
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, List, Tuple
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

from social_evaluation.utils.metrics_utils import combine_df_robot_agents
from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.animation import FuncAnimation, ArtistAnimation
import utils as pu
from tqdm import tqdm
import argcomplete, argparse
import yaml
import social_evaluation


def plot_experiment_aggregate(data: dict, exp: List[str], ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    pass


def plot_experiment_single(
    data: dict, exp: str, ax: plt.Axes, fig: plt.Figure, time_freq: float = 0.1
):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    df_odometry = pd.DataFrame(data[exp]["odometry"])
    df_people = pd.DataFrame(data[exp]["people"])
    df_agents_list = data[exp]["agents"]
    df_obstacles = data[exp]["obstacles"]
    transforms = data[exp]["transforms"]
    goal = data[exp]["goal"]
    map = data[exp]["map"]

    # Plot the data
    ani = plot_data(
        df_odometry,
        df_people,
        df_agents_list,
        df_obstacles,
        transforms,
        goal,
        map,
        ax,
        fig,
        time_freq=time_freq,
    )
    return ani


def plot_data(
    df_odometry: pd.DataFrame,
    df_people: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
    transforms: pd.DataFrame,
    goal: Tuple[float, float],
    map: np.ndarray,
    ax: plt.Axes,
    fig: plt.Figure,
    time_freq: float = 0.1,
) -> None:
    """Plot the data
    :param df_odometry: Dataframe with the odometry
    :param df_people: Dataframe with the people
    :param df_agents_list: List of dataframes with the agents
    :param df_obstacles: Dataframe with the obstacles
    :param transforms: Dataframe with the transforms
    :param goal: Tuple with the goal position
    :param map: Numpy array with the map
    :param ax: Matplotlib axis
    :param fig: Matplotlib figure
    :param time_freq: Frequency of the time to plot
    """
    pu.plot_map(map, ax)
    pu.plot_goal(goal, ax)
    df_robot = transforms["map_to_base_link"]
    df_trajectories = combine_df_robot_agents(
        df_robot=df_robot, df_agents_list=df_agents_list
    )

    ani = pu.plot_trajectories(df_trajectories, ax, fig=fig, time_freq=time_freq)
    return ani


def main():
    parser = argparse.ArgumentParser()

    package_path = Path(social_evaluation.__file__).parent

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Config file for all the plots parameters",
        default=package_path.joinpath("../config/trajectory_plots.yaml"),
    )

    parser.add_argument(
        "--pickle-file",
        type=Path,
        help="Bag file path to extract data from",
        default="None",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory to save the plots",
        default="plotted_trajectories",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    config = yaml.load(config_path.open("r"), Loader=yaml.FullLoader)

    if args.pickle_file != PosixPath("None"):
        pickle_file = Path(args.pickle_file)
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = {}
        for exp in config["experiments"]:
            print(f"Loading experiment {config['experiments']}")
            print(f"Loading experiment {config['experiments'][exp]['path']}")
            exp_path = Path(config["experiments"][exp]["path"])
            with open(exp_path, "rb") as f:
                pickle_data = pickle.load(f)
                for key in pickle_data.keys():
                    data[key] = pickle_data[key]

    for i, exp in enumerate(tqdm(data.keys(), desc=f"Plotting experiments")):

        experiment_name = exp.split("/")[-4]
        algorithm_name = exp.split("/")[-3]

        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        ani = plot_experiment_single(data, exp, ax, fig)

        if not args.output_dir.joinpath(experiment_name, algorithm_name).exists():
            os.makedirs(args.output_dir.joinpath(experiment_name, algorithm_name))

        # save the animation as gif
        ani.save(
            args.output_dir.joinpath(
                experiment_name, algorithm_name, f"scenario_{i}.gif"
            ),
            writer="imagemagick",
            fps=10,
        )
        # save the last frame of the animation
        fig.savefig(
            args.output_dir.joinpath(
                experiment_name, algorithm_name, f"scenario_{i}.png"
            )
        )


if __name__ == "__main__":
    main()
