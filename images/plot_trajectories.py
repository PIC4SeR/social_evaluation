# %%
import pandas as pd
from pathlib import Path
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
from tqdm import trange

import utils as pu


# %%


def plot_experiment_aggregate(data: dict, exp: List[str], ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    pass


def plot_from_start_up_to_time(data: dict, exp: str, time: float, ax: plt.Axes):
    pass


def plot_from_time_to_time(
    data: dict, exp: str, time_start: float, time_end: float, ax: plt.Axes
):
    pass


def plot_from_time_to_end(data: dict, exp: str, time_start: float, ax: plt.Axes):
    pass


def plot_experiment_single(data: dict, exp: str, ax: plt.Axes, fig: plt.Figure):
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
        time_freq=0.1,
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
    """
    pu.plot_map(map, ax)
    pu.plot_goal(goal, ax)
    df_robot = transforms["map_to_base_link"]
    df_trajectories = combine_df_robot_agents(
        df_robot=df_robot, df_agents_list=df_agents_list
    )

    ani = pu.plot_trajectories(df_trajectories, ax, fig=fig, time_freq=0.1)
    return ani


# %%
# Load the data

pickle_file = Path("/home/andrea/workspaces/appl_ws/output.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)


# %%
# Plot the experiment

exp = list(data.keys())[1]


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

ani = plot_experiment_single(data, exp, ax, fig)

plt.show()

# save the animation as gif
# %%
ani.save("test.gif", writer="imagemagick")
# %%
# save the last frame of the animation
fig.savefig("test.png")

# %%
## convert data[time] to datatime
# print(data[exp]['transforms']['map_to_base_link']['time'])
# data[exp]['transforms']['map_to_base_link']['time'] = pd.to_datetime(data[exp]['transforms']['map_to_base_link']['time'], unit='s')
