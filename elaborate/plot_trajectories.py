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
    plot_map(map, ax)
    plot_goal(goal, ax, color="red")
    df_robot = transforms["map_to_base_link"]
    df_trajectories = combine_df_robot_agents(
        df_robot=df_robot, df_agents_list=df_agents_list
    )

    ani = plot_trajectories(df_trajectories, ax, fig=fig)
    return ani


def plot_trajectories(
    df_trajectories: pd.DataFrame,
    ax: plt.Axes,
    time_freq: float = 0.1,
    time_start: float = -np.inf,
    time_end: float = np.inf,
    fig: plt.Figure = None,
) -> ArtistAnimation:
    """Plot the trajectories
    :param df_trajectories: Dataframe with the trajectories
    :param ax: Matplotlib axis
    :param time_freq: Frequency of the time to plot
    :param time_start: Start time
    :param time_end: End time
    """

    # Filter the data
    df_trajectories = df_trajectories[
        (df_trajectories["time"] >= time_start) & (df_trajectories["time"] <= time_end)
    ]

    max_velocity = 0
    min_velocity = 0

    max_x = df_trajectories["position_x"].max()
    min_x = df_trajectories["position_x"].min()
    max_y = df_trajectories["position_y"].max()
    min_y = df_trajectories["position_y"].min()

    # robot
    df_trajectories.dropna(subset=["velocity_x", "velocity_y"], inplace=True)
    df_trajectories["velocity"] = np.linalg.norm(
        df_trajectories[["velocity_x", "velocity_y"]].values, axis=1
    )

    max_velocity = max(max_velocity, df_trajectories["velocity"].max())
    min_velocity = min(min_velocity, df_trajectories["velocity"].min())

    # agents
    agent_num = df_trajectories.keys().str.contains("position_x_agent").sum()

    for i in range(agent_num):
        df_trajectories[f"velocity_agent{i}"] = np.linalg.norm(
            df_trajectories[[f"velocity_x_agent{i}", f"velocity_y_agent{i}"]].values,
            axis=1,
        )
        max_velocity = max(max_velocity, df_trajectories[f"velocity_agent{i}"].max())
        min_velocity = min(min_velocity, df_trajectories[f"velocity_agent{i}"].min())
        max_x = max(max_x, df_trajectories[f"position_x_agent{i}"].max())
        min_x = min(min_x, df_trajectories[f"position_x_agent{i}"].min())
        max_y = max(max_y, df_trajectories[f"position_y_agent{i}"].max())
        min_y = min(min_y, df_trajectories[f"position_y_agent{i}"].min())

    norm = mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.margins(0.1)

    # add offset image for the robot
    robot_image_path = "/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/robot.png"  # Replace with your PNG file path
    image_robot = np.rot90(plt.imread(robot_image_path, format="png"), 3)

    agents_image_path_stand = [
        f"/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/agent{i}_stand.png"
        for i in range(agent_num)
    ]
    agents_image_path_walk = [
        f"/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/agent{i}_walk.png"
        for i in range(agent_num)
    ]

    image_agents_stand = [
        np.rot90(plt.imread(image_path, format="png"), 3)
        for image_path in agents_image_path_stand
    ]
    image_agents_walk = [
        np.rot90(plt.imread(image_path, format="png"), 3)
        for image_path in agents_image_path_walk
    ]

    ani = pu.animate_trajectories(
        df_trajectories,
        fig,
        ax,
        agent_num,
        norm,
        image_robot,
        image_agents_walk,
        static_image_size=4,
    )

    plt.show()
    ani.save("test.mp4")
    return ani


def plot_goal(
    goal: Tuple[float, float], ax: plt.Axes, color: str = "green", radius: float = 0.3
) -> None:
    """Plot the goal
    :param goal: Tuple with the goal position
    """
    # goal_circle = patches.Circle(
    #     (goal[0], goal[1]), radius, edgecolor="black", fill=False, color=color
    # )

    # add annotation  with the goal image
    goal_image_path = "/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/goal.png"  # Replace with your PNG file path
    image_goal = plt.imread(goal_image_path, format="png")

    desired_width = 2 * radius
    desired_width_px = pu.meters_to_pixels(desired_width, ax)

    ab_goal = AnnotationBbox(
        pu.create_image_marker(image_goal, desired_width_px=desired_width_px),
        goal,
        frameon=False,
    )
    ax.add_artist(ab_goal)

    # ax.add_patch(goal_circle)


def plot_first_position(df_agents_list: List[pd.DataFrame], ax: plt.Axes) -> None:
    """
    Plot the first position of the agents
    """
    for df_agent in df_agents_list:
        ax.scatter(
            df_agent["position_x"].iloc[0],
            df_agent["position_y"].iloc[0],
            c="black",
            s=10,
        )


def plot_map(
    map: np.ndarray,
    ax: plt.Axes,
    resolution: float = 0.05,
    origin: Tuple[float, float] = (-8.41, -12.7),
) -> None:
    """Plot the map
    :param map: Numpy array with the map
    """
    color_map = mcolors.ListedColormap(["white", "black"])
    extent = [
        origin[0],
        origin[0] + map.shape[1] * resolution,
        origin[1],
        origin[1] + map.shape[0] * resolution,
    ]
    # use a resolution of 0.1 and set the origin of the map to the bottom left corner
    ax.imshow(map, cmap=color_map, extent=extent)


# %%
# Load the data

pickle_file = Path("/home/andrea/workspaces/appl_ws/output.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)


# %%
# Plot the experiment

exp = list(data.keys())[0]


fig, ax = plt.subplots()
fig.set_size_inches(10, 10)

ani = plot_experiment_single(data, exp, ax, fig)

plt.show()
