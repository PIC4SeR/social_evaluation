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


# %%


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image data to ensure valid RGB ranges.
    Floats are clipped to [0, 1], integers to [0, 255].
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Clip float data to [0, 1]
        image = np.clip(image, 0, 1)
    elif image.dtype == np.int32 or image.dtype == np.int64 or np.any(image > 255):
        # Clip integer data to [0, 255] and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def plot_experiment_aggregate(data: dict, exp: List[str], ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    pass


def create_line_collection(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    norm: mcolors.Normalize,
    cmap: str = "viridis",
) -> mcoll.LineCollection:
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(velocity)
    return lc


def create_annotation_box(
    image: np.ndarray,
    x: float,
    y: float,
    desired_size: float = 100,
    angle: float = 0,
    alpha: float = 1,
) -> AnnotationBbox:
    image_rot = create_image_marker(
        image, desired_size=desired_size, angle=angle, alpha=alpha
    )
    return AnnotationBbox(image_rot, (x, y), frameon=False)


# Function to handle the plotting of an agent (robot or person)
def plot_agent(
    ax: plt.Axes,
    df_trajectories: pd.DataFrame,
    i: int,
    artist: List,
    norm: mcolors.Normalize,
    image_agent: np.ndarray,
    static_image_size: int = 5,
    desired_size: float = 0.3,
) -> None:
    x = np.array(df_trajectories["position_x"].iloc[: i + 1])
    y = np.array(df_trajectories["position_y"].iloc[: i + 1])

    # Get the bounding box of the axes in display (pixel) coordinates
    bbox = ax.get_window_extent()

    # Calculate the pixel width and height of the axes
    ax_xbounds = ax.get_xbound()
    ax_width = ax_xbounds[1] - ax_xbounds[0]
    ax_width_in_pixels = bbox.width

    desired_size_pixels = desired_size * ax_width_in_pixels / ax_width

    # Plot the line collection for the agnet
    lc_agent = create_line_collection(
        x, y, df_trajectories["velocity"].iloc[: i + 1], norm
    )
    artist.append(ax.add_collection(lc_agent))

    # Add the agent image with rotation
    ab = create_annotation_box(
        image_agent,
        x[-1],
        y[-1],
        desired_size=desired_size_pixels,
        angle=df_trajectories["yaw"].iloc[i],
        alpha=0.5 + 0.5 * i / len(df_trajectories),
    )
    artist.append(ax.add_artist(ab))

    if static_image_size > 0:
        len_df = len(df_trajectories)
        for j in np.arange(0, len_df, len_df // static_image_size):
            if j < i:
                image_agent_rot = create_image_marker(
                    image_agent,
                    desired_size=desired_size_pixels,
                    angle=df_trajectories["yaw"].iloc[j],
                    alpha=0.3 + 0.7 * (j / len_df),
                )
                ab_agent = AnnotationBbox(
                    image_agent_rot,
                    (
                        x[j],
                        y[j],
                    ),
                    frameon=False,
                )
                artist.append(ax.add_artist(ab_agent))


def plot_agents(
    ax: plt.Axes,
    df_trajectories: pd.DataFrame,
    i: int,
    artist: list,
    agent_num: int,
    norm: mcolors.Normalize,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 5,
    desired_size: float = 0.3,
) -> None:
    for j in range(agent_num):
        plot_agent(
            ax,
            df_trajectories[
                [
                    f"position_x_agent{j}",
                    f"position_y_agent{j}",
                    f"velocity_agent{j}",
                    f"yaw_agent{j}",
                ]
            ].rename(
                columns={
                    f"position_x_agent{j}": "position_x",
                    f"position_y_agent{j}": "position_y",
                    f"velocity_agent{j}": "velocity",
                    f"yaw_agent{j}": "yaw",
                }
            ),
            i,
            artist,
            norm,
            image_agents_walk[j],
            static_image_size,
            desired_size,
        )


def create_image_marker(
    image: np.ndarray, desired_size: float = 100, angle: float = 0, alpha: float = 1
) -> OffsetImage:
    rotated_image = normalize_image(rotate(image, np.degrees(angle), reshape=True))
    return OffsetImage(
        rotated_image,
        zoom=desired_size / max(image.shape[0], image.shape[1]),
        alpha=alpha,
    )


def animate_trajectories(
    df_trajectories: pd.DataFrame,
    ax: plt.Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 4,
) -> ArtistAnimation:
    artists = []
    for i in trange(len(df_trajectories)):
        artist = []
        plot_agent(
            ax,
            df_trajectories[["position_x", "position_y", "velocity", "yaw"]],
            i,
            artist,
            norm,
            image_robot,
            static_image_size,
            desired_size=0.3,
        )
        plot_agents(
            ax,
            df_trajectories,
            i,
            artist,
            agent_num,
            norm,
            image_agents_walk,
            static_image_size,
            desired_size=0.3,
        )
        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=100, blit=True)
    cb = plt.colorbar(artists[-1][0], ax=ax)
    cb.ax.set_ylabel("Velocity [m/s]")
    return ani


def plot_from_start_up_to_time(data: dict, exp: str, time: float, ax: plt.Axes):
    pass


def plot_from_time_to_time(
    data: dict, exp: str, time_start: float, time_end: float, ax: plt.Axes
):
    pass


def plot_from_time_to_end(data: dict, exp: str, time_start: float, ax: plt.Axes):
    pass


def plot_experiment_single(data: dict, exp: str, ax: plt.Axes):
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
        df_odometry, df_people, df_agents_list, df_obstacles, transforms, goal, map, ax
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

    ani = plot_trajectories(df_trajectories, ax)
    return ani


def plot_trajectories(
    df_trajectories: pd.DataFrame,
    ax: plt.Axes,
    time_freq: float = 0.1,
    time_start: float = -np.inf,
    time_end: float = np.inf,
):
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

    ani = animate_trajectories(
        df_trajectories,
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
    goal_circle = patches.Circle(
        (goal[0], goal[1]), radius, edgecolor="black", fill=False, color=color
    )

    # add annotation  with the goal image
    goal_image_path = "/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/goal.png"  # Replace with your PNG file path
    image_goal = plt.imread(goal_image_path, format="png")

    # Get the bounding box of the axes in display (pixel) coordinates
    bbox = ax.get_window_extent()

    # Calculate the pixel width and height of the axes
    ax_xbounds = ax.get_xbound()
    ax_width = ax_xbounds[1] - ax_xbounds[0]
    ax_width_in_pixels = bbox.width

    desired_size_pixels = radius * 2 * ax_width_in_pixels / ax_width

    ab_goal = AnnotationBbox(
        create_image_marker(image_goal, desired_size=desired_size_pixels),
        goal,
        frameon=False,
    )
    ax.add_artist(ab_goal)

    ax.add_patch(goal_circle)


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

ani = plot_experiment_single(data, exp, ax)
# ani.save("test.mp4")
plt.show()
