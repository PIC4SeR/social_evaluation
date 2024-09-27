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


# %%


def plot_experiment_aggregate(data: dict, exp: List[str], ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    pass


def create_image_marker(image, zoom=0.2, angle=0, alpha=1):
    rotated_image = rotate(image, np.degrees(angle), reshape=True)
    # return OffsetImage(rotated_image.astype(np.uint8), zoom=zoom, alpha=alpha)
    return OffsetImage(rotated_image, zoom=zoom, alpha=alpha)


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
    # plot_agents(df_agents_list, ax)
    # plot_transforms(transforms, ax)
    df_robot = transforms["map_to_base_link"]
    df_trajectories = combine_df_robot_agents(
        df_robot=df_robot, df_agents_list=df_agents_list
    )

    ani = plot_trajectories(df_trajectories, ax)
    return ani


def update_robot(frame, ab, df):
    # update the position of the robot
    ab.xy = (
        df["position_x"].iloc[frame],
        df["position_y"].iloc[frame],
    )
    # update the angle of the robot
    image_robot = plt.imread(
        "/home/andrea/workspaces/appl_ws/social_evaluation/elaborate/images/robot.png"
    )
    image_robot = create_image_marker(
        image_robot, zoom=0.015, angle=df["yaw"].iloc[frame]
    )
    # ab.set_offset(image_robot)
    return (ab,)


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

    artists = []

    for i in range(len(df_trajectories)):

        x = np.array(df_trajectories["position_x"].iloc[: i + 1])
        y = np.array(df_trajectories["position_y"].iloc[: i + 1])

        artist = ax.plot(x, y, c="grey", alpha=0.0)

        points_robot = np.array([x, y]).T.reshape(-1, 1, 2)
        segments_robot = np.concatenate([points_robot[:-1], points_robot[1:]], axis=1)

        # # Create a LineCollection object with the segments
        lc_robot = mcoll.LineCollection(segments_robot, cmap="viridis", norm=norm)
        lc_robot.set_array(df_trajectories["velocity"].iloc[: i + 1])
        artist.append(ax.add_collection(lc_robot))

        image_robot_rot = create_image_marker(
            image_robot,
            zoom=0.1,
            angle=df_trajectories["yaw"].iloc[i],
            alpha=0.5 + 0.5 * i / len(df_trajectories),
        )
        ab = AnnotationBbox(
            image_robot_rot,
            (
                x[-1],
                y[-1],
            ),
            frameon=False,
        )
        artist.append(ax.add_artist(ab))

        for j in range(agent_num):
            x_agent = np.array(df_trajectories[f"position_x_agent{j}"].iloc[: i + 1])
            y_agent = np.array(df_trajectories[f"position_y_agent{j}"].iloc[: i + 1])

            points_agent = np.array([x_agent, y_agent]).T.reshape(-1, 1, 2)
            segments_agent = np.concatenate(
                [points_agent[:-1], points_agent[1:]], axis=1
            )

            # # Create a LineCollection object with the segments

            lc_agent = mcoll.LineCollection(segments_agent, cmap="viridis", norm=norm)
            lc_agent.set_array(df_trajectories[f"velocity_agent{j}"].iloc[: i + 1])
            artist.append(ax.add_collection(lc_agent))
            image_agent = image_agents_walk[j]
            # image_agent = image_agents_walk[j] if i % 2 == 0 else np.flip(image_agents_walk[j], 1)
            image_agent_rot = create_image_marker(
                image_agent,
                zoom=0.1,
                angle=df_trajectories[f"yaw_agent{j}"].iloc[i],
                alpha=0.5 + 0.5 * i / len(df_trajectories),
            )
            ab_agent = AnnotationBbox(
                image_agent_rot,
                (
                    x_agent[-1],
                    y_agent[-1],
                ),
                frameon=False,
            )

            artist.append(ax.add_artist(ab_agent))

        len_df = len(df_trajectories)
        for j in np.arange(0, len_df, len_df // 5):
            if j < i:
                image_robot_rot = create_image_marker(
                    image_robot,
                    zoom=0.1,
                    angle=df_trajectories["yaw"].iloc[j],
                    alpha=0.3 + 0.7 * (j / len_df),
                )
                ab = AnnotationBbox(
                    image_robot_rot,
                    (
                        x[j],
                        y[j],
                    ),
                    frameon=False,
                )
                artist.append(ax.add_artist(ab))

                for k in range(agent_num):
                    image_agent = image_agents_stand[k]
                    image_agent_rot = create_image_marker(
                        image_agent,
                        zoom=0.1,
                        angle=df_trajectories[f"yaw_agent{k}"].iloc[j],
                        alpha=0.3 + 0.7 * (j / len_df),
                    )
                    ab_agent = AnnotationBbox(
                        image_agent_rot,
                        (
                            x_agent[j],
                            y_agent[j],
                        ),
                        frameon=False,
                    )
                    artist.append(ax.add_artist(ab_agent))

        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=100, blit=True)
    cb = plt.colorbar(lc_robot, ax=ax)
    cb.ax.set_ylabel("Velocity [m/s]")

    plt.show()
    ani.save("test.mp4")
    return ani


def plot_obstacles(df_obstacles: pd.DataFrame, ax: plt.Axes):
    """Plot the obstacles
    :param df_obstacles: Dataframe with the obstacles
    """
    ax.scatter(df_obstacles["x"], df_obstacles["y"], c="black", s=2)

    pass


def plot_goal(
    goal: Tuple[float, float], ax: plt.Axes, color: str = "green", radius: float = 0.3
) -> None:
    """Plot the goal
    :param goal: Tuple with the goal position
    """
    goal_circle = patches.Circle((goal[0], goal[1]), radius, fill=False, color=color)
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


def plot_agents(df_agents_list: List[pd.DataFrame], ax: plt.Axes) -> None:
    """Plot the agents
    :param df_agents_list: List of dataframes with the agents
    :param ax: Matplotlib axis
    """

    # find the max/min velocity to normalize the arrow size
    max_velocity = 0
    min_velocity = 0

    for df_agent in df_agents_list:
        df_agent["velocity"] = np.linalg.norm(
            df_agent[["velocity_x", "velocity_y"]].values, axis=1
        )
        max_velocity = max(max_velocity, df_agent["velocity"].max())
        min_velocity = min(min_velocity, df_agent["velocity"].min())
    norm = mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)

    for df_agent in df_agents_list:

        df_agent["velocity"] = np.linalg.norm(
            df_agent[["velocity_x", "velocity_y"]].values, axis=1
        )
        # sample the points to plot (to avoid plotting all the points)
        df_agent_plot = df_agent.copy().iloc[0::10]
        # create a scatter plot
        ax.scatter(
            df_agent_plot["position_x"].values,
            df_agent_plot["position_y"].values,
            c=df_agent_plot["velocity"],
            cmap="viridis",
            norm=norm,
            marker="x",
            s=20,
            alpha=0.5,
        )


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
