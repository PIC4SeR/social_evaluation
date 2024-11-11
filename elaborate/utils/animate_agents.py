from typing import List
import matplotlib.patches as patches
import pandas as pd
from matplotlib.path import Path
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
from scipy.ndimage import gaussian_filter1d

from tqdm import trange

import utils.plot_utils as pu


def plot_agent(
    ax: plt.Axes,
    df_trajectories: pd.DataFrame,
    i: int,
    artist: List,
    norm: mcolors.Normalize,
    image_agent: np.ndarray,
    static_image_size: int = 5,
    desired_width: float = 0.3,
) -> None:
    x = np.array(df_trajectories["position_x"].iloc[: i + 1])
    y = np.array(df_trajectories["position_y"].iloc[: i + 1])
    yaw = np.array(df_trajectories["yaw"].iloc[: i + 1])

    x_smooth = gaussian_filter1d(x, sigma=2)
    y_smooth = gaussian_filter1d(y, sigma=2)
    yaw_smooth = gaussian_filter1d(yaw, sigma=2)

    desired_width_px = pu.meters_to_pixels(desired_width, ax)

    # Plot the line collection for the agnet
    lc_agent = pu.create_line_collection(
        x_smooth, y_smooth, df_trajectories["velocity"].iloc[: i + 1], norm
    )
    artist.append(ax.add_collection(lc_agent))

    alpha = lambda i: min(0.2 + 0.8 * i / len(df_trajectories), 1)
    # alphas = np.array([alpha(j) for j in range(min(i + 1, len(df_trajectories)))])
    lc_agent.set_alpha(alpha(i))
    lc_agent.set_linewidth(3)

    # Add the agent image with rotation
    ab = pu.create_annotation_box(
        image_agent,
        x_smooth[-1],
        y_smooth[-1],
        desired_width_px=desired_width_px,
        angle=yaw_smooth[-1],
        alpha=alpha(i),
    )
    artist.append(ax.add_artist(ab))

    if static_image_size > 0:
        len_df = len(df_trajectories)
        for j in np.arange(0, len_df, len_df // static_image_size):
            if j < i:
                image_agent_rot = pu.create_image_marker(
                    image_agent,
                    desired_width_px=desired_width_px,
                    angle=yaw_smooth[j],
                    alpha=alpha(j),
                )
                ab_agent = AnnotationBbox(
                    image_agent_rot,
                    (
                        x_smooth[j],
                        y_smooth[j],
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
    desired_width: float = 0.3,
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
            desired_width=desired_width,
        )


def animate_trajectories(
    df_trajectories: pd.DataFrame,
    fig: plt.Figure,
    ax: plt.Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 3,
    time_freq: float = 0.1,
) -> ArtistAnimation:
    artists = []
    df_trajectories_t = df_trajectories.copy()
    df_trajectories_t["time"] = pd.to_datetime(df_trajectories["time"], unit="s")
    df_trajectories_t.set_index("time", inplace=True)
    # Resample the data to the time frequency

    df_trajectories_rs = df_trajectories_t.asfreq(
        f"{int(1000 * time_freq)}ms", method="ffill"
    )
    for i in trange(len(df_trajectories_rs)):
        artist = []
        plot_agent(
            ax,
            df_trajectories_rs[["position_x", "position_y", "velocity", "yaw"]],
            i,
            artist,
            norm,
            image_robot,
            static_image_size,
            desired_width=0.1,
        )
        plot_agents(
            ax,
            df_trajectories_rs,
            i,
            artist,
            agent_num,
            norm,
            image_agents_walk,
            static_image_size,
            desired_width=0.1,
        )
        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=int(1000 * time_freq), blit=True)
    cb = plt.colorbar(artists[-1][0], ax=ax)
    cb.ax.set_ylabel("Velocity [m/s]")
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

    # agents
    agent_num = df_trajectories.keys().str.contains("position_x_agent").sum()
    # Set the limits of the axis
    pu.set_ax_limits_from_trajectories([df_trajectories], agent_num, ax)
    # find the interval for the velocity
    norm = pu.find_norm_for_velocity([df_trajectories], agent_num)

    image_robot, image_agents_walk, image_agents_stand = pu.get_images_for_experiment(
        agent_num
    )

    ani = animate_trajectories(
        df_trajectories,
        fig,
        ax,
        agent_num,
        norm,
        image_robot,
        image_agents_walk,
        static_image_size=4,
        time_freq=time_freq,
    )
    return ani


def plot_trajectories_aggregated(
    df_trajectories_list: List[pd.DataFrame],
    ax: plt.Axes,
    time_freq: float = 0.1,
    time_start: float = -np.inf,
    time_end: float = np.inf,
    fig: plt.Figure = None,
):
    """Plot the trajectories
    :param df_trajectories_list: List of dataframes with the trajectories
    :param ax: Matplotlib axis
    :param time_freq: Frequency of the time to plot
    :param time_start: Start time
    :param time_end: End time
    """

    agent_num = df_trajectories_list[0].keys().str.contains("position_x_agent").sum()
    pu.set_ax_limits_from_trajectories(df_trajectories_list, agent_num, ax)
    norm = pu.find_norm_for_velocity(df_trajectories_list, agent_num)

    ani = animate_trajectories_aggregated(
        df_trajectories_list, fig, ax, agent_num, norm, time_freq=time_freq
    )
    return ani


def animate_trajectories_aggregated(
    df_trajectories_list: List[pd.DataFrame],
    fig: plt.Figure,
    ax: plt.Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    time_freq: float = 0.1,
) -> ArtistAnimation:
    df_trajectories_rs_list = []
    for df_trajectories in df_trajectories_list:
        df_trajectories_t = df_trajectories.copy()
        df_trajectories_t["time"] = pd.to_datetime(df_trajectories["time"], unit="s")
        df_trajectories_t.set_index("time", inplace=True)
        # Resample the data to the time frequency

        df_trajectories_rs = df_trajectories_t.asfreq(
            f"{int(1000 * time_freq)}ms", method="ffill"
        )
        df_trajectories_rs_list.append(df_trajectories_rs)

    artists = []
    for i in trange(max([len(df) for df in df_trajectories_rs_list])):
        artist = []
        plot_robot_aggregated(
            ax,
            df_trajectories_rs_list,
            i,
            artist,
            norm,
        )

        plot_agents_aggregated(
            ax,
            df_trajectories_rs_list,
            i,
            artist,
            agent_num,
            norm,
        )

        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=int(1000 * time_freq), blit=True)
    cb = plt.colorbar(artists[-1][0], ax=ax)
    cb.ax.set_ylabel("Time [s]")
    return ani


def plot_robot_aggregated(
    ax: plt.Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: List,
    norm: mcolors.Normalize,
) -> None:
    for df_trajectories in df_trajectories_list:

        x = np.array(df_trajectories["position_x"].iloc[: i + 1])
        y = np.array(df_trajectories["position_y"].iloc[: i + 1])

        # get a vector with the time steps from the start in seconds
        time_steps_from_start = df_trajectories.index - df_trajectories.index[0]
        time_steps_from_start = time_steps_from_start.total_seconds().values

        norm = mcolors.Normalize(vmin=0, vmax=time_steps_from_start.max())
        # Plot the line collection for the agnet
        # lc_agent = pu.create_line_collection(
        #     x, y, df_trajectories["velocity"].iloc[: i + 1], norm
        # )
        lc_agent = pu.create_line_collection(x, y, time_steps_from_start, norm)
        # alpha = lambda i: min(0.2 + 0.8 * i / len(df_trajectories), 1)
        # alphas = np.array([alpha(j) for j in range(min(i + 1, len(df_trajectories)))])
        alphas = 0.7
        lc_agent.set_alpha(alphas)
        lc_agent.set_linewidth(3)
        artist.append(ax.add_collection(lc_agent))


def plot_agents_aggregated(
    ax: plt.Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: list,
    agent_num: int,
    norm: mcolors.Normalize,
) -> None:
    df_trajectories_list_rn = []
    for j in range(agent_num):
        for df_trajectories in df_trajectories_list:
            df_trajectories_list_rn.append(
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
                )
            )
        plot_agent_aggregated(
            ax,
            df_trajectories_list_rn,
            i,
            artist,
            norm,
        )


def plot_agent_aggregated(
    ax: plt.Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: List,
    norm: mcolors.Normalize,
) -> None:
    trajectories_x = []
    trajectories_y = []
    norm = mcolors.Normalize(vmin=0, vmax=1)
    # find the time steps from the start
    max_time = 0
    max_len = 0

    for df_trajectories in df_trajectories_list:
        time_steps_from_start = df_trajectories.index - df_trajectories.index[0]
        time_steps_from_start = time_steps_from_start.total_seconds().values
        max_time = max(max_time, time_steps_from_start.max())
        max_len = max(len(time_steps_from_start), max_len)

    time_steps_from_start = np.linspace(0, max_time, max_len)
    norm = mcolors.Normalize(vmin=0, vmax=time_steps_from_start.max())
    for df_trajectories in df_trajectories_list:

        x_smooth = gaussian_filter1d(
            np.array(df_trajectories["position_x"].iloc[: i + 1]), sigma=2
        )
        y_smooth = gaussian_filter1d(
            np.array(df_trajectories["position_y"].iloc[: i + 1]), sigma=2
        )

        # get a vector with the time steps from the start in seconds

        # trajectories_x.append(x_smooth)
        # trajectories_y.append(y_smooth)
        lc_agent = pu.create_line_collection(
            x_smooth,
            y_smooth,
            time_steps_from_start,
            norm,
        )
        alphas = 0.3
        lc_agent.set_array(time_steps_from_start)
        lc_agent.set_alpha(alphas)
        lc_agent.set_linewidth(10)
        lc_agent.set_capstyle = "round"
        lc_agent.set_linestyle((0, (1, 10)))
        artist.append(ax.add_collection(lc_agent))
