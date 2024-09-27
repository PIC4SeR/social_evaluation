from typing import List
import matplotlib.patches as patches
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

    desired_width_px = pu.meters_to_pixels(desired_width, ax)

    # Plot the line collection for the agnet
    lc_agent = pu.create_line_collection(
        x, y, df_trajectories["velocity"].iloc[: i + 1], norm
    )
    artist.append(ax.add_collection(lc_agent))

    # Add the agent image with rotation
    ab = pu.create_annotation_box(
        image_agent,
        x[-1],
        y[-1],
        desired_width_px=desired_width_px,
        angle=df_trajectories["yaw"].iloc[i],
        alpha=0.5 + 0.5 * i / len(df_trajectories),
    )
    artist.append(ax.add_artist(ab))

    if static_image_size > 0:
        len_df = len(df_trajectories)
        for j in np.arange(0, len_df, len_df // static_image_size):
            if j < i:
                image_agent_rot = pu.create_image_marker(
                    image_agent,
                    desired_width_px=desired_width_px,
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
            desired_width=0.1,
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
            desired_width=0.1,
        )
        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=100, blit=True)
    cb = plt.colorbar(artists[-1][0], ax=ax)
    cb.ax.set_ylabel("Velocity [m/s]")
    return ani
