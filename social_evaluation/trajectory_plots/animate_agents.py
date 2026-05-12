from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.offsetbox import AnnotationBbox
from matplotlib.animation import ArtistAnimation
from scipy.ndimage import gaussian_filter1d

from tqdm import trange
import social_evaluation.trajectory_plots as pu
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist


def _prepare_trajectory_time_index(df_trajectories: pd.DataFrame) -> pd.DataFrame:
    """Return a trajectory frame with a sorted, unique datetime index."""

    df_trajectories_t = df_trajectories.copy()
    df_trajectories_t["time"] = pd.to_datetime(df_trajectories_t["time"], unit="s")
    df_trajectories_t.sort_values("time", kind="stable", inplace=True)
    df_trajectories_t = df_trajectories_t[
        ~df_trajectories_t["time"].duplicated(keep="first")
    ]
    df_trajectories_t.set_index("time", inplace=True)
    return df_trajectories_t


def smooth_angle_series(values: np.ndarray, sigma: float = 2) -> np.ndarray:
    """Smooth angular data without introducing wraparound spins."""
    if values.size == 0:
        return values
    sin_smooth = gaussian_filter1d(np.sin(values), sigma=sigma)
    cos_smooth = gaussian_filter1d(np.cos(values), sigma=sigma)
    return np.arctan2(sin_smooth, cos_smooth)


def plot_agent(
    ax: Axes,
    df_trajectories: pd.DataFrame,
    i: int,
    artist: List[Artist],
    norm: mcolors.Normalize,
    image_agent: np.ndarray,
    static_image_size: int = 5,
    desired_width: float = 0.3,
    image_width_multi: float = 1.0,
    cmap: mcolors.Colormap = plt.cm.Blues,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
) -> None:
    """Plot the agent in the trajectory plot

    :param ax: Matplotlib axis
    :param df_trajectories: Dataframe with the trajectories
    :param i: Index of the trajectory
    :param artist: List of artists
    :param norm: Normalize object for the trajectory time
    :param image_agent: Image for the agent
    :param static_image_size: Number of static images to plot
    :param desired_width: Desired width of the agent in meters
    :param image_width_multi: Multiplier for the image width
    :param cmap: Colormap for the trajectory line
    :param dynamic_alpha: Whether to use dynamic alpha values
    """
    x = np.array(df_trajectories["position_x"].iloc[: i + 1])
    y = np.array(df_trajectories["position_y"].iloc[: i + 1])
    yaw = np.array(df_trajectories["yaw"].iloc[: i + 1])

    x_smooth = gaussian_filter1d(x, sigma=2)
    y_smooth = gaussian_filter1d(y, sigma=2)
    yaw_smooth = smooth_angle_series(yaw, sigma=2)

    desired_width_px = pu.meters_to_pixels(desired_width, ax) * image_width_multi

    elapsed_time = pu.extract_elapsed_time_seconds(df_trajectories, i)
    # Plot the line collection for the agent
    lc_agent = pu.create_line_collection(
        x_smooth, y_smooth, elapsed_time, norm, cmap=cmap
    )
    artist.append(ax.add_collection(lc_agent)) if plot_trajectory else None

    alpha = lambda i: min(0.2 + 0.8 * i / len(df_trajectories), 1) if dynamic_alpha else 1
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
    ab.set_clip_on(True)
    ab.set_clip_path(ax.patch)
    artist.append(ax.add_artist(ab))

    if static_image_size > 0 and plot_trajectory:
        len_df = len(df_trajectories)
        if len_df == 0:
            return
        step = max(len_df // static_image_size, 1)
        for j in np.arange(0, len_df, step):
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
                ab_agent.set_clip_on(True)
                ab_agent.set_clip_path(ax.patch)
                artist.append(ax.add_artist(ab_agent))


def plot_agents(
    ax: Axes,
    df_trajectories: pd.DataFrame,
    i: int,
    artist: List[Artist],
    agent_num: int,
    norm: mcolors.Normalize,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 5,
    desired_width: float = 0.3,
    image_width_multi: float = 1.0,
    cmap: mcolors.Colormap = plt.cm.Blues,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
) -> None:
    """Plot the agents in the trajectory plot

    :param ax: Matplotlib axis
    :param df_trajectories: Dataframe with the trajectories
    :param i: Index of the trajectory
    :param artist: List of artists
    :param agent_num: Number of agents
    :param norm: Normalize object for the trajectory time
    :param image_agents_walk: List of images for the agents
    :param static_image_size: Number of static images to plot
    :param desired_width: Desired width of the agent
    :param image_width_multi: Multiplier for the image width
    :param cmap: Colormap for the trajectory line
    :param plot_trajectory: Whether to plot the trajectory line
    :param dynamic_alpha: Whether to use dynamic alpha values
    """
    for j in range(agent_num):
        agent_column_map = {
            f"position_x_agent{j}": "position_x",
            f"position_y_agent{j}": "position_y",
            f"velocity_agent{j}": "velocity",
            f"yaw_agent{j}": "yaw",
        }
        agent_df = df_trajectories.reindex(columns=agent_column_map.keys()).rename(
            columns=agent_column_map
        )
        plot_agent(
            ax,
            agent_df,
            i,
            artist,
            norm,
            image_agents_walk[j],
            static_image_size,
            desired_width=desired_width,
            image_width_multi=image_width_multi,
            cmap=cmap,
            plot_trajectory=plot_trajectory,
            dynamic_alpha=dynamic_alpha,
        )


def animate_trajectories(
    df_trajectories: pd.DataFrame,
    fig: Figure,
    ax: Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 3,
    time_freq: float = 0.1,
    colorbar: bool = False,
    animate: bool = True,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
) -> Optional[ArtistAnimation]:
    """Animate the trajectories of the agents

    :param df_trajectories: Dataframe with the trajectories
    :param fig: Matplotlib figure
    :param ax: Matplotlib axis
    :param agent_num: Number of agents
    :param norm: Normalize object for the trajectory time
    :param image_robot: Image for the robot
    :param image_agents_walk: List of images for the agents
    :param static_image_size: Number of static images to plot
    :param time_freq: Frequency of the time to plot
    :param colorbar: Whether to plot the colorbar or not
    :param animate: Whether to animate the plot or not
    :param image_width_multi: Multiplier for the image width
    :param plot_trajectory: Whether to plot the trajectory line or not
    :param dynamic_alpha: Whether to use dynamic alpha values
    :return: Matplotlib animation object
    """
    artists = []
    # Create the animation
    ani = None
    df_trajectories_t = _prepare_trajectory_time_index(df_trajectories)
    # Resample the data to the time frequency

    df_trajectories_rs = df_trajectories_t.asfreq(
        f"{int(1000 * time_freq)}ms", method="ffill"
    )
    if df_trajectories_rs.empty:
        print("No trajectory samples after resampling; skipping animation.")
        return None
    if animate:
        for i in trange(len(df_trajectories_rs)):
            artist = plot_trajectory_frame(
                df_trajectories_rs,
                i,
                ax,
                agent_num,
                norm,
                image_robot,
                image_agents_walk,
                static_image_size,
                colorbar,
                image_width_multi,
                plot_trajectory=plot_trajectory,
                dynamic_alpha=dynamic_alpha,
            )
            artists.append(artist)
        ani = ArtistAnimation(fig, artists, interval=int(1000 * time_freq), blit=True)
    else:
        artist = plot_trajectory_frame(
            df_trajectories_rs,
            len(df_trajectories_rs) - 1,
            ax,
            agent_num,
            norm,
            image_robot,
            image_agents_walk,
            static_image_size,
            colorbar,
            image_width_multi,
            plot_trajectory=plot_trajectory,
            dynamic_alpha=dynamic_alpha,
        )
        for art in artist:
            art.set_animated(False)
            ax.add_artist(art)
        fig.canvas.draw()

    if colorbar and artists:
        cb = plt.colorbar(artists[-1][0], ax=ax)
        cb.ax.set_ylabel(r"$\mathrm{ Time\ [s]}$")
    return ani


def plot_trajectory_frame(
    df_trajectories: pd.DataFrame,
    index: int,
    ax: Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray,
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 3,
    colorbar: bool = False,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
) -> List[Artist]:
    """Plot the trajectory frame at a specific index
    :param df_trajectories: Dataframe with the trajectories
    :param index: Index of the trajectory to plot
    :param ax: Matplotlib axis
    :param agent_num: Number of agents
    :param norm: Normalize object for the trajectory time
    :param image_robot: Image for the robot
    :param image_agents_walk: List of images for the agents
    :param static_image_size: Number of static images to plot
    :param colorbar: Whether to plot the colorbar or not
    :param image_width_multi: Multiplier for the image width
    :param plot_trajectory: Whether to plot the trajectory line or not
    :param dynamic_alpha: Whether to use dynamic alpha values

    :return: List of artists for the trajectory frame
    """
    artist = []
    robot_columns = ["position_x", "position_y", "velocity", "yaw"]
    plot_agent(
        ax,
        df_trajectories.reindex(columns=robot_columns),
        index,
        artist,
        norm,
        image_robot,
        static_image_size,
        desired_width=0.15,
        image_width_multi=image_width_multi,
        cmap=plt.cm.Reds,
        plot_trajectory=plot_trajectory,
        dynamic_alpha=dynamic_alpha,
    )
    plot_agents(
        ax,
        df_trajectories,
        index,
        artist,
        agent_num,
        norm,
        image_agents_walk,
        static_image_size,
        desired_width=0.13,
        image_width_multi=image_width_multi,
        plot_trajectory=plot_trajectory,
        dynamic_alpha=dynamic_alpha,
    )

    return artist


def plot_trajectories(
    df_trajectories: pd.DataFrame,
    ax: Axes,
    time_freq: float = 0.1,
    time_start: float = -np.inf,
    time_end: float = np.inf,
    fig: Figure = Figure(),
    colorbar: bool = False,
    animate: bool = True,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
    ax_limits: Optional[List[float]] = None,
) -> Optional[ArtistAnimation]:
    """Plot the trajectories of the agents

    :param df_trajectories: Dataframe with the trajectories
    :param ax: Matplotlib axis
    :param time_freq: Frequency of the time to plot
    :param time_start: Start time
    :param time_end: End time
    :param fig: Matplotlib figure
    :param colorbar: Whether to plot the colorbar or not
    :param animate: Whether to animate the plot or not
    :param image_width_multi: Multiplier for the image width
    :param plot_trajectory: Whether to plot the trajectory line or not
    :param dynamic_alpha: Whether to use dynamic alpha values
    :param ax_limits: Limits for the axis
    :return: Animation object with the trajectories of the experiment
    """

    # Filter the data
    df_trajectories = df_trajectories[
        (df_trajectories["time"] >= time_start) & (df_trajectories["time"] <= time_end)
    ]

    # agents
    agent_num = df_trajectories.keys().str.contains("position_x_agent").sum()
    norm = pu.find_norm_for_time([df_trajectories], agent_num)

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
        colorbar=colorbar,
        animate=animate,
        image_width_multi=image_width_multi,
        plot_trajectory=plot_trajectory,
        dynamic_alpha=dynamic_alpha,
    )
    if ax_limits is not None:
        pu.set_ax_limits(ax, ax_limits)
    else:
        pu.set_ax_limits_from_trajectories([df_trajectories], agent_num, ax)
    return ani


def plot_trajectories_aggregated(
    df_trajectories_list: List[pd.DataFrame],
    ax: Axes,
    time_freq: float = 0.1,
    time_start: float = -np.inf,
    time_end: float = np.inf,
    fig: Figure = Figure(),
    animate: bool = True,
    dynamic_alpha: bool = True,
    plot_trajectory: bool = True,
):
    """Plot the trajectories of the agents aggregated

    :param df_trajectories_list: List of dataframes with the trajectories
    :param ax: Matplotlib axis
    :param time_freq: Frequency of the time to plot
    :param time_start: Start time
    :param time_end: End time
    :param fig: Matplotlib figure
    :param animate: Whether to animate the plot or not
    :param dynamic_alpha: Whether to use dynamic alpha values
    :param plot_trajectory: Whether to plot the trajectory line or not
    """

    agent_num = df_trajectories_list[0].keys().str.contains("position_x_agent").sum()
    pu.set_ax_limits_from_trajectories(df_trajectories_list, agent_num, ax)
    norm = pu.find_norm_for_time(df_trajectories_list, agent_num)

    ani = animate_trajectories_aggregated(
        df_trajectories_list,
        fig,
        ax,
        agent_num,
        norm,
        time_freq=time_freq,
        animate=animate,
        image_width_multi=1.0,
        plot_trajectory=plot_trajectory,
    )
    return ani


def animate_trajectories_aggregated(
    df_trajectories_list: List[pd.DataFrame],
    fig: Figure,
    ax: Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    time_freq: float = 0.1,
    animate: bool = True,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
) -> Optional[ArtistAnimation]:
    """Animate the trajectories of the agents aggregated

    :param df_trajectories_list: List of dataframes with the trajectories
    :param fig: Matplotlib figure
    :param ax: Matplotlib axis
    :param agent_num: Number of agents
    :param norm: Normalize object for the trajectory time
    :param time_freq: Frequency of the time to plot
    :return: Matplotlib animation object
    """

    df_trajectories_rs_list = []
    for df_trajectories in df_trajectories_list:
        df_trajectories_t = _prepare_trajectory_time_index(df_trajectories)
        # Resample the data to the time frequency

        df_trajectories_rs = df_trajectories_t.asfreq(
            f"{int(1000 * time_freq)}ms", method="ffill"
        )
        df_trajectories_rs_list.append(df_trajectories_rs)

    if not df_trajectories_rs_list:
        print("No trajectories provided for aggregation; skipping animation.")
        return None

    max_len = max(len(df) for df in df_trajectories_rs_list)
    if max_len == 0:
        print("No aggregated trajectory samples after resampling; skipping animation.")
        return None

    artists = []
    for i in trange(max_len):
        artist = []
        plot_robot_aggregated(
            ax,
            df_trajectories_rs_list,
            i,
            artist,
            norm,
            plot_trajectory=plot_trajectory,
        )

        plot_agents_aggregated(
            ax,
            df_trajectories_rs_list,
            i,
            artist,
            agent_num,
            norm,
            plot_trajectory=plot_trajectory,
        )

        artists.append(artist)

    ani = ArtistAnimation(fig, artists, interval=int(1000 * time_freq), blit=True)
    if artists:
        cb = plt.colorbar(artists[-1][0], ax=ax)
        cb.ax.set_ylabel(r"$\mathrm{Time\ [s]}$")
    return ani


def plot_robot_aggregated(
    ax: Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: List[Artist],
    norm: mcolors.Normalize,
    plot_trajectory: bool = True,
) -> None:
    """
    Plot the robot trajectory aggregated

    :param ax: Matplotlib axis
    :param df_trajectories_list: List of dataframes with the trajectories
    :param i: Index of the trajectory
    :param artist: List of artists
    :param norm: Normalize object for the trajectory time
    :param plot_trajectory: Whether to plot the trajectory line or not
    """
    for df_trajectories in df_trajectories_list:

        x = np.array(df_trajectories["position_x"].iloc[: i + 1])
        y = np.array(df_trajectories["position_y"].iloc[: i + 1])

        # get a vector with the time steps from the start in seconds
        time_steps_from_start = df_trajectories.index - df_trajectories.index[0]
        time_steps_from_start = time_steps_from_start.total_seconds().values

        norm = mcolors.Normalize(vmin=0, vmax=time_steps_from_start.max())
        # Plot the line collection for the agnet
        lc_agent = pu.create_line_collection(x, y, time_steps_from_start, norm)
        alphas = 0.7
        lc_agent.set_alpha(alphas)
        lc_agent.set_linewidth(3)
        artist.append(ax.add_collection(lc_agent))


def plot_agents_aggregated(
    ax: Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: List[Artist],
    agent_num: int,
    norm: mcolors.Normalize,
) -> None:
    """
    Plot the agents trajectory aggregated

    :param ax: Matplotlib axis
    :param df_trajectories_list: List of dataframes with the trajectories
    :param i: Index of the trajectory
    :param artist: List of artists
    :param agent_num: Number of agents
    :param norm: Normalize object for the trajectory time
    """

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
    ax: Axes,
    df_trajectories_list: List[pd.DataFrame],
    i: int,
    artist: List[Artist],
    norm: mcolors.Normalize,
) -> None:
    """
    Plot the agent trajectory aggregated

    :param ax: Matplotlib axis
    :param df_trajectories_list: List of dataframes with the trajectories
    :param i: Index of the trajectory
    :param artist: List of artists
    :param norm: Normalize object for the trajectory time
    """

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
        # lc_agent.set_capstyle = "round"
        lc_agent.set_linestyle((0, (1, 10)))
        artist.append(ax.add_collection(lc_agent))
