from typing import Any, Callable, Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.offsetbox import AnnotationBbox
from matplotlib.patches import Circle
from matplotlib.animation import Animation, FuncAnimation
from scipy.ndimage import gaussian_filter1d

from tqdm import tqdm
import social_evaluation.trajectory_plots as pu
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist


def _visible(cmap: mcolors.Colormap, low: float = 0.35) -> mcolors.Colormap:
    """The ``[low, 1]`` part of a sequential colormap.

    Reds, Oranges, ... start at near-white, so a trail coloured by elapsed time
    was invisible on a white background for its first stretch.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_visible", cmap(np.linspace(low, 1.0, 256))
    )


def _frame_by_frame(
    fig: Figure,
    n_frames: int,
    draw: Callable[[int], List[Artist]],
    time_freq: float,
) -> FuncAnimation:
    """An animation whose frames are built only when they are drawn.

    ArtistAnimation needs every frame's artists up front, and each frame holds
    its own rotated copy of every sprite (a TB2 sprite is 3.4 MB before
    rotation): a 592-frame four-robot run grew by 60-130 MB a frame and was
    killed at frame 435. Here only the current frame's artists exist; the
    previous frame's are removed before the next one is built.
    """
    current: List[Artist] = []
    progress = tqdm(total=n_frames, unit="frame")

    def update(i: int) -> List[Artist]:
        for art in current:
            art.remove()
        current[:] = draw(i)
        if i < progress.n:  # a second save starts over
            progress.reset()
        progress.update(i + 1 - progress.n)
        return list(current)

    def keep() -> List[Artist]:
        # Also runs on the first plain canvas draw (the PNG's bounding box),
        # which must not replace the last frame the PNG is meant to show.
        return list(current)

    return FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=keep,
        interval=int(1000 * time_freq),
        blit=True,
    )


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


def _merge_fleet_trajectories(
    df_trajectories: pd.DataFrame,
    df_robots: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align robot trajectories with the existing combined human frame."""
    agent_columns = [c for c in df_trajectories if c == "time" or "_agent" in c]
    merged = df_trajectories[agent_columns].copy()
    for index, trajectory in enumerate(df_robots.values()):
        columns = trajectory[["time", "position_x", "position_y", "yaw"]].rename(
            columns={
                "position_x": f"position_x_robot{index}",
                "position_y": f"position_y_robot{index}",
                "yaw": f"yaw_robot{index}",
            }
        )
        merged = merged.merge(columns, on="time", how="outer")
    merged.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    numeric_columns = merged.select_dtypes(include=np.number).columns.drop("time")
    merged[numeric_columns] = merged[numeric_columns].interpolate(
        limit_direction="both"
    )
    merged.ffill(inplace=True)
    merged.bfill(inplace=True)
    return merged


# Legacy marker sizes [m], used when no ``marker_sizes`` are configured.
LEGACY_ROBOT_WIDTH, LEGACY_AGENT_WIDTH = 0.15, 0.13


def marker_size(sizes: Optional[Dict[str, Any]], name: Optional[str]) -> float:
    """Icon size [m] for a robot ``name`` (None = a pedestrian) under ``sizes``.

    ``sizes`` is the trajectory config's ``marker_sizes`` block::

        marker_sizes:
          robots: {turtlebot2: 0.356, jackal: 0.49, default: 0.4}
          agent: 0.5
          agent_collision_radius: 0.4

    A robot takes the longest ``robots`` key its name starts with (so
    ``turtlebot2_1`` takes ``turtlebot2``), else ``default``. The size is the
    icon's vertical side, which is the robot's length for the robot icons.
    """
    if sizes is None:
        return LEGACY_AGENT_WIDTH if name is None else LEGACY_ROBOT_WIDTH
    if name is None:
        return float(sizes.get("agent", LEGACY_AGENT_WIDTH))
    robots = sizes.get("robots") or {}
    keys = [k for k in robots if k != "default" and str(name).startswith(k)]
    if keys:
        return float(robots[max(keys, key=len)])
    return float(robots.get("default", LEGACY_ROBOT_WIDTH))


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
    label: Optional[str] = None,
    exact_scale: bool = False,
    collision_radius: float = 0.0,
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
    :param exact_scale: Draw the icon at exactly ``desired_width`` metres
    :param collision_radius: If > 0, also draw a circle of this radius [m]
        around the current position (e.g. HuNav's collision cylinder)
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
    if label is not None and i == len(df_trajectories) - 1:
        lc_agent.set_label(label)

    # Add the agent image with rotation
    ab = pu.create_annotation_box(
        image_agent,
        x_smooth[-1],
        y_smooth[-1],
        desired_width_px=desired_width_px,
        angle=yaw_smooth[-1],
        alpha=alpha(i),
        exact_scale=exact_scale,
    )
    ab.set_clip_on(True)
    ab.set_clip_path(ax.patch)
    artist.append(ax.add_artist(ab))
    if collision_radius > 0:
        ring = Circle((x_smooth[-1], y_smooth[-1]), collision_radius, fill=False,
                      edgecolor=cmap(0.75), linewidth=1.2, alpha=0.6)
        ring.set_clip_path(ax.patch)
        artist.append(ax.add_patch(ring))

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
                    exact_scale=exact_scale,
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
    exact_scale: bool = False,
    collision_radius: float = 0.0,
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
            exact_scale=exact_scale,
            collision_radius=collision_radius,
        )


def animate_trajectories(
    df_trajectories: pd.DataFrame,
    fig: Figure,
    ax: Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray | List[np.ndarray],
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 3,
    time_freq: float = 0.1,
    colorbar: bool = False,
    animate: bool = True,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
    robot_names: Optional[List[str]] = None,
    sizes: Optional[Dict[str, Any]] = None,
) -> Optional[Animation]:
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
    ani = None
    df_trajectories_t = _prepare_trajectory_time_index(df_trajectories)
    # Resample the data to the time frequency

    df_trajectories_rs = df_trajectories_t.asfreq(
        f"{int(1000 * time_freq)}ms", method="ffill"
    )
    if df_trajectories_rs.empty:
        print("No trajectory samples after resampling; skipping animation.")
        return None

    def draw(i: int) -> List[Artist]:
        return plot_trajectory_frame(
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
            robot_names=robot_names,
            sizes=sizes,
        )

    # The last frame carries the legend labels and the colorbar's mappable.
    artist = draw(len(df_trajectories_rs) - 1)
    if not animate:
        for art in artist:
            art.set_animated(False)
            ax.add_artist(art)
    # Also when animating: the draw maps the trails' colours, which the legend
    # copies -- before it every entry was the default blue.
    fig.canvas.draw()

    if robot_names and plot_trajectory:
        handles, labels = ax.get_legend_handles_labels()
        unique_handles = dict(zip(labels, handles))
        ax.legend(
            unique_handles.values(), unique_handles.keys(), fontsize=9, ncol=2
        )

    if colorbar and animate:
        cb = plt.colorbar(artist[0], ax=ax)
        cb.ax.set_ylabel(r"$\mathrm{ Time\ [s]}$")
    if animate:
        for art in artist:
            art.remove()
        ani = _frame_by_frame(fig, len(df_trajectories_rs), draw, time_freq)
    return ani


def plot_trajectory_frame(
    df_trajectories: pd.DataFrame,
    index: int,
    ax: Axes,
    agent_num: int,
    norm: mcolors.Normalize,
    image_robot: np.ndarray | List[np.ndarray],
    image_agents_walk: List[np.ndarray],
    static_image_size: int = 3,
    colorbar: bool = False,
    image_width_multi: float = 1.0,
    plot_trajectory: bool = True,
    dynamic_alpha: bool = True,
    robot_names: Optional[List[str]] = None,
    sizes: Optional[Dict[str, Any]] = None,
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

    :param sizes: Marker sizes in metres, drawn to scale (see marker_size);
        None keeps the legacy fixed sizes (0.15 m robots, 0.13 m agents)
    :return: List of artists for the trajectory frame
    """
    artist = []
    exact = sizes is not None
    agent_width = marker_size(sizes, None)
    collision_radius = float((sizes or {}).get("agent_collision_radius", 0.0))
    if robot_names:
        robot_cmaps = [
            _visible(cmap)
            for cmap in (
                plt.cm.Reds,
                plt.cm.Oranges,
                plt.cm.Greens,
                plt.cm.Purples,
                plt.cm.Greys,
                plt.cm.YlOrBr,
            )
        ]
        for robot_index, robot_name in enumerate(robot_names):
            robot_column_map = {
                f"position_x_robot{robot_index}": "position_x",
                f"position_y_robot{robot_index}": "position_y",
                f"yaw_robot{robot_index}": "yaw",
            }
            robot_df = df_trajectories.reindex(
                columns=robot_column_map.keys()
            ).rename(columns=robot_column_map)
            plot_agent(
                ax,
                robot_df,
                index,
                artist,
                norm,
                image_robot[robot_index],
                static_image_size,
                desired_width=marker_size(sizes, robot_name),
                image_width_multi=image_width_multi,
                cmap=robot_cmaps[robot_index % len(robot_cmaps)],
                plot_trajectory=plot_trajectory,
                dynamic_alpha=dynamic_alpha,
                label=robot_name,
                exact_scale=exact,
            )
    else:
        robot_columns = ["position_x", "position_y", "velocity", "yaw"]
        plot_agent(
            ax,
            df_trajectories.reindex(columns=robot_columns),
            index,
            artist,
            norm,
            image_robot,
            static_image_size,
            desired_width=marker_size(sizes, "robot"),
            image_width_multi=image_width_multi,
            cmap=plt.cm.Reds,
            plot_trajectory=plot_trajectory,
            dynamic_alpha=dynamic_alpha,
            exact_scale=exact,
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
        desired_width=agent_width,
        image_width_multi=image_width_multi,
        plot_trajectory=plot_trajectory,
        dynamic_alpha=dynamic_alpha,
        exact_scale=exact,
        collision_radius=collision_radius,
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
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    sizes: Optional[Dict[str, Any]] = None,
) -> Optional[Animation]:
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

    agent_num = df_trajectories.keys().str.contains("position_x_agent").sum()
    if df_robots:
        filtered_robots = {
            name: trajectory[
                (trajectory["time"] >= time_start)
                & (trajectory["time"] <= time_end)
            ]
            for name, trajectory in df_robots.items()
        }
        df_trajectories = _merge_fleet_trajectories(
            df_trajectories, filtered_robots
        )
        norm = pu.find_norm_for_time([df_trajectories], agent_num)
        robot_names = list(filtered_robots)
        image_robot, image_agents_walk, _ = pu.get_images_for_experiment(
            agent_num, robot_names
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
            robot_names=robot_names,
            sizes=sizes,
        )
        if ax_limits is not None:
            pu.set_ax_limits(ax, ax_limits)
        else:
            x_columns = [c for c in df_trajectories if c.startswith("position_x_")]
            y_columns = [c for c in df_trajectories if c.startswith("position_y_")]
            ax.set_xlim(
                df_trajectories[x_columns].min().min() - 1,
                df_trajectories[x_columns].max().max() + 1,
            )
            ax.set_ylim(
                df_trajectories[y_columns].min().min() - 1,
                df_trajectories[y_columns].max().max() + 1,
            )
        return ani

    # agents
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
        sizes=sizes,
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
) -> Optional[Animation]:
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

    def draw(i: int) -> List[Artist]:
        artist: List[Artist] = []
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
        return artist

    last = draw(max_len - 1)
    if last:
        cb = plt.colorbar(last[0], ax=ax)
        cb.ax.set_ylabel(r"$\mathrm{Time\ [s]}$")
    for art in last:
        art.remove()
    return _frame_by_frame(fig, max_len, draw, time_freq)


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
