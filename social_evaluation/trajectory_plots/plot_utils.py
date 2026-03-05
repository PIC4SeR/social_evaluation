from matplotlib.transforms import Bbox
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, List, Optional, Tuple, Union
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pkg_dir = Path(__file__).resolve().parents[0]
images_dir = Path(pkg_dir, "images")


def apply_trajectory_plot_rcparams() -> None:
    """Use LaTeX serif styling for trajectory plots."""
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["font.weight"] = "bold"


def get_bbox_for_figure(fig: Figure, ax: Axes) -> Bbox:
    """
    Get the bounding box for the figure that includes all elements of the plot.

    :param fig: Figure object
    :param ax: Axis object
    :return: Bounding box object
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # start from the axes rectangle
    bbox = ax.get_window_extent(renderer)

    # include title, labels, and ticks in the bounding box
    extras = []
    for t in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        if t.get_text():
            extras.append(t.get_window_extent(renderer))

    bbox = Bbox.union([bbox, *extras]).transformed(fig.dpi_scale_trans.inverted())
    # add padding
    return bbox


def extract_elapsed_time_seconds(df: pd.DataFrame, i: int) -> np.ndarray:
    """Return elapsed time in seconds for samples up to index *i*."""
    time_source = df.index[: i + 1]

    raw_values = time_source.to_numpy()

    if raw_values.size == 0:
        return np.array([], dtype=float)

    dtype = raw_values.dtype

    if dtype == object:
        try:
            datetime_values = pd.to_datetime(raw_values)
            raw_values = datetime_values.to_numpy()
            dtype = raw_values.dtype
        except (ValueError, TypeError):
            numeric_values = pd.to_numeric(raw_values, errors="coerce")
            if np.isnan(numeric_values).all():
                return np.linspace(0, raw_values.size - 1, raw_values.size)
            raw_values = numeric_values.to_numpy()
            dtype = raw_values.dtype

    if np.issubdtype(dtype, np.datetime64):
        deltas = raw_values - raw_values[0]
        return (deltas / np.timedelta64(1, "s")).astype(float)

    if np.issubdtype(dtype, np.timedelta64):
        return (raw_values / np.timedelta64(1, "s")).astype(float)

    numeric_values = raw_values.astype(float, copy=False)
    return numeric_values - numeric_values[0]


def normalize_goal(goal_value: Any) -> Optional[Tuple[float, float]]:
    """Convert supported goal specs into a numeric tuple."""
    if goal_value is None:
        return None
    if isinstance(goal_value, dict):
        if "x" in goal_value and "y" in goal_value:
            return (float(goal_value["x"]), float(goal_value["y"]))
        print(f"Goal dict must contain 'x' and 'y' keys. Skipping: {goal_value}")
        return None
    if isinstance(goal_value, (list, tuple)):
        if len(goal_value) != 2:
            print(
                f"Goal list/tuple must have exactly two entries. Skipping: {goal_value}"
            )
            return None
        return (float(goal_value[0]), float(goal_value[1]))
    print(
        f"Unsupported goal specification type {type(goal_value)}. Expected dict, list, or tuple."
    )
    return None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image data to ensure valid RGB ranges.
    Floats are clipped to [0, 1], integers to [0, 255].

    :param image: Image data as numpy array
    :return: Normalized image data
    """
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Clip float data to [0, 1]
        image = np.clip(image, 0, 1)
    elif image.dtype == np.int32 or image.dtype == np.int64 or np.any(image > 255):
        # Clip integer data to [0, 255] and convert to uint8
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def meters_to_pixels(meters: float, ax: Axes) -> float:
    """
    Convert meters to pixels using the current axis limits.

    :param meters: Length in meters
    :param ax: Axis object
    :return: Length in pixels
    """
    x0, x1 = ax.get_xbound()
    width_px = ax.get_window_extent().width
    width_m = x1 - x0
    return meters * width_px / width_m


def create_line_collection(
    x: np.ndarray,
    y: np.ndarray,
    velocity: np.ndarray,
    norm: mcolors.Normalize,
    cmap: Optional[mcolors.Colormap] = None,
    color: Optional[str] = None,
) -> mcoll.LineCollection:
    """
    Create a line collection for a trajectory.

    :param x: X-coordinates of the trajectory
    :param y: Y-coordinates of the trajectory
    :param velocity: Velocity values for each point
    :param norm: Normalize object for the velocity
    :param cmap: Colormap name
    :param color: Single color for the line (overrides colormap if provided)
    :return: Line collection object
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)

    if color is not None:
        lc.set_color(color)
    else:
        lc.set_array(velocity)
    return lc


def create_image_marker(
    image: np.ndarray, desired_width_px: float = 20, angle: float = 0, alpha: float = 1
) -> OffsetImage:
    """
    Create an image marker for a point on the plot.

    :param image: Image data as numpy array
    :param desired_width_px: Desired width of the image in pixels
    :param angle: Rotation angle in radians
    :param alpha: Transparency value
    :return: OffsetImage object
    """
    rotated_image = normalize_image(rotate(image, np.degrees(angle), reshape=True))
    return OffsetImage(
        rotated_image,
        zoom=desired_width_px / image.shape[0],
        alpha=alpha,
    )


def create_annotation_box(
    image: np.ndarray,
    x: float,
    y: float,
    desired_width_px: float = 20,
    angle: float = 0,
    alpha: float = 1,
) -> AnnotationBbox:
    """
    Create an annotation box with an image marker for a point on the plot.

    :param image: Image data as numpy array
    :param x: X-coordinate of the point
    :param y: Y-coordinate of the point
    :param desired_width_px: Desired width of the image in pixels
    :param angle: Rotation angle in radians
    :param alpha: Transparency value
    :return: AnnotationBbox object
    """
    image_rot = create_image_marker(
        image, desired_width_px=desired_width_px, angle=angle, alpha=alpha
    )
    return AnnotationBbox(image_rot, (x, y), frameon=False)


def plot_map(
    map: np.ndarray,
    ax: Axes,
    resolution: float = 0.05,
    origin: Tuple[float, float] = (-8.41, -12.7),
) -> None:
    """Plot the map

    :param map: Numpy array with the map
    :param ax: Axis object
    :param resolution: Resolution of the map
    :param origin: Origin of the map
    """
    color_map = mcolors.ListedColormap(["white", "black"])
    extent = (
        origin[0],
        origin[0] + map.shape[1] * resolution,
        origin[1],
        origin[1] + map.shape[0] * resolution,
    )
    # use a resolution of 0.1 and set the origin of the map to the bottom left corner
    img = ax.imshow(map, cmap=color_map, extent=extent)
    img.set_in_layout(
        False
    )  # Exclude the image from the layout to prevent tight_layout issues
    img.set_clip_on(True)
    img.set_clip_path(ax.patch)


def plot_goal(goal: Tuple[float, float], ax: Axes, radius: float = 0.2) -> None:
    """Plot the goal position on the map

    :param goal: Tuple with the goal position
    :param ax: Axis object
    :param radius: Radius of the goal
    """
    goal_image_path = Path(images_dir, "goal.png")
    image_goal = plt.imread(goal_image_path, format="png")

    desired_width = 2 * radius
    desired_width_px = meters_to_pixels(desired_width, ax)

    ab_goal = AnnotationBbox(
        create_image_marker(image_goal, desired_width_px=desired_width_px),
        goal,
        frameon=False,
    )
    ab_goal.set_clip_on(True)
    ab_goal.set_clip_path(ax.patch)
    ax.add_artist(ab_goal)


def plot_path(
    df_paths: pd.DataFrame, ax: Axes, color: str = "green", linewidth: float = 1
) -> None:
    """Plot the path of the robot on the map

    :param df_paths: Dataframe with the path
    :param ax: Axis object
    :param color: Color of the path
    :param linewidth: Width of the path
    """
    # take the first path
    if df_paths.empty:
        return
    df_path = df_paths.iloc[-1]["poses"]
    # transform the path to dataframe
    df_path = pd.DataFrame(df_path)
    df_path.plot(
        x="position_x",
        y="position_y",
        ax=ax,
        color=color,
        linewidth=linewidth,
        alpha=0.5,
    )
    # remove the legend
    ax.get_legend().remove()


def retrieve_image_from_path(image_path: str | Path) -> np.ndarray:
    """Retrieve image from path

    :param image_path: Path to the image
    :return: Numpy array with the image
    """

    return np.rot90(plt.imread(image_path, format="png"), 3)


def get_images_for_experiment(
    agent_num: int,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Get the images for the robot and agents

    :param agent_num: Number of agents
    :return: Tuple with the robot image, list with the images for the agents walking and list with the images for the agents standing
    """
    robot_image_path = Path(pkg_dir, "images", "robot.png")
    robot_image = retrieve_image_from_path(robot_image_path)

    agents_image_path_stand = [
        Path(pkg_dir, "images", f"agent{i % 3}_stand.png") for i in range(agent_num)
    ]
    agents_image_path_walk = [
        Path(pkg_dir, "images", f"agent{i % 3}_walk.png") for i in range(agent_num)
    ]
    image_agents_stand = [
        retrieve_image_from_path(image_path) for image_path in agents_image_path_stand
    ]
    image_agents_walk = [
        retrieve_image_from_path(image_path) for image_path in agents_image_path_walk
    ]

    return robot_image, image_agents_walk, image_agents_stand


def find_ax_limits(
    df_trajectories: pd.DataFrame, agent_num: int
) -> Tuple[float, float, float, float]:
    """
    Find the limits for the axis

    :param df_trajectories: Dataframe with the trajectories
    :param agent_num: Number of agents
    :return: Tuple with the limits for the axis (min_x, max_x, min_y, max_y)
    """
    min_x = df_trajectories["position_x"].min()
    max_x = df_trajectories["position_x"].max()
    min_y = df_trajectories["position_y"].min()
    max_y = df_trajectories["position_y"].max()

    for i in range(0, agent_num):
        min_x = min(min_x, df_trajectories[f"position_x_agent{i}"].min())
        max_x = max(max_x, df_trajectories[f"position_x_agent{i}"].max())
        min_y = min(min_y, df_trajectories[f"position_y_agent{i}"].min())
        max_y = max(max_y, df_trajectories[f"position_y_agent{i}"].max())

    return min_x - 1, max_x + 1, min_y - 1, max_y + 1


def set_ax_limits_from_trajectories(
    df_trajectories_list: List[pd.DataFrame], agent_num: int, ax: Axes
) -> None:
    """
    Set the limits for the axis from the trajectories

    :param df_trajectories_list: List with the dataframes with the trajectories
    :param agent_num: Number of agents
    :param ax: Axis object
    """
    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for df_trajectories in df_trajectories_list:
        min_x_, max_x_, min_y_, max_y_ = find_ax_limits(df_trajectories, agent_num)
        min_x = min(min_x, min_x_)
        max_x = max(max_x, max_x_)
        min_y = min(min_y, min_y_)
        max_y = max(max_y, max_y_)
    try:
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)
    except ValueError:
        print(
            f"Could not set axis limits: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}"
        )


def set_ax_limits(ax: Axes, ax_limits: List[float]) -> None:
    """
    Set the limits for the axis

    :param ax: Axis object
    :param ax_limits: List with the limits for the axis [min_x, max_x, min_y, max_y]
    """
    if ax_limits is not None and len(ax_limits) == 4:
        ax.set_xlim(ax_limits[0], ax_limits[1])
        ax.set_ylim(ax_limits[2], ax_limits[3])


def find_min_max_velocity(
    df_trajectories: pd.DataFrame, agent_num: int
) -> Tuple[float, float]:
    """
    Find the minimum and maximum velocity for the agents

    :param df_trajectories: Dataframe with the trajectories
    :param agent_num: Number of agents

    return: Tuple with the minimum and maximum velocity
    """
    df_trajectories.dropna(subset=["velocity_x", "velocity_y"], inplace=True)
    df_trajectories["velocity"] = np.linalg.norm(
        df_trajectories[["velocity_x", "velocity_y"]].values, axis=1
    )

    min_velocity = df_trajectories["velocity"].min()
    max_velocity = df_trajectories["velocity"].max()

    for i in range(0, agent_num):
        df_trajectories[f"velocity_agent{i}"] = np.linalg.norm(
            df_trajectories[[f"velocity_x_agent{i}", f"velocity_y_agent{i}"]].values,
            axis=1,
        )
        min_velocity = min(min_velocity, df_trajectories[f"velocity_agent{i}"].min())
        max_velocity = max(max_velocity, df_trajectories[f"velocity_agent{i}"].max())

    return min_velocity, max_velocity


def find_norm_for_velocity(
    df_trajectories_list: List[pd.DataFrame], agent_num: int
) -> mcolors.Normalize:
    """Find the norm for the velocity

    :param df_trajectories: Dataframe with the trajectories
    :param agent_num: Number of agents
    :return: Normalize object for the velocity
    """
    min_velocity, max_velocity = np.inf, -np.inf
    for df_trajectories in df_trajectories_list:
        min_velocity_, max_velocity_ = find_min_max_velocity(df_trajectories, agent_num)
        min_velocity = min(min_velocity, min_velocity_)
        max_velocity = max(max_velocity, max_velocity_)
    return mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)


def _duration_seconds_from_source(
    source: Union[pd.Series, pd.Index, np.ndarray],
) -> Optional[float]:
    """Compute span in seconds represented by a time-like source."""
    if isinstance(source, pd.Index):
        values = source.to_series(index=None)
    elif isinstance(source, pd.Series):
        values = source.copy()
    else:
        values = pd.Series(source)

    values = values.dropna()
    if values.empty:
        return None

    dtype = values.dtype

    if np.issubdtype(dtype, np.datetime64):
        diff = values.max() - values.min()
        return float(diff.total_seconds())

    if np.issubdtype(dtype, np.timedelta64):
        diff = values.max() - values.min()
        return float(diff.total_seconds())

    if np.issubdtype(dtype, np.number):
        return float(values.max() - values.min())

    datetime_values = pd.to_datetime(values, errors="coerce")
    datetime_values = datetime_values.dropna()
    if not datetime_values.empty:
        diff = datetime_values.max() - datetime_values.min()
        return float(diff.total_seconds())

    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_values = numeric_values.dropna()
    if not numeric_values.empty:
        return float(numeric_values.max() - numeric_values.min())

    return None


def find_norm_for_time(
    df_trajectories_list: List[pd.DataFrame], agent_num: int
) -> mcolors.Normalize:
    """Find the norm for the time

    :param df_trajectories: Dataframe with the trajectories
    :param agent_num: Number of agents
    :return: Normalize object for the time
    """
    max_duration = 0.0
    for df_trajectories in df_trajectories_list:
        if "time" in df_trajectories.columns:
            time_source = df_trajectories["time"]
        elif isinstance(df_trajectories.index, pd.Index):
            time_source = df_trajectories.index
        else:
            continue

        duration = _duration_seconds_from_source(time_source)
        if duration is None:
            continue
        max_duration = max(max_duration, duration)

    if not np.isfinite(max_duration) or max_duration <= 0:
        max_duration = 1.0

    return mcolors.Normalize(vmin=0.0, vmax=max_duration)
