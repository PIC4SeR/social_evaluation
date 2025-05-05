import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

from scipy.ndimage import rotate
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

pkg_dir = Path(__file__).resolve().parents[0]
images_dir = Path(pkg_dir, "images")


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


def meters_to_pixels(meters: float, ax: plt.Axes) -> float:
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
    cmap: str = "viridis",
) -> mcoll.LineCollection:
    """
    Create a line collection for a trajectory.

    :param x: X-coordinates of the trajectory
    :param y: Y-coordinates of the trajectory
    :param velocity: Velocity values for each point
    :param norm: Normalize object for the velocity
    :param cmap: Colormap name
    :return: Line collection object
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, cmap=cmap, norm=norm)
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
    ax: plt.Axes,
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
    extent = [
        origin[0],
        origin[0] + map.shape[1] * resolution,
        origin[1],
        origin[1] + map.shape[0] * resolution,
    ]
    # use a resolution of 0.1 and set the origin of the map to the bottom left corner
    ax.imshow(map, cmap=color_map, extent=extent)


def plot_goal(goal: Tuple[float, float], ax: plt.Axes, radius: float = 0.3) -> None:
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
    ax.add_artist(ab_goal)


def plot_path(
    df_paths: pd.DataFrame, ax: plt.Axes, color: str = "green", linewidth: float = 1
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


def retrieve_image_from_path(image_path: str) -> np.ndarray:
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
) -> Tuple[float, float]:
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
    df_trajectories_list: List[pd.DataFrame], agent_num: int, ax: plt.Axes
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
    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)


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
