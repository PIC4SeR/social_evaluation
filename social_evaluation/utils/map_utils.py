import pandas as pd
import numpy as np

from enum import Enum
from typing import List, Tuple, Dict
from pathlib import Path
from yaml import safe_load
from cv2 import imread, IMREAD_GRAYSCALE, IMREAD_COLOR, IMREAD_UNCHANGED
from tf_transformations import euler_from_quaternion, quaternion_from_euler


class GridmapMapCost(Enum):
    """Enum class for the cost of the gridmap"""

    FREE = 0
    OCCUPIED = 100
    UNKNOWN = -1


def orientation_around_z_axis(angle: float) -> np.ndarray:
    """
    Create a quaternion orientation around the Z axis

    :param angle: Angle to rotate
    :return: Quaternion orientation
    """
    q = quaternion_from_euler(0, 0, angle)
    return np.array(q)


def get_map_data(map_path: Path) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Get the map data from a map file

    :param map_path: Path to the map file
    :return: Occupancy gridmap and metadata
    """
    # Load data
    with open(map_path, "r") as file:
        data = safe_load(file)
    metadata = data
    # Get map data
    metadata["image_path"] = Path(metadata.pop("image"))
    assert metadata["image_path"].exists()
    im = imread(str(metadata["image_path"]), IMREAD_UNCHANGED)
    metadata["image_raw"] = im
    metadata["occupancy_gridmap"] = image_to_occupancy_gridmap(im, metadata)
    metadata["width"] = metadata["occupancy_gridmap"].shape[1]
    metadata["height"] = metadata["occupancy_gridmap"].shape[0]

    return metadata["occupancy_gridmap"], metadata


def image_to_occupancy_gridmap(im: np.ndarray, metadata: Dict[str, any]) -> np.ndarray:
    """
    Convert an image to an occupancy gridmap

    :param im: Image to convert
    :param metadata: Metadata of the image
    :return: Occupancy gridmap
    """
    # Get map data
    mode = metadata["mode"]
    negate = metadata["negate"]
    occupied_thresh = metadata["occupied_thresh"]
    free_thresh = metadata["free_thresh"]

    # Check if image is grayscale
    assert len(im.shape) == 2
    # check if image is normalized
    if im.max() > 1:
        im = im / 255.0

    # Convert image to occupancy gridmap
    map_data = np.full(im.shape, GridmapMapCost.UNKNOWN.value)

    image = im if negate else 1 - im
    match mode:
        case "trinary":
            map_data = np.where(
                image < free_thresh, GridmapMapCost.FREE.value, map_data
            )
            map_data = np.where(
                image > occupied_thresh, GridmapMapCost.OCCUPIED.value, map_data
            )
        case "scale":
            map_data = np.where(
                image < free_thresh, GridmapMapCost.FREE.value, map_data
            )
            map_data = np.where(
                image > occupied_thresh, GridmapMapCost.OCCUPIED.value, map_data
            )
            map_data = np.where(
                (image >= free_thresh) & (image <= occupied_thresh),
                (image - free_thresh) / (occupied_thresh - free_thresh),
                map_data,
            )
        case _:
            raise ValueError(f"Invalid mode: {mode}")
    return map_data


def retrieve_obstacle_positions(
    occupancy_gridmap: np.ndarray, metadata: Dict[str, any]
) -> pd.DataFrame:
    """Retrieve the positions of the obstacles in the occupancy gridmap

    :param occupancy_gridmap: Occupancy gridmap
    :param metadata: Metadata of the occupancy gridmap
    :return: DataFrame with the positions of the obstacles
    """

    # Get map data
    resolution = metadata["resolution"]
    origin = np.array(metadata["origin"])

    orientation = orientation_around_z_axis(origin[2])

    # Get obstacle positions
    obstacle_positions = []

    for y in range(metadata["height"]):
        for x in range(metadata["width"]):
            if occupancy_gridmap[y, x] == GridmapMapCost.OCCUPIED.value:
                obstacle_positions.append(
                    {
                        "x": round(x * resolution + origin[0], 1),
                        "y": round(
                            -y * resolution
                            + origin[1]
                            + resolution * metadata["height"],
                            1,
                        ),
                    }
                )

    return pd.DataFrame(obstacle_positions).drop_duplicates(
        subset=["x", "y"], ignore_index=True
    )


def find_closest_obstacle_single(
    agent: pd.Series,
    df_obstacles: pd.DataFrame,
    robot_radius: float = 0.2,
    max_radius: float = 10.0,
):
    """Find the closest obstacle to an agent"""
    df_obstacles_copy = df_obstacles.copy()

    df_obstacles_copy["dist"] = df_obstacles_copy.apply(
        lambda x: np.linalg.norm(
            np.array([x["x"], x["y"]])
            - np.array([agent["position_x"], agent["position_y"]]),
            axis=0,
        ),
        axis=1,
    )
    return df_obstacles_copy.loc[df_obstacles_copy["dist"].idxmin()]


def find_closest_obstacle(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    robot_radius: float = 0.2,
    max_radius: float = 10.0,
):
    """Find the closest obstacle to an agent"""
    df_agent_ = df_agent.copy()
    df_agent_.dropna(subset=["time", "position_x"], inplace=True)

    df_agent_[["closest_obs_x", "closest_obs_y", "closest_obs_dist"]] = df_agent_.apply(
        lambda x: find_closest_obstacle_single(
            x, df_obstacles, robot_radius, max_radius
        ),
        axis=1,
    )
    return df_agent_[["closest_obs_x", "closest_obs_y", "closest_obs_dist"]]
