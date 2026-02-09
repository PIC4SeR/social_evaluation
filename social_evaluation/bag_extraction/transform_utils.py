from logging import warning
import pandas as pd

# get the path to the bag file

from tf_transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    compose_matrix,
    decompose_matrix,
    inverse_matrix,
    concatenate_matrices,
)
import numpy as np
from typing import List, Tuple, Union, Dict


def _spline_interpolate_with_fallback(
    series: pd.Series, order: int = 3
) -> pd.Series:
    """Interpolate using a spline when possible, otherwise fall back gracefully."""

    valid_points = series.count()
    if valid_points == 0:
        return series
    if valid_points == 1:
        return series.ffill().bfill()

    effective_order = min(max(1, valid_points - 1), order)

    try:
        return series.interpolate(
            method="spline",
            order=effective_order,
            limit_direction="both",
        )
    except (ValueError, TypeError) as exc:
        warning(
            "Falling back to linear interpolation for %s due to: %s",
            getattr(series, "name", "series"),
            exc,
        )
        return series.interpolate(method="linear", limit_direction="both")


def get_transform_between_frames(
    df: pd.DataFrame, frame_id: str, child_frame_id: str, time: float
) -> np.ndarray:
    """Get the transform between two frames at a given time in the tf tree.

    :param df: DataFrame containing the tf tree
    :param frame_id: The frame_id of the parent frame
    :param child_frame_id: The frame_id of the child frame
    :param time: The time at which to get the transform
    :return: The transform between the two frames
    """

    if frame_id == child_frame_id:
        warning("frame_id and child_frame_id are the same.")
        return np.eye(4)
    ordered_sequence = find_ordered_sequence_between_frames(
        df, frame_id, child_frame_id, time
    )
    if ordered_sequence is None:
        raise ValueError(
            f"Could not find a transform between {frame_id} and {child_frame_id} at time {time}."
        )
    return concatenate_matrices(
        *[transform["transform"] for transform in ordered_sequence]
    )


def get_pose_data_from_transform(transform: np.ndarray) -> pd.Series:
    """Get the position and orientation data from a transform matrix.

    :param transform: The transform matrix
    :return: A pandas Series containing the position and orientation data
    """
    _, _, rotation, translation, _ = decompose_matrix(transform)
    quaternion_rotation = quaternion_from_euler(*rotation)
    return pd.Series(
        {
            "position_x": translation[0],
            "position_y": translation[1],
            "position_z": translation[2],
            "qx": quaternion_rotation[0],
            "qy": quaternion_rotation[1],
            "qz": quaternion_rotation[2],
            "qw": quaternion_rotation[3],
            "roll": rotation[0],
            "pitch": rotation[1],
            "yaw": rotation[2],
        }
    )


def find_ordered_sequence_between_frames(
    df: pd.DataFrame, frame_id: str, child_frame_id: str, time: float
) -> List[Dict[str, Union[str, np.ndarray]]]:
    """Find the ordered sequence of transforms between two frames at a given time in the tf tree.

    :param df: DataFrame containing the tf tree
    :param frame_id: The frame_id of the parent frame
    :param child_frame_id: The frame_id of the child frame
    :param time: The time at which to get the transform
    :return: The ordered sequence of transforms between the two frames
    """
    if (
        ordered_sequence := _find_ordered_sequence_between_frames(
            df, frame_id, child_frame_id, time
        )
    ) is not None:
        return ordered_sequence

    return [
        _invert_transform_data(transform)
        for transform in _find_ordered_sequence_between_frames(
            df, child_frame_id, frame_id, time
        )
    ]


def _invert_transform_data(
    transform: Dict[str, Union[str, np.ndarray]],
) -> Dict[str, Union[str, np.ndarray]]:
    """Invert the transform data from child_frame_id to frame_id

    :param transform: The transform data
    :return: The inverted transform data
    """
    transform["transform"] = inverse_matrix(transform["transform"])
    transform["frame_id"], transform["child_frame_id"] = (
        transform["child_frame_id"],
        transform["frame_id"],
    )
    return transform


def _find_ordered_sequence_between_frames(
    df: pd.DataFrame, frame_id: str, child_frame_id: str, time: float
) -> List[Dict[str, Union[str, np.ndarray]]]:
    """
    Find the ordered sequence of transforms between two frames at a given time in the tf tree.
    use an Dijkstra's algorithm to find the ordered sequence of transforms between frame_id and child_frame_id

    :param df: DataFrame containing the tf tree
    :param frame_id: The frame_id of the parent frame
    :param child_frame_id: The frame_id of the child frame
    :param time: The time at which to get the transform
    :return: The ordered sequence of transforms between the two frames
    """

    df = df[df["time"] == time]
    ordered_sequence = []
    visited = set()
    queue = [(frame_id, None)]
    predecessors = {}

    while queue:
        current_frame, _ = queue.pop(0)
        if current_frame == child_frame_id:
            break
        visited.add(current_frame)
        for column in df.columns:
            if current_frame in column:
                next_frame = column.split(" -> ")[1]
                if next_frame not in visited:
                    queue.append((next_frame, current_frame))
                    predecessors[next_frame] = current_frame
    if child_frame_id not in predecessors:
        return None
    current_frame = child_frame_id
    while current_frame != frame_id:
        ordered_sequence.append(
            df[f"{predecessors[current_frame]} -> {current_frame}"].iloc[-1]
        )
        current_frame = predecessors[current_frame]
    return ordered_sequence[::-1]


def compute_odometry_data(
    df: pd.DataFrame,
    period: int = 1,
    max_velocity: float = 2.0,
    max_angular_vel: float = 1.0,
    alpha: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the linear and angular velocity from successive transforms

    :param df: DataFrame containing the tf tree
    :param period: The number of time steps to use for the velocity calculation
    :param max_velocity: The maximum velocity of the robot
    :param max_angular_vel: The maximum angular velocity of the robot
    :param alpha: The smoothing factor for the exponential moving average filter
    :return: A DataFrame containing the odometry data
    """

    # remove duplicate rows on the time column
    df = df.drop_duplicates(subset=["time"]).copy()

    # add columns to the dataframe that contain the velocity of the robot

    df["velocity_x"] = df.position_x.diff(periods=period)
    df["velocity_y"] = df.position_y.diff(periods=period)
    df["velocity_z"] = df.position_z.diff(periods=period)

    dt = df.time.diff(periods=period)
    dt[dt < 1e-3] = np.nan

    # rotate the velocity vector to the robot's frame

    df_vel = df.apply(
        lambda x: pd.Series(
            np.array([x["velocity_x"], x["velocity_y"], x["velocity_z"]])
            @ compose_matrix(angles=(x["roll"], x["pitch"], x["yaw"]))[:3, :3],
            index=["velocity_x", "velocity_y", "velocity_z"],
        ),
        axis=1,
    )

    df["velocity_x"] = df_vel["velocity_x"] / dt
    df["velocity_y"] = df_vel["velocity_y"] / dt
    df["velocity_z"] = df_vel["velocity_z"] / dt

    df[df["velocity_x"].abs() > max_velocity] = np.nan
    df[df["velocity_y"].abs() > max_velocity] = np.nan
    df[df["velocity_z"].abs() > max_velocity] = np.nan

    # replace the zeros in the velocity columns with the interpolation of the previous and next values
    df.infer_objects(copy=False)
    df["velocity_x"] = _spline_interpolate_with_fallback(
        df.loc[:, "velocity_x"].replace(0, np.nan)
    )
    df["velocity_y"] = _spline_interpolate_with_fallback(
        df.loc[:, "velocity_y"].replace(0, np.nan)
    )
    df["velocity_z"] = _spline_interpolate_with_fallback(
        df.loc[:, "velocity_z"].replace(0, np.nan)
    )

    # smooth the velocity data using an exponential moving average filter

    df["velocity_x"] = df["velocity_x"].ewm(alpha=alpha).mean()
    df["velocity_y"] = df["velocity_y"].ewm(alpha=alpha).mean()
    df["velocity_z"] = df["velocity_z"].ewm(alpha=alpha).mean()

    # add columns to the dataframe that contain the angular velocity of the robot

    df["angular_velocity_x"] = (
        2.0
        / dt
        * (
            df["qw"].shift(periods=period) * df["qx"]
            - df["qx"].shift(periods=period) * df["qw"]
            - df["qy"].shift(periods=period) * df["qz"]
            + df["qz"].shift(periods=period) * df["qy"]
        )
    )

    df["angular_velocity_y"] = (
        2.0
        / dt
        * (
            df["qw"].shift(periods=period) * df["qy"]
            + df["qx"].shift(periods=period) * df["qz"]
            - df["qy"].shift(periods=period) * df["qw"]
            - df["qz"].shift(periods=period) * df["qx"]
        )
    )

    df["angular_velocity_z"] = (
        2.0
        / dt
        * (
            df["qw"].shift(periods=period) * df["qz"]
            - df["qx"].shift(periods=period) * df["qy"]
            + df["qy"].shift(periods=period) * df["qx"]
            - df["qz"].shift(periods=period) * df["qw"]
        )
    )

    df[df["angular_velocity_x"].abs() > max_angular_vel] = np.nan
    df[df["angular_velocity_y"].abs() > max_angular_vel] = np.nan
    df[df["angular_velocity_z"].abs() > max_angular_vel] = np.nan

    # replace the zeros in the angular velocity columns with the interpolation of the previous and next values
    df["angular_velocity_x"] = _spline_interpolate_with_fallback(
        df.loc[:, "angular_velocity_x"].replace(0, np.nan)
    )
    df["angular_velocity_y"] = _spline_interpolate_with_fallback(
        df.loc[:, "angular_velocity_y"].replace(0, np.nan)
    )
    df["angular_velocity_z"] = _spline_interpolate_with_fallback(
        df.loc[:, "angular_velocity_z"].replace(0, np.nan)
    )

    # smooth the angular velocity data using an exponential moving average filter

    df["angular_velocity_x"] = df["angular_velocity_x"].ewm(alpha=alpha).mean()
    df["angular_velocity_y"] = df["angular_velocity_y"].ewm(alpha=alpha).mean()
    df["angular_velocity_z"] = df["angular_velocity_z"].ewm(alpha=alpha).mean()

    return df, dt
