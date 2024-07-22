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


def get_transform_between_frames(df, frame_id, child_frame_id, time):
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


def get_pose_data_from_transform(transform):
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


def find_ordered_sequence_between_frames(df, frame_id, child_frame_id, time):
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


def _invert_transform_data(transform):
    # invert the transform data from child_frame_id to frame_id
    transform["transform"] = inverse_matrix(transform["transform"])
    transform["frame_id"], transform["child_frame_id"] = (
        transform["child_frame_id"],
        transform["frame_id"],
    )
    return transform


def _find_ordered_sequence_between_frames(df, frame_id, child_frame_id, time):
    # use an Dijkstra's algorithm to find the ordered sequence of transforms between frame_id and child_frame_id
    # at the given time in the tf tree
    df = df[df["time"] == time]
    # find the ordered sequence of transforms between frame_id and child_frame_id
    ordered_sequence = []
    visited = set()
    queue = [(frame_id, None)]
    predecessors = {}

    while queue:
        current_frame, parent_frame = queue.pop(0)
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


def compute_odometry_data(df, period=1, max_velocity=2, alpha=0.1):
    # remove duplicate rows on the time column
    df = df.drop_duplicates(subset=["time"])

    # add columns to the dataframe that contain the velocity of the robot

    df.loc[:, "velocity_x"] = df["position_x"].diff(periods=period)
    df.loc[:, "velocity_y"] = df["position_y"].diff(periods=period)
    df.loc[:, "velocity_z"] = df["position_z"].diff(periods=period)

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
    df["velocity_x"] = (
        df.loc[:, "velocity_x"].replace(0, np.nan).interpolate("spline", order=3)
    )
    df["velocity_y"] = (
        df.loc[:, "velocity_y"].replace(0, np.nan).interpolate("spline", order=3)
    )
    df["velocity_z"] = (
        df.loc[:, "velocity_z"].replace(0, np.nan).interpolate("spline", order=3)
    )

    # smooth the velocity data using an exponential moving average filter

    df["velocity_x"] = df["velocity_x"].ewm(alpha=alpha).mean()
    df["velocity_y"] = df["velocity_y"].ewm(alpha=alpha).mean()
    df["velocity_z"] = df["velocity_z"].ewm(alpha=alpha).mean()

    return df, dt
