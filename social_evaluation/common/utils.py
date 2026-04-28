import pandas as pd

# get the path to the bag file
from pathlib import Path as OSPath
import os
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion, compose_matrix
import numpy as np

from people_msgs.msg import People, Person
from nav_msgs.msg import Odometry, Path as NavPath
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from typing import List, Tuple, Union, Dict, Type


def _interpolate_with_fallback(
    df: pd.DataFrame, columns: List[str], limit_direction: str = "both"
) -> None:
    """Interpolate dataframe columns using progressive fallbacks."""
    if not columns:
        return

    methods = [
        ("cubic", {"order": 3}),
        ("quadratic", {"order": 2}),
        ("linear", {}),
    ]

    for method, extra_args in methods:
        try:
            df.loc[:, columns] = df.loc[:, columns].interpolate(
                method=method, limit_direction=limit_direction, **extra_args
            )
        except Exception as exc:
            print(
                f"Interpolation with method '{method}' failed ({exc}). "
                "Falling back to simpler method."
            )
            continue
        remaining_nan = df.loc[:, columns].isna().any().any()
        if not remaining_nan:
            return

    # Final safety net
    df.loc[:, columns] = df.loc[:, columns].ffill().bfill()


def time_goal_reached(
    df: pd.DataFrame,
    goal_position: Tuple[float, float],
    threshold: float = 0.5,
    robot_radius: float = 0.3,
) -> Union[float, None]:
    """Get the time when the robot reached the goal position

    :param df: Dataframe with the robot data
    :param goal_position: The goal position
    :param threshold: The threshold distance to consider the goal reached
    :param robot_radius: The radius of the robot
    :return: The time when the goal was reached
    """

    df["distance_to_goal"] = (
        (df["position_x"] - goal_position[0]) ** 2
        + (df["position_y"] - goal_position[1]) ** 2
    ) ** 0.5
    # return the time when the goal was reached
    if df["distance_to_goal"].min() < threshold + robot_radius:
        return df[df["distance_to_goal"] < threshold + robot_radius].iloc[0]["time"]


def normalize_angle(angle: float) -> float:
    """Normalize the angle to be between -pi and pi

    :param angle: The angle to normalize
    :return: The normalized angle
    """
    return np.arctan2(np.sin(angle), np.cos(angle))


def combine_df_robot_agents(
    df_robot: pd.DataFrame | pd.Series, df_agents_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """Combine the robot and agents dataframes

    :param df_robot: Dataframe with the robot data
    :param df_agents_list: List of dataframes with the agents data
    :return: Dataframe with the combined data
    """
    df_agents = df_robot.copy()
    df_agents = df_agents.merge(
        combine_df_agents(df_agents_list), on="time", how="outer"
    )
    df_agents.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    # Remove columns that are entirely NaN (e.g., missing robot velocities)
    df_agents.dropna(axis=1, how="all", inplace=True)
    df_agents.infer_objects(copy=False)
    attributes = [
        attribute
        for attribute in df_agents.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behavior" not in attribute
        and "frame_id" not in attribute
    ]
    _interpolate_with_fallback(df_agents, attributes)
    df_agents.ffill(inplace=True)
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents


def combine_df_robot_and_agent(
    df_robot: pd.DataFrame, df_agent: pd.DataFrame
) -> pd.DataFrame:
    """Combine the robot and agents dataframes

    :param df_robot: Dataframe with the robot data
    :param df_agent: Dataframe with the agent data
    :return: Dataframe with the combined data
    """
    df_interaction = df_robot
    df_interaction = df_robot.merge(
        df_agent, on="time", how="outer", suffixes=("_1", "_2")
    )
    df_interaction.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    df_interaction.infer_objects(copy=False)
    attributes = [
        attribute
        for attribute in df_interaction.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behavior" not in attribute
        and "frame_id" not in attribute
    ]
    _interpolate_with_fallback(df_interaction, attributes)
    df_interaction.ffill(inplace=True)
    df_interaction.dropna(inplace=True)
    df_interaction.reset_index(drop=True, inplace=True)
    return df_interaction


def combine_df_agents(df_agents_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine the agents dataframes interpolating the missing values and filling the NaN values

    :param df_agents_list: List of dataframes with the agents data
    :return: Dataframe with the combined data
    """
    df_agents = df_agents_list[0].add_suffix("_agent0")
    df_agents.rename(columns={"time_agent0": "time"}, inplace=True)
    for i, df_agent in enumerate(df_agents_list[1:]):
        df_agents = df_agents.merge(
            df_agent.add_suffix(f"_agent{i+1}").rename(
                columns={f"time_agent{i+1}": "time"}
            ),
            on="time",
            how="outer",
        )
    df_agents.sort_values("time", kind="stable", inplace=True, ignore_index=True)

    attributes = [
        attribute
        for attribute in df_agents.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behavior" not in attribute
    ]
    df_agents.infer_objects(copy=False)
    _interpolate_with_fallback(df_agents, attributes)
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents


def align_df_agents_on_time(df_agents_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Align the agents dataframes on time

    :param df_agents_list: List of dataframes with the agents data
    :return: List of dataframes with the aligned data"""

    if not df_agents_list:
        return []

    # Get the union of all timestamps
    all_times = pd.concat([df["time"] for df in df_agents_list]).sort_values().unique()

    aligned_agents = []
    for agent_df in df_agents_list:
        # Reindex requires a unique index. Some bags can contain repeated
        # samples at the same timestamp, so collapse those first.
        agent_df = (
            agent_df.sort_values("time", kind="stable")
            .drop_duplicates(subset="time", keep="last")
            .set_index("time")
            .reindex(all_times)
        )
        agent_df.infer_objects(copy=False)
        # interpolate missing values

        attributes = [
            attribute
            for attribute in agent_df.columns.to_list()
            if "time" not in attribute
            and "id" not in attribute
            and "behavior" not in attribute
            and "frame_id" not in attribute
        ]

        _interpolate_with_fallback(agent_df, attributes)
        # forward-fill and backward-fill missing values
        agent_df.ffill(inplace=True)
        agent_df.bfill(inplace=True)
        # reset index and rename the time column
        agent_df = agent_df.reset_index().rename(columns={"index": "time"})

        aligned_agents.append(agent_df)
    return aligned_agents


def align_df_robot_agents(
    df_robot: pd.DataFrame, df_agents_list: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """Align the robot and agents dataframes on time

    :param df_robot: Dataframe with the robot data
    :param df_agents_list: List of dataframes with the agents data
    :return: Tuple of dataframes with the aligned data"""

    df_agents_list_ = df_agents_list.copy()
    # Align agents on time
    df_agents_list_.insert(0, df_robot)

    aligned_agents = align_df_agents_on_time(df_agents_list_)

    return aligned_agents[0], aligned_agents[1:]
