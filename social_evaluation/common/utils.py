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
    df_robot: pd.DataFrame, df_agents_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """Combine the robot and agents dataframes

    :param df_robot: Dataframe with the robot data
    :param df_agents_list: List of dataframes with the agents data
    :return: Dataframe with the combined data
    """
    df_agents = df_robot
    df_agents = df_agents.merge(
        combine_df_agents(df_agents_list), on="time", how="outer"
    )
    df_agents.infer_objects(copy=False)
    attributes = [
        attribute
        for attribute in df_agents.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behavior" not in attribute
        and "behaviour" not in attribute
    ]
    df_agents.loc[:, attributes] = df_agents.loc[:, attributes].interpolate(
        method="cubic", order=3
    )
    df_agents.ffill(inplace=True)
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents


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

    attributes = [
        attribute
        for attribute in df_agents.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behavior" not in attribute
        and "behaviour" not in attribute
    ]
    df_agents.infer_objects(copy=False)
    df_agents.loc[:, attributes] = df_agents.loc[:, attributes].interpolate(
        method="cubic", order=3
    )
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents
