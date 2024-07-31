import numpy as np
import math
import pandas as pd
from typing import List, Tuple, Dict, Any, Union
from enum import Enum
from social_evaluation.utils.utils import time_goal_reached
from social_evaluation.utils.sfm import *

# Teaching Robot Navigation Behaviors to Optimal RRT Planners
# Noé Pérez-Higueras, Fernando Caballero & Luis Merino


class SocialSpace(Enum):
    """Social space categories
    INTIMATE: 0.45 meters
    PERSONAL: 1.2 meters
    SOCIAL: 3.6 meters
    PUBLIC: 3.6 meters or more"""

    INTIMATE = 0
    PERSONAL = 1
    SOCIAL = 2
    PUBLIC = 3


def indicator_function(distance: float, k: SocialSpace) -> int:
    match k:
        case SocialSpace.INTIMATE:
            return 1 if distance < 0.45 else 0
        case SocialSpace.PERSONAL:
            return 1 if 0.45 <= distance < 1.2 else 0
        case SocialSpace.SOCIAL:
            return 1 if 1.2 <= distance < 3.6 else 0
        case SocialSpace.PUBLIC:
            return 1 if 3.6 <= distance else 0
        case _:
            return 0


metrics = {}


def metric(name: str) -> callable:
    def decorator(func) -> callable:
        metrics[name] = func
        return func

    return decorator


def normalize_angle(angle: float) -> float:
    """Normalize the angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def combine_df_agents(df_agents_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine the agents dataframes"""
    df_agents = df_agents_list[0].add_suffix("_agent0")
    df_agents.rename(columns={"time_agent0": "time"}, inplace=True)
    for i, df_agent in enumerate(df_agents_list[1:]):
        df_agents = df_agents.merge(
            df_agent, on="time", how="outer", suffixes=("", f"_agent{i}")
        )
    df_agents.infer_objects(copy=False)

    attributes = [
        attribute
        for attribute in df_agents.columns.to_list()
        if "time" not in attribute
        and "id" not in attribute
        and "behaviour" not in attribute
    ]

    df_agents.loc[:, attributes] = df_agents.loc[:, attributes].interpolate(
        method="cubic", order=3
    )
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents


def combine_df_robot_agents(
    df_robot: pd.DataFrame, df_agents_list: List[pd.DataFrame]
) -> pd.DataFrame:
    """Combine the robot and agents dataframes"""
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
        and "behaviour" not in attribute
    ]
    df_agents.loc[:, attributes] = df_agents.loc[:, attributes].interpolate(
        method="cubic", order=3
    )
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)
    return df_agents


def get_group_center(
    df_agents_list: List[pd.DataFrame], group_id: int, distance: float
) -> pd.DataFrame:
    """Get the group center of the agents belonging to the same group
    The group center is the 'average' position of the agents in the group"""

    group = [df_agent for df_agent in df_agents_list if df_agent.group_id == group_id]
    df_agents = combine_df_agents(group)

    df_agents = df_agents.interpolate(method="cubic", order=3)
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)

    for i in range(len(df_agents_list)):
        df_agents[f"group_point_x_agent{i}"] = df_agents.apply(
            lambda x: [
                x[f"position_x_agent{i}"] + distance * math.cos(x[f"yaw_agent{i}"])
            ],
            axis=1,
        )
        df_agents[f"group_point_y_agent{i}"] = df_agents.apply(
            lambda x: [
                x[f"position_y_agent{i}"] + distance * math.sin(x[f"yaw_agent{i}"])
            ],
            axis=1,
        )

    df_agents["group_center_x"] = df_agents.apply(
        lambda x: np.mean(
            [x[f"group_point_agent{i}"] for i in range(len(df_agents_list))], axis=0
        ),
        axis=1,
    )
    df_agents["group_center_y"] = df_agents.apply(
        lambda x: np.mean(
            [x[f"group_point_y_agent{i}"] for i in range(len(df_agents_list))], axis=0
        ),
        axis=1,
    )
    return df_agents["time", "group_center_x", "group_center_y"]


def distance_to_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
) -> pd.DataFrame:
    """Compute the distance to the closest person"""
    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    for i in range(len(df_agents_list)):
        df_agents[f"distance_agent{i}"] = df_agents.apply(
            lambda x: np.linalg.norm(
                [
                    x["position_x"] - x[f"position_x_agent{i}"],
                    x["position_y"] - x[f"position_y_agent{i}"],
                ]
            ),
            axis=1,
        )
    df_agents["min_distance"] = (
        df_agents[[f"distance_agent{i}" for i in range(len(df_agents_list))]].min(
            axis=1
        )
        - robot_radius
        - agent_radius
    )
    df_agents[["time", "min_distance"]]
    return df_agents[["time", "min_distance"]]


def space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame, k: SocialSpace
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of space intrusions"""
    df_min_dist = distance_to_closest_person(df_agents_list, df_robot)
    df_min_dist["intrusion"] = df_min_dist["min_distance"].apply(
        lambda x: indicator_function(x, k)
    )
    intrusion_percentage = df_min_dist["intrusion"].mean() * 100
    return intrusion_percentage, df_min_dist[["time", "intrusion"]]


def detect_groups(df_agents_list: List[pd.DataFrame]) -> List[int]:
    """Detect the groups in the agents"""
    group_ids = []
    for df_agent in df_agents_list:
        df_agent.dropna(inplace=True)
        group_ids.extend(df_agent.group_id.unique())
    group_ids = [int(group_id) for group_id in group_ids if group_id != "-1"]
    return group_ids


def group_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    k: SocialSpace,
    group_radius: float = 1.5,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of space intrusions for each group"""
    group_ids = detect_groups(df_agents_list)
    if len(group_ids) == 0:
        return [0.0, pd.DataFrame()]

    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    df_group_intrusions = pd.DataFrame()
    df_group_intrusions["time"] = df_agents["time"]
    df_group_intrusions["group_intrusion"] = 0
    for group_id in group_ids:
        df_group_center = get_group_center(df_agents_list, group_id, group_radius)
        df_group_center.merge(df_robot, on="time", how="outer")
        df_group_center = df_group_center.interpolate(method="cubic", order=3)
        df_group_center.dropna(inplace=True)
        df_group_center.reset_index(drop=True, inplace=True)

        df_group_center["distance"] = df_group_center.apply(
            lambda x: np.linalg.norm(
                [
                    x["group_center_x"] - x["position_x"],
                    x["group_center_y"] - x["position_y"],
                ]
            ),
            axis=1,
        )
        df_group_center["intrusion"] = df_group_center["distance"].apply(
            lambda x: indicator_function(x, k)
        )
        df_group_intrusions["group_intrusion"] += df_group_center["intrusion"]

    df_group_intrusions["group_intrusion"] = df_group_intrusions[
        "group_intrusion"
    ].apply(lambda x: 1 if x > 0 else 0)
    intrusion_percentage = df_group_intrusions["group_intrusion"].mean() * 100
    return intrusion_percentage, df_group_intrusions


def check_collisions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[int, int, pd.DataFrame]:
    """Check for collisions between the robot and people"""
    robot_collisions = 0
    person_collisions = 0

    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    for i in range(len(df_agents_list)):
        df_agents[f"distance_agent{i}"] = df_agents.apply(
            lambda x: np.linalg.norm(
                [
                    x["position_x"] - x[f"position_x_agent{i}"],
                    x["position_y"] - x[f"position_y_agent{i}"],
                ]
            ),
            axis=1,
        )
        df_agents[f"collision_agent{i}"] = df_agents[f"distance_agent{i}"].apply(
            lambda x: 1 if x < 0.6 else 0
        )
        # we want to distinguish between the robot colliding with a person and a person colliding with the robot

        # find the direction of the robot with respect to the person
        df_agents[f"angle_agent{i}"] = df_agents.apply(
            lambda x: np.arctan2(
                x[f"position_y_agent{i}"] - x["position_y"],
                x[f"position_x_agent{i}"] - x["position_x"],
            ),
            axis=1,
        )
        df_agents[f"angle_robot{i}"] = df_agents.apply(
            lambda x: np.arctan2(
                x["position_y"] - x[f"position_y_agent{i}"],
                x["position_x"] - x[f"position_x_agent{i}"],
            ),
            axis=1,
        )

        df_agents[f"collision_agent{i}"] = df_agents.apply(
            lambda x: (
                1
                if (
                    abs(x[f"angle_agent{i}"]) < abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] > x[f"velocity_x_agent{i}"]
                )
                or (
                    abs(x[f"angle_agent{i}"]) > abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] < x[f"velocity_x_agent{i}"]
                )
                or (
                    abs(x[f"angle_agent{i}"]) == abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] == x[f"velocity_x_agent{i}"]
                )
                else 0
            ),
            axis=1,
        )

        df_agents[f"collision_robot{i}"] = df_agents.apply(
            lambda x: (
                1
                if (
                    abs(x[f"angle_agent{i}"]) > abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] < x[f"velocity_x_agent{i}"]
                )
                or (
                    abs(x[f"angle_agent{i}"]) < abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] > x[f"velocity_x_agent{i}"]
                )
                or (
                    abs(x[f"angle_agent{i}"]) == abs(x[f"angle_robot{i}"])
                    and x["velocity_x"] == x[f"velocity_x_agent{i}"]
                )
                else 0
            ),
            axis=1,
        )

        robot_collisions += df_agents[f"collision_robot{i}"].sum()
        person_collisions += df_agents[f"collision_agent{i}"].sum()

    return robot_collisions, person_collisions, df_agents


def closest_agent_to_robot(
    df_robot: pd.DataFrame, df_agents_list: List[pd.DataFrame]
) -> List[float]:
    """Compute the closest pedestrian to the robot as a list of indexes"""
    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    df_agents["closest_agent"] = df_agents.apply(
        lambda x: np.argmin(
            [
                np.linalg.norm(
                    [
                        x["position_x"] - x[f"position_x_agent{i}"],
                        x["position_y"] - x[f"position_y_agent{i}"],
                    ]
                )
                for i in range(len(df_agents_list))
            ]
        ),
        axis=1,
    )
    return df_agents["closest_agent"]
