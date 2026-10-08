import numpy as np
import pandas as pd
from typing import List, Tuple
from enum import Enum
from social_evaluation.common import combine_df_robot_agents, combine_df_agents


# Teaching Robot Navigation Behaviors to Optimal RRT Planners
# Noé Pérez-Higueras, Fernando Caballero & Luis Merino


class SocialSpace(Enum):
    """
    Social space categories
    """

    INTIMATE = 0  #: less than 0.45 meters
    PERSONAL = 1  #: from 0.45 to 1.2 meters
    SOCIAL = 2  #: from 1.2 to 3.6 meters
    PUBLIC = 3  #: more than 3.6 meters


def indicator_function(distance: float, k: SocialSpace) -> int:
    """Indicator function for the social space categories

    :param distance: Distance between the robot and the person
    :param k: Social space category
    :return: 1 if the distance is within the category, 0 otherwise"""
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
    """Decorator to add a metric to the metrics dictionary

    :param name: Name of the metric
    :return: Decorator function"""

    def wrapper(func) -> callable:
        metrics[name] = func
        return func

    return wrapper


def get_group_center(
    df_agents_list: List[pd.DataFrame], group_id: int, distance: float
) -> pd.DataFrame:
    """Get the group center of the agents belonging to the same group
    The group center is the 'average' position of the agents in the group

    :param df_agents_list: List of dataframes with the agents data
    :param group_id: Group id
    :param distance: Distance to the group center
    :return: Dataframe with the group center data"""

    # Select only the agents belonging to this group. ``group_id`` is a column,
    # so compare element-wise and keep the agent if any sample matches.
    # ``group_id`` is a column and may be stored as str (bag extraction casts it)
    # while ``group_id`` here is an int from detect_groups, so compare as strings.
    group = [
        df_agent
        for df_agent in df_agents_list
        if (df_agent.group_id.astype(str) == str(group_id)).any()
    ]
    df_agents = combine_df_agents(group)

    df_agents = df_agents.interpolate(method="cubic", order=3)
    df_agents.dropna(inplace=True)
    df_agents.reset_index(drop=True, inplace=True)

    # combine_df_agents re-indexes the selected agents as agent0..agent{len(group)-1},
    # so iterate over the group members only.
    for i in range(len(group)):
        df_agents[f"group_point_x_agent{i}"] = (
            df_agents[f"position_x_agent{i}"]
            + distance * np.cos(df_agents[f"yaw_agent{i}"])
        )
        df_agents[f"group_point_y_agent{i}"] = (
            df_agents[f"position_y_agent{i}"]
            + distance * np.sin(df_agents[f"yaw_agent{i}"])
        )

    df_agents["group_center_x"] = df_agents[
        [f"group_point_x_agent{i}" for i in range(len(group))]
    ].mean(axis=1)
    df_agents["group_center_y"] = df_agents[
        [f"group_point_y_agent{i}" for i in range(len(group))]
    ].mean(axis=1)
    return df_agents[["time", "group_center_x", "group_center_y"]]


def radius_for(robot_name: str, config: dict) -> float:
    """This robot's radius under ``config`` (a ``metrics_config`` block).

    ``robot_radii`` maps model names to radii and is matched by the longest key
    that prefixes the robot's name, so fleet members ``turtlebot2_1`` and
    ``turtlebot2_2`` take the ``turtlebot2`` entry. Anything unmatched falls
    back to the scalar ``robot_radius``.
    """
    radii = config.get("robot_radii") or {}
    matches = [key for key in radii if str(robot_name).startswith(key)]
    if matches:
        return float(radii[max(matches, key=len)])
    return float(config.get("robot_radius", 0.3))


def distance_to_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    robot_radius: float = 0.3,
    agent_radius: float = 0.2,
) -> pd.DataFrame:
    """Compute the distance to the closest person

    :param df_agents_list: List of dataframes with the agents data
    :param df_robot: Dataframe with the robot data
    :param robot_radius: Radius of the robot
    :param agent_radius: Radius of the agent
    :return: Dataframe with the time and distance to the closest person"""
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
    return df_agents[["time", "min_distance"]]


def space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    k: SocialSpace,
    robot_radius: float = 0.3,
    agent_radius: float = 0.2,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of space intrusions

    :param df_agents_list: List of dataframes with the agents data
    :param df_robot: Dataframe with the robot data
    :param k: Social space category
    :param robot_radius: Radius of the robot
    :param agent_radius: Radius of the agent
    :return: Percentage of space intrusions and dataframe with the time and intrusion"""

    df_min_dist = distance_to_closest_person(
        df_agents_list, df_robot, robot_radius, agent_radius
    )
    df_min_dist["intrusion"] = df_min_dist["min_distance"].apply(
        lambda x: indicator_function(x, k)
    )
    intrusion_percentage = df_min_dist["intrusion"].mean() * 100
    return intrusion_percentage, df_min_dist[["time", "intrusion"]]


def detect_groups(df_agents_list: List[pd.DataFrame]) -> List[int]:
    """Detect the groups in the agents

    :param df_agents_list: List of dataframes with the agents data
    :return: List of group ids"""
    group_ids = []
    for df_agent in df_agents_list:
        df_agent.dropna(inplace=True)
        group_ids.extend(df_agent.group_id.unique())
    # Deduplicate: several agents share the same group_id, but each group must be
    # processed once (otherwise group intrusions are double-counted). Exclude the
    # ungrouped sentinel (-1).
    return sorted({int(group_id) for group_id in group_ids if int(group_id) != -1})


def group_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    k: SocialSpace,
    group_radius: float = 1.5,
    robot_radius: float = 0.3,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of space intrusions for each group

    :param df_agents_list: List of dataframes with the agents data
    :param df_robot: Dataframe with the robot data
    :param k: Social space category
    :param group_radius: Radius of the group
    :param robot_radius: Radius of the robot
    :return: Percentage of space intrusions and dataframe with the time and intrusion"""
    group_ids = detect_groups(df_agents_list)
    if len(group_ids) == 0:
        # Return a tuple (not a list) so the metric dispatch stores the scalar
        # 0.0 rather than the raw [value, df] pair.
        return 0.0, pd.DataFrame()

    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    df_group_intrusions = pd.DataFrame()
    df_group_intrusions["time"] = df_agents["time"]
    df_group_intrusions["group_intrusion"] = 0
    for group_id in group_ids:
        df_group_center = get_group_center(df_agents_list, group_id, group_radius)
        # Attach the robot position (the merge result must be assigned back, else
        # position_x/position_y are missing when the distance is computed below).
        df_group_center = df_group_center.merge(
            df_robot[["time", "position_x", "position_y"]], on="time", how="outer"
        )
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
    """Check for collisions between the robot and people

    :param df_agents_list: List of dataframes with the agents data
    :param df_robot: Dataframe with the robot data
    :return: Number of robot collisions, number of person collisions and dataframe with the collisions
    """
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
        df_agents[f"collision_agent_robot{i}"] = df_agents[f"distance_agent{i}"].apply(
            lambda x: 1 if x < 0.02 else 0
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
                x[f"collision_agent_robot{i}"]
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
                x[f"collision_agent_robot{i}"]
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
    """Compute the closest pedestrian to the robot as a list of indexes

    :param df_robot: Dataframe with the robot data
    :param df_agents_list: List of dataframes with the agents data
    :return: List of indexes with the closest agent to the robot at each time step"""
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
