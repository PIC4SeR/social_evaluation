import numpy as np
import math
import rclpy
from hunav_msgs.msg import Agent
from geometry_msgs.msg import Pose
from hunav_evaluator.sfm import SFM
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
    df_agents = df_agents.interpolate(method="cubic", order=3)
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

    df_agents = df_agents.interpolate(method="cubic", order=3)
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


### Metrics ###


@metric("time_to_reach_goal")
def total_time(df_robot: pd.DataFrame) -> float:
    """Compute the total time of the robot's trajectory"""
    return df_robot.time.iloc[-1] - df_robot.time.iloc[0]


@metric("path_length")
def robot_path_length(df_robot: pd.DataFrame, period: int = 4) -> float:
    """Compute the total length of the robot's trajectory"""
    step_x = df_robot.position_x.diff(periods=period)
    step_y = df_robot.position_y.diff(periods=period)
    return np.sum(np.sqrt(step_x**2 + step_y**2))


@metric("cumulative_heading_changes")
def cumulative_heading_changes(df_robot: pd.DataFrame) -> float:
    """Compute the cumulative heading changes of the robot"""
    # normalize the angles
    yaw = df_robot.yaw.apply(normalize_angle)

    return np.sum(np.abs(np.diff(yaw)))


@metric("avg_distance_to_closest_person")
def avg_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average distance to the closest person"""
    df_min_dist = distance_to_closest_person(
        df_agents_list, df_robot, robot_radius, agent_radius
    )
    min_dist_mean = np.mean(df_min_dist["min_distance"])
    return min_dist_mean, df_min_dist


@metric("minimum_distance_to_people")
def minimum_distance_to_people(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
) -> Tuple[float, pd.DataFrame]:
    """Compute the minimum distance to people"""
    df_min_dist = distance_to_closest_person(
        df_agents_list, df_robot, robot_radius, agent_radius
    )
    min_dist = np.min(df_min_dist["min_distance"])

    return min_dist, df_min_dist


@metric("maximum_distance_to_people")
def maximum_distance_to_people(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
) -> Tuple[float, pd.DataFrame]:
    """Compute the maximum distance to people"""
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
    df_agents["max_distance"] = (
        df_agents[[f"distance_agent{i}" for i in range(len(df_agents_list))]].max(
            axis=1
        )
        - robot_radius
        - agent_radius
    )

    return np.max(df_agents["max_distance"]), df_agents[["time", "max_distance"]]


def space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame, k: SocialSpace
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of space intrusions"""
    df_min_dist = distance_to_closest_person(df_agents_list, df_robot)
    df_min_dist["intrusion"] = df_min_dist["min_distance"].apply(
        lambda x: indicator_function(x, k)
    )
    intrusion_percentage = df_min_dist["intrusion"].mean() * 100
    return intrusion_percentage, df_min_dist["time", "intrusion"]


@metric("intimate_space_intrusions")
def intimate_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of intimate space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.INTIMATE
    )
    return percentage, df_intrusion


@metric("personal_space_intrusions")
def personal_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of personal space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.PERSONAL
    )
    return percentage, df_intrusion


@metric("social_space_intrusions")
def social_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of social space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.SOCIAL
    )
    return percentage, df_intrusion


@metric("public_space_intrusions")
def public_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of public space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.PUBLIC
    )
    return percentage, df_intrusion


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


@metric("group_intimate_space_intrusions")
def group_intimate_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of intimate space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.INTIMATE
    )
    return percentage, df_group_intrusions


@metric("group_personal_space_intrusions")
def group_personal_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of personal space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.PERSONAL
    )
    return percentage, df_group_intrusions


@metric("group_social_space_intrusions")
def group_social_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of social space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.SOCIAL
    )
    return percentage, df_group_intrusions


@metric("group_public_space_intrusions")
def group_public_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of public space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.PUBLIC
    )
    return percentage, df_group_intrusions


# SEAN 2.0: Formalizing and Generating Social Situations for Robot Navigation
# Nathan Tsoi, Alec Xiang, Peter Yu, Samuel S. Sohn, Greg Schwartz, Subashri Ramesh, Mohamed Hussein, Anjali W. Gupta, Mubbasir Kapadia, and Marynel Vázquez

# The metrics Robot on Person Personal Distance Violation, Person on Robot Personal Distance Violation, Intimate Distance Violation and
# Person on Robot Intimate Distance Violation have already been implemented in the Personal_space_intrusions function.
# Instead of returning the number of times, it returns a percentage of distance violation.


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


@metric("robot_on_person_collision")
def robot_on_person_collision(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people"""
    robot_collisions, _, df_agents = check_collisions(df_agents_list, df_robot)
    return robot_collisions, df_agents


@metric("person_on_robot_collision")
def person_on_robot_collision(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people"""
    _, person_collisions, df_agents = check_collisions(df_agents_list, df_robot)
    return person_collisions


@metric("time_not_moving")
def time_not_moving(
    df_robot: pd.DataFrame,
    period: int = 1,
    min_velocity: float = 0.01,
    min_angular_velocity: float = 0.02,
) -> Tuple[float, pd.DataFrame]:
    """Compute the time the robot is not moving"""
    not_moving = df_robot.apply(
        lambda x: (
            1
            if abs(x["velocity_x"]) < min_velocity
            and abs(abs(x["angular_velocity_z"]) < min_angular_velocity)
            else 0
        ),
        axis=1,
    )
    time_stopped = df_robot.time.diff(periods=period) * not_moving
    return time_stopped.sum(), not_moving


@metric("completed")
def goal_reached(
    df_robot: pd.DataFrame,
    goal: Tuple[float, float],
    goal_tolerance: float = 0.5,
    robot_radius: float = 0.3,
) -> bool:
    """Check if the robot reached the goal"""
    return time_goal_reached(df_robot, goal, goal_tolerance, robot_radius) is not None


@metric("final_goal_distance")
def final_goal_distance(df_robot: pd.DataFrame, goal: Tuple[float, float]) -> float:
    """Compute the distance between the last robot position and the target position."""
    x = df_robot.position_x.iloc[-1]
    y = df_robot.position_y.iloc[-1]
    return np.linalg.norm([x - goal[0], y - goal[1]])


@metric("minimum_goal_distance")
def minimum_goal_distance(df_robot: pd.DataFrame, goal: Tuple[float, float]) -> float:
    """Compute the closest the robot passes to the target position."""
    x = df_robot.position_x
    y = df_robot.position_y
    return np.min(np.linalg.norm([x - goal[0], y - goal[1]], axis=1))


# SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
# ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University


@metric("avg_robot_linear_speed")
def avg_robot_linear_speed(df_robot: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot linear speed"""
    return df_robot.velocity_x.abs().mean(), df_robot.velocity_x.abs()


@metric("avg_robot_angular_speed")
def avg_robot_angular_speed(df_robot: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot angular speed"""
    return df_robot.angular_velocity_z.abs().mean(), df_robot.angular_velocity_z


@metric("avg_acceleration")
def avg_acceleration(
    df_robot: pd.DataFrame, period: int = 5, max_accel: float = 1.0, alpha: float = 0.1
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot acceleration"""
    df_robot["acceleration"] = df_robot.velocity_x.abs().diff(
        periods=period
    ) / df_robot.time.diff(periods=period)
    df_robot["acceleration" > max_accel] = np.nan
    df_robot["acceleration"].interpolate(method="cubic", order=3, inplace=True)
    df_robot.dropna(inplace=True)
    df_robot.reset_index(drop=True, inplace=True)
    df_robot["acceleration"].ewm(alpha=alpha).mean()

    return df_robot["acceleration"].mean(), df_robot[["time", "acceleration"]]


@metric("avg_overacceleration")
def avg_overacceleration(
    df_robot: pd.DataFrame,
    period: int = 5,
    max_accel: float = 1.0,
    max_jerk: float = 1.0,
    alpha: float = 0.1,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot over-acceleration"""
    _, df_acceleration = avg_acceleration(df_robot, period, max_accel, alpha)
    df_acceleration["jerk"] = df_robot.acceleration.diff(
        periods=period
    ) / df_acceleration.time.diff(periods=period)
    df_acceleration["jerk" > max_jerk] = np.nan
    df_acceleration["jerk"].interpolate(method="cubic", order=3, inplace=True)
    df_acceleration.dropna(inplace=True)
    df_acceleration.reset_index(drop=True, inplace=True)
    df_acceleration["jerk"].ewm(alpha=alpha).mean()

    return df_acceleration["jerk"].mean(), df_acceleration[["time", "jerk"]]


# Learning a Group-Aware Policy for Robot Navigation
# Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2


@metric("avg_pedestrian_velocity")
def avg_pedestrian_velocity(
    df_agents_list: List[pd.DataFrame],
) -> Tuple[float, List[float]]:
    """Compute the average pedestrian velocity"""

    df_agents = combine_df_agents(df_agents_list)
    df_agents["avg_pedestrian_velocity"] = df_agents.apply(
        lambda x: np.mean(
            [
                np.linalg.norm([x[f"velocity_x_agent{i}"], x[f"velocity_y_agent{i}"]])
                for i in range(len(df_agents_list))
            ]
        ),
        axis=1,
    )
    return (
        df_agents["avg_pedestrian_velocity"].mean(),
        df_agents["avg_pedestrian_velocity"],
    )


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


@metric("avg_closest_pedestrian_velocity")
def avg_closest_pedestrian_velocity(
    df_robot: pd.DataFrame, df_agents_list: List[pd.DataFrame]
) -> Tuple[float, List[float]]:
    """Compute the average velocity of the closest pedestrian to the robot"""
    df_agents = combine_df_robot_agents(df_robot, df_agents_list)
    df_agents["closest_agent"] = closest_agent_to_robot(df_robot, df_agents_list)
    df_agents["avg_closest_pedestrian_velocity"] = df_agents.apply(
        lambda x: np.linalg.norm(
            [
                x[f"velocity_x_agent{x['closest_agent']}"],
                x[f"velocity_y_agent{x['closest_agent']}"],
            ]
        ),
        axis=1,
    )
    return (
        df_agents["avg_closest_pedestrian_velocity"].mean(),
        df_agents["avg_closest_pedestrian_velocity"],
    )


def avg_pedestrian_angle(agents, robot):
    # TODO: compute the average angle of the pedestrians with respect to the robot
    pass


# metrics based on social force model


@metric("social_force_on_agents")
def social_force_on_agents(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the social force provoked by the robot to the agents"""
    df_robot["social_force_norm"] = modulus_social_force_2(df_robot, df_agents_list)
    return df_robot["social_force_norm"].sum(), df_robot[["time", "social_force_norm"]]


@metric("social_force_on_robot")
def social_force_on_robot(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the social force provoked by the agents to the robot"""
    df_robot["social_force_norm"] = modulus_social_force(df_robot, df_agents_list)
    return df_robot["social_force_norm"].sum(), df_robot[["time", "social_force_norm"]]


# cumulative social work employed in this planner:
# https://github.com/robotics-upo/social_force_window_planner


@metric("social_work")
def social_work(df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame) -> float:
    """Compute the cumulative social work employed in this planner"""
    df_robot["social_work"] = social_work(df_robot, df_agents_list)
    return df_robot["social_work"].sum(), df_robot[["time", "social_work"]]


# social work per second or social power
@metric("social_work_per_second")
def social_work_per_second(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame
) -> float:
    """Compute the cumulative social work employed in this planner per second"""
    df_robot["social_work"] = social_work(df_robot, df_agents_list)
    return (
        df_robot["social_work"].sum()
        / (df_robot["time"].iloc[-1] - df_robot["time"].iloc[0]),
        df_robot[["time", "social_work"]],
    )


@metric("obstacle_force_on_agents")
def obstacle_force_on_agents(
    df_agents_list: List[pd.DataFrame], df_obstacles: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the obstacle force provoked by the obstacles to the agents"""
    for df_agent in df_agents_list:
        df_agent["obstacle_force_norm"] = modulus_obstacle_force(df_agent, df_obstacles)
    # mean force on agents
    # combine all agents
    df_agents = combine_df_agents(df_agents_list)
    return (
        np.mean([df_agent["obstacle_force_norm"].sum() for df_agent in df_agents_list]),
        df_agents[
            [
                "time",
                *[f"obstacle_force_norm_agent{i}" for i in range(len(df_agents_list))],
            ]
        ],
    )


def obstacle_force_on_robot(
    df_robot: pd.DataFrame, df_obstacles: pd.DataFrame
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the obstacle force provoked by the obstacles to the robot"""
    df_robot["obstacle_force_norm"] = modulus_obstacle_force(df_robot, df_obstacles)
    return (
        df_robot["obstacle_force_norm"].sum(),
        df_robot[["time", "obstacle_force_norm"]],
    )


# TODO
def path_irregularity(agents, robot):
    pass


# TODO
def path_efficiency(agents, robot):
    pass


@metric("static_obstacle_collision")
def static_obs_collision(
    df_robot: pd.DataFrame, df_obstacles: pd.DataFrame, dist_threshold: float = 0.5
) -> int:
    """Compute the number of times the robot collides with a static obstacle"""

    # count the number of times the robot collides with a static obstacle
    df_robot = find_closest_obstacle(df_robot, df_obstacles)
    df_robot["collision"] = df_robot["closest_obstacle_distance"] < dist_threshold
    return df_robot["collision"].sum()


# Evaluation of Socially-Aware Robot Navigation
# Yuxiang Gao * and Chien-Ming Huang
# Department of Computer Science, The Johns Hopkins University, Baltimore, MD, United States

# TODO:
# Average Displacement Error --> Trajectory needed
# Final Displacement Error --> Trajectory needed
# Asymmetric Dynamic Time Warping --> Trajectory needed
# Topological Complexity --> Path needed
# Path irregularity and Path efficiency are similar to the ones in SEAN paper .
# Personal space and o/p/r-space metrics are similar to the ones in Teaching Robot Navigation Behaviors to Optimal RRT Planners paper.


# metrics = {
#     # N. Perez-Higueras, F. Caballero, and L. Merino, “Teaching Robot Nav-
#     # igation Behaviors to Optimal RRT Planners,” International Journal of
#     # Social Robotics, vol. 10, no. 2, pp. 235–249, 2018.
#     "time_to_reach_goal": total_time,
#     "path_length": robot_path_length,
#     "cumulative_heading_changes": cumulative_heading_changes,
#     "avg_distance_to_closest_person": avg_closest_person,
#     "minimum_distance_to_people": minimum_distance_to_people,
#     "maximum_distance_to_people": maximum_distance_to_people,
#     "intimate_space_intrusions": intimate_space_intrusions,
#     "personal_space_intrusions": personal_space_intrusions,
#     "social_space_intrusions": social_space_intrusions,
#     "group_intimate_space_intrusions": group_intimate_space_intrusions,
#     "group_personal_space_intrusions": group_personal_space_intrusions,
#     "group_social_space_intrusions": group_social_space_intrusions,
#     # N. Tsoi, A. Xiang, P. Yu, S. S. Sohn, G. Schwartz, S. Ramesh,
#     # M. Hussein, A. W. Gupta, M. Kapadia, and M. V ́azquez, “Sean 2.0:
#     # Formalizing and generating social situations for robot navigation,”
#     # IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 047–
#     # 11 054, 2022
#     #   - 'Total Path Length' (meters): similar to 'path_length'
#     #   - 'Path Irregularity': (radians): total rotations in the robot's
#     #       traveled path greater than the total rotations in the search-based
#     #       path from the starting pose.
#     #   - 'Path Efficiency': (meters): ratio between robot's traveled path and
#     #       geodesic distance of the search-based path from the starting pose.
#     # true when the robot's final pose is within a specified distance of the goal.
#     # The final distance threshold is easily adjustable by the user, but defaults
#     # to 1.2m.
#     "completed": goal_reached,
#     # (meters): the closest the robot passes to the target position.
#     "minimum_distance_to_target": minimum_goal_distance,
#     # (meters): distance between the last robot position and the target position.
#     "final_distance_to_target": final_goal_distance,
#     #   - 'Robot on Person Personal Distance Violation': number of times a robot
#     # approaches a person within the personal distance of the robot.
#     # Similar to 'personal_space_intrusions'
#     #   - 'Person on Robot Personal Distance Violation': number of times a person
#     # approaches the robot within the personal distance of the robot.
#     #   - 'Intimate Distance Violation': number of times the robot approached within
#     # the intimate distance of a person.
#     #   - 'Person on Robot Intimate Distance Violation': number of times a person
#     # approaches the robot within the intimate distance of the robot.
#     "robot_on_person_collision": robot_on_person_collision,
#     "person_on_robot_collision": person_on_robot_collision,
#     "time_not_moving": time_not_moving,
#     "static_obstacle_collision": static_obs_collision,
#     # number of times the robot collides with a static obstacle.
#     # SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
#     # ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University
#     "avg_robot_linear_speed": avg_robot_linear_speed,
#     "avg_robot_angular_speed": avg_robot_angular_speed,
#     "avg_acceleration": avg_acceleration,
#     "avg_overacceleration": avg_overacceleration,
#     # Learning a Group-Aware Policy for Robot Navigation
#     # Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2
#     "avg_pedestrian_velocity": avg_pedestrian_velocity,
#     "avg_closest_pedestrian_velocity": avg_closest_pedestrian_velocity,
#     # metrics based on Social Force Model employed in different papers
#     "social_force_on_agents": social_force_on_agents,
#     "social_force_on_robot": social_force_on_robot,
#     "social_work": social_work,
#     "obstacle_force_on_robot": obstacle_force_on_robot,
#     "obstacle_force_on_agents": obstacle_force_on_agents,
# }
