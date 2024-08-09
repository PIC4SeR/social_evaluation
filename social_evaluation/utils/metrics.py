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
from social_evaluation.utils.metrics_utils import *

# Teaching Robot Navigation Behaviors to Optimal RRT Planners
# Noé Pérez-Higueras, Fernando Caballero & Luis Merino


""" Metrics for evaluating social navigation 
    The metrics are based on the following papers:
    - Teaching Robot Navigation Behaviors to Optimal RRT Planners
    - SEAN 2.0: Formalizing and Generating Social Situations for Robot Navigation
    - SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
    - Learning a Group-Aware Policy for Robot Navigation
    - Evaluation of Socially-Aware Robot Navigation

    The metrics are divided into the following categories:
    - Navigation metrics
    - Social space metrics
    - Collision metrics
    - Robot dynamics metrics
    - Group dynamics metrics
    - Social force model metrics
    - Path planning metrics
    
    The metrics are implemented as functions that take the following arguments:
    - df_robot: pandas DataFrame with the robot's data
    - df_agents_list: list of pandas DataFrames with the agents' data
    - goal: tuple with the goal position (x, y)
    - robot_radius: float with the robot's radius
    - agent_radius: float with the agent's radius
    - period: int with the period for computing the metrics
    - min_velocity: float with the minimum velocity for considering the robot not moving
    - min_angular_velocity: float with the minimum angular velocity for considering the robot not moving
    - goal_tolerance: float with the goal tolerance for considering the goal reached
    - max_accel: float with the maximum acceleration
    - max_jerk: float with the maximum jerk
    - alpha: float with the exponential moving average factor
    - dist_threshold: float with the distance threshold for considering a collision with a static obstacle

    The metrics return a tuple with the metric value and a pandas DataFrame with the metric values over time.
    
    Metrics:
    - time_to_reach_goal: Compute the total time of the robot's trajectory
    - path_length: Compute the total length of the robot's trajectory
    - cumulative_heading_changes: Compute the cumulative heading changes of the robot
    - avg_distance_to_closest_person: Compute the average distance to the closest person
    - minimum_distance_to_people: Compute the minimum distance to people
    - maximum_distance_to_people: Compute the maximum distance to people
    - intimate_space_intrusions: Compute the percentage of intimate space intrusions
    - personal_space_intrusions: Compute the percentage of personal space intrusions
    - social_space_intrusions: Compute the percentage of social space intrusions
    - public_space_intrusions: Compute the percentage of public space intrusions
    - group_intimate_space_intrusions: Compute the percentage of intimate space intrusions for each group
    - group_personal_space_intrusions: Compute the percentage of personal space intrusions for each group
    - group_social_space_intrusions: Compute the percentage of social space intrusions for each group
    - group_public_space_intrusions: Compute the percentage of public space intrusions for each group
    - robot_on_person_collision: Check for collisions between the robot and people
    - person_on_robot_collision: Check for collisions between the robot and people
    - time_not_moving: Compute the time the robot is not moving
    - completed: Check if the robot reached the goal
    - final_goal_distance: Compute the distance between the last robot position and the target position
    - minimum_goal_distance: Compute the closest the robot passes to the target position
    - avg_robot_linear_speed: Compute the average robot linear speed
    - avg_robot_angular_speed: Compute the average robot angular speed
    - avg_acceleration: Compute the average robot acceleration
    - avg_overacceleration: Compute the average robot jerk
    - avg_pedestrian_velocity: Compute the average pedestrian velocity
    - avg_closest_pedestrian_velocity: Compute the average velocity of the closest pedestrian to the robot
    - social_force_on_agents: Compute the cumulative modulus of the social force provoked by the robot to the agents
    - social_force_on_robot: Compute the cumulative modulus of the social force provoked by the agents to the robot
    - social_work: Compute the cumulative social work employed in this planner
    - social_work_per_second: Compute the cumulative social work employed in this planner per second
    - obstacle_force_on_agents: Compute the cumulative modulus of the obstacle force provoked by the obstacles to the agents
    - obstacle_force_on_robot: Compute the cumulative modulus of the obstacle force provoked by the obstacles to the robot

"""


@metric("time_to_reach_goal")
def total_time(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> float:
    """Compute the total time of the robot's trajectory"""
    return df_robot.time.iloc[-1] - df_robot.time.iloc[0]


@metric("path_length")
def robot_path_length(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    period: int = 4,
    **kwargs,
) -> float:
    """Compute the total length of the robot's trajectory"""
    step_x = df_robot.position_x.diff(periods=period)
    step_y = df_robot.position_y.diff(periods=period)
    return np.sum(np.sqrt(step_x**2 + step_y**2))


@metric("cumulative_heading_changes")
def cumulative_heading_changes(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> float:
    """Compute the cumulative heading changes of the robot"""
    # normalize the angles
    yaw = df_robot.yaw.apply(normalize_angle)

    return np.sum(np.abs(np.diff(yaw)))


@metric("avg_distance_to_closest_person")
def avg_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
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
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
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
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
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


@metric("intimate_space_intrusions")
def intimate_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame, *args, **kwargs
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of intimate space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.INTIMATE
    )
    return percentage, df_intrusion


@metric("personal_space_intrusions")
def personal_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame, *args, **kwargs
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of personal space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.PERSONAL
    )
    return percentage, df_intrusion


@metric("social_space_intrusions")
def social_space_intrusions(
    df_agents_list: List[pd.DataFrame], df_robot: pd.DataFrame, *args, **kwargs
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of social space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.SOCIAL
    )
    return percentage, df_intrusion


@metric("public_space_intrusions")
def public_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of public space intrusions"""
    percentage, df_intrusion = space_intrusions(
        df_agents_list, df_robot, SocialSpace.PUBLIC
    )
    return percentage, df_intrusion


@metric("group_intimate_space_intrusions")
def group_intimate_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of intimate space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.INTIMATE
    )
    return percentage, df_group_intrusions


@metric("group_personal_space_intrusions")
def group_personal_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of personal space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.PERSONAL
    )
    return percentage, df_group_intrusions


@metric("group_social_space_intrusions")
def group_social_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the percentage of social space intrusions for each group"""
    percentage, df_group_intrusions = group_space_intrusions(
        df_agents_list, df_robot, SocialSpace.SOCIAL
    )
    return percentage, df_group_intrusions


@metric("group_public_space_intrusions")
def group_public_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
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


@metric("robot_on_person_collision")
def robot_on_person_collision(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people"""
    robot_collisions, _, df_agents = check_collisions(df_agents_list, df_robot)
    return robot_collisions, df_agents


@metric("person_on_robot_collision")
def person_on_robot_collision(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people"""
    _, person_collisions, df_agents = check_collisions(df_agents_list, df_robot)
    return person_collisions


@metric("time_not_moving")
def time_not_moving(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    period: int = 1,
    min_velocity: float = 0.01,
    min_angular_velocity: float = 0.02,
    **kwargs,
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
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    goal_tolerance: float = 0.5,
    robot_radius: float = 0.3,
    **kwargs,
) -> bool:
    """Check if the robot reached the goal"""
    return time_goal_reached(df_robot, goal, goal_tolerance, robot_radius) is not None


@metric("final_goal_distance")
def final_goal_distance(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    **kwargs,
) -> float:
    """Compute the distance between the last robot position and the target position."""
    x = df_robot.position_x.iloc[-1]
    y = df_robot.position_y.iloc[-1]
    return np.linalg.norm([x - goal[0], y - goal[1]])


@metric("minimum_goal_distance")
def minimum_goal_distance(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    **kwargs,
) -> float:
    """Compute the closest the robot passes to the target position."""
    x = df_robot.position_x
    y = df_robot.position_y
    return np.min(np.linalg.norm([x - goal[0], y - goal[1]], axis=1))


# SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
# ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University


@metric("avg_robot_linear_speed")
def avg_robot_linear_speed(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot linear speed"""
    return df_robot.velocity_x.abs().mean(), df_robot.velocity_x.abs()


@metric("avg_robot_angular_speed")
def avg_robot_angular_speed(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot angular speed"""
    return df_robot.angular_velocity_z.abs().mean(), df_robot.angular_velocity_z


@metric("avg_acceleration")
def avg_acceleration(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    period: int = 5,
    max_accel: float = 1.0,
    alpha: float = 0.1,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot acceleration"""
    df_robot_ = df_robot.copy()
    df_robot_["acceleration"] = df_robot.velocity_x.abs().diff(
        periods=period
    ) / df_robot.time.diff(periods=period)
    df_robot_[df_robot_["acceleration"].abs() > max_accel] = np.nan
    df_robot_["acceleration"].interpolate(method="cubic", order=3, inplace=True)
    df_robot_.dropna(inplace=True)
    df_robot_.reset_index(drop=True, inplace=True)
    df_robot_["acceleration"].ewm(alpha=alpha).mean()

    return df_robot_["acceleration"].mean(), df_robot_[["time", "acceleration"]]


@metric("avg_overacceleration")
def avg_overacceleration(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    period: int = 5,
    max_accel: float = 1.0,
    max_jerk: float = 1.0,
    alpha: float = 0.1,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot over-acceleration"""
    df_robot_ = df_robot.copy()
    _, df_acceleration = avg_acceleration(
        df_agents_list, df_robot_, period, max_accel, alpha
    )
    df_acceleration["jerk"] = df_acceleration.acceleration.diff(
        periods=period
    ) / df_acceleration.time.diff(periods=period)
    df_acceleration[df_acceleration["jerk"].abs() > max_jerk] = np.nan
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
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
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


@metric("avg_closest_pedestrian_velocity")
def avg_closest_pedestrian_velocity(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
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
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the social force provoked by the robot to the agents"""
    df_robot_ = df_robot.copy()
    df_robot_["social_force_dual_norm"] = modulus_social_force_dual(
        df_robot_, df_agents_list
    )
    return (
        df_robot_["social_force_dual_norm"].sum(),
        df_robot_[["time", "social_force_dual_norm"]],
    )


@metric("social_force_on_robot")
def social_force_on_robot(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    df_robot_ = df_robot.copy()
    """Compute the cumulative modulus of the social force provoked by the agents to the robot"""
    df_robot_["social_force_norm"] = modulus_social_force(df_robot_, df_agents_list)
    return (
        df_robot_["social_force_norm"].sum(),
        df_robot_[["time", "social_force_norm"]],
    )


# cumulative social work employed in this planner:
# https://github.com/robotics-upo/social_force_window_planner


@metric("social_work")
def social_work(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    **kwargs,
) -> float:
    df_robot_ = df_robot.copy()
    """Compute the cumulative social work employed in this planner"""
    df_robot_["social_work"] = compute_social_work(
        df_robot_, df_agents_list, df_obstacles
    )
    return df_robot_["social_work"].sum(), df_robot_[["time", "social_work"]]


# social work per second or social power
@metric("social_work_per_second")
def social_work_per_second(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    **kwargs,
) -> float:
    """Compute the cumulative social work employed in this planner per second"""
    df_robot_ = df_robot.copy()
    _, df_social_work = social_work(
        df_agents_list=df_agents_list, df_robot=df_robot_, df_obstacles=df_obstacles
    )
    df_robot_["social_work"] = df_social_work["social_work"]
    return (
        df_robot_["social_work"].sum()
        / (df_robot_["time"].iloc[-1] - df_robot_["time"].iloc[0]),
        df_robot_[["time", "social_work"]],
    )


@metric("obstacle_force_on_agents")
def obstacle_force_on_agents(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the obstacle force provoked by the obstacles to the agents"""
    for df_agent in df_agents_list:
        df_agent[["obstacle_force_norm", "obstacle_force"]] = modulus_obstacle_force(
            df_agent, df_obstacles
        )
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


@metric("obstacle_force_on_robot")
def obstacle_force_on_robot(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the cumulative modulus of the obstacle force provoked by the obstacles to the robot"""
    df_robot_ = df_robot.copy()
    df_robot_[["obstacle_force_norm", "obstacle_force"]] = modulus_obstacle_force(
        df_robot_, df_obstacles
    )
    return (
        df_robot_["obstacle_force_norm"].sum(),
        df_robot_[["time", "obstacle_force_norm"]],
    )


# TODO
def path_irregularity(agents, robot):
    pass


# TODO
def path_efficiency(agents, robot):
    pass


@metric("static_obstacle_collision")
def static_obs_collision(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    distance_threshold: float = 0.5,
    **kwargs,
) -> int:
    """Compute the number of times the robot collides with a static obstacle"""

    # count the number of times the robot collides with a static obstacle
    df_robot_ = df_robot.copy()
    df_robot_[["closest_obs_x", "closest_obs_y", "closest_obs_dist"]] = (
        find_closest_obstacle(df_robot_, df_obstacles)
    )
    df_robot_["collision"] = df_robot_["closest_obs_dist"] < distance_threshold
    return df_robot_["collision"].sum()


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
