import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from social_evaluation.common import (
    time_goal_reached,
    combine_df_robot_agents,
    normalize_angle,
    combine_df_agents,
    find_closest_obstacle,
)
from social_evaluation.metrics_evaluation.metrics_utils import (
    metric,
    SocialSpace,
    space_intrusions,
    group_space_intrusions,
    check_collisions,
    distance_to_closest_person,
    indicator_function,
    closest_agent_to_robot,
    time_integral,
)

from social_evaluation.metrics_evaluation.sfm import (
    modulus_social_force_dual,
    modulus_social_force,
    compute_social_work,
    modulus_obstacle_force,
    robot_velocity_to_world,
)

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
    - df_obstacles: pandas DataFrame with the obstacles' data
    - df_results: pandas DataFrame with the results of the experiment
    - goal: tuple with the goal position (x, y)
    - robot_radius: float with the robot's radius
    - agent_radius: float with the agent's radius
    - min_velocity: float with the minimum velocity for considering the robot not moving
    - min_angular_velocity: float with the minimum angular velocity for considering the robot not moving
    - goal_tolerance: float with the goal tolerance for considering the goal reached
    - max_accel: float with the acceleration above which a value is an outlier
    - max_jerk: float with the jerk above which a value is an outlier
    - rate_window: float with the seconds acceleration and jerk are differenced over
    - smoothing_time: float with the time constant (s) of their moving average
    - dist_threshold: float with the distance threshold for considering a collision with a static obstacle

    The metrics return a tuple with the metric value and a pandas DataFrame with the metric values over time.
    
    Metrics:
    - experiment_result: Compute the experiment result
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
    - final_distance_to_target: Compute the distance between the last robot position and the target position
    - minimum_distance_to_target: Compute the closest the robot passes to the target position
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


def robot_speed(df_robot: pd.DataFrame) -> pd.Series:
    """The robot's speed, whatever the frame of its velocity (hunav_evaluator's linear_vel)

    :param df_robot: pandas DataFrame with the robot's data
    :return: Series with the robot's speed
    """
    velocity_y = df_robot["velocity_y"] if "velocity_y" in df_robot else 0.0
    return np.hypot(df_robot["velocity_x"], velocity_y)


@metric("experiment_result")
def experiment_result(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_results: Optional[pd.DataFrame] = None,
    **kwargs,
) -> str:
    """Compute the experiment result

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_results: pandas DataFrame with the results of the experiment
    :return: str with the experiment result
    """
    if df_results is None:
        print("No results dataframe provided")
        return ""
    return df_results.iloc[0]["data"]


@metric("time_to_reach_goal")
def total_time(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    goal_tolerance: float = 0.3,
    **kwargs,
) -> float:
    """Compute the time until the robot first comes within ``goal_tolerance`` of
    the goal (its centre), or the whole run when it never does

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param goal: tuple with the goal position (x, y)
    :param goal_tolerance: float with the goal tolerance for considering the goal reached
    :return: float with the total time of the robot's trajectory
    """

    reached = time_goal_reached(df_robot, goal, goal_tolerance)
    if reached is not None:
        return reached - df_robot.time.iloc[0]
    return df_robot.dropna().time.iloc[-1] - df_robot.time.iloc[0]


@metric("path_length")
def robot_path_length(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> float:
    """Compute the total length of the robot's trajectory

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the total length of the robot's trajectory
    """

    step_x = df_robot.position_x.diff().dropna()
    step_y = df_robot.position_y.diff().dropna()

    # compute the step as the mean difference on a period
    step = np.array([step_x, step_y]).T
    # compute the length of each step
    step_length = np.linalg.norm(step, axis=1)
    return np.sum(step_length)


@metric("cumulative_heading_changes")
def cumulative_heading_changes(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> float:
    """Compute the cumulative heading changes of the robot

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the cumulative heading changes of the robot
    """
    # wrap every heading change to [-pi, pi] so crossing +-pi is not a full turn
    df_robot_ = df_robot.copy().dropna()
    return np.sum(np.abs(normalize_angle(np.diff(df_robot_.yaw.to_numpy()))))


@metric("avg_distance_to_closest_person")
def avg_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average distance to the closest person

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param robot_radius: float with the robot's radius
    :param agent_radius: float with the agent's radius
    :return: float with the average distance to the closest
    """

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
    """Compute the minimum distance to people

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param robot_radius: float with the robot's radius
    :param agent_radius: float with the agent's radius
    :return: float with the minimum distance to people
    """

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
    """Compute the maximum distance to people

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param robot_radius: float with the robot's radius
    :param agent_radius: float with the agent's radius
    :return: float with the maximum distance to people
    """

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


def _make_space_intrusion_metric(space: SocialSpace):
    """Build a per-agent space-intrusion metric for a given social space."""

    def _metric(
        df_agents_list: List[pd.DataFrame],
        df_robot: pd.DataFrame,
        *args,
        robot_radius: float = 0.3,
        agent_radius: float = 0.3,
        **kwargs,
    ) -> Tuple[float, pd.DataFrame]:
        return space_intrusions(
            df_agents_list, df_robot, space, robot_radius, agent_radius
        )

    return _metric


def _make_group_space_intrusion_metric(space: SocialSpace):
    """Build a per-group space-intrusion metric for a given social space."""

    def _metric(
        df_agents_list: List[pd.DataFrame],
        df_robot: pd.DataFrame,
        *args,
        robot_radius: float = 0.3,
        **kwargs,
    ) -> Tuple[float, pd.DataFrame]:
        return group_space_intrusions(
            df_agents_list, df_robot, space, robot_radius=robot_radius
        )

    return _metric


@metric("intimate_personal_space_intrusions")
def intimate_personal_space_intrusions(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Percentage of time with a person closer than 1.2 m (body gap): the intimate
    (< 0.45 m) and personal (0.45-1.2 m) zones together. The zones are disjoint, so this
    is intimate_ + personal_space_intrusions -- and unlike personal alone it cannot drop
    when the robot moves closer still."""
    df_min_dist = distance_to_closest_person(df_agents_list, df_robot, robot_radius, agent_radius)
    df_min_dist["intrusion"] = df_min_dist["min_distance"].apply(
        lambda d: indicator_function(d, SocialSpace.INTIMATE) + indicator_function(d, SocialSpace.PERSONAL)
    )
    return df_min_dist["intrusion"].mean() * 100, df_min_dist[["time", "intrusion"]]


# Register the four proxemic zones for both per-agent and per-group intrusions.
for _space in SocialSpace:
    _zone = _space.name.lower()
    metric(f"{_zone}_space_intrusions")(_make_space_intrusion_metric(_space))
    metric(f"group_{_zone}_space_intrusions")(
        _make_group_space_intrusion_metric(_space)
    )


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
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param robot_radius: float with the robot's radius
    :param agent_radius: float with the agent's radius
    :return: int with the number of collisions between the robot and people
    """
    robot_collisions, _, df_agents = check_collisions(
        df_agents_list, df_robot, robot_radius, agent_radius
    )
    return robot_collisions, df_agents


@metric("person_on_robot_collision")
def person_on_robot_collision(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    robot_radius: float = 0.3,
    agent_radius: float = 0.3,
    **kwargs,
) -> Tuple[int, pd.DataFrame]:
    """Check for collisions between the robot and people

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param robot_radius: float with the robot's radius
    :param agent_radius: float with the agent's radius
    :return: int with the number of collisions between the robot and people
    """
    _, person_collisions, df_agents = check_collisions(
        df_agents_list, df_robot, robot_radius, agent_radius
    )
    return person_collisions, df_agents


@metric("time_not_moving")
def time_not_moving(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    min_velocity: float = 0.01,
    min_angular_velocity: float = 0.02,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the time the robot is not moving

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param min_velocity: float with the minimum velocity for considering the robot not moving
    :param min_angular_velocity: float with the minimum angular velocity for considering the robot not moving
    :return: float with the time the robot is not moving
    """
    not_moving = (
        (robot_speed(df_robot) < min_velocity)
        & (df_robot["angular_velocity_z"].abs() < min_angular_velocity)
    ).astype(int)
    time_stopped = df_robot.time.diff() * not_moving
    return time_stopped.sum(), not_moving


@metric("completed")
def goal_reached(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    goal_tolerance: float = 0.3,
    **kwargs,
) -> bool:
    """Check if the robot ended the run at the goal (SEAN 2.0): its final
    position within ``goal_tolerance`` of the goal (its centre)

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param goal: tuple with the goal position (x, y)
    :param goal_tolerance: float with the goal tolerance for considering the goal reached
    :return: bool with True if the robot reached the goal, False otherwise
    """
    final = df_robot.dropna(subset=["position_x", "position_y"]).iloc[-1]
    return bool(
        np.hypot(final.position_x - goal[0], final.position_y - goal[1]) < goal_tolerance
    )


@metric("final_distance_to_target")
def final_goal_distance(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    **kwargs,
) -> float:
    """Compute the distance between the last robot position and the target position.

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param goal: tuple with the goal position (x, y)
    :return: float with the distance between the last robot position and the target position
    """

    df_robot_ = df_robot.copy().dropna()
    x = df_robot_.position_x.iloc[-1]
    y = df_robot_.position_y.iloc[-1]
    return np.linalg.norm([x - goal[0], y - goal[1]])


@metric("minimum_distance_to_target")
def minimum_goal_distance(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    goal: Tuple[float, float],
    **kwargs,
) -> float:
    """Compute the closest the robot passes to the target position.

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param goal: tuple with the goal position (x, y)
    :return: float with the minimum distance to the target position
    """

    df_robot_ = df_robot.copy().dropna()
    x = df_robot_.position_x
    y = df_robot_.position_y
    return np.min(np.linalg.norm([x - goal[0], y - goal[1]], axis=0))


# SocNavBench: A Grounded Simulation Testing Framework for Evaluating Social Navigation
# ABHIJAT BISWAS, ALLAN WANG, GUSTAVO SILVERA, AARON STEINFELD, and HENNY AD-MONI, Carnegie Mellon University


@metric("avg_robot_linear_speed")
def avg_robot_linear_speed(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot linear speed

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the average robot linear speed
    """
    speed = robot_speed(df_robot)
    return speed.mean(), speed


@metric("avg_robot_angular_speed")
def avg_robot_angular_speed(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average robot angular speed

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the average robot angular speed
    """
    return df_robot.angular_velocity_z.abs().mean(), df_robot.angular_velocity_z


def rate_filter(
    df: pd.DataFrame, rate_window: float, smoothing_time: float
) -> Tuple[int, float]:
    """The sample count and moving-average factor of a rate over ``rate_window``
    seconds smoothed with time constant ``smoothing_time``, at this data's median
    sample step. Fixing them in seconds keeps acceleration and jerk comparable
    between data recorded at 10 Hz and at 100 Hz.

    :param df: pandas DataFrame with a time column
    :param rate_window: seconds each rate is differenced over
    :param smoothing_time: time constant (s) of the exponential moving average
    :return: (samples per rate window, moving-average factor)
    """
    times = np.unique(df["time"].dropna().to_numpy(dtype=float))
    dt = float(np.median(np.diff(times))) if len(times) > 1 else rate_window
    period = max(1, int(round(rate_window / dt)))
    return period, float(1.0 - np.exp(-dt / smoothing_time))


@metric("avg_acceleration")
def avg_acceleration(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    rate_window: float = 0.5,
    max_accel: float = 2.5,
    smoothing_time: float = 0.949,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average magnitude of the robot's acceleration

    The speed is differenced over ``rate_window`` seconds, values above
    ``max_accel`` are treated as outliers and interpolated, then smoothed with
    an exponential moving average of time constant ``smoothing_time`` seconds.

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param rate_window: float with the seconds the speed is differenced over
    :param max_accel: float with the acceleration (m/s^2) above which a value is an outlier
    :param smoothing_time: float with the time constant (s) of the moving average
    :return: float with the average robot acceleration
    """
    period, alpha = rate_filter(df_robot, rate_window, smoothing_time)
    df_robot_ = df_robot.copy()
    df_robot_["acceleration"] = robot_speed(df_robot).diff(
        periods=period
    ) / df_robot.time.diff(periods=period)
    # Null only the outlier acceleration values (not the whole row) so they can
    # be interpolated; masking the whole row would blank `time` and drop them.
    df_robot_.loc[df_robot_["acceleration"].abs() > max_accel, "acceleration"] = np.nan
    df_robot_["acceleration"] = df_robot_["acceleration"].interpolate(
        method="cubic", order=3
    )
    df_robot_.dropna(inplace=True)
    df_robot_.reset_index(drop=True, inplace=True)
    df_robot_["acceleration"] = df_robot_["acceleration"].ewm(alpha=alpha).mean()

    return df_robot_["acceleration"].abs().mean(), df_robot_[["time", "acceleration"]]


@metric("avg_overacceleration")
def avg_overacceleration(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    rate_window: float = 0.5,
    max_accel: float = 2.5,
    max_jerk: float = 2.5,
    smoothing_time: float = 0.949,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the average magnitude of the robot's jerk (over-acceleration): the
    rate of change of the filtered acceleration, filtered the same way

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param rate_window: float with the seconds each rate is differenced over
    :param max_accel: float with the acceleration (m/s^2) above which a value is an outlier
    :param max_jerk: float with the jerk (m/s^3) above which a value is an outlier
    :param smoothing_time: float with the time constant (s) of the moving average
    :return: float with the average robot over-acceleration
    """

    period, alpha = rate_filter(df_robot, rate_window, smoothing_time)
    df_robot_ = df_robot.copy()
    _, df_acceleration = avg_acceleration(
        df_agents_list,
        df_robot_,
        rate_window=rate_window,
        max_accel=max_accel,
        smoothing_time=smoothing_time,
    )
    df_acceleration["jerk"] = df_acceleration.acceleration.diff(
        periods=period
    ) / df_acceleration.time.diff(periods=period)
    # Null only the outlier jerk values (not the whole row) so they can be
    # interpolated instead of dropped.
    df_acceleration.loc[df_acceleration["jerk"].abs() > max_jerk, "jerk"] = np.nan
    df_acceleration["jerk"] = df_acceleration["jerk"].interpolate(
        method="cubic", order=3
    )
    df_acceleration.dropna(inplace=True)
    df_acceleration.reset_index(drop=True, inplace=True)
    df_acceleration["jerk"] = df_acceleration["jerk"].ewm(alpha=alpha).mean()

    return df_acceleration["jerk"].abs().mean(), df_acceleration[["time", "jerk"]]


# Learning a Group-Aware Policy for Robot Navigation
# Kapil Katyal ∗1,2 , Yuxiang Gao ∗2 , Jared Markowitz 1 , Sara Pohland 3 , Corban Rivera 1 , I-Jeng Wang 1 , Chien-Ming Huang 2


@metric("avg_pedestrian_velocity")
def avg_pedestrian_velocity(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, List[float]]:
    """Compute the average pedestrian velocity

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the average pedestrian velocity
    """

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
    """Compute the average velocity of the closest pedestrian to the robot

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the average velocity of the closest pedestrian to the robot
    """

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
    """Compute the modulus of the social force provoked by the robot to the agents,
    integrated over time

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the time integral (N s) of the social force provoked by the robot to the agents
    """

    df_robot_ = robot_velocity_to_world(df_robot)
    social_force_dual_norm = modulus_social_force_dual(df_robot_, df_agents_list)
    df_robot_ = df_robot_.merge(social_force_dual_norm, on="time", how="left")
    return (
        time_integral(df_robot_, "social_force_dual_norm"),
        df_robot_[["time", "social_force_dual_norm"]],
    )


@metric("social_force_on_robot")
def social_force_on_robot(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the modulus of the social force provoked by the agents to the robot,
    integrated over time

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :return: float with the time integral (N s) of the social force provoked by the agents to the robot
    """
    df_robot_ = robot_velocity_to_world(df_robot)

    social_force_norm = modulus_social_force(df_robot_, df_agents_list)
    df_robot_ = df_robot_.merge(social_force_norm, on="time", how="left")
    return (
        time_integral(df_robot_, "social_force_norm"),
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
    robot_radius: float = 0.3,
    **kwargs,
) -> float:
    """Compute the social work employed in this planner, integrated over time

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_obstacles: pandas DataFrame with the obstacles' data
    :param robot_radius: float with the robot's radius
    :return: float with the time integral (N s) of the social work employed in this planner
    """
    df_robot_ = df_robot.copy()

    social_work = compute_social_work(
        robot_velocity_to_world(df_robot), df_agents_list, df_obstacles, robot_radius
    )
    df_robot_ = df_robot_.merge(social_work, on="time", how="left")
    return time_integral(df_robot_, "social_work"), df_robot_[["time", "social_work"]]


# social work per second or social power
@metric("social_work_per_second")
def social_work_per_second(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    robot_radius: float = 0.3,
    **kwargs,
) -> float:
    """Compute the social work employed in this planner per second: its time
    integral over the time it was measured

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_obstacles: pandas DataFrame with the obstacles' data
    :param robot_radius: float with the robot's radius
    :return: float with the social work employed in this planner per second (N)
    """

    df_robot_ = df_robot.copy().dropna()
    total, df_social_work = social_work(
        df_agents_list=df_agents_list,
        df_robot=df_robot_,
        df_obstacles=df_obstacles,
        robot_radius=robot_radius,
    )
    measured = df_social_work.dropna()
    duration = measured["time"].max() - measured["time"].min() if len(measured) else 0.0
    if not duration > 0:
        return 0, df_social_work
    return total / duration, df_social_work


@metric("obstacle_force_on_agents")
def obstacle_force_on_agents(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    agent_radius: float = 0.3,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the modulus of the obstacle force provoked by the obstacles to the agents,
    integrated over time and averaged over the agents

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_obstacles: pandas DataFrame with the obstacles' data
    :param agent_radius: float with the agent's radius
    :return: float with the cumulative modulus of the obstacle force provoked by the obstacles to the agents
    """

    new_df_agents_list = []
    for df_agent in df_agents_list:
        obstacle_force_norm = modulus_obstacle_force(
            df_agent, df_obstacles, radius=agent_radius
        )
        obstacle_force_norm.drop(columns=["obstacle_force"], inplace=True)
        df_agent = df_agent.merge(obstacle_force_norm, on="time", how="left")
        new_df_agents_list.append(df_agent)
    df_agents = combine_df_agents(new_df_agents_list)
    return (
        np.mean(
            [
                time_integral(df_agent, "obstacle_force_norm")
                for df_agent in new_df_agents_list
            ]
        ),
        df_agents[
            [
                "time",
                *[
                    f"obstacle_force_norm_agent{i}"
                    for i in range(len(new_df_agents_list))
                ],
            ]
        ],
    )


@metric("obstacle_force_on_robot")
def obstacle_force_on_robot(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_obstacles: pd.DataFrame,
    robot_radius: float = 0.3,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Compute the modulus of the obstacle force provoked by the obstacles to the robot,
    integrated over time

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_obstacles: pandas DataFrame with the obstacles' data
    :param robot_radius: float with the robot's radius
    :return: float with the cumulative modulus of the obstacle force provoked by the obstacles to the robot
    """

    df_robot_ = df_robot.copy()
    obstacle_force_norm = modulus_obstacle_force(
        df_robot, df_obstacles, radius=robot_radius
    )
    df_robot_ = df_robot_.merge(obstacle_force_norm, on="time", how="left")
    return (
        time_integral(df_robot_, "obstacle_force_norm"),
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
    robot_radius: float = 0.3,
    **kwargs,
) -> int:
    """Compute the number of times the robot collides with a static obstacle

    :param df_agents_list: list of pandas DataFrames with the agents' data
    :param df_robot: pandas DataFrame with the robot's data
    :param df_obstacles: pandas DataFrame with the obstacles' data
    :param distance_threshold: float with the distance threshold for considering a collision
    :param robot_radius: float with the robot's radius
    :return: int with the number of times the robot collides with a static obstacle
    """

    # count the number of times the robot collides with a static obstacle
    df_robot_ = df_robot.copy()
    closeset_obstacle = find_closest_obstacle(
        df_robot, df_obstacles, robot_radius=robot_radius
    )
    df_robot_ = df_robot_.merge(closeset_obstacle, on="time", how="left")
    df_robot_["collision"] = df_robot_["closest_obs_dist"] < distance_threshold
    return df_robot_["collision"].sum(), df_robot_[["time", "collision"]]


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
