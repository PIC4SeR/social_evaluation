import numpy as np
import math
import pandas as pd
from typing import List, Tuple, Dict, Any, Union, Optional
from social_evaluation.utils.map_utils import find_closest_obstacle


def normalize_angle(angle: float) -> float:
    """Normalize the angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


config = {
    "force_factor_desired": 2.0,
    "force_factor_obstacle": 10,
    "force_factor_social": 2.1,
    "force_factor_group_gaze": 3.0,
    "force_factor_group_coherence": 2.0,
    "force_factor_group_repulsion": 1.0,
    "force_sigma_obstacle": 0.2,
    "lambda_": 2.0,
    "gamma": 0.35,
    "n": 2.0,
    "n_prime": 3.0,
    "relaxation_time": 0.5,
    "robot_radius": 0.2,  # agent radius
    "goal_radius": 0.2,  # agent goal radius
    "desired_velocity": 0.5,  # agent desired velocity
}


def compute_desired_force(
    df_agent: pd.DataFrame,
    agent_goal: Optional[np.ndarray] = None,
    relaxation_time: float = config["relaxation_time"],
    force_factor_desired: float = config["force_factor_desired"],
    goal_radius: float = config["goal_radius"],
):
    """Compute the desired force for an agent"""
    df_desired = pd.DataFrame()
    df_desired["time"] = df_agent["time"]
    df_desired["agent_position"] = df_agent.apply(
        lambda x: np.array([x["position_x"], x["position_y"]]), axis=1
    )
    df_desired["agent_velocity"] = df_agent.apply(
        lambda x: np.array([x["velocity_x"], x["velocity_y"]]), axis=1
    )

    if agent_goal is None:
        df_desired["desired_force"] = -df_desired["agent_velocity"] / relaxation_time
        return df_desired["desired_force"]
    df_desired["goal_position"] = agent_goal
    df_desired["dist"] = df_desired.apply(
        lambda x: np.linalg.norm(x["goal_position"] - x["agent_position"]), axis=1
    )
    df_desired["desired_direction"] = df_desired.apply(
        lambda x: (x["goal_position"] - x["agent_position"]) / x["dist"], axis=1
    )
    df_desired["desired_force"] = df_agent.apply(
        lambda x: (
            -x["agent_velocity"] / relaxation_time
            if x["dist"] < goal_radius
            else force_factor_desired
            * (x["desired_direction"] * x["desired_velocity"] - x["agent_velocity"])
            / relaxation_time
        ),
        axis=1,
    )
    return df_desired["desired_force"]


def compute_obstacle_force(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    force_factor_obstacle: float = config["force_factor_obstacle"],
    force_sigma_obstacle: float = config["force_sigma_obstacle"],
) -> pd.DataFrame:
    """Compute the obstacle force for an agent"""
    df_agent_ = df_agent.copy()
    df_agent_[["closest_obs_x", "closest_obs_y", "closest_obs_dist"]] = (
        find_closest_obstacle(df_agent_, df_obstacles)
    )
    df_agent_["obstacle_force"] = df_agent_.apply(
        lambda x: force_factor_obstacle
        * np.exp(-x["closest_obs_dist"] / force_sigma_obstacle)
        * (np.array(x[["closest_obs_x", "closest_obs_y"]]) / x["closest_obs_dist"]),
        axis=1,
    )
    return df_agent_["obstacle_force"]


def modulus_obstacle_force(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    force_factor_obstacle: float = config["force_factor_obstacle"],
    force_sigma_obstacle: float = config["force_sigma_obstacle"],
):
    """Compute the obstacle force norm for an agent"""
    df_agent_ = df_agent.copy()
    df_agent_["obstacle_force"] = compute_obstacle_force(
        df_agent_, df_obstacles, force_factor_obstacle, force_sigma_obstacle
    )
    df_agent_["obstacle_force_norm"] = df_agent_.apply(
        lambda x: np.linalg.norm(x["obstacle_force"]), axis=1
    )
    return df_agent_[["obstacle_force_norm", "obstacle_force"]]


def compute_social_force_two_agents(
    df_agent1: pd.DataFrame,
    df_agent2: pd.DataFrame,
    lambda_: float = config["lambda_"],
    gamma: float = config["gamma"],
    n: float = config["n"],
    n_prime: float = config["n_prime"],
    force_factor_social: float = config["force_factor_social"],
):
    """Compute the social force between two agents"""
    df_interaction = df_agent1.merge(df_agent2, on="time", suffixes=("_1", "_2"))
    df_interaction.dropna(inplace=True)
    if df_interaction.empty:
        return pd.Series([np.nan] * len(df_agent1))
    df_interaction.reset_index(drop=True, inplace=True)

    df_interaction["diff"] = df_interaction[["position_x_1", "position_y_1"]].apply(
        lambda x: np.array([x["position_x_1"], x["position_y_1"]]), axis=1
    ) - df_interaction[["position_x_2", "position_y_2"]].apply(
        lambda x: np.array([x["position_x_2"], x["position_y_2"]]), axis=1
    )
    df_interaction["diff_norm"] = df_interaction.apply(
        lambda x: np.linalg.norm(x["diff"]), axis=1
    )
    df_interaction["diff_direction"] = df_interaction.apply(
        lambda x: x["diff"] / x["diff_norm"], axis=1
    )

    df_interaction["vel_diff"] = df_interaction[["velocity_x_1", "velocity_y_1"]].apply(
        lambda x: np.array([x["velocity_x_1"], x["velocity_y_1"]]), axis=1
    ) - df_interaction[["velocity_x_2", "velocity_y_2"]].apply(
        lambda x: np.array([x["velocity_x_2"], x["velocity_y_2"]]), axis=1
    )
    df_interaction["interaction_vector"] = (
        df_interaction["diff"] + lambda_ * df_interaction["vel_diff"]
    )
    df_interaction["interaction_length"] = df_interaction["interaction_vector"].apply(
        np.linalg.norm
    )
    df_interaction["interaction_direction"] = df_interaction[
        "interaction_vector"
    ].apply(lambda x: x / np.linalg.norm(x))
    df_interaction["theta"] = df_interaction.apply(
        lambda x: np.arctan2(x["diff"][1], x["diff"][0])
        - np.arctan2(x["interaction_direction"][1], x["interaction_direction"][0]),
        axis=1,
    )
    df_interaction["theta"] = df_interaction["theta"].apply(normalize_angle)

    df_interaction["B"] = gamma * df_interaction["interaction_length"]
    df_interaction["force_velocity_amount"] = df_interaction.apply(
        lambda x: -np.exp(
            -x["diff_norm"] / x["B"] - (n_prime * x["B"] * x["theta"]) ** 2
        ),
        axis=1,
    )
    df_interaction["force_angle_amount"] = df_interaction.apply(
        lambda x: -np.sign(x["theta"])
        * np.exp(-x["diff_norm"] / x["B"] - (n * x["B"] * x["theta"]) ** 2),
        axis=1,
    )

    df_interaction["force_velocity"] = (
        df_interaction["force_velocity_amount"]
        * df_interaction["interaction_direction"]
    )
    df_interaction["interaction_direction_left_normal"] = df_interaction[
        "interaction_direction"
    ].apply(lambda x: np.array([-x[1], x[0]]))
    df_interaction["force_angle"] = (
        df_interaction["force_angle_amount"]
        * df_interaction["interaction_direction_left_normal"]
    )
    df_interaction["social_force"] = force_factor_social * (
        df_interaction["force_velocity"] + df_interaction["force_angle"]
    )
    return df_interaction["social_force"]


def compute_social_force(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the social force provoked by the other agents to the agent examined
    """
    for i, agent in enumerate(df_agents):
        df_agent[f"social_force_{i}"] = compute_social_force_two_agents(df_agent, agent)
    df_agent["social_force"] = df_agent[
        [f"social_force_{i}" for i in range(len(df_agents))]
    ].sum(axis=1)
    df_agent["social_force"] = df_agent["social_force"].apply(
        lambda x: np.nan if not isinstance(x, np.ndarray) else x
    )
    return df_agent["social_force"]


def modulus_social_force(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the cumulative modulus of the social force provoked by the agents in the agent
    """
    df_agent["social_force"] = compute_social_force(df_agent, df_agents)
    return df_agent["social_force"].apply(
        lambda x: np.linalg.norm(x) if isinstance(x, np.ndarray) else np.nan
    )


def compute_social_force_dual(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the social force provoked by the agent in examination to the other agents
    """
    for i, agent in enumerate(df_agents):
        df_agent[f"social_force_{i}_dual"] = compute_social_force_two_agents(
            agent, df_agent
        )
    df_agent["social_force_dual"] = df_agent[
        [f"social_force_{i}_dual" for i in range(len(df_agents))]
    ].sum(axis=1)
    df_agent["social_force_dual"] = df_agent["social_force_dual"].apply(
        lambda x: np.nan if not isinstance(x, np.ndarray) else x
    )
    return df_agent["social_force_dual"]


def modulus_social_force_dual(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the cumulative modulus of the social force provoked by the agent in the rest of agents
    """
    df_agent["social_force_dual"] = compute_social_force_dual(df_agent, df_agents)
    return (
        df_agent["social_force_dual"]
        .dropna()
        .apply(lambda x: np.linalg.norm(x) if isinstance(x, np.ndarray) else np.nan)
    )


def compute_social_work(
    df_robot: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
) -> pd.DataFrame:
    """
    It computes the social work of the robot in the environment
    plus the social work provoked by the robot in the other agents
    """
    df_robot_ = df_robot.copy()
    df_agents_list_ = df_agents_list.copy()
    df_robot_["social_force_norm"] = modulus_social_force(df_robot, df_agents_list_)
    df_robot_[["obstacle_force_norm", "obstacle_force"]] = modulus_obstacle_force(
        df_robot, df_obstacles
    )
    df_robot_.ffill(inplace=True)
    for df_agent in df_agents_list_:
        df_agent = df_agent.ffill(inplace=True)

    df_robot_["wr"] = df_robot_["social_force_norm"] + df_robot_["obstacle_force_norm"]

    for i, agent in enumerate(df_agents_list_):
        df_robot_[f"wa_{i}"] = compute_social_force_two_agents(df_robot_, agent)
    df_robot_["wa"] = df_robot_[[f"wa_{i}" for i in range(len(df_agents_list_))]].sum(
        axis=1
    )
    df_robot_["wa"] = df_robot_["wa"].apply(
        lambda x: np.linalg.norm(x) if isinstance(x, np.ndarray) else np.nan
    )

    return df_robot_["wr"] + df_robot_["wa"]
