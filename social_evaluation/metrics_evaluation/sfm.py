import numpy as np
import pandas as pd
from typing import List, Optional
from social_evaluation.common import (
    find_closest_obstacle,
    normalize_angle,
    combine_df_robot_and_agent,
)

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
    """Compute the desired force for an agent

    :param df_agent: DataFrame with the agent data
    :param agent_goal: Goal position for the agent
    :param relaxation_time: Relaxation time for the agent
    :param force_factor_desired: Desired force factor
    :param goal_radius: Radius of the goal
    :return: DataFrame with the desired force for the agent
    """

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
    """Compute the obstacle force for an agent

    :param df_agent: DataFrame with the agent data
    :param df_obstacles: DataFrame with the obstacles data
    :param force_factor_obstacle: Obstacle force factor
    :param force_sigma_obstacle: Obstacle force sigma
    :return: DataFrame with the obstacle force for the agent
    """

    df_agent_ = df_agent.copy()
    if df_obstacles is None or df_obstacles.empty or not {"x", "y"} <= set(df_obstacles.columns):
        # A map without static obstacles (an open field): no obstacle force.
        df_agent_.dropna(subset=["time", "position_x"], inplace=True)
        df_agent_["obstacle_force"] = [np.zeros(2) for _ in range(len(df_agent_))]
        return df_agent_[["obstacle_force", "time"]]
    closest_obstacle = find_closest_obstacle(df_agent_, df_obstacles)
    df_agent_ = df_agent_.merge(closest_obstacle, on="time", how="right")
    df_agent_["obstacle_force"] = df_agent_.apply(
        lambda x: -force_factor_obstacle
        * np.exp(-x["closest_obs_dist"] / force_sigma_obstacle)
        * (np.array(x[["closest_obs_x", "closest_obs_y"]]) / x["closest_obs_dist"]),
        axis=1,
    )
    return df_agent_[["obstacle_force", "time"]]


def modulus_obstacle_force(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    force_factor_obstacle: float = config["force_factor_obstacle"],
    force_sigma_obstacle: float = config["force_sigma_obstacle"],
):
    """Compute the obstacle force norm for an agent

    :param df_agent: DataFrame with the agent data
    :param df_obstacles: DataFrame with the obstacles data
    :param force_factor_obstacle: Obstacle force factor
    :param force_sigma_obstacle: Obstacle force sigma
    :return: DataFrame with the obstacle force norm for the agent
    """

    df_agent_ = df_agent.copy()
    obstacle_force = compute_obstacle_force(
        df_agent_, df_obstacles, force_factor_obstacle, force_sigma_obstacle
    )
    df_agent_ = df_agent_.merge(obstacle_force, on="time", how="right")
    df_agent_["obstacle_force_norm"] = df_agent_.apply(
        lambda x: np.linalg.norm(x["obstacle_force"]), axis=1
    )
    return df_agent_[["obstacle_force_norm", "obstacle_force", "time"]]


def compute_social_force_two_agents(
    df_interaction: pd.DataFrame,
    lambda_: float = config["lambda_"],
    gamma: float = config["gamma"],
    n: float = config["n"],
    n_prime: float = config["n_prime"],
    force_factor_social: float = config["force_factor_social"],
) -> pd.DataFrame:
    """Compute the social force between two agents

    :param df_interaction: DataFrame with the interaction data
    :param lambda_: Lambda parameter
    :param gamma: Gamma parameter
    :param n: N parameter
    :param n_prime: N prime parameter
    :param force_factor_social: Social force factor
    :return: DataFrame with the social force between two agents
    """

    if df_interaction.empty:
        return pd.Series(
            {
                "social_force": np.array([np.nan, np.nan]),
                "time": np.nan,
            }
        )
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
    return df_interaction[["time", "social_force"]]


def compute_social_force(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the social force provoked by the other agents to the agent examined

    :param df_agent: DataFrame with the agent data
    :param df_agents: List of DataFrames with the other agents data
    :return: DataFrame with the social force provoked by the other agents to the agent examined
    """

    df_agent_ = df_agent.copy()
    valid_agents = []
    for i, agent in enumerate(df_agents):
        df_interaction = combine_df_robot_and_agent(df_agent, agent)
        df_interaction.dropna(inplace=True)
        if df_interaction.empty:
            continue
        social_force = compute_social_force_two_agents(df_interaction)
        df_agent_ = df_agent_.merge(
            social_force, on="time", how="right", suffixes=("", f"_{i}")
        )
        df_agent_.rename(columns={"social_force": f"social_force_{i}"}, inplace=True)
        valid_agents.append(f"social_force_{i}")
    df_agent_["social_force"] = df_agent_[[agent for agent in valid_agents]].sum(axis=1)
    df_agent_["social_force"] = df_agent_["social_force"].apply(
        lambda x: np.nan if not isinstance(x, np.ndarray) else x
    )
    return df_agent_[["social_force", "time"]]


def modulus_social_force(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the cumulative modulus of the social force provoked by the agents in the agent

    :param df_agent: DataFrame with the agent data
    :param df_agents: List of DataFrames with the other agents data
    :return: DataFrame with the cumulative modulus of the social force provoked by the agents in the agent
    """

    df_agent_ = df_agent.copy()
    valid_agents = []
    for i, agent in enumerate(df_agents):
        df_interaction = combine_df_robot_and_agent(df_agent, agent)
        df_interaction.dropna(inplace=True)
        if df_interaction.empty:
            continue
        social_force = compute_social_force_two_agents(df_interaction)
        df_agent_ = df_agent_.merge(
            social_force, on="time", how="right", suffixes=("", f"_{i}")
        )
        df_agent_.rename(columns={"social_force": f"social_force_{i}"}, inplace=True)
        df_agent_[f"social_force_{i}_norm"] = df_agent_.apply(
            lambda x: np.linalg.norm(x[f"social_force_{i}"]), axis=1
        )
        valid_agents.append(f"social_force_{i}_norm")
    df_agent_["social_force_norm"] = df_agent_[[agent for agent in valid_agents]].sum(
        axis=1
    )
    return df_agent_[["social_force_norm", "time"]]


def compute_social_force_dual(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the social force provoked by the agent in examination to the other agents

    :param df_agent: DataFrame with the agent data
    :param df_agents: List of DataFrames with the other agents data
    :return: DataFrame with the social force provoked by the agent in examination to the other agents
    """

    df_agent_ = df_agent.copy()
    for i, agent in enumerate(df_agents):
        df_interaction = combine_df_robot_and_agent(df_agent, agent)
        df_interaction.dropna(inplace=True)
        # use time as index to join the dataframes
        social_force = compute_social_force_two_agents(df_interaction)
        df_agent_ = df_agent_.merge(social_force, on="time", how="right")
        # rename the column to avoid conflicts
        df_agent_.rename(
            columns={"social_force": f"social_force_{i}_dual"}, inplace=True
        )
    df_agent_["social_force_dual"] = df_agent_[
        [f"social_force_{i}_dual" for i in range(len(df_agents))]
    ].sum(axis=1)
    df_agent_["social_force_dual"] = df_agent_["social_force_dual"].apply(
        lambda x: np.nan if not isinstance(x, np.ndarray) else x
    )
    return df_agent_[["social_force_dual", "time"]]


def modulus_social_force_dual(df_agent: pd.DataFrame, df_agents: List[pd.DataFrame]):
    """
    It computes the cumulative modulus of the social force provoked by the agent in the rest of agents

    :param df_agent: DataFrame with the agent data
    :param df_agents: List of DataFrames with the other agents data
    :return: DataFrame with the cumulative modulus of the social force provoked by the agent in the rest of agents
    """

    df_agent_ = df_agent.copy()
    valid_agents = []
    for i, agent in enumerate(df_agents):
        df_interaction = combine_df_robot_and_agent(df_agent, agent)
        df_interaction.dropna(inplace=True)
        if df_interaction.empty:
            continue
        social_force = compute_social_force_two_agents(df_interaction)
        df_agent_ = df_agent_.merge(
            social_force, on="time", how="right", suffixes=("", f"_{i}")
        )
        df_agent_.rename(
            columns={"social_force": f"social_force_{i}_dual"}, inplace=True
        )
        df_agent_[f"social_force_{i}_dual_norm"] = df_agent_.apply(
            lambda x: np.linalg.norm(x[f"social_force_{i}_dual"]), axis=1
        )
        valid_agents.append(f"social_force_{i}_dual_norm")
    df_agent_["social_force_dual_norm"] = df_agent_[
        [agent for agent in valid_agents]
    ].sum(axis=1)
    return df_agent_[["social_force_dual_norm", "time"]]


def compute_social_work(
    df_robot: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
) -> pd.DataFrame:
    """
    It computes the social work of the robot in the environment plus the social work provoked by the robot in the other agents

    :param df_robot: DataFrame with the robot data
    :param df_agents_list: List of DataFrames with the other agents data
    :param df_obstacles: DataFrame with the obstacles data
    :return: DataFrame with the social work of the robot in the environment plus the social work provoked by the robot in the other agents
    """

    df_robot_ = df_robot.copy()
    df_agents_list_ = df_agents_list.copy()
    social_force = compute_social_force(df_robot, df_agents_list_)
    df_robot_ = df_robot_.merge(social_force, on="time", how="right")
    obstacle_force_norm = modulus_obstacle_force(df_robot, df_obstacles)
    df_robot_ = df_robot_.merge(obstacle_force_norm, on="time", how="right")
    df_robot_.ffill(inplace=True)
    for df_agent in df_agents_list_:
        df_agent = df_agent.ffill(inplace=True)

    df_robot_["social_force_norm"] = df_robot_["social_force"].apply(
        lambda x: np.linalg.norm(x) if isinstance(x, np.ndarray) else np.nan
    )
    df_robot_["wr"] = df_robot_["social_force_norm"] + df_robot_["obstacle_force_norm"]

    social_force_dual_norm = modulus_social_force_dual(df_robot, df_agents_list_)
    df_robot_ = df_robot_.merge(social_force_dual_norm, on="time", how="right")
    df_robot_.rename(columns={"social_force_dual_norm": "wa"}, inplace=True)
    df_robot_["social_work"] = df_robot_["wr"] + df_robot_["wa"]
    return df_robot_[["social_work", "time"]]
