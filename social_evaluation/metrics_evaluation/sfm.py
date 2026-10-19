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


def robot_velocity_to_world(df_robot: pd.DataFrame) -> pd.DataFrame:
    """Copy of the robot data with ``velocity_x/y`` rotated into the world frame

    The robot's velocity is extracted in its own frame (odometry twist, ground
    truth and the transform-derived velocity alike), the people's velocities are
    in the world frame, and the social force needs both in the same frame.

    :param df_robot: DataFrame with the robot data (``yaw`` and body-frame velocity)
    :return: DataFrame with the robot data and world-frame velocity
    """
    df_robot_ = df_robot.copy()
    cos_yaw = np.cos(df_robot_["yaw"])
    sin_yaw = np.sin(df_robot_["yaw"])
    velocity_x = df_robot_["velocity_x"]
    velocity_y = df_robot_["velocity_y"]
    df_robot_["velocity_x"] = velocity_x * cos_yaw - velocity_y * sin_yaw
    df_robot_["velocity_y"] = velocity_x * sin_yaw + velocity_y * cos_yaw
    return df_robot_


def compute_obstacle_force(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    force_factor_obstacle: float = config["force_factor_obstacle"],
    force_sigma_obstacle: float = config["force_sigma_obstacle"],
    radius: float = config["robot_radius"],
) -> pd.DataFrame:
    """Compute the obstacle force for an agent

    The obstacle is the nearest occupied cell of the map image. hunav_evaluator
    averages the force of HuNav's closest-obstacle points instead (computed from
    the Gazebo world), so the two evaluators differ on this force.

    :param df_agent: DataFrame with the agent data
    :param df_obstacles: DataFrame with the obstacles data
    :param force_factor_obstacle: Obstacle force factor
    :param force_sigma_obstacle: Obstacle force sigma
    :param radius: Radius of the agent's body, subtracted from the obstacle distance
    :return: DataFrame with the obstacle force for the agent
    """

    df_agent_ = df_agent.copy()
    if df_obstacles is None or df_obstacles.empty or not {"x", "y"} <= set(df_obstacles.columns):
        # A map without static obstacles (an open field): no obstacle force.
        df_agent_.dropna(subset=["time", "position_x"], inplace=True)
        df_agent_["obstacle_force"] = [np.zeros(2) for _ in range(len(df_agent_))]
        return df_agent_[["obstacle_force", "time"]]
    closest_obstacle = find_closest_obstacle(df_agent_, df_obstacles, robot_radius=radius)
    df_agent_ = df_agent_.merge(closest_obstacle, on="time", how="right")
    # Repulsion away from the obstacle: the unit vector obstacle -> agent, scaled
    # by the distance between the obstacle and the agent's body.
    away = df_agent_[["position_x", "position_y"]].to_numpy(dtype=float) - df_agent_[
        ["closest_obs_x", "closest_obs_y"]
    ].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        away = away / np.linalg.norm(away, axis=1)[:, None]
    amount = force_factor_obstacle * np.exp(
        -df_agent_["closest_obs_dist"].to_numpy(dtype=float) / force_sigma_obstacle
    )
    df_agent_["obstacle_force"] = list(amount[:, None] * away)
    return df_agent_[["obstacle_force", "time"]]


def modulus_obstacle_force(
    df_agent: pd.DataFrame,
    df_obstacles: pd.DataFrame,
    force_factor_obstacle: float = config["force_factor_obstacle"],
    force_sigma_obstacle: float = config["force_sigma_obstacle"],
    radius: float = config["robot_radius"],
):
    """Compute the obstacle force norm for an agent

    :param df_agent: DataFrame with the agent data
    :param df_obstacles: DataFrame with the obstacles data
    :param force_factor_obstacle: Obstacle force factor
    :param force_sigma_obstacle: Obstacle force sigma
    :param radius: Radius of the agent's body, subtracted from the obstacle distance
    :return: DataFrame with the obstacle force norm for the agent
    """

    df_agent_ = df_agent.copy()
    obstacle_force = compute_obstacle_force(
        df_agent_, df_obstacles, force_factor_obstacle, force_sigma_obstacle, radius
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
    """Compute the social force exerted on agent ``_1`` by agent ``_2``

    Moussaid et al. (2009), as in lightsfm and hunav_evaluator's
    ``computeSocialForceTwoAgents``: the ``_1`` columns are the agent receiving
    the force, the ``_2`` columns the agent provoking it. Velocities must be in
    the world frame.

    :param df_interaction: DataFrame with the interaction data
    :param lambda_: Lambda parameter
    :param gamma: Gamma parameter
    :param n: N parameter
    :param n_prime: N prime parameter
    :param force_factor_social: Social force factor
    :return: DataFrame with the social force on agent ``_1``
    """

    if df_interaction.empty:
        return pd.Series(
            {
                "social_force": np.array([np.nan, np.nan]),
                "time": np.nan,
            }
        )
    df_interaction.reset_index(drop=True, inplace=True)

    position_1 = df_interaction[["position_x_1", "position_y_1"]].to_numpy(dtype=float)
    position_2 = df_interaction[["position_x_2", "position_y_2"]].to_numpy(dtype=float)
    velocity_1 = df_interaction[["velocity_x_1", "velocity_y_1"]].to_numpy(dtype=float)
    velocity_2 = df_interaction[["velocity_x_2", "velocity_y_2"]].to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # receiver -> source
        diff = position_2 - position_1
        diff_norm = np.linalg.norm(diff, axis=1)
        diff_direction = diff / diff_norm[:, None]

        interaction_vector = lambda_ * (velocity_1 - velocity_2) + diff_direction
        interaction_length = np.linalg.norm(interaction_vector, axis=1)
        interaction_direction = interaction_vector / interaction_length[:, None]
        theta = normalize_angle(
            np.arctan2(diff_direction[:, 1], diff_direction[:, 0])
            - np.arctan2(interaction_direction[:, 1], interaction_direction[:, 0])
        )

        B = gamma * interaction_length
        force_velocity_amount = -np.exp(-diff_norm / B - (n_prime * B * theta) ** 2)
        force_angle_amount = -np.sign(theta) * np.exp(
            -diff_norm / B - (n * B * theta) ** 2
        )
        interaction_direction_left_normal = np.column_stack(
            [-interaction_direction[:, 1], interaction_direction[:, 0]]
        )
        social_force = force_factor_social * (
            force_velocity_amount[:, None] * interaction_direction
            + force_angle_amount[:, None] * interaction_direction_left_normal
        )
    # Coincident agents or a vanishing interaction vector leave the direction
    # undefined: no force there (missing inputs stay missing).
    inputs_known = np.isfinite(
        np.hstack([position_1, position_2, velocity_1, velocity_2])
    ).all(axis=1)
    social_force[inputs_known & ~np.isfinite(social_force).all(axis=1)] = 0.0
    df_interaction["social_force"] = list(social_force)
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
        # the other agent receives the force (``_1``), the agent examined provokes it
        df_interaction = combine_df_robot_and_agent(agent, df_agent)
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
        # the other agent receives the force (``_1``), the agent examined provokes it
        df_interaction = combine_df_robot_and_agent(agent, df_agent)
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
    robot_radius: float = config["robot_radius"],
) -> pd.DataFrame:
    """
    It computes the social work of the robot in the environment plus the social work provoked by the robot in the other agents

    :param df_robot: DataFrame with the robot data (world-frame velocity)
    :param df_agents_list: List of DataFrames with the other agents data
    :param df_obstacles: DataFrame with the obstacles data
    :param robot_radius: Radius of the robot
    :return: DataFrame with the social work of the robot in the environment plus the social work provoked by the robot in the other agents
    """

    df_robot_ = df_robot.copy()
    df_agents_list_ = df_agents_list.copy()
    social_force = compute_social_force(df_robot, df_agents_list_)
    df_robot_ = df_robot_.merge(social_force, on="time", how="right")
    obstacle_force_norm = modulus_obstacle_force(
        df_robot, df_obstacles, radius=robot_radius
    )
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
