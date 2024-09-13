# %%
import pandas as pd
from pathlib import Path
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Union, List, Tuple
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.collections as mcoll

# %%


def plot_experiment_aggregate(data: dict, exp: List[str], ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    pass


def plot_experiment_single(data: dict, exp: str, ax: plt.Axes):
    """Plot the experiment
    :param data: Dictionary with the data
    :param exp: Experiment to plot
    """
    df_odometry = pd.DataFrame(data[exp]["odometry"])
    df_people = pd.DataFrame(data[exp]["people"])
    df_agents_list = data[exp]["agents"]
    df_obstacles = data[exp]["obstacles"]
    transforms = data[exp]["transforms"]
    goal = data[exp]["goal"]
    map = data[exp]["map"]

    # Plot the data
    plot_data(
        df_odometry, df_people, df_agents_list, df_obstacles, transforms, goal, map, ax
    )


def plot_data(
    df_odometry: pd.DataFrame,
    df_people: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
    transforms: pd.DataFrame,
    goal: Tuple[float, float],
    map: np.ndarray,
    ax: plt.Axes,
) -> None:
    """Plot the data
    :param df_odometry: Dataframe with the odometry
    :param df_people: Dataframe with the people
    :param df_agents_list: List of dataframes with the agents
    :param df_obstacles: Dataframe with the obstacles
    :param transforms: Dataframe with the transforms
    :param goal: Tuple with the goal position
    :param map: Numpy array with the map
    """
    plot_map(map, ax)
    plot_goal(goal, ax, color="red")
    plot_agents(df_agents_list, ax)
    plot_transforms(transforms, ax)


def plot_obstacles(df_obstacles: pd.DataFrame, ax: plt.Axes):
    """Plot the obstacles
    :param df_obstacles: Dataframe with the obstacles
    """
    ax.scatter(df_obstacles["x"], df_obstacles["y"], c="black", s=2)

    pass


def plot_transforms(transforms: pd.DataFrame, ax: plt.Axes):
    """Plot the transforms
    :param transforms: Dataframe with the transforms
    :param ax: Matplotlib axis
    """
    max_velocity = 0
    min_velocity = 0

    for transform in transforms:
        df_transform = transforms[transform].copy().dropna()
        df_transform["velocity"] = np.linalg.norm(
            df_transform[["velocity_x", "velocity_y"]].values, axis=1
        )
        max_velocity = max(max_velocity, df_transform["velocity"].max())
        min_velocity = min(min_velocity, df_transform["velocity"].min())

    norm = mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)

    for transform in transforms:
        df_transform = transforms[transform].copy().dropna()
        df_transform["velocity"] = np.linalg.norm(
            df_transform[["velocity_x", "velocity_y"]].values, axis=1
        )
        df_transform_plot = df_transform.copy().iloc[0::10]
        ax.scatter(
            df_transform_plot["position_x"].values,
            df_transform_plot["position_y"].values,
            c=df_transform_plot["velocity"],
            cmap="hsv",
            norm=norm,
            s=20,
            alpha=0.5,
        )
        df_transform_orientation = df_transform_plot.copy().iloc[0::10]

        for i, row in df_transform_orientation.iterrows():
            ax.add_patch(
                patches.FancyArrow(
                    row["position_x"],
                    row["position_y"],
                    np.cos(row["yaw"]) * max_velocity,
                    np.sin(row["yaw"]) * max_velocity,
                    color="black",
                    head_width=0.1 * max_velocity,
                    head_length=0.1 * max_velocity,
                )
            )


def plot_goal(
    goal: Tuple[float, float], ax: plt.Axes, color: str = "green", radius: float = 0.3
) -> None:
    """Plot the goal
    :param goal: Tuple with the goal position
    """
    goal_circle = patches.Circle((goal[0], goal[1]), radius, fill=False, color=color)
    ax.add_patch(goal_circle)


def plot_map(
    map: np.ndarray,
    ax: plt.Axes,
    resolution: float = 0.05,
    origin: Tuple[float, float] = (-8.41, -12.7),
) -> None:
    """Plot the map
    :param map: Numpy array with the map
    """
    color_map = mcolors.ListedColormap(["white", "black"])
    extent = [
        origin[0],
        origin[0] + map.shape[1] * resolution,
        origin[1],
        origin[1] + map.shape[0] * resolution,
    ]
    # use a resolution of 0.1 and set the origin of the map to the bottom left corner
    ax.imshow(map, cmap=color_map, extent=extent)


def plot_agents(df_agents_list: List[pd.DataFrame], ax: plt.Axes) -> None:
    """Plot the agents
    :param df_agents_list: List of dataframes with the agents
    :param ax: Matplotlib axis
    """

    # find the max/min velocity to normalize the arrow size
    max_velocity = 0
    min_velocity = 0

    for df_agent in df_agents_list:
        df_agent["velocity"] = np.linalg.norm(
            df_agent[["velocity_x", "velocity_y"]].values, axis=1
        )
        max_velocity = max(max_velocity, df_agent["velocity"].max())
        min_velocity = min(min_velocity, df_agent["velocity"].min())
    norm = mcolors.Normalize(vmin=min_velocity, vmax=max_velocity)

    for df_agent in df_agents_list:

        df_agent["velocity"] = np.linalg.norm(
            df_agent[["velocity_x", "velocity_y"]].values, axis=1
        )
        # sample the points to plot (to avoid plotting all the points)
        df_agent_plot = df_agent.copy().iloc[0::10]
        # create a scatter plot
        ax.scatter(
            df_agent_plot["position_x"].values,
            df_agent_plot["position_y"].values,
            c=df_agent_plot["velocity"],
            cmap="viridis",
            norm=norm,
            marker="x",
            s=20,
            alpha=0.5,
        )
        # add a patch for the orientation to a subset of the points
        # df_agent_orientation = df_agent_plot.copy().iloc[0::10]

        # the arrow is drawn from the position to the position + a vector of length 1 in the direction of the yaw
        # for i, row in df_agent_orientation.iterrows():
        #     ax.add_patch(
        #         patches.FancyArrow(
        #             row["position_x"],
        #             row["position_y"],
        #             np.cos(row["yaw"]) * max_velocity,
        #             np.sin(row["yaw"]) * max_velocity,
        #             color="black",
        #             head_width=0.1 * max_velocity,
        #             head_length=0.1 * max_velocity,
        #         )
        #     )


# %%
# Load the data

pickle_file = Path("/home/andrea/workspaces/appl_ws/output.pickle")
with open(pickle_file, "rb") as f:
    data = pickle.load(f)


# %%
# Plot the experiment

exp = list(data.keys())[0]


fig, ax = plt.subplots()
# plot_map(list(data.values())[0]["map"], ax)
# for exp in data.keys():
#     # print(exp.keys())
#     # print(exp)
#     plot_experiment(data, exp, ax)

plot_experiment_single(data, exp, ax)
plt.show()
# %%
# print(data[exp].keys())

# # %%
# print(data[exp]["goal"])
# # %%


# plot_map(data[exp]["map"], ax)

# # ax.set_xlim(-10, 10)
# # ax.set_ylim(-10, 10)
# ax.set_aspect("equal")

# # Plot the goal
# goal = data[exp]["goal"]
# plot_goal(goal, ax, color="red")
# # plot_obstacles(data[exp]["obstacles"], ax)
# # %%
# # Plot the agents
# df_agents_list = data[exp]["agents"]
# plot_agents(df_agents_list, ax)
# plot_transforms(data[exp]["transforms"], ax)
# # create a colorbar
# # sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
# # sm.set_array([])
# # plt.colorbar(sm)


# # %%

plt.show()
