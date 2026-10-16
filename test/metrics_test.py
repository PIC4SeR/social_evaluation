import pandas as pd
import pytest


@pytest.fixture
def quaternion():
    return [0, 0, 0, 1]


@pytest.fixture
def df_robot():
    """Fixture for the robot dataframe
    it has the following columns:
    - time
    - position_x
    - position_y
    - position_z
    - qx
    - qy
    - qz
    - qw
    - roll
    - pitch
    - yaw
    - velocity_x
    - velocity_y
    - velocity_z
    - angular_velocity_x
    - angular_velocity_y
    - angular_velocity_z
    """

    return pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "position_x": [0.0, 1.0, 2.0, 3.0],
            "position_y": [0.0, 1.0, 2.0, 3.0],
            "position_z": [0.0, 1.0, 2.0, 3.0],
            "qx": [0.0, 1.0, 2.0, 3.0],
            "qy": [0.0, 1.0, 2.0, 3.0],
            "qz": [0.0, 1.0, 2.0, 3.0],
            "qw": [0.0, 1.0, 2.0, 3.0],
            "roll": [0.0, 1.0, 2.0, 3.0],
            "pitch": [0.0, 1.0, 2.0, 3.0],
            "yaw": [0.0, 1.0, 2.0, 3.0],
            "velocity_x": [0.0, 1.0, 2.0, 3.0],
            "velocity_y": [0.0, 1.0, 2.0, 3.0],
            "velocity_z": [0.0, 1.0, 2.0, 3.0],
            "angular_velocity_x": [0.0, 1.0, 2.0, 3.0],
            "angular_velocity_y": [0.0, 1.0, 2.0, 3.0],
            "angular_velocity_z": [0.0, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def df_human():
    """Fixture for the human dataframe
    it has the following columns:
    - time
    - position_x
    - position_y
    - yaw
    - velocity_x
    - velocity_y
    - angular_velocity_z
    """
    return pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0, 3.0],
            "position_x": [0.0, 1.0, 2.0, 3.0],
            "position_y": [0.0, 1.0, 2.0, 3.0],
            "yaw": [0.0, 1.0, 2.0, 3.0],
            "velocity_x": [0.0, 1.0, 2.0, 3.0],
            "velocity_y": [0.0, 1.0, 2.0, 3.0],
            "angular_velocity_z": [0.0, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def df_obstacles():
    """Fixture for the obstacles dataframe
    it has the following columns:
    - x
    - y
    """
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def df_agent_list():
    """Fixture for the agents list
    it is a list of dataframes containing df_human
    """
    return [df_human, df_human]


def test_obstacle_force_is_zero_without_obstacles(df_robot):
    """An open-field map has no static obstacles: an empty obstacle frame gives zero obstacle
    force (it used to raise KeyError on the missing x / y columns, e.g. in social_work)."""
    import numpy as np
    from social_evaluation.metrics_evaluation.sfm import modulus_obstacle_force

    for empty in (pd.DataFrame(), pd.DataFrame({"x": [], "y": []})):
        out = modulus_obstacle_force(df_robot, empty)
        assert len(out) == len(df_robot)
        assert np.allclose(out["obstacle_force_norm"], 0.0)
