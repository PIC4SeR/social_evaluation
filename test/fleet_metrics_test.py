import numpy as np
import pandas as pd
import pytest

from social_evaluation.metrics_evaluation import metrics_dict
from social_evaluation.metrics_evaluation.fleet import (
    centroid_drift_series,
    social_potential_series,
)


def _robot_df(xs, ys, times=None):
    times = times if times is not None else np.arange(len(xs), dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "position_x": np.asarray(xs, dtype=float),
            "position_y": np.asarray(ys, dtype=float),
        }
    )


def _agent_df(xs, ys, yaw=0.0, vx=0.0, vy=0.0, times=None):
    times = times if times is not None else np.arange(len(xs), dtype=float)
    n = len(xs)
    return pd.DataFrame(
        {
            "time": times,
            "position_x": np.asarray(xs, dtype=float),
            "position_y": np.asarray(ys, dtype=float),
            "yaw": np.full(n, yaw),
            "velocity_x": np.full(n, vx),
            "velocity_y": np.full(n, vy),
            "angular_velocity_z": np.zeros(n),
            "behavior": ["1"] * n,
            "group_id": ["-1"] * n,
            "id": ["1"] * n,
        }
    )


def test_centroid_drift_symmetric_fleet_stays_at_zero():
    # Two robots moving symmetrically about (0, 0): centroid never moves.
    df_robots = {
        "r1": _robot_df([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        "r2": _robot_df([-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]),
    }
    series = centroid_drift_series(df_robots)
    assert series["drift"].max() == pytest.approx(0.0)


def test_centroid_drift_against_explicit_hold_point():
    # Both robots shift +1 m in x: centroid drift from the start point is 1.
    df_robots = {
        "r1": _robot_df([1.0, 2.0], [1.0, 1.0]),
        "r2": _robot_df([-1.0, 0.0], [-1.0, -1.0]),
    }
    scalar, series = metrics_dict["centroid_max_drift"](
        [], pd.DataFrame(), df_robots=df_robots, hold_point=(0.0, 0.0)
    )
    assert scalar == pytest.approx(1.0)
    assert list(series.columns[:3]) == ["time", "centroid_x", "centroid_y"]


def test_social_potential_matches_closed_form():
    # One static robot 1 m in front of a static human at the origin:
    # J = A_soc * exp(-0.5 * (1 / sigma_front0)^2).
    df_robots = {"r1": _robot_df([1.0, 1.0], [0.0, 0.0])}
    agents = [_agent_df([0.0, 0.0], [0.0, 0.0])]
    series = social_potential_series(
        df_robots, agents, A_soc=2.0, sigma_front0=1.0, k_v=0.5
    )
    assert series["potential"].iloc[0] == pytest.approx(2.0 * np.exp(-0.5))


def test_social_potential_uses_rear_sigma_behind_the_human():
    # Robot 1 m behind the human (facing +x): rear variance applies.
    df_robots = {"r1": _robot_df([-1.0, -1.0], [0.0, 0.0])}
    agents = [_agent_df([0.0, 0.0], [0.0, 0.0])]
    series = social_potential_series(df_robots, agents, A_soc=2.0, sigma_rear=0.75)
    assert series["potential"].iloc[0] == pytest.approx(
        2.0 * np.exp(-0.5 / 0.75**2)
    )


def test_fleet_min_distance_takes_the_closest_robot():
    # r1 passes 1 m from the agent, r2 stays 5 m away; radii shrink the gap.
    df_robots = {
        "r1": _robot_df([1.0, 1.0], [0.0, 0.0]),
        "r2": _robot_df([5.0, 5.0], [0.0, 0.0]),
    }
    agents = [_agent_df([0.0, 0.0], [0.0, 0.0])]
    scalar, _ = metrics_dict["fleet_min_distance_to_people"](
        agents, pd.DataFrame(), df_robots=df_robots, robot_radius=0.3, agent_radius=0.2
    )
    assert scalar == pytest.approx(0.5)


def test_fleet_metrics_return_nan_without_fleet_data():
    scalar, series = metrics_dict["centroid_max_drift"](
        [], pd.DataFrame(), df_robots=None
    )
    assert np.isnan(scalar) and series.empty
