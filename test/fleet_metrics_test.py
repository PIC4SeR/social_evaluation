import numpy as np
import pandas as pd
import pytest

from social_evaluation.metrics_evaluation import metrics_dict
from social_evaluation.metrics_evaluation.fleet import (
    centroid_drift_series,
    social_memory_series,
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


def _memory_df(xs, ys, times=None):
    times = times if times is not None else np.arange(len(xs), dtype=float)
    return pd.DataFrame(
        {
            "time": times,
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "z": np.zeros(len(xs)),
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


def test_settled_motion_accepts_a_centroid_parked_somewhere_new():
    # Pushed off the hold point and left there, but steady once the humans are
    # gone: that is convergence. Drift still reads 5 m, which is why it is not
    # the verdict.
    xs = [0.0, 2.0, 5.0, 5.0, 5.0, 5.0]
    df_robots = {"r1": _robot_df(xs, [0.0] * len(xs))}
    peak, _ = metrics_dict["centroid_max_drift"](
        [], pd.DataFrame(), df_robots=df_robots, hold_point=(0.0, 0.0)
    )
    settled, _ = metrics_dict["centroid_settled_motion"](
        [], pd.DataFrame(), df_robots=df_robots, settle_window=2.0
    )
    assert peak == pytest.approx(5.0)
    assert settled == pytest.approx(0.0)


def test_settled_motion_flags_a_centroid_still_moving():
    # Same run, but the centroid never stops: the tail still swings +-0.5 m
    # about its own mean of 4.5.
    xs = [0.0, 2.0, 5.0, 4.0, 5.0, 4.0]
    df_robots = {"r1": _robot_df(xs, [0.0] * len(xs))}
    settled, _ = metrics_dict["centroid_settled_motion"](
        [], pd.DataFrame(), df_robots=df_robots, settle_window=3.0
    )
    assert settled == pytest.approx(0.5)


def test_spread_growth_flags_a_diverging_formation():
    # Two robots holding 2 m apart, then drifting to 6 m apart: 3x the size.
    held = {
        "r1": _robot_df([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
        "r2": _robot_df([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]),
    }
    diverging = {
        "r1": _robot_df([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]),
        "r2": _robot_df([-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]),
    }
    assert metrics_dict["formation_spread_growth"](
        [], pd.DataFrame(), df_robots=held
    )[0] == pytest.approx(1.0)
    assert metrics_dict["formation_spread_growth"](
        [], pd.DataFrame(), df_robots=diverging
    )[0] == pytest.approx(3.0)


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


def test_social_memory_merges_the_recorded_workers():
    # Two workers' recorded z on different time bases: interpolated onto one,
    # per-robot norms kept, fleet means alongside.
    df_memory = {
        "r1": _memory_df([0.0, 3.0], [0.0, 4.0], times=[0.0, 2.0]),
        "r2": _memory_df([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
    }
    series = social_memory_series(df_memory)
    assert series["z_norm_r1"].iloc[-1] == pytest.approx(5.0)
    # r1 is interpolated to (1.5, 2.0) at t=1, so the mean x is (1.5 + 1)/2.
    at_one = series[series["time"] == 1.0].iloc[0]
    assert at_one["z_x"] == pytest.approx(1.25)
    assert at_one["z_y"] == pytest.approx(1.0)


def test_social_memory_empty_without_a_swarm_controller():
    # Any controller other than swarm_control never publishes the topic.
    assert social_memory_series({}).empty


def _stamped_like_a_bag(df, copies=4, columns=("position_x", "x")):
    """Every row repeated at the same stamp with a drifting value -- what a
    20-40 Hz topic looks like on the simulator's 10 Hz clock."""
    rows = []
    for _, row in df.iterrows():
        for c in range(copies):
            row = row.copy()
            for column in columns:
                if column in row:
                    row[column] = row[column] + 0.01
            rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def test_duplicate_stamps_do_not_multiply_the_merged_series():
    """Outer merges on a non-unique time are cartesian products: three robots
    at 4 rows per stamp used to give 64 rows per stamp per series, compounded
    again when the series were merged (46 GB on one 40 s run)."""
    from social_evaluation.formation_timeseries_ import run_timeseries
    from social_evaluation.metrics_evaluation.fleet import (
        command_input_series,
        merge_fleet_positions,
    )

    stamps = 30
    robots = {
        name: _stamped_like_a_bag(_robot_df(np.linspace(0.0, 1.0, stamps) + k,
                                            np.zeros(stamps)))
        for k, name in enumerate(("r1", "r2", "r3"))
    }
    merged = merge_fleet_positions(robots)
    assert len(merged) == stamps
    # The LAST message at a stamp is kept.
    assert merged["position_x_r1"].iloc[0] == pytest.approx(0.04)

    vectors = {name: _stamped_like_a_bag(_memory_df(np.ones(stamps), np.zeros(stamps)))
               for name in robots}
    assert len(social_memory_series(vectors)) == stamps
    assert len(command_input_series(vectors, vectors)) == stamps
    agents = [_agent_df(np.linspace(3.0, 4.0, stamps), np.zeros(stamps))]
    assert len(social_potential_series(robots, agents)) == stamps

    run = {"robots": robots, "agents": agents, "memory": vectors,
           "command": vectors, "u_common": vectors}
    assert len(run_timeseries(run)) == stamps
