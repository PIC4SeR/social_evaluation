"""Fleet-level metrics for formation / perturbation experiments.

These metrics consume the per-robot dataframes produced by bag extraction when
the experiment config declares a ``robots: {name: odometry_topic}`` mapping
(passed to every metric as the ``df_robots`` keyword). Three families:

* **Centroid drift** -- distance of the fleet's live centroid from a hold
  point (``hold_point`` from the metrics config, defaulting to the centroid at
  the first timestamp), the direct offline counterpart of the perturbation
  task's drift monitor.
* **Social potential** -- the same asymmetric Gaussian potential
  ``J = A_soc * exp(-0.5 r' Sigma^-1 r)`` that drives ``swarm_control``
  (`controller.py`), summed over robots x humans at each timestep. Parameter
  defaults mirror ``swarm_control.controller.SocialCost``.
* **Fleet aggregations** -- existing single-robot metrics evaluated per robot
  and reduced over the fleet (e.g. min distance to any human by any robot).

Each metric returns ``(scalar, time-series dataframe)`` like the rest of the
metric library, so the series are also reusable by the ``formation_timeseries``
exporter. On single-robot data (no ``df_robots``) they return NaN.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from social_evaluation.common import combine_df_robot_agents
from social_evaluation.metrics_evaluation.metrics import (
    avg_closest_person,
    minimum_distance_to_people,
)
from social_evaluation.metrics_evaluation.metrics_utils import metric


def merge_fleet_positions(df_robots: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge every robot's (x, y) onto one time base and interpolate.

    :param df_robots: Mapping of robot name to its odometry dataframe
    :return: Dataframe with ``time`` plus ``position_x_<name>``/``position_y_<name>``
    """
    merged: Optional[pd.DataFrame] = None
    for name, df in df_robots.items():
        if df.empty:
            continue
        cols = df[["time", "position_x", "position_y"]].rename(
            columns={
                "position_x": f"position_x_{name}",
                "position_y": f"position_y_{name}",
            }
        )
        cols = cols.astype(float)
        merged = cols if merged is None else merged.merge(cols, on="time", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    position_cols = [c for c in merged.columns if c != "time"]
    merged[position_cols] = merged[position_cols].interpolate(
        limit_direction="both"
    )
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def centroid_drift_series(
    df_robots: Dict[str, pd.DataFrame],
    hold_point: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """Fleet centroid and its distance from the hold point over time.

    :param df_robots: Mapping of robot name to its odometry dataframe
    :param hold_point: (x, y) reference; defaults to the first centroid sample
    :return: Dataframe with ``time``, ``centroid_x``, ``centroid_y``, ``drift``
    """
    merged = merge_fleet_positions(df_robots)
    if merged.empty:
        return pd.DataFrame()
    x_cols = [c for c in merged.columns if c.startswith("position_x_")]
    y_cols = [c for c in merged.columns if c.startswith("position_y_")]
    series = merged[["time"]].copy()
    series["centroid_x"] = merged[x_cols].mean(axis=1)
    series["centroid_y"] = merged[y_cols].mean(axis=1)
    if hold_point is None:
        hold_point = (series["centroid_x"].iloc[0], series["centroid_y"].iloc[0])
    series["drift"] = np.hypot(
        series["centroid_x"] - hold_point[0], series["centroid_y"] - hold_point[1]
    )
    return series


def social_potential_series(
    df_robots: Dict[str, pd.DataFrame],
    df_agents_list: List[pd.DataFrame],
    A_soc: float = 2.0,
    sigma_front0: float = 1.0,
    k_v: float = 0.5,
    sigma_rear: float = 0.75,
    sigma_left: float = 0.75,
    sigma_right: float = 0.75,
    **_,
) -> pd.DataFrame:
    """Total asymmetric Gaussian social potential over time.

    Mirrors the cost whose anti-gradient drives ``swarm_control`` (same
    parameters and defaults as its ``SocialCost``): for each robot/human pair
    the human-frame offset picks a frontal variance that grows with the human's
    speed and side/rear variances otherwise. The total is the sum over all
    pairs; per-robot columns ``potential_<name>`` are kept for inspection.

    :param df_robots: Mapping of robot name to its odometry dataframe
    :param df_agents_list: List of dataframes with the agents' data
    :return: Dataframe with ``time``, ``potential_<name>`` ... and ``potential``
    """
    if not df_agents_list:
        return pd.DataFrame()

    per_robot: Optional[pd.DataFrame] = None
    for name, df_robot in df_robots.items():
        if df_robot.empty:
            continue
        combined = combine_df_robot_agents(df_robot, df_agents_list)
        potential = np.zeros(len(combined))
        for i in range(len(df_agents_list)):
            rx = combined["position_x"] - combined[f"position_x_agent{i}"]
            ry = combined["position_y"] - combined[f"position_y_agent{i}"]
            theta = combined[f"yaw_agent{i}"]
            speed = np.hypot(
                combined[f"velocity_x_agent{i}"], combined[f"velocity_y_agent{i}"]
            )
            # Human-frame offset (rotation by -theta).
            r_front = np.cos(theta) * rx + np.sin(theta) * ry
            r_side = -np.sin(theta) * rx + np.cos(theta) * ry
            sigma_front = sigma_front0 * (1.0 + k_v * speed)
            sigma_x = np.where(r_front >= 0.0, sigma_front, sigma_rear)
            sigma_y = np.where(r_side >= 0.0, sigma_left, sigma_right)
            exponent = 0.5 * ((r_front / sigma_x) ** 2 + (r_side / sigma_y) ** 2)
            potential += A_soc * np.exp(-exponent)
        robot_series = pd.DataFrame(
            {"time": combined["time"].astype(float), f"potential_{name}": potential}
        )
        per_robot = (
            robot_series
            if per_robot is None
            else per_robot.merge(robot_series, on="time", how="outer")
        )
    if per_robot is None:
        return pd.DataFrame()
    per_robot.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    potential_cols = [c for c in per_robot.columns if c != "time"]
    per_robot[potential_cols] = per_robot[potential_cols].interpolate(
        limit_direction="both"
    )
    per_robot.dropna(inplace=True)
    per_robot["potential"] = per_robot[potential_cols].sum(axis=1)
    per_robot.reset_index(drop=True, inplace=True)
    return per_robot


def _no_fleet_data(metric_name: str) -> Tuple[float, pd.DataFrame]:
    print(f"{metric_name}: no fleet data ('robots' missing/empty), returning NaN")
    return np.nan, pd.DataFrame()


@metric("centroid_max_drift")
def centroid_max_drift(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    hold_point: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Maximum distance of the fleet centroid from the hold point."""
    if not df_robots:
        return _no_fleet_data("centroid_max_drift")
    series = centroid_drift_series(df_robots, hold_point)
    if series.empty:
        return _no_fleet_data("centroid_max_drift")
    return float(series["drift"].max()), series


@metric("centroid_mean_drift")
def centroid_mean_drift(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    hold_point: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Mean distance of the fleet centroid from the hold point."""
    if not df_robots:
        return _no_fleet_data("centroid_mean_drift")
    series = centroid_drift_series(df_robots, hold_point)
    if series.empty:
        return _no_fleet_data("centroid_mean_drift")
    return float(series["drift"].mean()), series


@metric("fleet_social_potential_mean")
def fleet_social_potential_mean(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Time-average of the total social potential over the fleet."""
    if not df_robots:
        return _no_fleet_data("fleet_social_potential_mean")
    series = social_potential_series(df_robots, df_agents_list, **kwargs)
    if series.empty:
        return _no_fleet_data("fleet_social_potential_mean")
    return float(series["potential"].mean()), series


@metric("fleet_social_potential_max")
def fleet_social_potential_max(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Peak of the total social potential over the fleet."""
    if not df_robots:
        return _no_fleet_data("fleet_social_potential_max")
    series = social_potential_series(df_robots, df_agents_list, **kwargs)
    if series.empty:
        return _no_fleet_data("fleet_social_potential_max")
    return float(series["potential"].max()), series


def _aggregate_over_robots(
    single_robot_metric,
    reduce_fn,
    df_agents_list: List[pd.DataFrame],
    df_robots: Dict[str, pd.DataFrame],
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Run a single-robot metric per robot and reduce the scalars over the fleet."""
    per_robot = {}
    for name, df_robot in df_robots.items():
        if df_robot.empty:
            continue
        result = single_robot_metric(df_agents_list, df_robot, **kwargs)
        per_robot[name] = result[0] if isinstance(result, tuple) else result
    if not per_robot:
        return np.nan, pd.DataFrame()
    return float(reduce_fn(list(per_robot.values()))), pd.DataFrame(
        [per_robot]
    )


@metric("fleet_min_distance_to_people")
def fleet_min_distance_to_people(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Minimum over the fleet of each robot's minimum distance to any human."""
    if not df_robots:
        return _no_fleet_data("fleet_min_distance_to_people")
    return _aggregate_over_robots(
        minimum_distance_to_people, np.min, df_agents_list, df_robots, **kwargs
    )


@metric("fleet_avg_distance_to_closest_person")
def fleet_avg_distance_to_closest_person(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Fleet mean of each robot's average distance to its closest human."""
    if not df_robots:
        return _no_fleet_data("fleet_avg_distance_to_closest_person")
    return _aggregate_over_robots(
        avg_closest_person, np.mean, df_agents_list, df_robots, **kwargs
    )
