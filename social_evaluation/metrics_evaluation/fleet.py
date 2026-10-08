"""Fleet-level metrics for formation / perturbation experiments.

These metrics consume the per-robot dataframes produced by bag extraction when
the experiment config declares a ``robots: {name: odometry_topic}`` mapping
(passed to every metric as the ``df_robots`` keyword). Three families:

* **Centroid drift** -- distance of the fleet's live centroid from a hold
  point (``hold_point`` from the metrics config, defaulting to the centroid at
  the first timestamp). Descriptive, not a verdict: the perturbation task lets
  the centroid settle away from the hold point, so its stability is scored by
  ``centroid_settled_motion`` (did it stop moving?) and
  ``formation_spread_growth`` (did the robots stay together?) instead.
* **Social potential** -- the same asymmetric Gaussian potential
  ``J = A_soc * exp(-0.5 r' Sigma^-1 r)`` that drives ``swarm_control``
  (`controller.py`), summed over robots x humans at each timestep. Parameter
  defaults mirror ``swarm_control.controller.SocialCost``.
* **Social memory** -- ``swarm_control``'s internal ``z`` state as recorded on
  ``/<robot>/social_memory`` (``run_data["memory"]``), merged over the fleet.
  Read, not recomputed; empty under any other controller.
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
from social_evaluation.metrics_evaluation.metrics_utils import metric, radius_for


def _one_row_per_time(df: pd.DataFrame) -> pd.DataFrame:
    """The last row at each ``time``, so that merging on ``time`` stays 1:1.

    Bag stamps come off the simulator clock, which ticks at 10 Hz, so a 20 Hz
    topic carries 2-6 messages per stamp. An outer merge on a non-unique key
    is a per-stamp cartesian product, compounded over every series merged:
    one 40 s run reached 46 GB and was killed (2026-09-10). The last message
    is the state at the end of the stamp.
    """
    df = df.astype({"time": float})
    return df.drop_duplicates("time", keep="last").reset_index(drop=True)


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
        cols = _one_row_per_time(cols.astype(float))
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


def formation_spread_series(df_robots: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """RMS distance of the robots from their own centroid over time.

    Size of the formation regardless of where it sits: constant while the shape
    holds, growing without bound if the fleet falls apart.

    :param df_robots: Mapping of robot name to its odometry dataframe
    :return: Dataframe with ``time`` and ``spread``
    """
    merged = merge_fleet_positions(df_robots)
    if merged.empty:
        return pd.DataFrame()
    xs = merged[[c for c in merged.columns if c.startswith("position_x_")]]
    ys = merged[[c for c in merged.columns if c.startswith("position_y_")]]
    dx = xs.sub(xs.mean(axis=1), axis=0)
    dy = ys.sub(ys.mean(axis=1), axis=0)
    series = merged[["time"]].copy()
    series["spread"] = np.sqrt((dx**2).mean(axis=1) + (dy**2).mean(axis=1))
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
        # One row per stamp here too, or the per-robot merges below multiply
        # (and the mean of the total is weighted by how many messages a stamp
        # happened to carry -- fleet_social_potential_mean changed slightly
        # when this was added, 2026-09-17).
        combined = _one_row_per_time(
            combine_df_robot_agents(_one_row_per_time(df_robot), df_agents_list)
        )
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


def social_memory_series(df_memory: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Each worker's social-memory state ``z`` over time, as recorded.

    ``z`` is internal ``swarm_control`` worker state (``z <- z(1 - k_z) +
    u_social``) that each worker publishes on ``/<robot>/social_memory``; bag
    extraction hands it over as ``run_data["memory"]``. Empty for runs under any
    other controller, and for bags recorded before the topic existed.

    :param df_memory: Mapping of robot name to its social-memory dataframe
        (``time``, ``x``, ``y``), as produced by bag extraction
    :return: Dataframe with ``time``, ``z_x_<name>``/``z_y_<name>``/
        ``z_norm_<name>`` per robot, and the fleet means ``z_x``, ``z_y``,
        ``z_norm``
    """
    if not df_memory:
        return pd.DataFrame()
    merged: Optional[pd.DataFrame] = None
    for name, df_z in df_memory.items():
        if df_z.empty:
            continue
        cols = _one_row_per_time(df_z[["time", "x", "y"]].astype(float)).rename(
            columns={"x": f"z_x_{name}", "y": f"z_y_{name}"}
        )
        merged = cols if merged is None else merged.merge(cols, on="time", how="outer")
    if merged is None:
        return pd.DataFrame()
    merged.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    value_cols = [c for c in merged.columns if c != "time"]
    merged[value_cols] = merged[value_cols].interpolate(limit_direction="both")
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    for name in [c.removeprefix("z_x_") for c in merged.columns if c.startswith("z_x_")]:
        merged[f"z_norm_{name}"] = np.hypot(merged[f"z_x_{name}"], merged[f"z_y_{name}"])
    for axis in ("x", "y", "norm"):
        merged[f"z_{axis}"] = merged[
            [c for c in merged.columns if c.startswith(f"z_{axis}_")]
        ].mean(axis=1)
    return merged


def command_input_series(
    df_command: Dict[str, pd.DataFrame],
    df_ucommon: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """The commanded input over time: the total ``u`` and the common input ``u_c``.

    Two different things that are easy to conflate:

    * ``u`` (``/<robot>/cmd_u``, ``run_data["command"]``) is what each worker
      asked of its base -- formation + common input + social + memory feedback.
      It differs per robot, because the formation and social terms do.
    * ``u_c`` (``/<robot>/cmd_u_c``, ``run_data["u_common"]``) is the centroid
      law's own output. It is IDENTICAL on every worker by construction, so the
      per-robot columns should lie on top of each other; a spread between them
      means the workers disagreed about the fleet state, which is a bug worth
      seeing. Zero under ``centroid_mode: feedback``, where the target is a
      potential term inside the law rather than a common input.

    Reading ``u_c`` is how you tell what the centroid mode actually did -- held
    ``uc_speed``, tapered, stopped -- which none of the other series show.

    :param df_command: Robot name -> ``cmd_u`` dataframe (``time``, ``x``, ``y``)
    :param df_ucommon: Robot name -> ``cmd_u_c`` dataframe; absent for bags
        recorded before the topic existed, in which case only ``u`` is returned
    :return: Dataframe with ``time``, per-robot ``u_norm_<name>`` and
        ``uc_norm_<name>``, and the fleet means ``u_norm``, ``uc_norm``
    """
    frames: List[pd.DataFrame] = []
    for prefix, source in (("u", df_command), ("uc", df_ucommon or {})):
        for name, df_vec in (source or {}).items():
            if df_vec.empty:
                continue
            frames.append(
                _one_row_per_time(df_vec[["time", "x", "y"]].astype(float)).rename(
                    columns={"x": f"{prefix}_x_{name}", "y": f"{prefix}_y_{name}"}
                )
            )
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")
    merged.sort_values("time", kind="stable", inplace=True, ignore_index=True)
    value_cols = [c for c in merged.columns if c != "time"]
    merged[value_cols] = merged[value_cols].interpolate(limit_direction="both")
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    for prefix in ("u", "uc"):
        names = [
            c.removeprefix(f"{prefix}_x_")
            for c in merged.columns
            if c.startswith(f"{prefix}_x_")
        ]
        for name in names:
            merged[f"{prefix}_norm_{name}"] = np.hypot(
                merged[f"{prefix}_x_{name}"], merged[f"{prefix}_y_{name}"]
            )
        if names:
            merged[f"{prefix}_norm"] = merged[
                [f"{prefix}_norm_{n}" for n in names]
            ].mean(axis=1)
    return merged


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


@metric("centroid_settled_motion")
def centroid_settled_motion(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    hold_point: Optional[Tuple[float, float]] = None,
    settle_window: float = 10.0,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Peak excursion of the centroid from its own mean over the last window.

    The convergence half of the stability check, and deliberately pose-free: the
    perturbation is allowed to leave the fleet parked somewhere other than where
    it started, so what is scored is that the centroid *stopped moving*, not
    where it stopped. Near zero => settled; large => still being pushed around
    at the end of the episode.
    """
    if not df_robots:
        return _no_fleet_data("centroid_settled_motion")
    series = centroid_drift_series(df_robots, hold_point)
    if series.empty:
        return _no_fleet_data("centroid_settled_motion")
    tail = series[series["time"] >= series["time"].iloc[-1] - float(settle_window)]
    excursion = np.hypot(
        tail["centroid_x"] - tail["centroid_x"].mean(),
        tail["centroid_y"] - tail["centroid_y"].mean(),
    )
    return float(excursion.max()), series


@metric("formation_spread_growth")
def formation_spread_growth(
    df_agents_list: List[pd.DataFrame],
    df_robot: pd.DataFrame,
    *args,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
    **kwargs,
) -> Tuple[float, pd.DataFrame]:
    """Peak formation size relative to its starting size (1.0 = shape held).

    The divergence half of the stability check: the centroid can settle while
    the robots themselves scatter, and this catches that.
    """
    if not df_robots:
        return _no_fleet_data("formation_spread_growth")
    series = formation_spread_series(df_robots)
    if series.empty or series["spread"].iloc[0] <= 0.0:
        return _no_fleet_data("formation_spread_growth")
    return float(series["spread"].max() / series["spread"].iloc[0]), series


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
    """Run a single-robot metric per robot and reduce the scalars over the fleet.

    Each robot is measured with its own radius (``radius_for``)."""
    per_robot = {}
    for name, df_robot in df_robots.items():
        if df_robot.empty:
            continue
        options = {**kwargs, "robot_radius": radius_for(name, kwargs)}
        result = single_robot_metric(df_agents_list, df_robot, **options)
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
