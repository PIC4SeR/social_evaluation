#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Export formation/perturbation time series.

Consumes the extracted/filtered pickles produced by ``bag_extraction`` /
``data_filtering`` for fleet experiments (those with a ``robots`` mapping) and
writes, per run, one CSV with the merged time series plus three PNGs: drift and
social potential, the centroid against its start point, and the social memory.
The memory panels need ``run_data["memory"]``, i.e. a run recorded under
``swarm_control`` with the ``social_memory`` topic in the bag; they are skipped
otherwise. The scalar summaries of the same series live in the metric library
(``centroid_settled_motion``, ``fleet_social_potential_mean``, ...); this tool is
for looking at the curves themselves.

``--goal X Y`` gives the centroid's set point -- a formation task's goal (crossing,
passing) or a perturbation task's hold point: the drift is then the centroid's distance
to it, the CSV gains ``goal_x``/``goal_y``, and the plots draw it (with
``--goal-tolerance``, the success radius). Without it the drift is from the first
sample. ``--control-point-offset L`` measures the centroid of the robots' control points
``q + L (cos yaw, sin yaw)`` (what an off-axle unicycle controller steers; swarm_control's
``unicycle_offset``) instead of their bodies.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import argcomplete, argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from social_evaluation.metrics_evaluation.fleet import (
    centroid_drift_series,
    command_input_series,
    social_memory_series,
    social_potential_series,
)

DRIFT_COLOR = "#2a78d6"
POTENTIAL_COLOR = "#1baf7a"
CENTROID_X_COLOR = "#2a78d6"
CENTROID_Y_COLOR = "#1baf7a"
Z_X_COLOR = "#2a78d6"
Z_Y_COLOR = "#d64545"
TEXT_COLOR = "#0b0b0b"
START_COLOR = "#4a4a4a"
GOAL_COLOR = "#c23b22"


def run_timeseries(
    run_data: Dict[str, Any],
    hold_point: Optional[Tuple[float, float]] = None,
    goal: Optional[Tuple[float, float]] = None,
    control_point_offset: float = 0.0,
) -> pd.DataFrame:
    """Merged drift + potential time series for one run (empty if no fleet data).

    With ``goal`` (the set point) the drift is the distance to it (``hold_point`` is then
    unused); ``control_point_offset`` > 0 moves the centroid to the control points."""
    df_robots = run_data.get("robots") or {}
    if not df_robots:
        return pd.DataFrame()
    agents = run_data.get("agents") or []
    parts = [
        centroid_drift_series(df_robots, goal if goal is not None else hold_point,
                              control_point_offset),
        social_potential_series(df_robots, agents),
        social_memory_series(run_data.get("memory") or {}),
        command_input_series(
            run_data.get("command") or {}, run_data.get("u_common") or {}
        ),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="time", how="outer")
    if len(parts) > 1:
        merged.sort_values("time", kind="stable", inplace=True, ignore_index=True)
        value_cols = [c for c in merged.columns if c != "time"]
        merged[value_cols] = merged[value_cols].interpolate(limit_direction="both")
    merged = merged.copy()
    merged["time"] -= merged["time"].iloc[0]
    if goal is not None:
        merged["goal_x"], merged["goal_y"] = float(goal[0]), float(goal[1])
    return merged


def _finish(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_timeseries(
    series: pd.DataFrame, title: str, output_file: Path,
    goal_tolerance: Optional[float] = None,
) -> None:
    """Stacked panels: centroid drift (or distance to the goal) and total social
    potential over time."""
    to_goal = "goal_x" in series.columns
    panels = [
        ("drift", "Distance to set point (m)" if to_goal else "Centroid drift (m)", DRIFT_COLOR),
        ("potential", "Social potential", POTENTIAL_COLOR),
    ]
    panels = [p for p in panels if p[0] in series.columns]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8, 3 * len(panels)), sharex=True, squeeze=False
    )
    for ax, (column, label, color) in zip(axes[:, 0], panels):
        ax.plot(series["time"], series[column], color=color, linewidth=2)
        if column == "drift" and to_goal and goal_tolerance is not None:
            ax.axhline(goal_tolerance, color=GOAL_COLOR, linestyle="--", linewidth=1.5,
                       label=f"tolerance {goal_tolerance:g} m")
            ax.legend(loc="upper right", frameon=False)
        _finish(ax, label)
        ax.set_ylim(bottom=0)
    axes[-1, 0].set_xlabel("Time (s)", color=TEXT_COLOR)
    axes[0, 0].set_title(title, color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_centroid(series: pd.DataFrame, title: str, output_file: Path) -> None:
    """Centroid x and y against their starting values (the hold point), and
    against the goal when the series has one.

    The drift panel folds both axes into one distance; this splits them, so a
    centroid that settles back on its start line (or on the goal) is visible per axis.
    """
    if "centroid_x" not in series.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, axis, color in ((axes[0], "x", CENTROID_X_COLOR), (axes[1], "y", CENTROID_Y_COLOR)):
        values = series[f"centroid_{axis}"]
        ax.plot(series["time"], values, color=color, linewidth=2)
        ax.axhline(
            values.iloc[0], color=START_COLOR, linestyle="--", linewidth=1.5,
            label=f"start {axis}",
        )
        if f"goal_{axis}" in series.columns:
            ax.axhline(series[f"goal_{axis}"].iloc[0], color=GOAL_COLOR, linestyle=":",
                       linewidth=2, label=f"set point {axis}")
        ax.legend(loc="upper right", frameon=False)
        _finish(ax, f"Centroid {axis} (m)")
    axes[1].set_xlabel("Time (s)", color=TEXT_COLOR)
    axes[0].set_title(f"{title} -- centroid", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_memory(series: pd.DataFrame, title: str, output_file: Path) -> None:
    """Social memory z: fleet-mean components, and |z| per robot."""
    if "z_norm" not in series.columns:
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for axis, color in (("x", Z_X_COLOR), ("y", Z_Y_COLOR)):
        axes[0].plot(
            series["time"], series[f"z_{axis}"], color=color, linewidth=2,
            label=f"mean $z_{axis}$",
        )
    axes[0].axhline(0.0, color=START_COLOR, linestyle="--", linewidth=1)
    axes[0].legend(loc="upper right", frameon=False)
    _finish(axes[0], "z (m)")

    per_robot = [c for c in series.columns if c.startswith("z_norm_")]
    for column in per_robot:
        axes[1].plot(
            series["time"], series[column], linewidth=1, alpha=0.7,
            label=column.removeprefix("z_norm_"),
        )
    axes[1].plot(
        series["time"], series["z_norm"], color="black", linestyle="--", linewidth=2,
        label="fleet mean",
    )
    axes[1].legend(loc="upper right", frameon=False, ncol=2, fontsize="small")
    _finish(axes[1], "|z| (m)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Time (s)", color=TEXT_COLOR)
    axes[0].set_title(f"{title} -- social memory", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_input(series: pd.DataFrame, title: str, output_file: Path) -> None:
    """Commanded input: the common input u_c, and the total command u per robot.

    The top panel is the centroid law's own output. It is identical on every
    worker by construction, so the per-robot traces should coincide -- a visible
    spread means the workers disagreed about the fleet state. Reading it is how
    you tell what the mode did: held uc_speed, tapered, stopped.

    The bottom panel is what each base was actually asked for, which legitimately
    differs per robot because the formation and social terms do. The gap between
    the two is everything the swarm added on top of the drive.
    """
    if "u_norm" not in series.columns:
        return
    has_uc = "uc_norm" in series.columns
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    if has_uc:
        for column in [c for c in series.columns if c.startswith("uc_norm_")]:
            axes[0].plot(
                series["time"], series[column], linewidth=1, alpha=0.7,
                label=column.removeprefix("uc_norm_"),
            )
        axes[0].plot(
            series["time"], series["uc_norm"], color="black", linestyle="--",
            linewidth=2, label="fleet mean",
        )
        axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize="small")
    else:
        axes[0].text(
            0.5, 0.5, "no cmd_u_c in this bag", ha="center", va="center",
            transform=axes[0].transAxes, color=TEXT_COLOR,
        )
    _finish(axes[0], "|u_c| (m/s)")
    axes[0].set_ylim(bottom=0)

    for column in [c for c in series.columns if c.startswith("u_norm_")]:
        axes[1].plot(
            series["time"], series[column], linewidth=1, alpha=0.7,
            label=column.removeprefix("u_norm_"),
        )
    axes[1].plot(
        series["time"], series["u_norm"], color="black", linestyle="--",
        linewidth=2, label="fleet mean",
    )
    axes[1].legend(loc="upper right", frameon=False, ncol=2, fontsize="small")
    _finish(axes[1], "|u| (m/s)")
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Time (s)", color=TEXT_COLOR)
    axes[0].set_title(f"{title} -- commanded input", color=TEXT_COLOR)
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def run_name(run_key: str, with_scenario: bool = False) -> str:
    """Compact run identifier from a bag-path key (episode/algorithm/run dir).

    ``with_scenario`` prepends the scenario directory above the episode, for a
    pickle holding several scenarios, whose names would otherwise collide.
    """
    parts = Path(str(run_key)).parent.parts
    tail = parts[-4:] if with_scenario else parts[-3:]
    return "_".join(tail) if tail else "run"


def run_names(run_keys) -> Dict[str, str]:
    """``run_name`` for every key, with the scenario prefixed only when the
    short names collide -- single-scenario pickles keep their usual names."""
    keys = list(run_keys)
    names = {key: run_name(key) for key in keys}
    if len(set(names.values())) < len(keys):
        names = {key: run_name(key, with_scenario=True) for key in keys}
    return names


def process_pickle(
    pickle_file: Path,
    output_dir: Path,
    hold_point: Optional[Tuple[float, float]] = None,
    goal: Optional[Tuple[float, float]] = None,
    goal_tolerance: Optional[float] = None,
    control_point_offset: float = 0.0,
) -> int:
    """Export CSV + PNG for every fleet run in one pickle. Returns runs exported."""
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
    exported = 0
    names = run_names(data)
    for run_key, run_data in data.items():
        series = run_timeseries(run_data, hold_point, goal, control_point_offset)
        if series.empty:
            continue
        name = names[run_key]
        output_dir.mkdir(parents=True, exist_ok=True)
        series.to_csv(output_dir / f"{name}_timeseries.csv", index=False)
        plot_timeseries(series, name, output_dir / f"{name}_timeseries.png", goal_tolerance)
        plot_centroid(series, name, output_dir / f"{name}_centroid.png")
        plot_memory(series, name, output_dir / f"{name}_memory.png")
        plot_input(series, name, output_dir / f"{name}_input.png")
        exported += 1
    if exported == 0:
        print(f"{pickle_file}: no runs with fleet ('robots') data, nothing exported")
    return exported


def main():
    parser = argparse.ArgumentParser(
        description="Export centroid-drift and social-potential time series "
        "for formation/perturbation fleet experiments"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pickle-file", type=Path, help="Single extracted/filtered pickle"
    )
    input_group.add_argument(
        "--base-path",
        type=Path,
        help="Directory with extracted/filtered pickles (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the CSV/PNG outputs",
    )
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument(
        "--hold-point",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Drift reference point (default: the centroid at the first sample)",
    )
    reference.add_argument(
        "--goal",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="The centroid's set point (a formation task's goal, a perturbation task's "
        "hold point): the drift is the distance to it, and the plots draw it",
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=None,
        metavar="M",
        help="With --goal: the success radius, drawn on the distance panel",
    )
    parser.add_argument(
        "--control-point-offset",
        type=float,
        default=0.0,
        metavar="L",
        help="Measure the centroid of the control points q + L (cos yaw, sin yaw) "
        "(swarm_control: unicycle_offset, 0.10) instead of the robot bodies",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.base_path is not None:
        pickle_files = sorted(
            f
            for f in args.base_path.rglob("*.pickle")
            if not f.name.endswith("_metrics.pickle")
        )
        if not pickle_files:
            parser.error(f"No .pickle files found under {args.base_path}")
    else:
        pickle_files = [args.pickle_file]

    hold_point = tuple(args.hold_point) if args.hold_point else None
    goal = tuple(args.goal) if args.goal else None
    total = 0
    for pickle_file in pickle_files:
        if args.base_path is not None:
            out = args.output_dir / pickle_file.relative_to(args.base_path).parent
        else:
            out = args.output_dir
        total += process_pickle(pickle_file, out, hold_point, goal, args.goal_tolerance,
                                args.control_point_offset)
    print(f"Exported time series for {total} run(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
