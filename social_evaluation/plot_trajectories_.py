#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
import pandas as pd
from pathlib import Path
import pickle
import os
import numpy as np
import matplotlib
import subprocess

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter, FFMpegWriter
from typing import Dict, Any, List, Tuple, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure


from social_evaluation.common import combine_df_robot_agents
import social_evaluation.trajectory_plots as pu
from tqdm import tqdm
import argcomplete, argparse
import yaml
import social_evaluation


def normalize_animation_formats(
    requested_formats: Optional[List[str] | str],
    save_animations: bool,
) -> List[str]:
    """Return the unique animation formats that should be written."""
    if requested_formats is None:
        return ["gif"] if save_animations else []

    if isinstance(requested_formats, str):
        formats_to_process = [requested_formats]
    elif isinstance(requested_formats, (list, tuple, set)):
        formats_to_process = list(requested_formats)
    else:
        raise TypeError(
            "save_animation_formats must be a string or a list of strings."
        )

    normalized_formats: List[str] = []
    for animation_format in formats_to_process:
        normalized_format = animation_format.strip().lower()
        if not normalized_format:
            continue
        if normalized_format not in {"gif", "webm", "mp4"}:
            raise ValueError(
                f"Unsupported animation format '{animation_format}'. "
                "Supported formats are 'gif', 'webm', and 'mp4'."
            )
        if normalized_format not in normalized_formats:
            normalized_formats.append(normalized_format)

    if not normalized_formats and save_animations:
        normalized_formats.append("gif")

    return normalized_formats


def save_animation_output(
    animation: ArtistAnimation,
    output_path: Path,
    fps: int,
    preferred_writer: Optional[str] = "pillow",
) -> None:
    """Save a single animation file using a writer that matches the suffix."""
    if animation is None:
        return

    output_format = output_path.suffix.lower().lstrip(".")
    if output_format == "gif":
        try:
            animation.save(
                output_path,
                writer=preferred_writer if preferred_writer == "pillow" else "pillow",
                fps=fps,
                savefig_kwargs={"transparent": True},
            )
            return
        except (
            RuntimeError,
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            print(
                f"Falling back to PillowWriter for {output_path} because "
                f"'{preferred_writer}' failed: {exc}"
            )
            animation.save(
                output_path,
                writer=PillowWriter(fps=fps),
                savefig_kwargs={"transparent": True},
            )
            return

    if output_format == "webm" or output_format == "mp4":
        try:
            animation.save(
                output_path,
                writer=preferred_writer if preferred_writer == "ffmpeg" else "ffmpeg",
                fps=fps,
                savefig_kwargs={"transparent": True},
            )
            return
        except (
            RuntimeError,
            subprocess.CalledProcessError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            print(
                f"Falling back to FFMpegWriter for {output_path} because "
                f"'{preferred_writer}' failed: {exc}"
            )
            animation.save(
                output_path,
                writer=FFMpegWriter(fps=fps, codec="libvpx-vp9", bitrate=1800),
                savefig_kwargs={"transparent": True},
            )
            return

    raise ValueError(
        f"Unsupported animation output format '{output_format}' for {output_path}."
    )


def save_animation_outputs(
    animation: ArtistAnimation,
    output_path: Path,
    fps: int,
    output_formats: List[str],
    preferred_writer: Optional[str] = "pillow",
) -> None:
    """Save an animation to one or more requested output formats."""
    if animation is None:
        return

    for output_format in output_formats:
        save_animation_output(
            animation,
            output_path.with_suffix(f".{output_format}"),
            fps=fps,
            preferred_writer=preferred_writer,
        )


def plot_experiment_aggregate(
    data: dict,
    exp: List[str],
    ax: Axes,
    fig: Figure,
    time_freq: float = 0.1,
    animate: bool = True,
    colorbar: bool = False,
    use_odometry: bool = False,
) -> ArtistAnimation | None:
    """Plot the experiment aggregated in a single plot

    :param data: Dictionary with the data
    :param exp: Experiment to plot
    :param ax: Matplotlib axis
    :param fig: Matplotlib figure
    :param time_freq: Frequency of the time to plot
    :return: Animation object with the aggregated trajectories
    """

    goal = data[exp[0]].get("goal")
    map = data[exp[0]]["map"]
    map_metadata = data[exp[0]]["map_metadata"]

    df_trajectories_list = []
    for i in range(1, len(exp)):
        source = data[exp[i]]["odometry"]
        if not use_odometry and "map_to_base_link" in data[exp[i]]["transforms"]:
            source = data[exp[i]]["transforms"]["map_to_base_link"]
        df_trajectories_list.append(
            combine_df_robot_agents(
                df_robot=source,
                df_agents_list=data[exp[i]]["agents"],
            )
        )

    # Plot the data
    pu.plot_map(map, ax, map_metadata["resolution"], map_metadata["origin"])
    if goal is not None:
        pu.plot_goal(goal, ax)

    ani = pu.plot_trajectories_aggregated(
        df_trajectories_list,
        ax,
        fig=fig,
        time_freq=time_freq,
        animate=animate,
    )
    return ani


def plot_experiment_single(
    data: dict,
    exp: str,
    ax: Axes,
    fig: Figure,
    time_freq: float = 0.1,
    animate: bool = True,
    colorbar: bool = False,
    image_width_multi: float = 1.0,
    fontsize: int = 16,
    ax_limits: Optional[List[float]] = None,
    use_odometry: bool = False,
    dynamic_alpha: bool = True,
    plot_trajectory: bool = True,
):
    """Plot the experiment in a single plot

    :param data: Dictionary with the data
    :param exp: Experiment to plot
    :param ax: Matplotlib axis
    :param fig: Matplotlib figure
    :param time_freq: Frequency of the time to plot
    :param animate: Whether to animate the plot or not
    :param colorbar: Whether to plot the colorbar or not
    :param image_width_multi: Multiplier for the image width
    :param fontsize: Font size for the plot
    :param ax_limits: Limits for the axis
    :param use_odometry: Whether to use odometry or map_to_base_link for the robot trajectory
    :param dynamic_alpha: Whether to use dynamic alpha values for the trajectories
    :param plot_trajectory: Whether to plot the trajectory line or not
    :return: Animation object with the trajectories of the experiment
    """
    df_odometry = pd.DataFrame(data[exp]["odometry"])
    df_people = pd.DataFrame(data[exp]["people"])
    df_agents_list = data[exp]["agents"]
    df_obstacles = data[exp]["obstacles"]
    df_path = data[exp]["path"]
    transforms = data[exp]["transforms"]
    goal = data[exp].get("goal")
    map = data[exp]["map"]
    map_metadata = data[exp]["map_metadata"]

    # Plot the data
    ani = plot_data(
        df_odometry,
        df_people,
        df_agents_list,
        df_obstacles,
        transforms,
        df_path,
        goal,
        map,
        map_metadata,
        ax,
        fig,
        time_freq=time_freq,
        animate=animate,
        colorbar=colorbar,
        image_width_multi=image_width_multi,
        fontsize=fontsize,
        ax_limits=ax_limits,
        use_odometry=use_odometry,
        dynamic_alpha=dynamic_alpha,
        plot_trajectory=plot_trajectory,
    )
    return ani


def plot_data(
    df_odometry: pd.DataFrame,
    df_people: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
    transforms: pd.DataFrame,
    df_path: Optional[pd.DataFrame],
    goal: Optional[Tuple[float, float]],
    map: np.ndarray,
    map_metadata: Dict[str, Any],
    ax: Axes,
    fig: Figure,
    time_freq: float = 0.1,
    colorbar: bool = False,
    animate: bool = True,
    image_width_multi: float = 1.0,
    fontsize: int = 16,
    ax_limits: Optional[List[float]] = None,
    use_odometry: bool = False,
    dynamic_alpha: bool = True,
    plot_trajectory: bool = True,
) -> ArtistAnimation | None:
    """Plot the data in a single plot

    :param df_odometry: Dataframe with the odometry
    :param df_people: Dataframe with the people
    :param df_agents_list: List of dataframes with the agents
    :param df_obstacles: Dataframe with the obstacles
    :param transforms: Dataframe with the transforms
    :param goal: Tuple with the goal position
    :param map: Numpy array with the map
    :param map_metadata: Dictionary with the map metadata
    :param ax: Matplotlib axis
    :param fig: Matplotlib figure
    :param time_freq: Frequency of the time to plot
    :param colorbar: Whether to plot the colorbar or not
    :param animate: Whether to animate the plot or not
    :param image_width_multi: Multiplier for the image width
    :param fontsize: Font size for the plot
    :param ax_limits: Limits for the axis
    :param use_odometry: Whether to use odometry or map_to_base_link for the robot trajectory
    :param dynamic_alpha: Whether to use dynamic alpha values for the trajectories
    :param plot_trajectory: Whether to plot the trajectory line or not
    :return: Animation object with the trajectories of the experiment
    """
    pu.plot_map(map, ax, map_metadata["resolution"], map_metadata["origin"])
    if goal is not None:
        pu.plot_goal(goal, ax)
    else:
        print("No goal to plot")
    if not use_odometry and "map_to_base_link" in transforms:
        df_robot = transforms["map_to_base_link"]
    else:
        df_robot = df_odometry
    df_trajectories = combine_df_robot_agents(
        df_robot=df_robot, df_agents_list=df_agents_list
    )

    ani = pu.plot_trajectories(
        df_trajectories,
        ax,
        fig=fig,
        time_freq=time_freq,
        colorbar=colorbar,
        animate=animate,
        image_width_multi=image_width_multi,
        ax_limits=ax_limits,
        dynamic_alpha=dynamic_alpha,
        plot_trajectory=plot_trajectory,
    )

    ax.set_xlabel("X (m)", fontsize=fontsize)
    ax.set_ylabel("Y (m)", fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
    return ani


def load_data_from_pickles(pickle_paths: List[Path]) -> Dict[str, Any]:
    """Merge experiment dictionaries loaded from multiple pickle files."""
    aggregated_data: Dict[str, Any] = {}
    for pickle_path in pickle_paths:
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)
        for key, value in pickle_data.items():
            aggregated_data[key] = value
    return aggregated_data


def save_animation_with_fallback(
    animation: ArtistAnimation,
    output_path: Path,
    fps: int,
    preferred_writer: Optional[str] = "pillow",
) -> None:
    """Backwards-compatible wrapper for saving a single animation file."""
    save_animation_output(
        animation,
        output_path,
        fps=fps,
        preferred_writer=preferred_writer,
    )


def override_missing_goals_with_config(
    data: Dict[str, Any], experiments_config: Dict[str, Any]
) -> None:
    """Fill missing goals in loaded data using experiment parameters."""
    if not experiments_config:
        return

    configured_goals: Dict[str, Tuple[float, float]] = {}
    for experiment_name, params in experiments_config.items():
        if not isinstance(params, dict):
            continue
        normalized_goal = pu.normalize_goal(params.get("goal"))
        if normalized_goal is not None:
            configured_goals[experiment_name] = normalized_goal

    if not configured_goals:
        return

    applied_overrides = set()
    for exp_key, exp_data in data.items():
        if exp_data.get("goal") is not None:
            continue
        parts = exp_key.split("/")
        if len(parts) < 4:
            continue
        experiment_name = parts[-4]
        goal_override = configured_goals.get(experiment_name)
        if goal_override is not None:
            exp_data["goal"] = goal_override
            if experiment_name not in applied_overrides:
                print(
                    f"Using configured goal {goal_override} for experiment {experiment_name}"
                )
                applied_overrides.add(experiment_name)


def main():
    parser = argparse.ArgumentParser()

    package_path = Path(social_evaluation.__file__).parent

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Config file for all the plots parameters",
        default=package_path.joinpath("../config/trajectory_plots.yaml"),
    )

    parser.add_argument(
        "--pickle-file",
        type=Path,
        help="Bag file path to extract data from",
        default=None,
    )

    parser.add_argument(
        "--pickle-base-path",
        type=Path,
        help="Directory containing per-experiment pickle files",
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory to save the plots",
        default=Path("plotted_trajectories"),
    )

    parser.add_argument(
        "--use-tex",
        action="store_true",
        help="Whether to use LaTeX for text rendering in plots (requires LaTeX installation)",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    config = yaml.load(config_path.open("r"), Loader=yaml.FullLoader)

    plot_single = config["plot_single"]
    plot_aggregated = config["plot_aggregated"]
    plot_colorbar = config.get("plot_colorbar", False)
    time_freq = config.get("time_freq", 0.1)
    dynamic_alpha = config.get("dynamic_alpha", True)
    plot_trajectory = config.get("plot_trajectory", True)
    use_odometry = config.get("use_odometry", False)
    save_gif = config.get("save_gif", True)
    save_webm = config.get("save_webm", False)
    requested_animation_formats = config.get("save_animation_formats")
    if requested_animation_formats is None:
        requested_animation_formats = []
        if save_gif:
            requested_animation_formats.append("gif")
        if save_webm:
            requested_animation_formats.append("webm")
    save_animation_formats = normalize_animation_formats(
        requested_animation_formats,
        save_gif or save_webm,
    )
    save_animation = bool(save_animation_formats)
    save_fig = config.get("save_fig", True)
    fontsize = config.get("fontsize", 16)
    image_width_multi = config.get("image_width_multi", 1.0)
    bbox_pad_inches = float(config.get("bbox_pad_inches", 0.05))
    animation_writer = config.get("animation_writer", "pillow")
    use_tex = args.use_tex or config.get("use_tex", False)
    if use_tex:
        pu.apply_trajectory_plot_rcparams()

    pickle_base_path = args.pickle_base_path
    config_pickle_base_path = config.get("pickle_base_path")
    if pickle_base_path is None and config_pickle_base_path:
        pickle_base_path = Path(config_pickle_base_path)

    if args.pickle_file:
        pickle_file = Path(args.pickle_file)
        data = load_data_from_pickles([pickle_file])
    elif pickle_base_path:
        pickle_base_path = Path(pickle_base_path)
        if not pickle_base_path.exists():
            raise FileNotFoundError(
                f"Pickle base path {pickle_base_path} does not exist."
            )
        pickle_files = sorted(pickle_base_path.rglob("*.pickle"))
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found under {pickle_base_path}.")
        print(f"Loading {len(pickle_files)} pickle files from {pickle_base_path}")
        data = load_data_from_pickles(pickle_files)
    else:
        data = {}
        for exp in config["experiments"]:
            print(f"Loading experiment {config['experiments'][exp]['path']}")
            exp_path = Path(config["experiments"][exp]["path"])
            with open(exp_path, "rb") as f:
                pickle_data = pickle.load(f)
                for key in pickle_data.keys():
                    data[key] = pickle_data[key]

    override_missing_goals_with_config(data, config.get("experiments", {}))

    experiment_algorithm = {}
    for exp in data.keys():

        experiment_name = exp.split("/")[-4]
        algorithm_name = exp.split("/")[-3]
        if f"{experiment_name}_{algorithm_name}" not in experiment_algorithm:
            experiment_algorithm[f"{experiment_name}_{algorithm_name}"] = []
        experiment_algorithm[f"{experiment_name}_{algorithm_name}"].append(exp)
    if plot_single:
        for i, exp in enumerate(tqdm(data.keys(), desc=f"Plotting experiments")):
            experiment_name = exp.split("/")[-4]
            algorithm_name = exp.split("/")[-3]
            exp_id = exp.split("_")[-2]

            fig, ax = plt.subplots()
            fig.set_size_inches(10, 10)

            ani = plot_experiment_single(
                data,
                exp,
                ax,
                fig,
                time_freq=time_freq,
                animate=save_animation,
                colorbar=plot_colorbar,
                image_width_multi=image_width_multi,
                fontsize=fontsize,
                ax_limits=config["experiments"]
                .get(experiment_name, {})
                .get("ax_limits", None),
                use_odometry=use_odometry,
                dynamic_alpha=dynamic_alpha,
                plot_trajectory=plot_trajectory,
            )

            if not args.output_dir.joinpath(experiment_name, algorithm_name).exists():
                os.makedirs(args.output_dir.joinpath(experiment_name, algorithm_name))

            # save the animation in the requested formats
            if save_animation:
                if ani is None:
                    print(
                        f"Skipping animation for {experiment_name}/{algorithm_name}/{exp_id}: no frames."
                    )
                else:
                    output_animation = args.output_dir.joinpath(
                        experiment_name, algorithm_name, f"experiment_{exp_id}"
                    )
                    save_animation_outputs(
                        ani,
                        output_animation,
                        fps=int(1 / time_freq),
                        output_formats=save_animation_formats,
                        preferred_writer=animation_writer,
                    )
            # save the last frame of the animation
            if save_fig:
                output_png = args.output_dir.joinpath(
                    experiment_name, algorithm_name, f"experiment_{exp_id}.png"
                )
                bbox = pu.get_bbox_for_figure(fig, ax)
                fig.savefig(
                    output_png,
                    bbox_inches=bbox,
                    pad_inches=bbox_pad_inches,
                    transparent=True,
                    format="png",
                )
            plt.close()
            ax.clear()
            fig.clear()
    if plot_aggregated:
        for i, exp in enumerate(
            tqdm(experiment_algorithm.keys(), desc=f"Plotting aggregated experiments")
        ):

            if plot_aggregated:
                fig, ax = plt.subplots()
                fig.set_size_inches(10, 10)
                experiment_name, algorithm_name = exp.split("_")

                ani = plot_experiment_aggregate(
                    data,
                    experiment_algorithm[exp],
                    ax,
                    fig,
                    animate=save_animation,
                    time_freq=time_freq,
                    colorbar=plot_colorbar,
                    use_odometry=use_odometry,
                )

                # save the animation in the requested formats
                if save_animation:
                    if ani is None:
                        print(
                            f"Skipping aggregated animation for {experiment_name}/{algorithm_name}: no frames."
                        )
                    else:
                        aggregated_animation = args.output_dir.joinpath(
                            experiment_name,
                            algorithm_name,
                            f"aggregated_trajectories",
                        )
                        save_animation_outputs(
                            ani,
                            aggregated_animation,
                            fps=int(1 / time_freq),
                            output_formats=save_animation_formats,
                            preferred_writer=animation_writer,
                        )
                # save the last frame of the animation
                if save_fig:
                    aggregated_png = args.output_dir.joinpath(
                        experiment_name,
                        algorithm_name,
                        f"aggregated_trajectories.png",
                    )
                    bbox = pu.get_bbox_for_figure(fig, ax)
                    fig.savefig(
                        aggregated_png,
                        bbox_inches=bbox,
                        pad_inches=bbox_pad_inches,
                        transparent=True,
                        format="png",
                    )
                plt.close()
                ax.clear()
                fig.clear()


if __name__ == "__main__":
    main()
