#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation.data_filtering import compute_ema, filtered_robot_agents
import pandas as pd
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import numpy as np
from tqdm import tqdm


def create_filtered_data(
    data: Dict[str, Any],
    alpha: float = 0.1,
    hampel_params: Optional[Dict[str, Union[int, float]]] = None,
) -> Dict[str, Any]:
    """Create a filtered version of the original data structure by applying
    smoothing techniques to the 'agents' and 'transforms' series.

    Args:
        data: Dictionary containing the original data with keys 'robot' and 'agents'.
        alpha: Smoothing factor for EMA (default: 0.1).
    Returns:
        Dictionary with processed 'robot' and 'agents' data.
    """

    filtered_data = data.copy()

    if "map_to_base_link" in data["transforms"]:
        robot_data = data["transforms"]["map_to_base_link"].copy()
        filtered_data["transforms"]["map_to_base_link"], filtered_data["agents"] = (
            filtered_robot_agents(
                df_robot=robot_data,
                df_agents_list=data["agents"],
                alpha=alpha,
                hampel_params=hampel_params,
            )
        )
    else:
        robot_data = data["odometry"].copy()
        filtered_data["odometry"], filtered_data["agents"] = filtered_robot_agents(
            df_robot=robot_data,
            df_agents_list=data["agents"],
            alpha=alpha,
            hampel_params=hampel_params,
        )

    # Multi-robot fleets: smooth each robot's frame the same way (agents were
    # already filtered above alongside the primary robot).
    if data.get("robots"):
        filtered_data["robots"] = {
            name: compute_ema(df.copy(), alpha=alpha, hampel_params=hampel_params)
            for name, df in data["robots"].items()
            if not df.empty
        }

    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Filter recorded social navigation data")

    # Exactly one input source is required.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--pickle-file",
        type=Path,
        help="Path to a single pickle file to be filtered",
    )
    input_group.add_argument(
        "--base-path",
        type=Path,
        help="Directory containing the pickle files to be filtered",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        help="Smoothing factor for EMA (default: 0.1)",
        default=0.1,
    )

    parser.add_argument(
        "--hampel-window-size",
        type=int,
        help="Window size for Hampel filter (default: 20)",
        default=20,
    )

    parser.add_argument(
        "--hampel-n-sigmas",
        type=float,
        help="Number of sigmas for Hampel filter (default: 1.5)",
        default=1.5,
    )

    # At most one output target: an explicit file (single input) or a directory.
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save the filtered file (only valid with --pickle-file)",
    )
    output_group.add_argument(
        "--base-output-dir",
        type=Path,
        default=None,
        help="Directory to save the filtered files (default: alongside each input)",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.output_file is not None and args.base_path is not None:
        parser.error("--output-file can only be used with --pickle-file, not --base-path.")

    def default_output(pickle_file: Path) -> Path:
        """Output path when no explicit --output-file is given."""
        if args.base_output_dir is not None:
            # Preserve the input's structure relative to --base-path so nested
            # files with the same name don't collide under the output directory.
            if args.base_path is not None:
                return args.base_output_dir / pickle_file.relative_to(args.base_path)
            return args.base_output_dir / pickle_file.name
        return pickle_file.with_name(
            pickle_file.name.replace(".pickle", "_filtered.pickle")
        )

    if args.base_path is not None:
        if not args.base_path.exists():
            parser.error(f"Base path {args.base_path} does not exist.")
        # Recurse to match the nested layout produced by bag_extraction
        # (get_experiment_output_path) and consumed by plot_trajectories'
        # --pickle-base-path. Skip already-filtered outputs so re-runs don't
        # re-filter their own results.
        pickle_files_to_process = sorted(
            f
            for f in args.base_path.rglob("*.pickle")
            if not f.name.endswith("_filtered.pickle")
        )
        if not pickle_files_to_process:
            parser.error(f"No .pickle files found under {args.base_path}")
        new_files = [default_output(pickle_file) for pickle_file in pickle_files_to_process]
    else:
        pickle_files_to_process = [args.pickle_file]
        new_files = [
            args.output_file
            if args.output_file is not None
            else default_output(args.pickle_file)
        ]

    # use tqdm to show a progress bar while processing the files
    for file_to_use, new_file in tqdm(zip(pickle_files_to_process, new_files), desc=f"Processing files", total=len(pickle_files_to_process)):
        with open(file_to_use, "rb") as file:
            data = pickle.load(file)

        filtered_data = {}
        for key in tqdm(data.keys(), desc=f"Filtering data for {file_to_use.name}", leave=False):
            filtered_data[key] = create_filtered_data(
                data[key],
                alpha=args.alpha,
                hampel_params={
                    "window": args.hampel_window_size,
                    "n_sigma": args.hampel_n_sigmas,
                },
            )
        print(f"Saving filtered data to {new_file}")
        new_file.parent.mkdir(parents=True, exist_ok=True)
        with open(new_file, "wb") as f_out:
            pickle.dump(filtered_data, f_out)


if __name__ == "__main__":
    main()
