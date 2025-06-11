#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation.data_filtering import filtered_robot_agents
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

    return filtered_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate social metrics")
    parser.add_argument(
        "--pickle-file",
        type=Path,
        help="Path to the file to be filtered",
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

    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save the filtered data",
        default=None,
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    file_to_use = args.pickle_file
    new_file = args.output_file
    if new_file is None:
        new_file = file_to_use.with_name(
            file_to_use.name.replace(".pickle", "_filtered.pickle")
        )

    print(f"Filtering file {file_to_use}")
    with open(file_to_use, "rb") as file:
        data = pickle.load(file)

    filtered_data = {}
    for key in tqdm(data.keys(), desc="Filtering data"):
        filtered_data[key] = create_filtered_data(
            data[key],
            alpha=args.alpha,
            hampel_params={
                "window": args.hampel_window_size,
                "n_sigma": args.hampel_n_sigmas,
            },
        )
    print(f"Saving filtered data to {new_file}")
    with open(new_file, "wb") as f_out:
        pickle.dump(filtered_data, f_out)


if __name__ == "__main__":
    main()
