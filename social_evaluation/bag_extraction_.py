#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation.bag_extraction import (
    get_messages_from_bag,
    odometry_to_df,
    people_to_df,
    path_to_df,
    get_goal_position,
    tf2_messages_to_df,
    get_first_timestamp,
    get_person_data,
    filter_df_by_time,
    get_bag_path,
    get_transform_between_frames,
    get_pose_data_from_transform,
    compute_odometry_data,
)
from social_evaluation.common import get_map_data, retrieve_obstacle_positions
import social_evaluation
from tqdm import tqdm
from rclpy.logging import set_logger_level

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import os


def extract_topics_from_bag(
    bag_file: Path, topics: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple[float, float]]:
    """
    Extract the topics specified in the config file from the bag file
    and return the dataframes for each topic.
    : param bag_file: Path to the bag file
    : param topics: List of topics to extract
    : return: Tuple with the dataframes for odometry, people, tf and goal
    """
    df_odometry = pd.DataFrame()
    df_people = pd.DataFrame()
    df_tf = pd.DataFrame()
    goal = None
    for topic in topics:
        messages, msg_type = get_messages_from_bag(bag_file, topic)
        match msg_type:
            case "nav_msgs/msg/Odometry":
                df_odometry = odometry_to_df(messages)
            case "people_msgs/msg/People":
                df_people = people_to_df(messages)
            case "geometry_msgs/msg/PoseStamped":
                goal = get_goal_position(messages)
            case "tf2_msgs/msg/TFMessage":
                df_tf = tf2_messages_to_df(messages)
            case "nav_msgs/msg/Path":
                df_path = path_to_df(messages)
            case _:
                print(f"Unknown message type: {msg_type}")
    return df_odometry, df_people, df_tf, df_path, goal


def extract_transforms_from_dataframes(
    df_tf: pd.DataFrame, config_tf: Dict[str, Any], config_odometry: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Extract the selected transforms from the tf dataframes
    : param df_tf: Dataframe with the tf data
    : param config_tf: Configuration for the transforms
    : return: Dictionary with the selected transforms
    """
    transforms = {}
    if df_tf.empty:
        print("No tf data available")
        return transforms
    for transform in config_tf:
        for transform_name, transform_config in transform.items():
            frame_id = transform_config["frame_id"]
            child_frame_id = transform_config["child_frame_id"]
            transformations_parent_child = []
            for timestamp in filter_df_by_time(df_tf)["time"]:
                transformations_parent_child.append(
                    pd.Series(
                        {
                            "time": timestamp,
                            **get_pose_data_from_transform(
                                get_transform_between_frames(
                                    df_tf, frame_id, child_frame_id, timestamp
                                )
                            ),
                        }
                    )
                )
            transforms[transform_name] = pd.DataFrame(transformations_parent_child)
            transforms[transform_name], _ = compute_odometry_data(
                transforms[transform_name],
                period=config_odometry["period_for_diff"],
                max_velocity=config_odometry["max_velocity"],
                max_angular_vel=config_odometry["max_angular_velocity"],
                alpha=config_odometry["alpha"],
            )
    return transforms


def extract_data_from_bag(
    bag_file: Path,
    topics: List[str],
    exp_config: Dict[str, Any],
    time_limit: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Extract the data from the bag file
    : param bag_file: Path to the bag file
    : param exp_config: Configuration for the experiment
    : param time_limit: Time limit for the experiment (relative to the start time)
    : return: Dictionary with the extracted data
    """
    # set logger level to warning to avoid printing too many messages
    set_logger_level("rosbag2_storage", 30)
    # get the first timestamp in the bag file
    start_time = get_first_timestamp(bag_file)

    df_odometry, df_people, df_tf, df_path, goal = extract_topics_from_bag(
        bag_file, topics
    )
    transforms = extract_transforms_from_dataframes(
        df_tf, exp_config["transforms"], exp_config["robot_dynamics"]
    )
    df_agents_list = []
    for person in exp_config["examined_people"]:
        df_agents_list.append(get_person_data(df_people, person))
    map_file = Path(social_evaluation.__file__).parent.parent.joinpath(
        exp_config["map_file"]
    )
    df_obs, map, map_metadata = extract_obstacle_data(map_file)

    if time_limit is not None:
        end_time = time_limit["end_time"] + start_time
        start_time = time_limit["start_time"] + start_time
        df_odometry = filter_df_by_time(df_odometry, start_time, end_time)
        df_people = filter_df_by_time(df_people, start_time, end_time)
        df_path = filter_df_by_time(df_path, start_time, end_time)
        for transform_name, transform in transforms.items():
            transforms[transform_name] = filter_df_by_time(
                transform, start_time, end_time
            )
        for agent in df_agents_list:
            agent = filter_df_by_time(agent, start_time, end_time)

    return {
        "odometry": df_odometry,
        "people": df_people,
        "path": df_path,
        "agents": df_agents_list,
        "goal": goal,
        "transforms": transforms,
        "obstacles": df_obs,
        "map": map,
        "map_metadata": map_metadata,
    }


def extract_data_from_experiments(config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Extract the data from the bag files
    : param config: Configuration for the data extraction
    : return: Dictionary with the extracted data
    """
    experiment_names = config["experiment_names"]
    algorithms = config["algorithms"]
    experiment_ids = config["experiment_ids"]
    base_path = config["base_path"]
    time_limits = None
    data = {}
    for experiment_name in tqdm(
        experiment_names,
        desc=f"Extracting bags",
        leave=False,
    ):
        for algorithm in tqdm(
            algorithms,
            desc=f"Extracting bags for {experiment_name}",
            leave=False,
        ):
            for i in tqdm(
                experiment_ids,
                desc=f"Extracting bags for {experiment_name} with {algorithm}",
                leave=False,
            ):
                bag_path = get_bag_path(experiment_name, algorithm, base_path, i)
                # bag_paths.append(bag_path)
                if "time_limits" in config and experiment_name in config["time_limits"]:
                    time_limits = config["time_limits"][experiment_name][algorithm][i]
                data[bag_path] = extract_data_from_bag(
                    bag_path, config["topics"], config[experiment_name], time_limits
                )
    return data


def extract_obstacle_data(
    map_file: Path,
    pool_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the map data from the map file
    : param map_file: Path to the map file
    : return: Tuple with the obstacle data and the occupancy gridmap
    """

    occupancy_gridmap, metadata = get_map_data(map_file)
    df_obstacles = retrieve_obstacle_positions(occupancy_gridmap, metadata, pool_size)
    map_metadata = {
        "resolution": metadata["resolution"],
        "origin": metadata["origin"],
        "width": metadata["width"],
        "height": metadata["height"],
    }
    return df_obstacles, occupancy_gridmap, map_metadata


def save_data(data: Dict[str, pd.DataFrame], output_file: Path):
    """Save the extracted data to a pickle file
    : param data: Dictionary with the extracted data
    : param output_file: Path to the output file
    """
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def main():
    parser = argparse.ArgumentParser()

    package_path = Path(social_evaluation.__file__).parent

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Config file for the extraction parameters",
        default=package_path.joinpath("../config/config.yaml"),
    )

    parser.add_argument(
        "--exp-config-file",
        type=Path,
        help="Config file for all the experiments parameters",
        default=package_path.joinpath("../config/experiments_config.yaml"),
    )

    parser.add_argument(
        "--bag-file-path",
        type=Path,
        help="Bag file path to extract data from",
        default="None",
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output file path to save the extracted data",
        default="output.pickle",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)

    env_config_path = Path(args.exp_config_file)
    env_config = yaml.load(env_config_path.read_text(), Loader=yaml.FullLoader)

    for experiment in config["experiment_names"]:
        config[experiment] = env_config[experiment]

    if args.bag_file_path != PosixPath("None"):
        bag_file = Path(args.bag_file_path)
        bag_file = os.path.join(bag_file)
        data = extract_data_from_bag(bag_file, config)
    else:
        data = extract_data_from_experiments(config)

    # Save the extracted data
    save_data(data, args.output_file)


if __name__ == "__main__":
    main()
