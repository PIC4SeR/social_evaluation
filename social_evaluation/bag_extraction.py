#!/usr/bin/env python3
from social_evaluation import utils

import pandas as pd
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import os


def extract_topics_from_bag(bag_file: Path, topics: List[str]) -> pd.DataFrame:
    """
    Extract the topics specified in the config file from the bag file
    and return the dataframes for each topic.
    : param bag_file: Path to the bag file
    : param topics: List of topics to extract
    """
    df_odometry = pd.DataFrame()
    df_people = pd.DataFrame()
    df_tf = pd.DataFrame()
    goal = None
    for topic in topics:
        messages, msg_type = utils.get_messages_from_bag(bag_file, topic)
        match msg_type:
            case "nav_msgs/msg/Odometry":
                df_odometry = utils.odometry_to_df(messages)
            case "people_msgs/msg/People":
                df_people = utils.people_to_df(messages)
            case "geometry_msgs/msg/PoseStamped":
                goal = utils.get_goal_position(messages)
            case "tf2_msgs/msg/TFMessage":
                df_tf = utils.tf2_messages_to_df(messages)
            case _:
                print(f"Unknown message type: {msg_type}")
    return df_odometry, df_people, df_tf, goal


def extract_transforms_from_dataframes(
    df_tf: pd.DataFrame, config_tf: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Extract the selected transforms from the tf dataframes
    : param df_tf: Dataframe with the tf data
    : param config_tf: Configuration for the transforms
    : return: Dictionary with the selected transforms
    """
    transforms = {}
    for transform in config_tf:
        for transform_name, transform_config in transform.items():
            frame_id = transform_config["frame_id"]
            child_frame_id = transform_config["child_frame_id"]
            transformations_parent_child = []
            for timestamp in utils.filter_df_by_time(df_tf)["time"]:
                transformations_parent_child.append(
                    pd.Series(
                        {
                            "time": timestamp,
                            **utils.get_pose_data_from_transform(
                                utils.get_transform_between_frames(
                                    df_tf, frame_id, child_frame_id, timestamp
                                )
                            ),
                        }
                    )
                )
            transforms[transform_name] = pd.DataFrame(transformations_parent_child)
    return transforms


def extract_data_from_bag(
    bag_file: Path, config: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Extract the data from the bag file
    : param bag_file: Path to the bag file
    : param config: Configuration for the data extraction
    : return: Dictionary with the extracted data
    """
    df_odometry, df_people, df_tf, goal = extract_topics_from_bag(
        bag_file, config["topics"]
    )
    transforms = extract_transforms_from_dataframes(df_tf, config["transforms"])
    return {
        "odometry": df_odometry,
        "people": df_people,
        "goal": goal,
        "transforms": transforms,
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
    bag_paths = []
    for experiment_name in experiment_names:
        for algorithm in algorithms:
            for i in experiment_ids:
                bag_path = utils.get_bag_path(experiment_name, algorithm, base_path, i)
                bag_paths.append(bag_path)

    data = {}
    for bag_file in bag_paths:
        data[bag_file] = extract_data_from_bag(bag_file, config)
    return data


def save_data(data: Dict[str, pd.DataFrame], output_file: Path):
    """Save the extracted data to a pickle file
    : param data: Dictionary with the extracted data
    : param output_file: Path to the output file
    """
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        type=Path,
        help="Config file",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--bag_file_path",
        type=Path,
        help="Bag file path to extract data from",
        default="None",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        help="Output file path to save the extracted data",
        default="output.pickle",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    config_path = Path(args.config_file)
    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)

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
