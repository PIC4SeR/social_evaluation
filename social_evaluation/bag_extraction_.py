#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation.bag_extraction import (
    odometry_to_df,
    vector3_to_df,
    people_to_df,
    path_to_df,
    string_to_df,
    tf2_messages_to_df,
    get_person_data,
    filter_df_by_time,
    get_bag_path,
    get_transform_between_frames,
    get_pose_data_from_transform,
    compute_odometry_data,
)
from social_evaluation.bag_extraction.bag_extraction_utils import (
    iterate_bag_messages,
    odometry_msg_to_row,
    people_msg_to_row,
    path_msg_to_row,
    tf_message_to_row,
    string_msg_to_row,
    vector3_msg_to_row,
)
from social_evaluation.common import get_map_data, retrieve_obstacle_positions
import social_evaluation
from tqdm import tqdm
from rclpy.logging import set_logger_level

import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Set
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import os


def extract_topics_from_bag(bag_file: Path, topics: List[str]) -> Tuple[
    float,
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Optional[Tuple[float, float]],
    pd.DataFrame,
]:
    """
    Extract the topics specified in the config file from the bag file and return the dataframes for each topic.

    :param bag_file: Path to the bag file
    :param topics: List of topics to extract

    :return: Tuple with the start time and dataframes for odometry and
        social memory (one per topic, keyed by topic name), people, tf, path,
        goal and result
    """

    odometry_rows: Dict[str, List[Dict[str, Any]]] = {}
    memory_rows: Dict[str, List[Dict[str, Any]]] = {}
    people_rows: List[Dict[str, Any]] = []
    tf_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    goal_positions: Set[Tuple[float, float]] = set()
    first_timestamp: Optional[float] = None

    for topic_name, msg_type, message, timestamp in iterate_bag_messages(
        bag_file, topics
    ):
        if first_timestamp is None:
            first_timestamp = timestamp
        match msg_type:
            case "nav_msgs/msg/Odometry":
                odometry_rows.setdefault(topic_name, []).append(
                    odometry_msg_to_row(message)
                )
            case "people_msgs/msg/People":
                people_rows.append(people_msg_to_row(message))
            case "geometry_msgs/msg/PoseStamped":
                goal_positions.add(
                    (
                        message.pose.position.x,
                        message.pose.position.y,
                    )
                )
            case "tf2_msgs/msg/TFMessage":
                row = tf_message_to_row(message)
                if row:
                    tf_rows.append(row)
            case "nav_msgs/msg/Path":
                path_rows.append(path_msg_to_row(message))
            case "std_msgs/msg/String":
                result_rows.append(string_msg_to_row(message))
            case "geometry_msgs/msg/Vector3Stamped":
                memory_rows.setdefault(topic_name, []).append(
                    vector3_msg_to_row(message)
                )
            case _:
                print(f"Unknown message type: {msg_type}")

    start_time = first_timestamp if first_timestamp is not None else 0.0
    odometry_by_topic = {
        topic: odometry_to_df(rows) for topic, rows in odometry_rows.items()
    }
    memory_by_topic = {
        topic: vector3_to_df(rows) for topic, rows in memory_rows.items()
    }
    df_people = people_to_df(people_rows)
    df_tf = tf2_messages_to_df(tf_rows)
    df_path = path_to_df(path_rows)
    df_result = string_to_df(result_rows)
    goal = None
    if goal_positions:
        if len(goal_positions) > 1:
            print("Warning: more than one goal position found, using the first entry.")
        goal = goal_positions.pop()

    return (
        start_time,
        odometry_by_topic,
        memory_by_topic,
        df_people,
        df_tf,
        df_path,
        goal,
        df_result,
    )


def extract_transforms_from_dataframes(
    df_tf: pd.DataFrame, config_tf: Dict[str, Any], config_odometry: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """Extract the selected transforms from the tf dataframes

    :param df_tf: Dataframe with the tf data
    :param config_tf: Configuration for the transforms
    :param config_odometry: Configuration for the odometry data

    :return: Dictionary with the selected transforms
    """

    transforms = {}
    if df_tf.empty:
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

    :param bag_file: Path to the bag file
    :param exp_config: Configuration for the experiment
    :param time_limit: Time limit for the experiment (relative to the start time)

    :return: Dictionary with the extracted data. If the experiment config has a
        ``robots: {name: odometry_topic}`` mapping (multi-robot fleets), the
        result also contains ``robots: {name: df_odometry}``.
    """

    # set logger level to warning to avoid printing too many messages
    set_logger_level("rosbag2_storage", 30)
    robots_config: Dict[str, str] = exp_config.get("robots") or {}
    # swarm_control publishes each worker's social memory under its own
    # namespace; the recorder writes it as /<robot>/social_memory. Absent for
    # any other controller, in which case df_memory is simply empty.
    memory_topics = {name: f"/{name}/social_memory" for name in robots_config}
    topics = list(topics) + [
        topic
        for topic in list(robots_config.values()) + list(memory_topics.values())
        if topic not in topics
    ]
    (
        start_time,
        odometry_by_topic,
        memory_by_topic,
        df_people,
        df_tf,
        df_path,
        goal,
        df_result,
    ) = extract_topics_from_bag(bag_file, topics)
    df_robots = {
        name: odometry_by_topic.get(topic, pd.DataFrame())
        for name, topic in robots_config.items()
    }
    df_memory = {
        name: memory_by_topic[topic]
        for name, topic in memory_topics.items()
        if not memory_by_topic.get(topic, pd.DataFrame()).empty
    }
    # Back-compat single-robot dataframe: the configured topic if given,
    # otherwise the only (or first) odometry topic in the bag.
    robot_odometry_topic = exp_config.get("robot_odometry_topic")
    if robot_odometry_topic is not None:
        df_odometry = odometry_by_topic.get(robot_odometry_topic, pd.DataFrame())
    elif odometry_by_topic:
        if len(odometry_by_topic) > 1 and not robots_config:
            print(
                "Warning: multiple odometry topics found "
                f"({sorted(odometry_by_topic)}); using the first. Set "
                "'robot_odometry_topic' or a 'robots' mapping in the experiment "
                "config to disambiguate."
            )
        df_odometry = next(iter(odometry_by_topic.values()))
    else:
        df_odometry = pd.DataFrame()
    transforms = extract_transforms_from_dataframes(
        df_tf, exp_config.get("transforms", []), exp_config["robot_dynamics"]
    )
    df_agents_list = []
    for person in exp_config["examined_people"]:
        person_data = get_person_data(df_people, person)
        if not person_data.empty:
            df_agents_list.append(person_data)
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
        for name, df_robot in df_robots.items():
            df_robots[name] = filter_df_by_time(df_robot, start_time, end_time)
        for name, df_z in df_memory.items():
            df_memory[name] = filter_df_by_time(df_z, start_time, end_time)
        for transform_name, transform in transforms.items():
            transforms[transform_name] = filter_df_by_time(
                transform, start_time, end_time
            )
        for i, agent in enumerate(df_agents_list):
            agent = filter_df_by_time(agent, start_time, end_time)
            df_agents_list[i] = agent

    return {
        "odometry": df_odometry,
        "robots": df_robots,
        "memory": df_memory,
        "people": df_people,
        "path": df_path,
        "agents": df_agents_list,
        "goal": goal,
        "result": df_result,
        "transforms": transforms,
        "obstacles": df_obs,
        "map": map,
        "map_metadata": map_metadata,
    }


def extract_data_from_experiments(
    config: Dict[str, Any],
    output_file: Optional[Path] = None,
    base_output_path: Optional[Path] = None,
    existing_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """Extract the data from the bag files and persist progress incrementally.

    :param config: Configuration for the data extraction
    :param output_file: Optional path used to save intermediate aggregated results
    :param base_output_path: Optional base directory to store one pickle per experiment
    :param existing_data: Previously extracted data to resume from
    :return: Dictionary with the extracted data
    """

    experiment_names = config["experiment_names"]
    algorithms = config["algorithms"]
    experiment_ids = config["experiment_ids"]
    base_path = config["base_path"]

    data = existing_data if existing_data is not None else {}
    for experiment_name in tqdm(
        experiment_names,
        desc="Extracting bags",
        leave=False,
    ):
        for algorithm in tqdm(
            algorithms,
            desc=f"Extracting bags for {experiment_name}",
            leave=False,
        ):
            if base_output_path is None:
                experiment_data = data
                experiment_output_path = None
            else:
                experiment_output_path = get_experiment_output_path(
                    base_output_path, experiment_name, algorithm
                )
                experiment_data = {}
                if experiment_output_path.exists():
                    with open(experiment_output_path, "rb") as file:
                        experiment_data = pickle.load(file)
                    data.update(experiment_data)
                    tqdm.write(
                        f"Loaded {len(experiment_data)} cached entries for "
                        f"{algorithm}/{experiment_name}"
                    )

            for i in tqdm(
                experiment_ids,
                desc=f"Extracting bags for {experiment_name} with {algorithm}",
                leave=False,
            ):
                bag_path = get_bag_path(experiment_name, algorithm, base_path, i)
                if bag_path in experiment_data:
                    tqdm.write(f"Skipping {bag_path} (already extracted)")
                    continue

                time_limits = (
                    config.get("time_limits", {})
                    .get(experiment_name, {})
                    .get(algorithm, {})
                    .get(i)
                )
                experiment_data[bag_path] = extract_data_from_bag(
                    bag_path,
                    config["topics"],
                    config[experiment_name],
                    time_limits,
                )
                if experiment_output_path is not None:
                    save_data(experiment_data, experiment_output_path)
                elif output_file is not None:
                    save_data(data, output_file)
            if base_output_path is not None:
                data.update(experiment_data)
    return data


def get_experiment_output_path(
    base_output_path: Path, experiment_name: str, algorithm: Optional[str] = None
) -> Path:
    """Return the per-algorithm, per-experiment pickle path, mirroring the
    extracted_bags layout: <base>/<algorithm>/<experiment_name>.pickle"""

    directory = base_output_path / algorithm if algorithm else base_output_path
    experiment_relative = Path(experiment_name)
    return directory / experiment_relative.parent / f"{experiment_relative.name}.pickle"


def extract_obstacle_data(
    map_file: Path,
    pool_size: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the map data from the map file

    :param map_file: Path to the map file
    :param pool_size: Pool size for the max pooling

    :return: Tuple with the obstacle data and the occupancy gridmap
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

    :param data: Dictionary with the extracted data
    :param output_file: Path to the output file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(output_path.name + ".tmp")
    with open(tmp_output, "wb") as file:
        pickle.dump(data, file)
    os.replace(tmp_output, output_path)


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

    parser.add_argument(
        "--base-output-path",
        type=Path,
        help="Base directory where per-experiment pickles will be stored",
        default=None,
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    config_path = Path(args.config_file)
    config = yaml.load(config_path.read_text(), Loader=yaml.FullLoader)

    env_config_path = Path(args.exp_config_file)
    env_config = yaml.load(env_config_path.read_text(), Loader=yaml.FullLoader)

    for experiment in config["experiment_names"]:
        config[experiment] = env_config[experiment]

    output_file = Path(args.output_file)
    base_output_path = args.base_output_path

    existing_data = None
    if base_output_path is None and output_file.exists():
        with open(output_file, "rb") as file:
            existing_data = pickle.load(file)
        print(
            f"Loaded {len(existing_data)} previously extracted scenarios from {output_file}"
        )

    if args.bag_file_path != PosixPath("None"):
        bag_file = Path(args.bag_file_path)
        bag_file = os.path.join(bag_file)
        data = extract_data_from_bag(bag_file, config)
        save_data(data, output_file)
        return

    data = extract_data_from_experiments(
        config,
        output_file=None if base_output_path is not None else output_file,
        base_output_path=base_output_path,
        existing_data=existing_data,
    )

    if base_output_path is None:
        save_data(data, output_file)
    else:
        print(f"Per-experiment pickles stored under {base_output_path.resolve()}")


if __name__ == "__main__":
    main()
