from logging import warning
import pandas as pd

# get the path to the bag file
from pathlib import Path as OSPath
import os
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion, compose_matrix
import numpy as np

from std_msgs.msg import String
from people_msgs.msg import People, Person
from nav_msgs.msg import Odometry, Path as NavPath
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from typing import List, Tuple, Union, Dict, Type, Iterable, Iterator, Optional, Set, Any

MsgType = Union[
    Odometry, People, LaserScan, PoseStamped, TFMessage, NavPath, String, Person
]


def get_bag_path(experiment: str, algorithm: str, base_path: str, index: int) -> OSPath:
    """Get the path to the bag file for a given experiment and algorithm.

    :param experiment: The experiment name
    :param algorithm: The algorithm name
    :param base_path: The base path
    :param index: The index of the experiment
    :return: The path to the bag file
    """
    model_path = base_path + "/" + experiment + "/" + algorithm
    model_path = OSPath(model_path)
    # Check if the path exists    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")
    # use experiment_ or scenario_ as prefix depending on the naming convention used in the bags
    experiment_prefix = "experiment_" if (model_path / f"experiment_{index}").exists() else "scenario_"
    bag_path = os.path.join(
        model_path, f"{experiment_prefix}{index}", f"{experiment_prefix}{index}_0.db3"
    )
    # Check if the bag file exists
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag file {bag_path} does not exist")
    return bag_path


def get_rosbag_options(path: OSPath, serialization_format: str = "cdr") -> Tuple:
    """Get the rosbag options for a given path.

    :param path: The path to the bag file
    :param serialization_format: The serialization format
    :return: A tuple containing the storage options and the converter options
    """
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )
    return storage_options, converter_options


def get_messages_from_bag(bag_path: OSPath, topic: str) -> Tuple[List[MsgType], Type]:
    """Get messages from a bag file.

    :param bag_path: The path to the bag file
    :param topic: The topic name
    :return: A tuple containing the messages and the message type
    """
    storage_options, converter_options = get_rosbag_options(bag_path)
    bag = rosbag2_py.SequentialReader()
    bag.open(storage_options, converter_options)
    topic_types = bag.get_all_topics_and_types()
    type_map = {
        topic_types[i].name: topic_types[i].type for i in range(len(topic_types))
    }

    messages = []
    while bag.has_next():
        topic_name, data, t = bag.read_next()
        if topic_name == topic:
            messages.append(deserialize_message(data, get_message(type_map[topic])))
    return messages, type_map[topic]


def iterate_bag_messages(
    bag_path: OSPath, topics: Optional[Iterable[str]] = None
) -> Iterator[Tuple[str, str, MsgType, float]]:
    """Iterate through bag messages once, optionally filtering by topics.

    :param bag_path: The path to the bag file
    :param topics: Optional iterable with topic names to keep
    :return: Generator yielding (topic_name, msg_type, message, timestamp_sec)
    """
    topic_filter = set(topics) if topics is not None else None
    storage_options, converter_options = get_rosbag_options(bag_path)
    bag = rosbag2_py.SequentialReader()
    bag.open(storage_options, converter_options)
    topic_types = {topic.name: topic.type for topic in bag.get_all_topics_and_types()}
    type_cache = {
        topic_name: get_message(msg_type) for topic_name, msg_type in topic_types.items()
    }

    while bag.has_next():
        topic_name, data, timestamp = bag.read_next()
        if topic_filter is not None and topic_name not in topic_filter:
            continue
        if topic_name not in type_cache:
            continue
        msg = deserialize_message(data, type_cache[topic_name])
        yield topic_name, topic_types[topic_name], msg, timestamp * 1e-9


def odometry_msg_to_row(msg: Odometry) -> Dict[str, Any]:
    time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    qx = msg.pose.pose.orientation.x
    qy = msg.pose.pose.orientation.y
    qz = msg.pose.pose.orientation.z
    qw = msg.pose.pose.orientation.w
    roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
    frame_id = msg.header.frame_id
    velocity_x = msg.twist.twist.linear.x
    velocity_y = msg.twist.twist.linear.y
    velocity_z = msg.twist.twist.linear.z
    angular_velocity_x = msg.twist.twist.angular.x
    angular_velocity_y = msg.twist.twist.angular.y
    angular_velocity_z = msg.twist.twist.angular.z
    return {
        "time": time,
        "position_x": x,
        "position_y": y,
        "position_z": z,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw,
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "frame_id": frame_id,
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "velocity_z": velocity_z,
        "angular_velocity_x": angular_velocity_x,
        "angular_velocity_y": angular_velocity_y,
        "angular_velocity_z": angular_velocity_z,
    }


def people_msg_to_row(msg: People) -> Dict[str, Any]:
    time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    row = {"time": time}
    for person in msg.people:
        tags = dict(zip(person.tagnames, person.tags))
        behavior_tag = "behavior" if "behavior" in tags else "behaviour"
        row[person.name] = {
            "name": person.name,
            "id": tags["id"],
            "group_id": tags["group_id"],
            "behavior": tags[behavior_tag],
            "position_x": person.position.x,
            "position_y": person.position.y,
            "yaw": person.position.z,
            "velocity_x": person.velocity.x,
            "velocity_y": person.velocity.y,
            "angular_velocity_z": person.velocity.z,
        }
    return row


def path_msg_to_row(msg: NavPath) -> Dict[str, Any]:
    time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    poses = []
    for pose_stamped in msg.poses:
        x = pose_stamped.pose.position.x
        y = pose_stamped.pose.position.y
        z = pose_stamped.pose.position.z
        qx = pose_stamped.pose.orientation.x
        qy = pose_stamped.pose.orientation.y
        qz = pose_stamped.pose.orientation.z
        qw = pose_stamped.pose.orientation.w
        roll, pitch, yaw = euler_from_quaternion([qx, qy, qz, qw])
        poses.append(
            {
                "position_x": x,
                "position_y": y,
                "position_z": z,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }
        )
    return {"time": time, "poses": poses}


def tf_message_to_row(msg: TFMessage) -> Dict[str, Any]:
    if not msg.transforms:
        return {}
    time = msg.transforms[0].header.stamp.sec + msg.transforms[0].header.stamp.nanosec * 1e-9
    row = {"time": time}
    for transform_stamped in msg.transforms:
        translation = np.array(
            [
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z,
            ]
        )
        rotation = np.array(
            [
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
                transform_stamped.transform.rotation.w,
            ]
        )
        euler = euler_from_quaternion(rotation)
        transform_matrix = compose_matrix(translate=translation, angles=euler)
        key = f"{transform_stamped.header.frame_id} -> {transform_stamped.child_frame_id}"
        row[key] = {
            "frame_id": transform_stamped.header.frame_id,
            "child_frame_id": transform_stamped.child_frame_id,
            "transform": transform_matrix,
        }
    return row


def string_msg_to_row(msg: String) -> Dict[str, Any]:
    return {"data": msg.data}


def odometry_to_df(
    odometry_msgs: Iterable[Union[Odometry, Dict[str, Any]]]
) -> pd.DataFrame:
    """Convert Odometry data to a DataFrame.

    :param odometry_msgs: Iterable of Odometry messages or prepared row dicts
    :return: DataFrame containing the odometry data
    """
    series_list = []
    for msg in odometry_msgs:
        row = odometry_msg_to_row(msg) if isinstance(msg, Odometry) else msg
        if not row:
            continue
        series_list.append(pd.Series(row))
    return pd.DataFrame(series_list)


def people_to_df(
    people_msgs: Iterable[Union[People, Dict[str, Any]]]
) -> pd.DataFrame:
    """Convert People data to a DataFrame.

    :param people_msgs: Iterable of People messages or prepared row dicts
    :return: DataFrame containing the people data
    """
    people_names: Set[str] = set()
    series_list = []
    for msg in people_msgs:
        row = people_msg_to_row(msg) if isinstance(msg, People) else msg
        if not row:
            continue
        people_names.update(key for key in row.keys() if key != "time")
        series_list.append(pd.Series(row))
    if not series_list:
        return pd.DataFrame()
    df_people = pd.DataFrame(series_list)
    df = pd.DataFrame(columns=["time", *people_names])
    df.time = df.time.astype(float)
    df = df.combine_first(df_people)
    return df


def laser_scan_to_df(laser_scan_msgs: List[LaserScan]) -> pd.DataFrame:
    """Convert a list of LaserScan messages to a DataFrame.

    :param laser_scan_msgs: List of LaserScan messages
    :return: DataFrame containing the laser scan data
    """

    series_list = []
    for msg in laser_scan_msgs:
        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_increment = msg.angle_increment
        time_increment = msg.time_increment
        scan_time = msg.scan_time
        range_min = msg.range_min
        range_max = msg.range_max
        ranges = msg.ranges
        intensities = msg.intensities
        series_list.append(
            pd.Series(
                {
                    "time": time,
                    "angle_min": angle_min,
                    "angle_max": angle_max,
                    "angle_increment": angle_increment,
                    "time_increment": time_increment,
                    "scan_time": scan_time,
                    "range_min": range_min,
                    "range_max": range_max,
                    "ranges": ranges,
                    "intensities": intensities,
                }
            )
        )
    df = pd.DataFrame(series_list)
    return df


def path_to_df(path_msgs: Iterable[Union[NavPath, Dict[str, Any]]]) -> pd.DataFrame:
    """Convert Path data to a DataFrame."""

    series_list = []
    for msg in path_msgs:
        row = path_msg_to_row(msg) if isinstance(msg, NavPath) else msg
        if not row:
            continue
        series_list.append(pd.Series(row))
    return pd.DataFrame(series_list)


def tf2_messages_to_df(
    tf_messages: Iterable[Union[TFMessage, Dict[str, Any]]]
) -> pd.DataFrame:
    """Convert TFMessage data to a DataFrame."""

    transforms_set: Set[str] = set()
    series_list = []
    for msg in tf_messages:
        row = tf_message_to_row(msg) if isinstance(msg, TFMessage) else msg
        if not row:
            continue
        transforms_set.update(key for key in row.keys() if key != "time")
        series_list.append(pd.Series(row))
    if not series_list:
        return pd.DataFrame()
    df_transforms = pd.DataFrame(series_list)
    df = pd.DataFrame(columns=["time", *transforms_set])
    df = df.combine_first(df_transforms)
    df.sort_values(by="time", inplace=True, ignore_index=True)
    df.infer_objects(copy=False)
    df = ffill_dicts(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def string_to_df(
    string_msgs: Iterable[Union[String, Dict[str, Any]]]
) -> pd.DataFrame:
    """Convert string messages to a DataFrame."""

    series_list = []
    for msg in string_msgs:
        row = string_msg_to_row(msg) if isinstance(msg, String) else msg
        if not row:
            continue
        series_list.append(pd.Series(row))
    return pd.DataFrame(series_list)


def get_person_data(df: pd.DataFrame, person_name: str) -> pd.DataFrame:
    """Get the data of a person from a DataFrame containing people data.

    :param df: DataFrame containing people data
    :param person_name: The name of the person
    :return: DataFrame containing the data of the person
    """
    if person_name not in df.columns:
        warning(f"Person {person_name} not found in DataFrame columns.")
        return pd.DataFrame()
    df_person = df[["time", person_name]].copy()
    df_person.time = df_person.time.astype(float)
    df_person = df_person.dropna()
    df_person = df_person.reset_index(drop=True)
    df_person["position_x"] = df_person[person_name].apply(lambda x: x["position_x"])
    df_person["position_y"] = df_person[person_name].apply(lambda x: x["position_y"])
    df_person["yaw"] = df_person[person_name].apply(lambda x: x["yaw"])
    df_person["velocity_x"] = df_person[person_name].apply(lambda x: x["velocity_x"])
    df_person["velocity_y"] = df_person[person_name].apply(lambda x: x["velocity_y"])
    df_person["angular_velocity_z"] = df_person[person_name].apply(
        lambda x: x["angular_velocity_z"]
    )
    df_person["behavior"] = (
        df_person[person_name].apply(lambda x: x["behavior"]).astype(str)
    )
    df_person["group_id"] = (
        df_person[person_name].apply(lambda x: x["group_id"]).astype(str)
    )
    df_person["id"] = df_person[person_name].apply(lambda x: x["id"]).astype(str)
    df_person.drop(columns=[person_name], inplace=True)
    return df_person


def get_goal_position(pose_stamped_msgs: List[PoseStamped]) -> Tuple[float, float]:
    """Get the goal position from a list of PoseStamped messages.

    :param pose_stamped_msgs: List of PoseStamped messages
    :return: The goal position
    """
    goals = {
        (pose_stamped_msg.pose.position.x, pose_stamped_msg.pose.position.y)
        for pose_stamped_msg in pose_stamped_msgs
    }
    if len(goals) > 1:
        Warning("More than one goal position found. Returning the first one.")
    return goals.pop()


def get_first_timestamp(bag_path: OSPath) -> float:
    """Get the timestamp of the first message in the bag file.

    :param bag_path: The path to the bag file
    :return: The timestamp of the first message in the bag file
    """
    storage_options, converter_options = get_rosbag_options(bag_path)
    bag = rosbag2_py.SequentialReader()
    bag.open(storage_options, converter_options)
    return bag.read_next()[2] * 1e-9


def filter_df_by_time(
    df: pd.DataFrame, start_time: float = -np.inf, end_time: float = np.inf
) -> pd.DataFrame:
    return df[(df["time"] >= start_time) & (df["time"] <= end_time)]


def ffill_dicts(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills dictionary values in object columns row by row.
    This function is useful when a dictionary is spread across multiple rows.

    :param df: DataFrame with object columns containing dictionaries
    :return: DataFrame with forward-filled dictionary values
    """
    # Detect columns that contain dictionaries
    dict_cols = [
        col
        for col in df.select_dtypes(include="object").columns
        if df[col].dropna().apply(lambda x: isinstance(x, dict)).any()
    ]

    # Forward-fill dictionary columns
    for col in dict_cols:
        df[col] = df[col].ffill()

    return df
