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

from people_msgs.msg import People, Person
from nav_msgs.msg import Odometry, Path as NavPath
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from typing import List, Tuple, Union, Dict, Type

MsgType = Union[Odometry, People, LaserScan, PoseStamped, TFMessage, NavPath]


def get_bag_path(experiment: str, algorithm: str, base_path: str, index: int) -> OSPath:
    model_path = base_path + "/" + experiment + "/" + algorithm
    model_path = OSPath(model_path)
    bag_path = os.path.join(model_path, f"experiment_{index}", f"experiment_{index}_0.db3")
    return bag_path


def get_rosbag_options(path: OSPath, serialization_format: str = "cdr") -> Tuple:
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )
    return storage_options, converter_options


def get_messages_from_bag(bag_path: OSPath, topic: str) -> Tuple[List[MsgType], Type]:
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


def odometry_to_df(odometry_msgs: List[Odometry]) -> pd.DataFrame:
    series_list = []
    for msg in odometry_msgs:
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
        series_list.append(
            pd.Series(
                {
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
            )
        )
    df = pd.DataFrame(series_list)
    return df


def people_to_df(people_msgs: List[People]) -> pd.DataFrame:
    people_names = set()
    for msg in people_msgs:
        for person in msg.people:
            people_names.add(person.name)
    df = pd.DataFrame(columns=["time", *people_names])
    df.time = df.time.astype(float)
    series_list = []
    for msg in people_msgs:
        time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        people = []
        for person in msg.people:
            tags = dict(zip(person.tagnames, person.tags))
            people.append(
                {
                    "name": person.name,
                    "id": tags["id"],
                    "group_id": tags["group_id"],
                    "behaviour": tags["behaviour"],
                    "position_x": person.position.x,
                    "position_y": person.position.y,
                    "yaw": person.position.z,
                    "velocity_x": person.velocity.x,
                    "velocity_y": person.velocity.y,
                    "angular_velocity_z": person.velocity.z,
                }
            )
        series_list.append(
            pd.Series(
                {"time": time, **{person["name"]: person for person in people}},
            )
        )
    df_people = pd.DataFrame(series_list)
    df = df.combine_first(df_people)
    return df


def laser_scan_to_df(laser_scan_msgs: List[LaserScan]) -> pd.DataFrame:
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

def path_to_df(path_msgs: List[NavPath]) -> pd.DataFrame:
    series_list = []
    for msg in path_msgs:
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
        series_list.append(
            pd.Series(
                {
                    "time": time,
                    "poses": poses,
                }
            )
        )
    df = pd.DataFrame(series_list)
    return df


def tf2_messages_to_df(tf_messages: List[TFMessage]) -> pd.DataFrame:
    transforms_set = set()
    for msg in tf_messages:
        for transform_stamped in msg.transforms:
            transforms_set.add(
                f"{transform_stamped.header.frame_id} -> {transform_stamped.child_frame_id}"
            )
    df = pd.DataFrame(columns=["time", *transforms_set])
    series_list = []
    for msg in tf_messages:
        time = (
            msg.transforms[0].header.stamp.sec
            + msg.transforms[0].header.stamp.nanosec * 1e-9
        )
        transforms_list = []
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
            transforms_list.append(
                {
                    "frame_id": transform_stamped.header.frame_id,
                    "child_frame_id": transform_stamped.child_frame_id,
                    "transform": transform_matrix,
                }
            )
        series_list.append(
            pd.Series(
                {
                    "time": time,
                    **{
                        f'{transform["frame_id"]} -> {transform["child_frame_id"]}': transform
                        for transform in transforms_list
                    },
                }
            )
        )
    df_transforms = pd.DataFrame(series_list)
    df = df.combine_first(df_transforms)
    df.sort_values(by="time", inplace=True, ignore_index=True)
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_person_data(df: pd.DataFrame, person_name: str) -> pd.DataFrame:
    df_person = df[["time", person_name]]
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
    df_person["behaviour"] = (
        df_person[person_name].apply(lambda x: x["behaviour"]).astype(str)
    )
    df_person["group_id"] = (
        df_person[person_name].apply(lambda x: x["group_id"]).astype(str)
    )
    df_person["id"] = df_person[person_name].apply(lambda x: x["id"]).astype(str)
    df_person.drop(columns=[person_name], inplace=True)
    return df_person


def get_goal_position(pose_stamped_msgs: List[PoseStamped]) -> Tuple[float, float]:
    goals = {
        (pose_stamped_msg.pose.position.x, pose_stamped_msg.pose.position.y)
        for pose_stamped_msg in pose_stamped_msgs
    }
    if len(goals) > 1:
        Warning("More than one goal position found. Returning the first one.")
    return goals.pop()


def time_goal_reached(
    df: pd.DataFrame,
    goal_position: Tuple[float, float],
    threshold: float = 0.5,
    robot_radius: float = 0.3,
) -> Union[float, None]:
    df["distance_to_goal"] = (
        (df["position_x"] - goal_position[0]) ** 2
        + (df["position_y"] - goal_position[1]) ** 2
    ) ** 0.5
    # return the time when the goal was reached
    if df["distance_to_goal"].min() < threshold + robot_radius:
        return df[df["distance_to_goal"] < threshold + robot_radius].iloc[0]["time"]
    
def get_first_timestamp(bag_path: OSPath) -> float:
    storage_options, converter_options = get_rosbag_options(bag_path)
    bag = rosbag2_py.SequentialReader()
    bag.open(storage_options, converter_options)
    return bag.read_next()[2] * 1e-9 
    

def filter_df_by_time(
    df: pd.DataFrame, start_time: float = -np.inf, end_time: float = np.inf
) -> pd.DataFrame:
    return df[(df["time"] >= start_time) & (df["time"] <= end_time)]


def get_last_row_in_interval(
    df: pd.DataFrame, start_time: float, end_time: float
) -> pd.DataFrame:
    return filter_df_by_time(df, start_time, end_time).tail(1)


def filter_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    return df[columns]
