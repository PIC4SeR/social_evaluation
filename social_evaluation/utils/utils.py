from logging import warning
import pandas as pd

# get the path to the bag file
from pathlib import Path
import os
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tf_transformations import euler_from_quaternion, compose_matrix
import numpy as np

from people_msgs.msg import People, Person
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped


def get_bag_path(experiment, algorithm, base_path, index):
    model_path = base_path + "/" + experiment + "/" + algorithm
    model_path = Path(model_path)
    bag_path = os.path.join(model_path, f"scenario_{index}", f"scenario_{index}_0.db3")
    return bag_path


def get_rosbag_options(path, serialization_format="cdr"):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3")

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format,
    )
    return storage_options, converter_options


def get_messages_from_bag(bag_path, topic):
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
    return messages


def odometry_to_df(odometry_msgs):
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
                    "x": x,
                    "y": y,
                    "z": z,
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


def person_to_df(person_msgs):
    series_list = []
    for msg in person_msgs:
        name = msg.name
        position_x = msg.position.x
        position_y = msg.position.y
        yaw = msg.position.z
        velocity_x = msg.velocity.x
        velocity_y = msg.velocity.y
        angular_velocity_z = msg.velocity.z
        series_list.append(
            pd.Series(
                {
                    "name": name,
                    "position_x": position_x,
                    "position_y": position_y,
                    "yaw": yaw,
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                    "angular_velocity_z": angular_velocity_z,
                }
            )
        )
    df = pd.DataFrame(series_list)
    return df


def people_to_df(people_msgs):
    people_names = set()
    for msg in people_msgs:
        for person in msg.people:
            people_names.add(person.name)
    df = pd.DataFrame(columns=["time", *people_names])
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


def laser_scan_to_df(laser_scan_msgs):
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


def tf2_messages_to_df(tf_messages):
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
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_goal_position(pose_stamped_msgs):
    goals = {
        (pose_stamped_msg.pose.position.x, pose_stamped_msg.pose.position.y)
        for pose_stamped_msg in pose_stamped_msgs
    }
    if len(goals) > 1:
        Warning("More than one goal position found. Returning the first one.")
    return goals.pop()


def time_goal_reached(df, goal_position, threshold=0.5):
    df["distance_to_goal"] = (
        (df["x"] - goal_position[0]) ** 2 + (df["y"] - goal_position[1]) ** 2
    ) ** 0.5
    # return the time when the goal was reached
    if df["distance_to_goal"].min() < threshold:
        return df[df["distance_to_goal"] < threshold].iloc[0]["time"]


def filter_df_by_time(df, start_time=-np.inf, end_time=np.inf):
    return df[(df["time"] >= start_time) & (df["time"] <= end_time)]


def get_last_row_in_interval(df, start_time, end_time):
    return filter_df_by_time(df, start_time, end_time).tail(1)


def filter_columns(df, columns):
    return df[columns]
