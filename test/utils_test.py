import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


def _stub_module(name: str, **attrs):
    module = types.ModuleType(name)
    for attr_name, attr_value in attrs.items():
        setattr(module, attr_name, attr_value)
    sys.modules[name] = module
    return module


def _load_utils_module():
    _stub_module("rosbag2_py")

    _stub_module("rclpy")
    _stub_module("rclpy.serialization", deserialize_message=lambda *args, **kwargs: None)

    _stub_module("rosidl_runtime_py")
    _stub_module("rosidl_runtime_py.utilities", get_message=lambda *args, **kwargs: None)

    _stub_module(
        "tf_transformations",
        euler_from_quaternion=lambda *args, **kwargs: (0.0, 0.0, 0.0),
        compose_matrix=lambda *args, **kwargs: None,
    )

    _stub_module("people_msgs")
    _stub_module("people_msgs.msg", People=type("People", (), {}), Person=type("Person", (), {}))

    _stub_module("nav_msgs")
    _stub_module("nav_msgs.msg", Odometry=type("Odometry", (), {}), Path=type("Path", (), {}))

    _stub_module("tf2_msgs")
    _stub_module("tf2_msgs.msg", TFMessage=type("TFMessage", (), {}))

    _stub_module("sensor_msgs")
    _stub_module("sensor_msgs.msg", LaserScan=type("LaserScan", (), {}))

    _stub_module("geometry_msgs")
    _stub_module("geometry_msgs.msg", PoseStamped=type("PoseStamped", (), {}))

    utils_path = Path(__file__).resolve().parents[1] / "social_evaluation" / "common" / "utils.py"
    spec = importlib.util.spec_from_file_location("social_evaluation_test_utils_module", utils_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_align_df_agents_on_time_deduplicates_before_reindex():
    utils = _load_utils_module()

    duplicated_times = pd.DataFrame(
        {
            "time": [0.0, 1.0, 1.0, 2.0],
            "position_x": [0.0, 1.0, 1.5, 2.0],
            "position_y": [0.0, 1.0, 1.5, 2.0],
        }
    )
    sparse_times = pd.DataFrame(
        {
            "time": [0.0, 2.0],
            "position_x": [10.0, 12.0],
            "position_y": [20.0, 22.0],
        }
    )

    aligned = utils.align_df_agents_on_time([duplicated_times, sparse_times])

    assert [df["time"].tolist() for df in aligned] == [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
    assert aligned[0].loc[aligned[0]["time"] == 1.0, "position_x"].item() == 1.5
    assert aligned[1].loc[aligned[1]["time"] == 1.0, "position_x"].item() == 11.0


def test_combine_df_robot_agents_sorts_before_interpolation():
    utils = _load_utils_module()

    df_robot = pd.DataFrame(
        {
            "time": [2.0, 0.0, 1.0],
            "position_x": [2.0, 0.0, 1.0],
            "position_y": [2.0, 0.0, 1.0],
            "yaw": [0.2, 0.0, 0.1],
        }
    )
    df_agent = pd.DataFrame(
        {
            "time": [1.0, 0.0, 2.0],
            "position_x": [11.0, 10.0, 12.0],
            "position_y": [21.0, 20.0, 22.0],
            "yaw": [0.0, 0.0, 0.0],
        }
    )

    combined = utils.combine_df_robot_agents(df_robot, [df_agent])

    assert combined["time"].tolist() == [0.0, 1.0, 2.0]
    assert combined["time"].is_monotonic_increasing
