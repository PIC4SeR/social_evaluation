"""Regression tests for the metric bugs found comparing against hunav_evaluator."""

import numpy as np
import pandas as pd
import pytest

from social_evaluation.metrics_evaluation.metrics_utils import metrics, detect_groups
from social_evaluation.metrics_evaluation.sfm import (
    compute_obstacle_force,
    compute_social_force_two_agents,
    robot_velocity_to_world,
)

ROBOT_RADIUS = 0.27
AGENT_RADIUS = 0.17
TIME = np.round(np.arange(0.0, 10.0, 0.1), 3)


def _track(time=TIME, **columns):
    """A trajectory frame with the bag-extraction columns, constant unless given."""
    data = {
        "time": time,
        "position_x": 0.0,
        "position_y": 0.0,
        "yaw": 0.0,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "angular_velocity_z": 0.0,
    }
    data.update(columns)
    return pd.DataFrame(
        {k: np.broadcast_to(np.asarray(v, dtype=float), len(time)) for k, v in data.items()}
    )


def _person(group_id=-1, time=TIME, **columns):
    df = _track(time, **columns)
    df["group_id"] = str(group_id)
    return df


def _hunav_social_force(source_p, source_v, receiver_p, receiver_v):
    """hunav_evaluator's SFM.computeSocialForceTwoAgents(source, receiver), in numpy."""
    diff = source_p - receiver_p
    diff_direction = diff / np.linalg.norm(diff)
    interaction = 2.0 * (receiver_v - source_v) + diff_direction
    length = np.linalg.norm(interaction)
    direction = interaction / length
    a = np.arctan2(diff_direction[1], diff_direction[0])
    b = np.arctan2(direction[1], direction[0])
    theta = np.arctan2(np.sin(b - a), np.cos(b - a))  # shortest_angular_distance(a, b)
    B = 0.35 * length
    d = np.linalg.norm(diff)
    velocity_amount = -np.exp(-d / B - (3.0 * B * theta) ** 2)
    angle_amount = -np.sign(theta) * -np.exp(-d / B - (2.0 * B * theta) ** 2)
    left_normal = np.array([-direction[1], direction[0]])
    return 2.1 * (velocity_amount * direction + angle_amount * left_normal)


def _force_on(receiver_p, receiver_v, source_p, source_v):
    df = pd.DataFrame(
        [
            {
                "time": 0.0,
                "position_x_1": receiver_p[0],
                "position_y_1": receiver_p[1],
                "velocity_x_1": receiver_v[0],
                "velocity_y_1": receiver_v[1],
                "position_x_2": source_p[0],
                "position_y_2": source_p[1],
                "velocity_x_2": source_v[0],
                "velocity_y_2": source_v[1],
            }
        ]
    )
    return compute_social_force_two_agents(df)["social_force"].iloc[0]


ORIGIN = np.zeros(2)


@pytest.mark.parametrize("distance", [3.0, 10.0, 50.0])
def test_social_force_decays_with_distance(distance):
    force = _force_on(ORIGIN, ORIGIN, np.array([distance, 0.0]), ORIGIN)
    assert np.linalg.norm(force) < 1e-3


def test_social_force_is_larger_for_an_approaching_person():
    person = np.array([1.5, 0.0])
    approaching = _force_on(ORIGIN, ORIGIN, person, np.array([-1.0, 0.0]))
    receding = _force_on(ORIGIN, ORIGIN, person, np.array([1.0, 0.0]))
    assert np.linalg.norm(approaching) > 100 * np.linalg.norm(receding)


def test_social_force_matches_hunav_evaluator():
    rng = np.random.default_rng(0)
    for _ in range(200):
        receiver_p, source_p = rng.uniform(-5, 5, (2, 2))
        receiver_v, source_v = rng.uniform(-1.5, 1.5, (2, 2))
        if np.linalg.norm(source_p - receiver_p) < 0.3:
            continue
        np.testing.assert_allclose(
            _force_on(receiver_p, receiver_v, source_p, source_v),
            _hunav_social_force(source_p, source_v, receiver_p, receiver_v),
            atol=1e-6,
        )


def test_vanishing_interaction_vector_gives_no_force():
    # lambda * relative velocity cancels the unit direction exactly
    force = _force_on(ORIGIN, ORIGIN, np.array([2.0, 0.0]), np.array([0.5, 0.0]))
    np.testing.assert_array_equal(force, [0.0, 0.0])


def test_robot_velocity_is_rotated_into_the_world_frame():
    df = robot_velocity_to_world(_track(yaw=np.pi / 2, velocity_x=1.0))
    np.testing.assert_allclose(df[["velocity_x", "velocity_y"]].iloc[0], [0.0, 1.0], atol=1e-12)


def test_social_force_on_robot_uses_the_world_frame_velocity():
    # robot heading +y at 1 m/s (body frame); a person ahead feels more than one behind
    robot = _track(yaw=np.pi / 2, velocity_x=1.0)
    ahead, _ = metrics["social_force_on_robot"]([_person(position_y=1.5)], robot)
    behind, _ = metrics["social_force_on_robot"]([_person(position_y=-1.5)], robot)
    assert ahead > 10 * behind


@pytest.mark.parametrize("position", [(1.0, 0.0), (10.0, 5.0), (30.0, 20.0)])
def test_obstacle_force_does_not_depend_on_the_map_position(position):
    robot = _track(position_x=position[0], position_y=position[1])
    obstacles = pd.DataFrame({"x": [position[0] + 0.5], "y": [position[1]]})
    force = compute_obstacle_force(robot, obstacles, radius=ROBOT_RADIUS)["obstacle_force"]
    expected = 10 * np.exp(-(0.5 - ROBOT_RADIUS) / 0.2)
    np.testing.assert_allclose(np.stack(force), np.tile([-expected, 0.0], (len(TIME), 1)))


def _collisions(robot, person):
    options = dict(robot_radius=ROBOT_RADIUS, agent_radius=AGENT_RADIUS)
    robot_on_person, _ = metrics["robot_on_person_collision"]([person], robot, **options)
    person_on_robot, _ = metrics["person_on_robot_collision"]([person], robot, **options)
    return robot_on_person, person_on_robot


def test_robot_running_into_a_persons_back():
    # bodies overlap by 0.14 m; the person faces away, the robot faces the person
    robot_on_person, person_on_robot = _collisions(_track(), _person(position_x=0.30))
    assert (robot_on_person, person_on_robot) == (1, 0)  # one collision, however long


def test_person_walking_into_the_robots_side():
    robot = _track(yaw=np.pi / 2)
    person = _person(position_x=0.30, yaw=np.pi)
    assert _collisions(robot, person) == (0, 1)


def test_head_on_contact_counts_for_both():
    person = _person(position_x=0.30, yaw=np.pi)
    assert _collisions(_track(), person) == (1, 1)


def test_each_new_contact_is_a_collision():
    # touch for 3 s, part for 4 s, touch again for 3 s
    x = np.where((TIME < 3.0) | (TIME >= 7.0), 0.30, 1.0)
    assert _collisions(_track(), _person(position_x=x)) == (2, 0)


def test_bodies_apart_are_no_collision():
    assert _collisions(_track(), _person(position_x=0.50)) == (0, 0)


def test_heading_changes_wrap_around_pi():
    # heading west, the yaw jitters 0.01 rad across +-pi at every step
    yaw = np.pi - 0.005 + 0.01 * (np.arange(len(TIME)) % 2)
    total = metrics["cumulative_heading_changes"]([], _track(yaw=yaw))
    assert total == pytest.approx(0.01 * (len(TIME) - 1))


def test_time_not_moving_counts_each_interval_once():
    stopped, _ = metrics["time_not_moving"]([], _track(), period=5)
    assert stopped == pytest.approx(TIME[-1] - TIME[0])


def test_acceleration_averages_magnitudes():
    # 0 -> 1 m/s and back to 0, |a| = 1 / 4.9 s throughout
    speed = np.concatenate([np.linspace(0, 1, 50), np.linspace(1, 0, 50)])
    average, series = metrics["avg_acceleration"](
        [], _track(velocity_x=speed), rate_window=0.5, max_accel=1.0, smoothing_time=0.949
    )
    assert average == pytest.approx(1 / 4.9, rel=0.15)
    assert abs(series["acceleration"].mean()) < 0.5 * average  # the series stays signed


def _group_facing_robot(group_id, x, time=TIME):
    # two people side by side facing -x: their interaction centre is 1.5 m ahead of them
    return [
        _person(group_id, time, position_x=x, position_y=0.5, yaw=np.pi),
        _person(group_id, time, position_x=x, position_y=-0.5, yaw=np.pi),
    ]


def test_group_intrusion_measures_from_the_robots_body():
    # interaction centre 0.5 m from the robot's centre, 0.23 m from its body
    people = _group_facing_robot(1, 2.0)
    options = dict(robot_radius=ROBOT_RADIUS)
    intimate, _ = metrics["group_intimate_space_intrusions"](people, _track(), **options)
    personal, _ = metrics["group_personal_space_intrusions"](people, _track(), **options)
    assert (intimate, personal) == (100.0, 0.0)


def test_group_intrusions_align_groups_on_time():
    # group 1 is close for the first half only; group 2 is far, on its own timestamps
    near = np.where(TIME < 5.0, 2.0, 20.0)
    people = _group_facing_robot(1, near) + _group_facing_robot(2, -20.0, TIME + 0.05)
    options = dict(robot_radius=ROBOT_RADIUS)
    intimate, _ = metrics["group_intimate_space_intrusions"](people, _track(), **options)
    public, _ = metrics["group_public_space_intrusions"](people, _track(), **options)
    assert intimate == pytest.approx(50.0, abs=5.0)
    assert public == pytest.approx(50.0, abs=5.0)  # only once group 1 has left


def test_group_intrusions_use_the_closest_group():
    # group 1 puts the robot in its intimate space, group 2 in its personal space:
    # only the closest one counts, so the zones stay exclusive
    people = _group_facing_robot(1, 2.0) + _group_facing_robot(2, -2.5)
    for person in people[2:]:
        person["yaw"] = 0.0  # group 2 faces the robot from the other side
    options = dict(robot_radius=ROBOT_RADIUS)
    zones = {
        zone: metrics[f"group_{zone}_space_intrusions"](people, _track(), **options)[0]
        for zone in ("intimate", "personal", "social", "public")
    }
    assert zones == {"intimate": 100.0, "personal": 0.0, "social": 0.0, "public": 0.0}


def _robot_through(end_x):
    # the robot drives along x from 0 to end_x over the run
    return _track(position_x=np.linspace(0.0, end_x, len(TIME)), velocity_x=end_x / TIME[-1])


def test_time_to_reach_goal_stops_at_the_first_entry():
    robot = _robot_through(4.0)  # passes the goal at x = 2 and drives on
    time = metrics["time_to_reach_goal"]([], robot, goal=(2.0, 0.0), goal_tolerance=0.3)
    # first sample with |x - 2| < 0.3
    entry = TIME[np.argmax(np.abs(np.linspace(0.0, 4.0, len(TIME)) - 2.0) < 0.3)]
    assert time == pytest.approx(entry)


def test_time_to_reach_goal_is_the_whole_run_when_never_reached():
    robot = _robot_through(4.0)
    time = metrics["time_to_reach_goal"]([], robot, goal=(5.0, 0.0), goal_tolerance=0.3)
    assert time == pytest.approx(TIME[-1])


def test_completed_checks_the_final_position():
    robot = _robot_through(4.0)
    passed_through = metrics["completed"]([], robot, goal=(2.0, 0.0), goal_tolerance=0.3)
    ended_there = metrics["completed"]([], robot, goal=(4.1, 0.0), goal_tolerance=0.3)
    assert (passed_through, ended_there) == (False, True)


def test_goal_tolerance_is_on_the_centre_distance():
    # 0.35 m short of the goal: outside a 0.3 m tolerance whatever the robot's radius
    robot = _robot_through(4.0)
    done = metrics["completed"]([], robot, goal=(4.35, 0.0), goal_tolerance=0.3, robot_radius=0.326)
    assert done is False


def test_overlap_gives_a_negative_minimum_distance():
    person = _person(position_x=0.40)
    distance, _ = metrics["minimum_distance_to_people"](
        [person], _track(), robot_radius=ROBOT_RADIUS, agent_radius=0.3
    )
    assert distance == pytest.approx(0.40 - ROBOT_RADIUS - 0.3)


def test_detect_groups_leaves_the_frames_untouched():
    person = _person(3)
    person.loc[4, "velocity_x"] = np.nan
    assert detect_groups([person]) == [3]
    assert len(person) == len(TIME)


@pytest.mark.parametrize("dt", [0.1, 0.05])
def test_cumulative_metrics_do_not_depend_on_the_sampling_rate(dt):
    time = np.round(np.arange(0.0, 10.0 + 1e-9, dt), 3)
    robot, person = _track(time), _person(time=time, position_x=1.0)
    force = np.linalg.norm(_force_on(ORIGIN, ORIGIN, np.array([1.0, 0.0]), ORIGIN))
    on_robot, _ = metrics["social_force_on_robot"]([person], robot)
    assert on_robot == pytest.approx(10.0 * force, rel=1e-6)
    options = dict(df_obstacles=pd.DataFrame(), robot_radius=ROBOT_RADIUS)
    work, _ = metrics["social_work"]([person], robot, **options)
    per_second, _ = metrics["social_work_per_second"]([person], robot, **options)
    assert work == pytest.approx(10.0 * 2 * force, rel=1e-6)  # force on the robot + its dual
    assert per_second == pytest.approx(work / 10.0, rel=1e-6)


def test_robot_speed_is_the_full_speed():
    # a base sliding sideways at 0.5 m/s is moving, at 0.5 m/s
    robot = _track(velocity_x=0.0, velocity_y=0.5)
    speed, _ = metrics["avg_robot_linear_speed"]([], robot)
    stopped, _ = metrics["time_not_moving"]([], robot, min_velocity=0.08, min_angular_velocity=0.01)
    assert (speed, stopped) == (pytest.approx(0.5), 0.0)


def test_acceleration_and_jerk_do_not_depend_on_the_sampling_rate():
    # the same smooth speed profile recorded at 10 Hz and at 100 Hz
    values = {}
    for dt in (0.1, 0.01):
        time = np.round(np.arange(0.0, 10.0 + 1e-9, dt), 3)
        robot = _track(time, velocity_x=0.3 * (1 - np.cos(2 * np.pi * time / 10.0)))
        values[dt] = (
            metrics["avg_acceleration"]([], robot)[0],
            metrics["avg_overacceleration"]([], robot)[0],
        )
    np.testing.assert_allclose(values[0.01], values[0.1], rtol=0.1)
