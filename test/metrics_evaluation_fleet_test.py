"""Fleet runs: every robot measured, each with its own radius; runs keyed by
their bag path; nothing crashes on an empty run."""

import numpy as np
import pandas as pd
import pytest

import social_evaluation.metrics_evaluation_ as ME
from social_evaluation.common import experiment_for_bag
from social_evaluation.formation_timeseries_ import (
    plot_centroid, plot_timeseries, run_names, run_timeseries)
from social_evaluation.metrics_evaluation.metrics_utils import radius_for

from fleet_metrics_test import _agent_df, _robot_df

CONFIG = {
    "robot_radius": 0.27,
    "robot_radii": {"tb": 0.2, "jk": 0.3},
    "agent_radius": 0.17,
}
METRICS = {
    "metrics_config": CONFIG,
    "metrics": ["minimum_distance_to_people", "path_length",
                "fleet_min_distance_to_people"],
}


def _fleet():
    """tb_1 walks 1 m along y=0, jk walks 3 m along y=5; one pedestrian stands
    at (0, 2). Closest approaches: tb_1 2 m, jk 3 m (centre to centre)."""
    n = 11
    robots = {
        "tb_1": _robot_df(np.linspace(0.0, 1.0, n), np.zeros(n)),
        "jk": _robot_df(np.linspace(0.0, 3.0, n), np.full(n, 5.0)),
    }
    agents = [_agent_df(np.zeros(n), np.full(n, 2.0))]
    return robots, agents


def _run(robots, agents):
    first = next(iter(robots.values()))
    return {
        "odometry": first, "people": pd.DataFrame({"time": first["time"]}),
        "agents": agents, "obstacles": pd.DataFrame(), "transforms": {},
        "goal": (0.0, 0.0), "robots": robots,
    }


def test_radius_by_longest_name_prefix():
    config = {"robot_radius": 0.27, "robot_radii": {"tb": 0.2, "tb_big": 0.5}}
    assert radius_for("tb_1", config) == 0.2
    assert radius_for("tb_big_2", config) == 0.5
    assert radius_for("jackal", config) == 0.27


def test_every_robot_is_measured_with_its_own_radius():
    robots, agents = _fleet()
    results, _ = ME.evaluate_metrics(
        robots["tb_1"], pd.DataFrame({"time": [0.0]}), agents, pd.DataFrame(), {},
        (0.0, 0.0), CONFIG, METRICS["metrics"], df_robots=robots)
    assert results["minimum_distance_to_people@tb_1"] == pytest.approx(2.0 - 0.2 - 0.17)
    assert results["minimum_distance_to_people@jk"] == pytest.approx(3.0 - 0.3 - 0.17)
    # The fleet value is the worst robot for a distance...
    assert results["minimum_distance_to_people"] == pytest.approx(1.63)
    # ...and the mean for everything else.
    assert results["path_length@tb_1"] == pytest.approx(1.0)
    assert results["path_length@jk"] == pytest.approx(3.0)
    assert results["path_length"] == pytest.approx(2.0)
    # The fleet_* metric agrees, radii included, and gets no per-robot columns.
    assert results["fleet_min_distance_to_people"] == pytest.approx(1.63)
    assert "fleet_min_distance_to_people@jk" not in results


def test_a_single_robot_run_is_unchanged():
    robots, agents = _fleet()
    lone = {"tb_1": robots["tb_1"]}
    results, _ = ME.evaluate_metrics(
        lone["tb_1"], pd.DataFrame({"time": [0.0]}), agents, pd.DataFrame(), {},
        (0.0, 0.0), CONFIG, ["minimum_distance_to_people"], df_robots=lone)
    # One robot: the scalar fallback radius, as before, and no per-robot column.
    assert results["minimum_distance_to_people"] == pytest.approx(2.0 - 0.27 - 0.17)
    assert list(results.index) == ["minimum_distance_to_people"]


def test_fleet_reducers():
    assert ME.fleet_value("minimum_distance_to_people", [1.0, 0.2, None]) == 0.2
    assert ME.fleet_value("personal_space_intrusions", [10.0, 40.0]) == 40.0
    assert ME.fleet_value("robot_on_person_collision", [0, 2]) == 2.0
    assert ME.fleet_value("path_length", [1.0, 3.0]) == 2.0
    assert np.isnan(ME.fleet_value("path_length", [None, np.nan]))


def test_runs_keep_their_own_ids_and_empty_runs_their_row():
    """experiment_2 is missing and experiment_4 is empty: runs 3 and 4 must
    still meet their own outcomes."""
    robots, agents = _fleet()
    base = "/r/no_checkpoint/algo/scene/episode_1/algo"
    empty = {**_run({"tb_1": robots["tb_1"].iloc[0:0], "jk": robots["jk"].iloc[0:0]},
                    agents), "people": pd.DataFrame()}
    data = {
        f"{base}/experiment_1/experiment_1_0.db3": _run(robots, agents),
        f"{base}/experiment_3/experiment_3_0.db3": _run(robots, agents),
        f"{base}/experiment_4/experiment_4_0.db3": empty,
    }
    table = ME.evaluate_algorithm(list(data), "scene/episode_1", "algo", data,
                                  CONFIG, METRICS)
    assert list(table["run"]) == [1, 3, 4]
    assert np.isnan(table.loc[table["run"] == 4, "path_length"].item())
    outcomes = pd.DataFrame({
        "experiment_tag": ["episode_1"] * 4, "algorithm": ["algo"] * 4,
        "run": [1, 2, 3, 4],
        "result": ["Success", "Failure: Timeout", "Failure: Collision with robot",
                   "Failure: Navigation failure"],
    })
    merged = ME.merge_outcomes(table, outcomes).set_index("run")
    assert merged.loc[3, "result"] == "Failure: Collision with robot"
    assert merged.loc[4, "result"] == "Failure: Navigation failure"


def test_a_bag_path_names_its_experiment():
    names = ["scene_a/episode_1", "scene_a/episode_2", "scene_b/episode_1"]
    bag = "/r/algo/scene_a/episode_2/algo/experiment_3/experiment_3_0.db3"
    assert experiment_for_bag(bag, names) == "scene_a/episode_2"
    for candidates in (["scene_c/episode_1"], ["scene_a", "episode_2"],
                       ["scene_a", "scene_a/episode_2"]):
        with pytest.raises(ValueError):
            experiment_for_bag(bag, candidates)


def test_timeseries_names_gain_the_scenario_only_when_they_collide():
    tail = "episode_1/algo/experiment_1/experiment_1_0.db3"
    one = [f"/r/algo/scene_a/{tail}"]
    assert run_names(one)[one[0]] == "episode_1_algo_experiment_1"
    two = [f"/r/algo/scene_a/{tail}", f"/r/algo/scene_b/{tail}"]
    assert sorted(run_names(two).values()) == [
        "scene_a_episode_1_algo_experiment_1",
        "scene_b_episode_1_algo_experiment_1",
    ]


def test_timeseries_drift_is_the_distance_to_the_goal_when_one_is_given(tmp_path):
    """Formation tasks (crossing, passing): the centroid's distance to the goal; without a
    goal the drift stays relative to the first centroid sample."""
    robots, agents = _fleet()          # centroid from (0, 2.5) to (2, 2.5)
    run = _run(robots, agents)
    plain = run_timeseries(run)
    assert "goal_x" not in plain and plain["drift"].iloc[-1] == pytest.approx(2.0)
    to_goal = run_timeseries(run, goal=(4.0, 2.5))
    assert to_goal["drift"].iloc[0] == pytest.approx(4.0)
    assert to_goal["drift"].iloc[-1] == pytest.approx(2.0)
    assert (to_goal["goal_x"] == 4.0).all() and (to_goal["goal_y"] == 2.5).all()
    plot_timeseries(to_goal, "run", tmp_path / "ts.png", goal_tolerance=0.25)
    plot_centroid(to_goal, "run", tmp_path / "c.png")
    assert (tmp_path / "ts.png").stat().st_size > 0 and (tmp_path / "c.png").stat().st_size > 0


def test_centroid_final_goal_distance_is_the_last_centroid_to_goal_distance():
    """Formation tasks: final centroid (2, 2.5) against goal (4, 2.5) -> 2 m; NaN without a goal."""
    robots, agents = _fleet()
    args = (robots["tb_1"], pd.DataFrame({"time": [0.0]}), agents, pd.DataFrame(), {})
    results, _ = ME.evaluate_metrics(*args, (4.0, 2.5), CONFIG, ["centroid_final_goal_distance"],
                                     df_robots=robots)
    assert results["centroid_final_goal_distance"] == pytest.approx(2.0)
    results, _ = ME.evaluate_metrics(*args, None, CONFIG, ["centroid_final_goal_distance"],
                                     df_robots=robots)
    assert np.isnan(results["centroid_final_goal_distance"])


def test_intimate_personal_intrusions_are_the_sum_of_the_two_zones_on_the_worst_robot():
    """tb_1 passes a standing person with body gaps 0.13 (intimate), 0.63 and 1.13 (personal)
    and 2.63 m (social): 75 % = 25 + 50; jk stays far away, the fleet takes the worst robot."""
    xs = np.array([0.5, 1.0, 1.5, 3.0])           # gap = x - 0.2 (tb) - 0.17 (agent)
    robots = {"tb_1": _robot_df(xs, np.zeros(4)), "jk": _robot_df(xs, np.full(4, 10.0))}
    agents = [_agent_df(np.zeros(4), np.zeros(4))]
    names = ["intimate_space_intrusions", "personal_space_intrusions", "intimate_personal_space_intrusions"]
    results, _ = ME.evaluate_metrics(robots["tb_1"], pd.DataFrame({"time": [0.0]}), agents, pd.DataFrame(),
                                     {}, (0.0, 0.0), CONFIG, names, df_robots=robots)
    assert results["intimate_personal_space_intrusions@tb_1"] == pytest.approx(75.0)
    assert results["intimate_personal_space_intrusions@tb_1"] == pytest.approx(
        results["intimate_space_intrusions@tb_1"] + results["personal_space_intrusions@tb_1"])
    assert results["intimate_personal_space_intrusions@jk"] == 0.0
    assert results["intimate_personal_space_intrusions"] == pytest.approx(75.0)


def _yawed(xs, ys, yaw):
    df = _robot_df(xs, ys)
    df["yaw"] = yaw
    return df


def test_drift_metrics_follow_the_control_points_and_the_set_point():
    """Two robots parked at (0, 0) facing +x and (2, 0) facing +y: the body centroid is (1, 0),
    the control-point centroid (offset 0.1) is (1.05, 0.05). The set point (goal) is the
    reference of both the max and the final drift."""
    from social_evaluation.metrics_evaluation.fleet import centroid_drift_series, control_points
    robots = {"a": _yawed(np.zeros(3), np.zeros(3), 0.0), "b": _yawed(np.full(3, 2.0), np.zeros(3), np.pi / 2)}
    moved = control_points(robots, 0.1)
    assert moved["a"]["position_x"].iloc[0] == pytest.approx(0.1)
    assert moved["b"]["position_y"].iloc[0] == pytest.approx(0.1)
    assert control_points(robots, 0.0) is robots
    series = centroid_drift_series(robots, (1.0, 0.0), control_point_offset=0.1)
    assert series["drift"].iloc[-1] == pytest.approx(np.hypot(0.05, 0.05))
    config = {**CONFIG, "control_point_offset": 0.1}
    args = (robots["a"], pd.DataFrame({"time": [0.0]}), [_agent_df(np.full(3, 9.0), np.full(3, 9.0))],
            pd.DataFrame(), {})
    names = ["centroid_max_drift", "centroid_final_goal_distance"]
    results, _ = ME.evaluate_metrics(*args, (1.0, 0.0), config, names, df_robots=robots)
    assert results["centroid_final_goal_distance"] == pytest.approx(np.hypot(0.05, 0.05))
    assert results["centroid_max_drift"] == pytest.approx(np.hypot(0.05, 0.05))
    body, _ = ME.evaluate_metrics(*args, (1.0, 0.0), CONFIG, names, df_robots=robots)
    assert body["centroid_final_goal_distance"] == pytest.approx(0.0)


def test_centroid_time_to_goal_is_the_first_arrival_within_tolerance():
    from social_evaluation.metrics_evaluation.fleet import centroid_time_to_goal
    import numpy as np
    import pandas as pd
    time = np.arange(0.0, 10.01, 0.5)
    # two robots 1 m apart in x, their centroid moving along y from 0 to 5 m at 0.5 m/s
    robots = {name: pd.DataFrame({"time": time, "position_x": np.full_like(time, x),
                                  "position_y": 0.5 * time, "yaw": np.zeros_like(time)})
              for name, x in (("r1", -0.5), ("r2", 0.5))}
    value, _ = centroid_time_to_goal([], robots["r1"], df_robots=robots, goal=(0.0, 4.0),
                                     centroid_goal_tolerance=0.25)
    assert abs(value - 7.5) < 1e-9          # y = 3.75 at t = 7.5 s
    never, _ = centroid_time_to_goal([], robots["r1"], df_robots=robots, goal=(3.0, 4.0))
    assert np.isnan(never)
    assert np.isnan(centroid_time_to_goal([], robots["r1"], df_robots=robots, goal=None)[0])


def _intrusion_metrics(robots, agents):
    names = ["intimate_personal_space_intrusions", "fleet_intimate_personal_space_intrusions",
             "fleet_person_intimate_personal_space_intrusions"]
    results, _ = ME.evaluate_metrics(robots["tb_1"], pd.DataFrame({"time": [0.0]}), agents, pd.DataFrame(),
                                     {}, (0.0, 0.0), CONFIG, names, df_robots=robots)
    return [results[n] for n in names]


def test_fleet_intrusions_add_up_robots_intruding_at_different_times():
    """10 s: tb_1 next to person 0 for the first half, jk next to person 1 for the second.
    Each robot intrudes half the time (worst robot 50 %), the fleet all the time (100 %),
    each person half the time (50 %)."""
    t = np.round(np.arange(0.0, 10.0 + 1e-9, 0.1), 3)
    first = t < 5.0
    robots = {"tb_1": _robot_df(np.where(first, 0.5, 50.0), np.zeros(t.size), t),
              "jk": _robot_df(np.where(first, 50.0, 100.5), np.zeros(t.size), t)}
    agents = [_agent_df(np.zeros(t.size), np.zeros(t.size), times=t),
              _agent_df(np.full(t.size, 100.0), np.zeros(t.size), times=t)]
    worst_robot, fleet, person = _intrusion_metrics(robots, agents)
    assert worst_robot == pytest.approx(50.0, abs=1.5)
    assert fleet == pytest.approx(100.0, abs=1.5)
    assert person == pytest.approx(50.0, abs=1.5)


def test_fleet_intrusions_count_one_person_crowded_by_two_robots_once():
    """Both robots next to person 0 for the first half, person 1 never approached: the fleet
    intrudes half the time, the people on average a quarter (person 0 half, person 1 never)."""
    t = np.round(np.arange(0.0, 10.0 + 1e-9, 0.1), 3)
    first = t < 5.0
    robots = {"tb_1": _robot_df(np.where(first, 0.5, 50.0), np.zeros(t.size), t),
              "jk": _robot_df(np.where(first, -0.6, -50.0), np.zeros(t.size), t)}
    agents = [_agent_df(np.zeros(t.size), np.zeros(t.size), times=t),
              _agent_df(np.full(t.size, 100.0), np.zeros(t.size), times=t)]
    worst_robot, fleet, person = _intrusion_metrics(robots, agents)
    assert worst_robot == pytest.approx(50.0, abs=1.5)
    assert fleet == pytest.approx(50.0, abs=1.5)
    assert person == pytest.approx(25.0, abs=1.5)


def test_fleet_intrusions_need_fleet_data():
    agents = [_agent_df(np.zeros(3), np.zeros(3))]
    value, _ = ME.metrics_dict["fleet_intimate_personal_space_intrusions"](agents, _robot_df(np.zeros(3), np.zeros(3)))
    assert np.isnan(value)


def _outcomes_dir(tmp_path):
    """Two runs of one algorithm, with the pairing columns and survey responses."""
    import json
    responses = [
        [{"participant_id": "P01", "session_id": "S01", "safety": "4", "overall": "5"},
         {"participant_id": "P02", "session_id": "S01", "safety": "2", "overall": "3"}],
        [{"participant_id": "P01", "session_id": "S01", "safety": "5", "overall": "x"}],
    ]
    table = pd.DataFrame({
        "experiment_tag": ["crossing1", "crossing1"], "run_id": [1, 2],
        "result": ["Success", "Failure: Timeout"], "session_id": ["S01", "S01"],
        "ped_draw": ["s11_e1", "s11_e2"], "unrelated": [7, 8],
        "survey_responses": [json.dumps(r) for r in responses]})
    (tmp_path / "algo").mkdir()
    table.to_csv(tmp_path / "algo" / "algo_outcomes.csv", index=False)
    return tmp_path


def test_outcomes_carry_the_pairing_columns(tmp_path):
    outcomes = ME.load_outcomes(_outcomes_dir(tmp_path)).set_index("run")
    assert outcomes.loc[1, "session_id"] == "S01" and outcomes.loc[2, "ped_draw"] == "s11_e2"
    assert "unrelated" not in outcomes and "ped_models" not in outcomes   # listed and present only
    assert outcomes.loc[1, "survey_safety"] == 3.0                         # per-run mean kept


def test_survey_responses_become_one_row_per_participant(tmp_path):
    survey = ME.load_survey_responses(_outcomes_dir(tmp_path))
    assert list(zip(survey["run"], survey["participant_id"])) == [(1, "P01"), (1, "P02"), (2, "P01")]
    assert list(survey["survey_safety"]) == [4, 2, 5]
    assert np.isnan(survey["survey_overall"].iloc[2])                      # non-numeric answer
    results = pd.DataFrame({"experiment": ["crossing1/crossing1"] * 2, "algorithm": ["algo"] * 2,
                            "run": [1, 2], "path_length": [3.0, 4.0]})
    expanded = ME.expand_survey(results, survey)
    assert set(expanded["experiment"]) == {"crossing1/crossing1"}
    assert "path_length" not in expanded and len(expanded) == 3
    assert ME.expand_survey(results.iloc[0:0], survey).empty
