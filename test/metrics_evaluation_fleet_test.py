"""Fleet runs: every robot measured, each with its own radius; runs keyed by
their bag path; nothing crashes on an empty run."""

import numpy as np
import pandas as pd
import pytest

import social_evaluation.metrics_evaluation_ as ME
from social_evaluation.common import experiment_for_bag
from social_evaluation.formation_timeseries_ import run_names
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
