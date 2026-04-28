from pathlib import Path

import pandas as pd

from social_evaluation.benchmark_report.base import BenchmarkReportBase
from social_evaluation.benchmark_report.config import BenchmarkConfig, format_metric_groups


def test_format_metric_groups_parses_aggregate_metric():
    metric_groups = {
        "metrics_social": {
            "Unobtrusiveness": {"name": r"\textbf{UO}", "direction": "UP"},
            "Friendiliness": {"name": r"\textbf{FR}", "direction": "UP"},
            "aggregate_metric": {
                "enabled": True,
                "key": "qualitative_score",
                "name": r"\textbf{QS}",
                "weights": {"Unobtrusiveness": 2.0},
            },
        }
    }

    (
        formatted,
        directions,
        discard_overrides,
        aggregate_specs,
        plot_metric_keys,
    ) = format_metric_groups(metric_groups)

    assert "qualitative_score" in formatted["metrics_social"]
    assert directions["qualitative_score"] == "UP"
    assert discard_overrides == {}
    assert aggregate_specs["qualitative_score"]["metrics"] == [
        "Unobtrusiveness",
        "Friendiliness",
    ]
    assert aggregate_specs["qualitative_score"]["weights"] == {
        "Unobtrusiveness": 2.0
    }
    assert plot_metric_keys == {}


def test_format_metric_groups_can_plot_only_the_aggregate():
    metric_groups = {
        "metrics_social": {
            "Unobtrusiveness": {"name": r"\textbf{UO}", "direction": "UP"},
            "Friendiliness": {"name": r"\textbf{FR}", "direction": "UP"},
            "aggregate_metric": {
                "enabled": True,
                "key": "qualitative_score",
                "name": r"\textbf{QS}",
                "plot_only": True,
            },
        }
    }

    _, _, _, aggregate_specs, plot_metric_keys = format_metric_groups(metric_groups)

    assert aggregate_specs["qualitative_score"]["plot_only"] is True
    assert plot_metric_keys == {"metrics_social": ["qualitative_score"]}


def test_format_metric_groups_can_plot_selected_metrics_including_aggregate():
    metric_groups = {
        "metrics_social": {
            "time_to_reach_goal": {"name": r"\textbf{TTG}", "direction": "DOWN"},
            "avg_distance_to_closest_person": {
                "name": r"\textbf{AMD}",
                "direction": "UP",
            },
            "Unobtrusiveness": {"name": r"\textbf{UO}", "direction": "UP"},
            "Friendiliness": {"name": r"\textbf{FR}", "direction": "UP"},
            "Smoothness": {"name": r"\textbf{SM}", "direction": "UP"},
            "Avoidance foresight": {"name": r"\textbf{AF}", "direction": "UP"},
            "aggregate_metric": {
                "enabled": True,
                "key": "human_evaluation",
                "name": r"\textbf{Human Evaluation}",
                "metrics": [
                    "Unobtrusiveness",
                    "Friendiliness",
                    "Smoothness",
                    "Avoidance foresight",
                ],
            },
            "plot_metrics": [
                "time_to_reach_goal",
                "avg_distance_to_closest_person",
                "human_evaluation",
            ],
        }
    }

    _, _, _, aggregate_specs, plot_metric_keys = format_metric_groups(metric_groups)

    assert aggregate_specs["human_evaluation"]["metrics"] == [
        "Unobtrusiveness",
        "Friendiliness",
        "Smoothness",
        "Avoidance foresight",
    ]
    assert plot_metric_keys == {
        "metrics_social": [
            "time_to_reach_goal",
            "avg_distance_to_closest_person",
            "human_evaluation",
        ]
    }


def test_format_metric_groups_parses_excluded_metrics_by_experiment():
    metric_groups = {
        "metrics_social": {
            "Unobtrusiveness": {"name": r"\textbf{UO}", "direction": "UP"},
            "Friendiliness": {"name": r"\textbf{FR}", "direction": "UP"},
            "Smoothness": {"name": r"\textbf{SM}", "direction": "UP"},
            "Avoidance foresight": {"name": r"\textbf{AF}", "direction": "UP"},
            "aggregate_metric": {
                "enabled": True,
                "key": "human_evaluation",
                "exclude_metrics_by_experiment": {
                    "advanced4": ["Unobtrusiveness", "unknown_metric"]
                },
            },
        }
    }

    _, _, _, aggregate_specs, _ = format_metric_groups(metric_groups)

    assert aggregate_specs["human_evaluation"]["exclude_metrics_by_experiment"] == {
        "advanced4": ["Unobtrusiveness"]
    }


def test_metric_group_aggregate_respects_metric_directions():
    config = BenchmarkConfig(
        root_dir=Path("."),
        scenario="demo",
        navigators=["A", "B"],
        episodes=["Exp"],
        display_episodes=["Exp"],
        navigator_names={"A": "A", "B": "B"},
        episode_names={"Exp": "Exp"},
        metric_groups={
            "metrics_social": {
                "Unobtrusiveness": "UO",
                "Avoidance foresight": "AF",
                "qualitative_score": "QS",
            }
        },
        metric_directions={
            "Unobtrusiveness": "UP",
            "Avoidance foresight": "DOWN",
            "qualitative_score": "UP",
        },
        metric_captions={},
        intrusion_labels={},
        metric_group_aggregates={
            "qualitative_score": {
                "group_key": "metrics_social",
                "metrics": ["Unobtrusiveness", "Avoidance foresight"],
                "weights": {},
                "direction": "UP",
                "normalize": "minmax",
                "require_complete": True,
            }
        },
    )

    metrics = pd.DataFrame(
        {
            "experiment": ["Exp", "Exp"],
            "algorithm": ["A", "B"],
            "run": [0, 0],
            "Unobtrusiveness": [0.0, 10.0],
            "Avoidance foresight": [10.0, 0.0],
        }
    )

    report = BenchmarkReportBase.__new__(BenchmarkReportBase)
    report.config = config
    report._all_results = metrics
    report._success_only = metrics
    report._derived_metric_results = {}

    aggregate_results = report._build_metric_group_aggregate_results()
    derived = aggregate_results["qualitative_score"].sort_values("algorithm")

    assert derived["qualitative_score"].tolist() == [0.0, 1.0]


def test_metric_group_aggregate_can_use_fixed_limits():
    config = BenchmarkConfig(
        root_dir=Path("."),
        scenario="demo",
        navigators=["A", "B"],
        episodes=["Exp"],
        display_episodes=["Exp"],
        navigator_names={"A": "A", "B": "B"},
        episode_names={"Exp": "Exp"},
        metric_groups={
            "metrics_social": {
                "Unobtrusiveness": "UO",
                "Friendiliness": "FR",
                "qualitative_score": "QS",
            }
        },
        metric_directions={
            "Unobtrusiveness": "UP",
            "Friendiliness": "UP",
            "qualitative_score": "UP",
        },
        metric_captions={},
        intrusion_labels={},
        metric_group_aggregates={
            "qualitative_score": {
                "group_key": "metrics_social",
                "metrics": ["Unobtrusiveness", "Friendiliness"],
                "weights": {},
                "direction": "UP",
                "normalize": "minmax",
                "require_complete": True,
                "limits": [1, 5],
            }
        },
    )

    metrics = pd.DataFrame(
        {
            "experiment": ["Exp", "Exp"],
            "algorithm": ["A", "B"],
            "run": [0, 0],
            "Unobtrusiveness": [1.0, 5.0],
            "Friendiliness": [3.0, 5.0],
        }
    )

    report = BenchmarkReportBase.__new__(BenchmarkReportBase)
    report.config = config
    report._all_results = metrics
    report._success_only = metrics
    report._derived_metric_results = {}

    aggregate_results = report._build_metric_group_aggregate_results()
    derived = aggregate_results["qualitative_score"].sort_values("algorithm")

    assert derived["qualitative_score"].tolist() == [0.25, 1.0]


def test_metric_group_aggregate_can_skip_normalization():
    config = BenchmarkConfig(
        root_dir=Path("."),
        scenario="demo",
        navigators=["A", "B"],
        episodes=["Exp"],
        display_episodes=["Exp"],
        navigator_names={"A": "A", "B": "B"},
        episode_names={"Exp": "Exp"},
        metric_groups={
            "metrics_social": {
                "Unobtrusiveness": "UO",
                "Avoidance foresight": "AF",
                "raw_score": "RS",
            }
        },
        metric_directions={
            "Unobtrusiveness": "UP",
            "Avoidance foresight": "DOWN",
            "raw_score": "UP",
        },
        metric_captions={},
        intrusion_labels={},
        metric_group_aggregates={
            "raw_score": {
                "group_key": "metrics_social",
                "metrics": ["Unobtrusiveness", "Avoidance foresight"],
                "weights": {},
                "direction": "UP",
                "normalize": "none",
                "require_complete": True,
            }
        },
    )

    metrics = pd.DataFrame(
        {
            "experiment": ["Exp", "Exp"],
            "algorithm": ["A", "B"],
            "run": [0, 0],
            "Unobtrusiveness": [4.0, 2.0],
            "Avoidance foresight": [1.0, 3.0],
        }
    )

    report = BenchmarkReportBase.__new__(BenchmarkReportBase)
    report.config = config
    report._all_results = metrics
    report._success_only = metrics
    report._derived_metric_results = {}

    aggregate_results = report._build_metric_group_aggregate_results()
    derived = aggregate_results["raw_score"].sort_values("algorithm")

    assert derived["raw_score"].tolist() == [1.5, -0.5]


def test_metric_group_aggregate_reports_missing_source_metrics():
    config = BenchmarkConfig(
        root_dir=Path("."),
        scenario="demo",
        navigators=["A", "B"],
        episodes=["Exp"],
        display_episodes=["Exp"],
        navigator_names={"A": "A", "B": "B"},
        episode_names={"Exp": "Exp"},
        metric_groups={
            "metrics_social": {
                "time_to_reach_goal": "TTG",
                "human_evaluation": "HE",
            }
        },
        metric_directions={
            "time_to_reach_goal": "DOWN",
            "human_evaluation": "UP",
        },
        metric_captions={},
        intrusion_labels={},
        metric_group_aggregates={
            "human_evaluation": {
                "group_key": "metrics_social",
                "metrics": ["Unobtrusiveness", "Friendiliness"],
                "weights": {},
                "direction": "UP",
                "normalize": "none",
                "require_complete": True,
            }
        },
    )

    metrics = pd.DataFrame(
        {
            "experiment": ["Exp", "Exp"],
            "algorithm": ["A", "B"],
            "run": [0, 0],
            "time_to_reach_goal": [1.0, 2.0],
        }
    )

    report = BenchmarkReportBase.__new__(BenchmarkReportBase)
    report.config = config
    report._all_results = metrics
    report._success_only = metrics
    report._derived_metric_results = {}
    report.aggregate_metric_warnings = []

    aggregate_results = report._build_metric_group_aggregate_results()

    assert aggregate_results == {}
    assert report.aggregate_metric_warnings == [
        "Aggregate metric 'human_evaluation' skipped because its source metrics "
        "are missing from the loaded results: Unobtrusiveness, Friendiliness"
    ]


def test_metric_group_aggregate_can_exclude_metric_for_specific_experiment():
    config = BenchmarkConfig(
        root_dir=Path("."),
        scenario="demo",
        navigators=["A", "B"],
        episodes=["Exp", "Advanced4"],
        display_episodes=["Exp", "Advanced4"],
        navigator_names={"A": "A", "B": "B"},
        episode_names={"Exp": "Exp", "Advanced4": "Advanced4"},
        metric_groups={
            "metrics_social": {
                "Unobtrusiveness": "UO",
                "Friendiliness": "FR",
                "Smoothness": "SM",
                "Avoidance foresight": "AF",
                "human_evaluation": "HE",
            }
        },
        metric_directions={
            "Unobtrusiveness": "UP",
            "Friendiliness": "UP",
            "Smoothness": "UP",
            "Avoidance foresight": "UP",
            "human_evaluation": "UP",
        },
        metric_captions={},
        intrusion_labels={},
        metric_group_aggregates={
            "human_evaluation": {
                "group_key": "metrics_social",
                "metrics": [
                    "Unobtrusiveness",
                    "Friendiliness",
                    "Smoothness",
                    "Avoidance foresight",
                ],
                "weights": {
                    "Unobtrusiveness": 1.0,
                    "Friendiliness": 1.0,
                    "Smoothness": 1.0,
                    "Avoidance foresight": 1.0,
                },
                "direction": "UP",
                "normalize": "none",
                "require_complete": True,
                "exclude_metrics_by_experiment": {
                    "Advanced4": ["Unobtrusiveness"]
                },
            }
        },
    )

    metrics = pd.DataFrame(
        {
            "experiment": ["Exp", "Advanced4"],
            "algorithm": ["A", "A"],
            "run": [0, 0],
            "Unobtrusiveness": [4.0, None],
            "Friendiliness": [2.0, 3.0],
            "Smoothness": [2.0, 4.0],
            "Avoidance foresight": [4.0, 5.0],
        }
    )

    report = BenchmarkReportBase.__new__(BenchmarkReportBase)
    report.config = config
    report._all_results = metrics
    report._success_only = metrics
    report._derived_metric_results = {}
    report.aggregate_metric_messages = []
    report.aggregate_metric_warnings = []

    aggregate_results = report._build_metric_group_aggregate_results()
    derived = aggregate_results["human_evaluation"].sort_values("experiment")

    assert derived["experiment"].tolist() == ["Advanced4", "Exp"]
    assert derived["human_evaluation"].tolist() == [4.0, 3.0]
    assert report.aggregate_metric_messages == [
        "Aggregate metric 'human_evaluation' applying experiment-specific exclusions: "
        "Advanced4: Unobtrusiveness"
    ]
