#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
from social_evaluation.common import path_contains
from social_evaluation.metrics_evaluation import metrics_dict
from social_evaluation.metrics_evaluation.metrics_utils import radius_for
import pandas as pd
from typing import Dict, Any, Union, List, Tuple, Optional
from pathlib import Path, PosixPath
import argcomplete, argparse
import yaml
import pickle
import json
import re
import numpy as np
from tqdm import tqdm

# Default Likert survey dimensions aggregated (mean over participants) per run.
# Override via the ``survey_fields`` key in the metrics YAML.
SURVEY_FIELDS = ["safety", "friendliness", "naturalness", "avoidance_foresight"]

# How a fleet's per-robot values of a single-robot metric become the fleet
# value reported under the metric's own name: the worst robot for distances
# (smallest), space intrusions and collisions (largest); the mean otherwise.
_FLEET_WORST_LOW = {"minimum_distance_to_people", "avg_distance_to_closest_person"}
_FLEET_WORST_HIGH_SUFFIXES = ("_space_intrusions", "_collision")


def fleet_value(metric: str, values: List[Any]) -> float:
    """The fleet value of ``metric`` from its per-robot values (NaN if none)."""
    numbers = np.array(
        [np.nan if v is None else v for v in values], dtype=float
    )
    if numbers.size == 0 or np.all(np.isnan(numbers)):
        return np.nan
    if metric in _FLEET_WORST_LOW:
        return float(np.nanmin(numbers))
    if metric.endswith(_FLEET_WORST_HIGH_SUFFIXES):
        return float(np.nanmax(numbers))
    return float(np.nanmean(numbers))


def _is_fleet_metric(metric: str) -> bool:
    """Metrics defined in metrics_evaluation/fleet.py already use every robot."""
    return metrics_dict[metric].__module__.endswith(".fleet")


def _run_id(path: Any, fallback: int) -> int:
    """The run number a bag path encodes (``.../experiment_<N>/...``).

    Numbering runs by their position in the list shifted every later run onto
    the wrong outcome as soon as one run was missing from the data."""
    found = re.findall(r"experiment_(\d+)", str(path))
    return int(found[-1]) if found else fallback


def _empty_run(run: Dict[str, Any]) -> bool:
    """A run with nothing to measure: no robot poses, or no pedestrian stamps
    (e.g. a run the harness cancelled a second after it started)."""
    people = run.get("people")
    if people is None or "time" not in getattr(people, "columns", ()):
        return True
    robots = run.get("robots") or {}
    odometry = run.get("odometry")
    has_poses = any(len(df) for df in robots.values()) or (
        odometry is not None and len(odometry) > 0
    )
    return not has_poses


def evaluate_algorithm(
    paths: List[PosixPath],
    experiment: str,
    algorithm: str,
    data: Dict[str, Any],
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluate the algorithm on the data

    :param paths: List of paths with the data
    :param data: Dictionary with the data
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return: Dataframe with the results of the evaluation for the algorithm
    """
    series = []
    for i, path in enumerate(
        tqdm(
            paths,
            desc=f"Evaluating scenarios for {experiment} with {algorithm}",
            leave=False,
        )
    ):
        if _empty_run(data[path]):
            # Keep the row, so the run still lines up with its outcome.
            tqdm.write(f"Warning: nothing to measure in {path}; metrics left NaN")
            results = pd.Series({m: np.nan for m in metrics["metrics"]}, dtype=float)
        else:
            results, _ = evaluate_selected_metrics(data[path], config, metrics)
        results["experiment"] = experiment
        results["algorithm"] = algorithm
        results["run"] = _run_id(path, i + 1)
        series.append(results)
    df = pd.DataFrame(series)
    return df


def evaluate_selected_metrics(
    data: Dict[str, Any], config: Dict[str, Any], metrics: Dict[str, Any]
) -> Tuple[pd.Series, pd.DataFrame]:
    """Evaluate the selected metrics on the data

    :param data: Dictionary with the data
    :param config: Dictionary with the configuration for the evaluation
    :param metrics: Dictionary with the metrics to evaluate
    :return: Series with the results of the evaluation
    """
    # Extract the data from the dictionary
    # print(data.keys())
    df_odometry = data["odometry"]
    df_people = data["people"]
    df_agents_list = data["agents"]
    df_obstacles = data["obstacles"]
    transforms = data["transforms"]
    goal = data["goal"]
    df_results = data["result"] if "result" in data else None
    df_robots = data.get("robots") or {}

    # check dtypes
    for df in df_agents_list:
        df.time = df.time.astype(float)
        df.behavior = df.behavior.astype(str)
        df.id = df.id.astype(str)
        df.group_id = df.group_id.astype(str)
    df_people.time = df_people.time.astype(float)

    metrics_list = metrics["metrics"]

    # Evaluate the metrics
    results, results_df = evaluate_metrics(
        df_odometry,
        df_people,
        df_agents_list,
        df_obstacles,
        transforms,
        goal,
        config,
        metrics_list,
        df_results=df_results,
        df_robots=df_robots,
    )

    return results, results_df


def evaluate_metrics(
    df_odometry: pd.DataFrame,
    df_people: pd.DataFrame,
    df_agents_list: List[pd.DataFrame],
    df_obstacles: pd.DataFrame,
    transforms: Dict[str, pd.DataFrame],
    goal: Union[np.ndarray, Tuple[float, float]],
    config: Dict[str, Any],
    metrics: List[str],
    df_results: Optional[pd.DataFrame] = None,
    df_robots: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Evaluate the metrics on the data

    :param df_odometry: Dataframe with the odometry data
    :param df_people: Dataframe with the people data
    :param df_agents_list: List of dataframes with the agents data
    :param df_obstacles: Dataframe with the obstacles data
    :param transforms: Dictionary with the transforms
    :param goal: Goal position
    :param config: Configuration for the evaluation
    :param metrics: List of metrics to evaluate
    :return: Series with the results of the evaluation
    """

    # Compute the metrics for the experiment. In a fleet run (two or more
    # robots) a single-robot metric is measured on EVERY robot, each with its
    # own radius: ``<metric>@<robot>`` per robot, and the fleet value under the
    # metric's own name (fleet_value). Before 2026-09-17 it was measured on the
    # first robot only.
    fleet = {name: df for name, df in (df_robots or {}).items() if not df.empty}
    context = dict(
        df_obstacles=df_obstacles,
        goal=goal,
        df_results=df_results,
        df_robots=df_robots,
    )
    metrics_results = {}
    metrics_df = {}
    for metric in metrics:
        function = metrics_dict[metric]
        if len(fleet) >= 2 and not _is_fleet_metric(metric):
            values, frames = {}, {}
            for name, df_robot in fleet.items():
                options = {**config, "robot_radius": radius_for(name, config)}
                result = function(df_agents_list, df_robot.copy(), **context, **options)
                values[name], frames[name] = _split_result(result)
                metrics_results[f"{metric}@{name}"] = values[name]
            metrics_results[metric] = fleet_value(metric, list(values.values()))
            metrics_df[metric] = frames
            continue

        robot_data = None
        if "map_to_base_link" in transforms:
            robot_data = transforms["map_to_base_link"].copy()
        else:
            robot_data = df_odometry.copy()
        result = function(df_agents_list, robot_data, **context, **config)
        metrics_results[metric], metrics_df[metric] = _split_result(result)

    metrics_results = pd.Series(metrics_results)
    return metrics_results, metrics_df


def _split_result(result: Any) -> Tuple[Any, pd.DataFrame]:
    """(scalar, series) from a metric's return value."""
    if isinstance(result, Tuple):
        return result[0], result[1]
    return result, pd.DataFrame()


def _aggregate_survey(raw: Any, survey_fields: List[str]) -> Dict[str, float]:
    """Mean Likert score per survey dimension over all participants of a run.

    :param raw: The ``survey_responses`` cell (a JSON list of participant dicts)
    :param survey_fields: Likert question ids to aggregate
    :return: ``{survey_<field>: mean}`` for each field (NaN if absent)
    """
    means = {f"survey_{field}": np.nan for field in survey_fields}
    if not isinstance(raw, str) or not raw.strip():
        return means
    try:
        responses = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return means
    if not responses:
        return means
    responses_df = pd.DataFrame(responses)
    for field in survey_fields:
        if field in responses_df.columns:
            values = pd.to_numeric(responses_df[field], errors="coerce")
            if values.notna().any():
                means[f"survey_{field}"] = float(values.mean())
    return means


def load_outcomes(
    outcomes_path: Path, survey_fields: List[str] = SURVEY_FIELDS
) -> pd.DataFrame:
    """Load ``result`` and aggregated survey scores from ``*_outcomes.csv`` file(s).

    :param outcomes_path: Directory (searched recursively) or a single CSV file
    :param survey_fields: Likert question ids to aggregate (mean per run)
    :return: One row per run, keyed by (experiment_tag, algorithm, run)
    """
    if outcomes_path.is_dir():
        csv_files = sorted(outcomes_path.rglob("*_outcomes.csv"))
    else:
        csv_files = [outcomes_path]
    rows: List[Dict[str, Any]] = []
    for csv_file in csv_files:
        outcomes_df = pd.read_csv(csv_file)
        # Match the benchmark loader convention: algorithm comes from the filename.
        algorithm = csv_file.stem.replace("_outcomes", "")
        for _, row in outcomes_df.iterrows():
            record: Dict[str, Any] = {
                "experiment_tag": str(row["experiment_tag"]),
                "algorithm": algorithm,
                "run": int(row["run_id"]),
                "result": row.get("result"),
            }
            record.update(_aggregate_survey(row.get("survey_responses"), survey_fields))
            rows.append(record)
    return pd.DataFrame(rows)


def merge_outcomes(results: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Left-join ``result`` + survey columns onto the per-run metrics.

    Metrics carry the full experiment name (e.g. ``crossing1/crossing1``); the
    outcomes are keyed by the experiment tag (the last path segment).
    """
    if results.empty or outcomes.empty:
        return results
    merged = results.copy()
    merged["experiment_tag"] = (
        merged["experiment"].astype(str).str.split("/").str[-1]
    )
    merged["run"] = merged["run"].astype(int)
    merged = merged.merge(
        outcomes, on=["experiment_tag", "algorithm", "run"], how="left"
    )
    merged.drop(columns=["experiment_tag"], inplace=True)
    return merged


def evaluate_data(
    data: Dict[str, Any], config: Dict[str, Any], metrics: Dict[str, Any]
) -> pd.DataFrame:
    """Evaluate the configured metrics over one loaded data dictionary.

    :param data: Extracted/filtered data keyed by bag path
    :param config: Parsed configuration (experiment_names, algorithms)
    :param metrics: Parsed metrics file (metrics_config + metric list)
    :return: DataFrame with one row per evaluated scenario
    """
    experiment_names = list(config["experiment_names"])
    algorithms = list(config["algorithms"])
    metrics_config = metrics["metrics_config"]

    # find all the paths that contain the experiment names and the algorithms
    paths_list = {}
    for experiment_name in experiment_names:
        for algorithm in algorithms:
            paths_list[experiment_name + ":" + algorithm] = [
                k
                for k in data.keys()
                if path_contains(k, experiment_name) and path_contains(k, algorithm)
            ]

    results_list = []
    for path in tqdm(paths_list, desc="Evaluating experiments", leave=False):
        experiment, algorithm = path.split(":")
        results_list.append(
            evaluate_algorithm(
                paths_list[path], experiment, algorithm, data, metrics_config, metrics
            )
        )
    return pd.concat(results_list, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate social metrics")
    parser.add_argument(
        "--config-file", type=Path, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--metrics-file", type=Path, required=True, help="Path to the metrics file"
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=None,
        help="Directory or CSV with *_outcomes.csv to merge (result + survey means) "
        "onto the metrics",
    )

    # Exactly one data source is required: a single pickle or a directory to scan.
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data-file", type=Path, help="Path to a single extracted/filtered data pickle"
    )
    input_group.add_argument(
        "--base-path",
        type=Path,
        help="Directory containing data pickle files (searched recursively)",
    )

    # At most one output target: an explicit file (single input) or a directory.
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Path to save the metrics (only valid with --data-file)",
    )
    output_group.add_argument(
        "--base-output-dir",
        type=Path,
        default=None,
        help="Directory to save one metrics pickle per input (default: alongside each input)",
    )
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    if args.output_file is not None and args.base_path is not None:
        parser.error("--output-file can only be used with --data-file, not --base-path.")

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    with open(args.metrics_file, "r") as file:
        metrics = yaml.safe_load(file)

    outcomes = None
    if args.outcomes is not None:
        if not args.outcomes.exists():
            parser.error(f"Outcomes path {args.outcomes} does not exist.")
        survey_fields = metrics.get("survey_fields") or SURVEY_FIELDS
        outcomes = load_outcomes(args.outcomes, survey_fields)
        if outcomes.empty:
            print(f"Warning: no *_outcomes.csv rows found under {args.outcomes}")

    def default_output(data_file: Path) -> Path:
        """Output path when no explicit --output-file is given."""
        if args.base_output_dir is not None:
            # Preserve the input's structure relative to --base-path so nested
            # files with the same name don't collide under the output directory.
            if args.base_path is not None:
                return args.base_output_dir / data_file.relative_to(args.base_path)
            return args.base_output_dir / data_file.name
        return data_file.with_name(data_file.name.replace(".pickle", "_metrics.pickle"))

    if args.base_path is not None:
        if not args.base_path.exists():
            parser.error(f"Base path {args.base_path} does not exist.")
        # Recurse to match the nested layout produced by bag_extraction and
        # data_filtering. Skip metrics outputs so re-runs don't re-evaluate them.
        data_files = sorted(
            f
            for f in args.base_path.rglob("*.pickle")
            if not f.name.endswith("_metrics.pickle")
        )
        if not data_files:
            parser.error(f"No .pickle files found under {args.base_path}")
        output_files = [default_output(data_file) for data_file in data_files]
    else:
        data_files = [args.data_file]
        output_files = [
            args.output_file
            if args.output_file is not None
            else default_output(args.data_file)
        ]

    # Evaluate metrics per input pickle and write one metrics pickle per input.
    for data_file, output_file in tqdm(
        zip(data_files, output_files), desc="Evaluating files", total=len(data_files)
    ):
        with open(data_file, "rb") as file:
            data = pickle.load(file)
        results = evaluate_data(data, config, metrics)
        if outcomes is not None:
            results = merge_outcomes(results, outcomes)
        if results.empty:
            print(
                f"Warning: no scenarios in {data_file} matched the configured "
                "experiment_names/algorithms; writing an empty metrics file."
            )
        print(f"Saving metrics to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as file:
            pickle.dump(results, file)


if __name__ == "__main__":
    main()
