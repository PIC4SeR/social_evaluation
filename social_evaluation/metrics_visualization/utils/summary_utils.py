import pickle
import pandas as pd
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional


def load_metrics(
    file: Path,
    experiment_name: Optional[str] = None,
    algorithm_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load metrics from a pickle file or csv file.

    Args:
        file (Path): Path to the pickle file containing metrics.
    Returns:
        pd.DataFrame: DataFrame containing the loaded metrics."""
    if not file.exists():
        raise FileNotFoundError(f"File {file} does not exist.")
    if file.suffix == ".pkl" or file.suffix == ".pickle":
        with open(file, "rb") as f:
            metrics = pickle.load(f)
    elif file.suffix == ".csv":
        metrics = pd.read_csv(file)
        metrics["algorithm"] = file.stem.replace("_outcomes", "")
    else:
        raise ValueError(f"Unsupported file format: {file.suffix}")
    if experiment_name:
        metrics["experiment"] = experiment_name
    if algorithm_name:
        metrics["algorithm"] = algorithm_name
    return metrics


def load_metrics_from_dir(
    dir_path: Union[str, Path],
    recursive: bool = False,
    use_experiment_folder_name: bool = False,
    use_algorithm_file_name: bool = False,
) -> pd.DataFrame:
    """
    Load metrics from a directory containing multiple metrics files.

    Args:
        dir_path (Union[str, Path]): Path to the directory containing metrics files.
        recursive (bool): Whether to search for files recursively in subdirectories.
        use_experiment_folder_name (bool): Whether to use the folder name as the experiment name.
        use_algorithm_file_name (bool): Whether to use the file name (without extension) as the algorithm name.
    Returns:
        pd.DataFrame: DataFrame containing all metrics loaded from the files.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise NotADirectoryError(
            f"Directory {dir_path} does not exist or is not a directory."
        )
    if recursive:
        dir_paths = [p for p in dir_path.rglob("*") if p.is_dir()]
    else:
        dir_paths = [dir_path]
    metrics_files = []
    for d in dir_paths:
        metrics_files.extend(
            list(d.glob("*.pkl")) + list(d.glob("*.pickle")) + list(d.glob("*.csv"))
        )
    all_metrics = []

    for file in metrics_files:
        if file.is_file():
            experiment_name = file.parent.name if use_experiment_folder_name else None
            algorithm_name = (
                file.stem.replace("_outcomes", "") if use_algorithm_file_name else None
            )
            metrics = load_metrics(
                file, experiment_name=experiment_name, algorithm_name=algorithm_name
            )
            all_metrics.append(metrics)

    if not all_metrics:
        raise ValueError(f"No metrics files found in directory: {dir_path}")
    all_metrics = pd.concat(all_metrics, ignore_index=True)
    return all_metrics


def get_mean_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the mean of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
    Returns:
        pd.DataFrame: DataFrame containing the mean metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=[col for col in metrics.columns if col not in req_columns],
        dropna=False,
        aggfunc="mean",
    ).reset_index()


def get_std_dev_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the standard deviation of metrics grouped by experiment and algorithm.
    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        Returns:
        pd.DataFrame: DataFrame containing the standard deviation of metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=[col for col in metrics.columns if col not in req_columns],
        dropna=False,
        aggfunc="std",
    ).reset_index()


def get_median_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the median of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
    Returns:
        pd.DataFrame: DataFrame containing the median metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=[col for col in metrics.columns if col not in req_columns],
        dropna=False,
        aggfunc="median",
    ).reset_index()


def get_max_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the maximum of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
    Returns:
        pd.DataFrame: DataFrame containing the maximum metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=[col for col in metrics.columns if col not in req_columns],
        dropna=False,
        aggfunc="max",
    ).reset_index()


def get_min_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Calculate the minimum of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
    Returns:
        pd.DataFrame: DataFrame containing the minimum metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=[col for col in metrics.columns if col not in req_columns],
        dropna=False,
        aggfunc="min",
    ).reset_index()


def get_metrics_summary(
    metrics: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Generate a summary of metrics including mean, std deviation, median, max, and min.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing summarized metrics.
    """
    return {
        "mean": get_mean_metrics(metrics),
        "std_dev": get_std_dev_metrics(metrics),
        "median": get_median_metrics(metrics),
        "max": get_max_metrics(metrics),
        "min": get_min_metrics(metrics),
    }


def filter_metrics_by_algorithm(metrics: pd.DataFrame, algorithm: str) -> pd.DataFrame:
    """
    Filter metrics by a specific algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        algorithm (str): The algorithm to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing metrics for the specified algorithm.
    """
    return metrics[metrics["algorithm"] == algorithm].reset_index(drop=True)


def filter_metrics_by_experiment(
    metrics: pd.DataFrame, experiment: str
) -> pd.DataFrame:
    """
    Filter metrics by a specific experiment.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiment (str): The experiment to filter by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing metrics for the specified experiment.
    """
    return metrics[metrics["experiment"] == experiment].reset_index(drop=True)


def filter_metrics_for_success(metrics: pd.DataFrame) -> pd.DataFrame:
    """Filter metrics for successful runs.
    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
    Returns:
        pd.DataFrame: Filtered DataFrame containing metrics for successful runs.
    """
    return metrics[metrics["result"] == "Success"].reset_index(drop=True)


def filter_metrics_for_failure(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Filter metrics for failed runs.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.

    Returns:
        pd.DataFrame: Filtered DataFrame containing metrics for failed runs.
    """
    return metrics[metrics["result"] != "Success"].reset_index(drop=True)


def select_metrics_by_columns(
    metrics: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Select specific columns from the metrics DataFrame.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        columns (List[str]): List of column names to select.

    Returns:
        pd.DataFrame: DataFrame containing only the selected columns.
    """

    # always include 'experiment', 'algorithm', 'run', and 'result' columns
    required_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        required_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    return metrics.copy()[required_columns + columns].reset_index(drop=True)


def rename_metrics_columns(
    metrics: pd.DataFrame, column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Rename columns in the metrics DataFrame.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        column_mapping (Dict[str, str]): Dictionary mapping old column names to new names.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    return metrics.rename(columns=column_mapping).reset_index(drop=True)


def parse_experiment_result(
    metrics: pd.DataFrame, result_column: str = "result"
) -> pd.DataFrame:
    """
    Parse the result column of the metrics DataFrame.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        result_column (str): Name of the column containing results.

    Returns:
        pd.DataFrame: DataFrame with parsed results.
    """
    if result_column not in metrics.columns:
        raise ValueError(f"Column '{result_column}' not found in metrics DataFrame.")

    metrics_ = metrics.copy()
    metrics_["success"] = metrics_[result_column].apply(
        lambda x: 1 if x == "Success" else 0
    )
    metrics_["timeout"] = metrics_[result_column].apply(
        lambda x: 1 if "Timeout" in x else 0
    )
    metrics_["collision"] = metrics_[result_column].apply(
        lambda x: 1 if "Collision" in x else 0
    )
    metrics_["navigation_failure"] = metrics_[result_column].apply(
        lambda x: 1 if "Navigation failure" in x else 0
    )

    return metrics_
