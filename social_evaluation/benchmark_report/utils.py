import math
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def export_latex_tabular(
    report_latex: pd.DataFrame,
    column_format: str = "lcccc",
    header: list = ["Precision", "Recall", "F1-score", "Support"],
    float_format="%.2f",
    bold_rows: bool = True,
    multirow: bool = False,
):
    processed_header = header
    if isinstance(header, (list, tuple)):
        if len(header) != len(report_latex.columns):
            processed_header = list(report_latex.columns)
        else:
            processed_header = [
                h.replace("{", "{{").replace("}", "}}") if isinstance(h, str) else h
                for h in header
            ]

    try:
        latex_tabular = report_latex.to_latex(
            column_format=column_format,
            header=processed_header,
            float_format=float_format,
            bold_rows=bold_rows,
            escape=False,
            na_rep="-",
            multicolumn_format="c",
        )
    except KeyError:
        if bold_rows:
            latex_tabular = report_latex.to_latex(
                column_format=column_format,
                header=processed_header,
                float_format=float_format,
                bold_rows=False,
                escape=False,
                na_rep="-",
                multicolumn_format="c",
            )
        else:
            raise
    # handle multirows
    if multirow:
        latex_tabular = latex_tabular.replace(
            "\\begin{tabular}{", "\\begin{tabular}{ll"
        )

    # handle nans
    latex_tabular = latex_tabular.replace("nan", "-")
    latex_tabular = latex_tabular.replace(r" \pm -", "")
    latex_tabular = latex_tabular.replace(r"- \pm -", "-")

    if multirow:
        index_names = list(report_latex.index.names)
        if len(index_names) == 2:
            new_header = " & ".join(index_names + processed_header) + r" \\"  # type: ignore
            # replace the first two lines of the latex_tabular with the new_header
            lines = latex_tabular.split("\n")
            # find the line that starts with \toprule
            top_rule_index = next(
                i for i, line in enumerate(lines) if line.startswith(r"\toprule")
            )
            # replace the line after \toprule with new_header
            lines[top_rule_index + 1] = new_header
            # remove the line after that (the one with the multirow)
            lines.pop(top_rule_index + 2)
            latex_tabular = "\n".join(lines)
        elif len(index_names) == 1:
            new_header = " & ".join(index_names + processed_header) + r" \\"  # type: ignore
            # replace the first two lines of the latex_tabular with the new_header
            lines = latex_tabular.split("\n")
            # find the line that starts with \toprule
            top_rule_index = next(
                i for i, line in enumerate(lines) if line.startswith(r"\toprule")
            )
            # replace the line after \toprule with new_header
            lines[top_rule_index + 1] = new_header
            latex_tabular = "\n".join(lines)
        else:
            raise ValueError("Multirow is only supported for 1 or 2 index levels.")

        # replace the \cline{1-{str(len(report_latex.columns))}} with \midrule
        latex_tabular = latex_tabular.replace(
            rf"\cline{{1-{str(len(report_latex.columns))}}}", r"\midrule"
        )
        # remove last \midrule
        if latex_tabular.strip().endswith(r"\midrule"):
            latex_tabular = latex_tabular.strip()[: -len(r"\midrule")].strip()
    return latex_tabular


def export_latex_table(
    report_latex: pd.DataFrame,
    filename: str,
    column_format: str = "lcccc",
    header: list = ["Precision", "Recall", "F1-score", "Support"],
    float_format: str = "%.2f",
    bold_rows: bool = True,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    scale: Optional[float] = 0.71,
    multirow: bool = False,
    colorbar_path: Optional[Union[str, Path]] = None,
    colorbar_width: Optional[str] = "0.45\\linewidth",
    colorbar_caption: Optional[str] = "Color scale",
):
    """
    Exports a styled DataFrame to a LaTeX table.
    """
    latex_tabular = export_latex_tabular(
        report_latex,
        column_format=column_format,
        header=header,
        float_format=float_format,
        bold_rows=bold_rows,
        multirow=multirow,
    )

    with open(filename, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n\t")
        f.write(r"\centering" + "\n\t")
        caption_text = caption or ""
        if colorbar_path:
            colorbar_path_obj = Path(colorbar_path)
            colorbar_caption_text = colorbar_caption or ""
            colorbar_snippet = rf"\newline\textit{{{colorbar_caption_text}: }}\includegraphics[width={colorbar_width}]{{{colorbar_path_obj.as_posix()}}}"
            caption_text = (
                f"{caption_text}{colorbar_snippet}"
                if caption_text
                else colorbar_snippet
            )
        if caption_text:
            f.write(r"\caption{" + caption_text + "}" + "\n\t")
        if label:
            f.write(r"\label{" + label + "}" + "\n\t")
        if scale:
            f.write(r"\scalebox{" + str(scale) + "}{" + "\n\t")
        # indent all lines of latex_tabular by one tab
        # latex_tabular = "\n".join(["\t" + line for line in latex_tabular.split("\n")])
        latex_tabular = "\n".join(["\t" + line for line in latex_tabular.split("\n")])
        f.write(latex_tabular)
        if scale:
            f.write("}" + "\n")
        f.write(r"\end{table}" + "\n")

    return latex_tabular


def style_report_latex(
    mean_df: pd.DataFrame,
    cmap_name: Optional[str] = "Blues",
    alpha: float = 0.7,
    std_dev_df: Optional[Union[float, pd.Series, pd.DataFrame]] = None,
    metric_directions: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    from matplotlib import cm

    styled_df = mean_df.copy().astype(object)

    def _resolve_direction(column: str) -> str:
        if metric_directions and column in metric_directions:
            return metric_directions[column].upper()
        lowered = column.lower()
        if "down" in lowered:
            return "DOWN"
        return "UP"

    def _get_std_cell(row, col):
        if std_dev_df is None:
            return None
        if isinstance(std_dev_df, pd.DataFrame):
            if row in std_dev_df.index and col in std_dev_df.columns:
                return std_dev_df.loc[row, col]
            return None
        if isinstance(std_dev_df, pd.Series):
            if row in std_dev_df.index:
                return std_dev_df.loc[row]
            if col in std_dev_df.index:
                return std_dev_df.loc[col]
            return None
        return std_dev_df

    def _format_plain_cell(
        value: Any, std_value: Any, *, bold: bool, underline: bool
    ) -> str:
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            value_float = float("nan")
        if math.isnan(value_float):
            return "-"
        value_text = f"{value_float:.2f}"
        std_text = ""
        if std_value is not None:
            try:
                std_float = float(std_value)
            except (TypeError, ValueError):
                std_float = float("nan")
            if not math.isnan(std_float) and std_float != 0:
                std_text = rf" \pm {std_float:.2f}"
        value_text = rf"{value_text}{std_text if std_text else ''}"
        if bold:
            value_text = rf"\mathbf{{{value_text}}}"
        if underline:
            value_text = rf"\underline{{{value_text}}}"
        return rf"${value_text}$"

    if not cmap_name:
        for col in mean_df.columns:
            direction = _resolve_direction(col)
            numeric_series = pd.to_numeric(mean_df[col], errors="coerce")
            highlight_rows = set()
            underline_rows = set()
            if not numeric_series.dropna().empty:
                if direction == "DOWN":
                    best_value = numeric_series.min()
                    second_best_value = numeric_series[
                        numeric_series != best_value
                    ].min()

                else:
                    best_value = numeric_series.max()
                    second_best_value = numeric_series[
                        numeric_series != best_value
                    ].max()
                try:
                    best_float = float(best_value)
                    second_best_float = float(second_best_value)
                except (TypeError, ValueError):
                    best_float = float("nan")
                    second_best_float = float("nan")
                if not math.isnan(best_float) and not math.isnan(second_best_float):
                    for row, candidate in numeric_series.items():
                        if pd.isna(candidate):
                            continue
                        if math.isclose(
                            float(candidate),
                            best_float,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ):
                            highlight_rows.add(row)
                        elif math.isclose(
                            float(candidate),
                            second_best_float,
                            rel_tol=1e-9,
                            abs_tol=1e-12,
                        ):
                            underline_rows.add(row)
            for row in mean_df.index:
                std_cell = _get_std_cell(row, col)
                bold = row in highlight_rows
                underline = row in underline_rows
                styled_df.loc[row, col] = _format_plain_cell(
                    mean_df.loc[row, col], std_cell, bold=bold, underline=underline
                )
        return styled_df

    cmap = cm.get_cmap(cmap_name)
    norms = []
    for col in mean_df.columns:
        if "%" in col or "Success Rate" in col or "SR" in col:
            norms.append(plt.Normalize(0, 100))  # type: ignore
        else:
            norms.append(plt.Normalize(mean_df[col].min(), mean_df[col].max()))  # type: ignore

    def blend_with_white(rgb, alpha_value):
        return tuple(alpha_value * c + (1 - alpha_value) * 1.0 for c in rgb)

    for row in mean_df.index:
        for col_idx, col in enumerate(mean_df.columns):
            val = mean_df.loc[row, col]
            direction = _resolve_direction(col)
            try:
                norm_val = norms[col_idx](val)
                if direction == "DOWN":
                    norm_val = 1 - norm_val
            except Exception:
                norm_val = 0.5
            rgb = cmap(norm_val)[:3]
            rgb = blend_with_white(rgb, alpha)
            color = f"{rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f}"
            try:
                styled_val = f"{float(val):.2f}"
            except Exception:
                styled_val = "-"
            styled_df.loc[row, col] = rf"\cellcolor[rgb]{{{color}}} {styled_val}"
            std_cell = _get_std_cell(row, col)
            if std_dev_df is not None and std_cell is not None:
                try:
                    std_float = float(std_cell)
                except (TypeError, ValueError):
                    std_float = float("nan")
                if math.isnan(std_float):
                    styled_std = "-"
                elif std_float == 0:
                    styled_std = ""
                else:
                    styled_std = rf"\pm {std_float:.2f}"
                styled_df.loc[row, col] = (
                    rf"\cellcolor[rgb]{{{color}}} ${styled_val} {styled_std}$"
                )
            elif std_dev_df is not None:
                styled_df.loc[row, col] = rf"\cellcolor[rgb]{{{color}}} ${styled_val}$"

    return styled_df


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
        # Always include the root directory so flat pickle exports are discovered.
        dir_paths = [dir_path]
        dir_paths.extend(p for p in dir_path.rglob("*") if p.is_dir())
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


def _prepare_numeric_metric_columns(
    metrics: pd.DataFrame, req_columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a DataFrame copy (if needed) and numeric metric columns ready for aggregation."""
    value_columns: List[str] = []
    mutated = False
    for col in metrics.columns:
        if col in req_columns:
            continue
        series = metrics[col]
        if pd.api.types.is_numeric_dtype(series):
            value_columns.append(col)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().any():
            if not mutated:
                metrics = metrics.copy()
                mutated = True
            metrics[col] = coerced
            value_columns.append(col)
    return metrics, value_columns


def get_mean_metrics(
    metrics: pd.DataFrame, experiments: Optional[List[str]]
) -> pd.DataFrame:
    """Calculate the mean of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
    Returns:
        pd.DataFrame: DataFrame containing the mean metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if experiments is not None:
        metrics = metrics[metrics["experiment"].isin(experiments)]
    metrics, value_columns = _prepare_numeric_metric_columns(metrics, req_columns)
    if not value_columns:
        raise ValueError("No numeric metric columns available for aggregation.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=value_columns,
        dropna=False,
        aggfunc="mean",
    ).reset_index()


def get_std_dev_metrics(
    metrics: pd.DataFrame, experiments: Optional[List[str]]
) -> pd.DataFrame:
    """Calculate the standard deviation of metrics grouped by experiment and algorithm.
    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
    Returns:
        pd.DataFrame: DataFrame containing the standard deviation of metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if experiments is not None:
        metrics = metrics[metrics["experiment"].isin(experiments)]
    metrics, value_columns = _prepare_numeric_metric_columns(metrics, req_columns)
    if not value_columns:
        raise ValueError("No numeric metric columns available for aggregation.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=value_columns,
        dropna=False,
        aggfunc="std",
    ).reset_index()


def get_median_metrics(
    metrics: pd.DataFrame, experiments: Optional[List[str]]
) -> pd.DataFrame:
    """Calculate the median of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
    Returns:
        pd.DataFrame: DataFrame containing the median metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if experiments is not None:
        metrics = metrics[metrics["experiment"].isin(experiments)]
    metrics, value_columns = _prepare_numeric_metric_columns(metrics, req_columns)
    if not value_columns:
        raise ValueError("No numeric metric columns available for aggregation.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=value_columns,
        dropna=False,
        aggfunc="median",
    ).reset_index()


def get_max_metrics(
    metrics: pd.DataFrame, experiments: Optional[List[str]]
) -> pd.DataFrame:
    """Calculate the maximum of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
    Returns:
        pd.DataFrame: DataFrame containing the maximum metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if experiments is not None:
        metrics = metrics[metrics["experiment"].isin(experiments)]
    metrics, value_columns = _prepare_numeric_metric_columns(metrics, req_columns)
    if not value_columns:
        raise ValueError("No numeric metric columns available for aggregation.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=value_columns,
        dropna=False,
        aggfunc="max",
    ).reset_index()


def get_min_metrics(
    metrics: pd.DataFrame, experiments: Optional[List[str]]
) -> pd.DataFrame:
    """Calculate the minimum of metrics grouped by experiment and algorithm.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
    Returns:
        pd.DataFrame: DataFrame containing the minimum metrics grouped by experiment and algorithm.
    """
    req_columns = ["experiment", "algorithm", "run"]
    if "result" in metrics.columns:
        req_columns.append("result")
    if not isinstance(metrics, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if experiments is not None:
        metrics = metrics[metrics["experiment"].isin(experiments)]
    metrics, value_columns = _prepare_numeric_metric_columns(metrics, req_columns)
    if not value_columns:
        raise ValueError("No numeric metric columns available for aggregation.")
    return metrics.pivot_table(
        index=["experiment", "algorithm"],
        values=value_columns,
        dropna=False,
        aggfunc="min",
    ).reset_index()


def get_group_mean_metrics(
    metrics: pd.DataFrame,
    episode_groups: Mapping[str, Union[str, List[str], Tuple[str, ...]]],
) -> pd.DataFrame:
    """Calculate mean metrics aggregated over user-defined episode groups as the mean of the means.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        episode_groups (Mapping[str, Union[str, List[str], Tuple[str, ...]]]): Mapping of group names to experiment names (each definition can be a string or an iterable of strings).
    Returns:
        pd.DataFrame: DataFrame containing the mean metrics grouped by episode_group and algorithm.
    """

    # first get the mean metrics by experiment and algorithm
    mean_metrics = get_mean_metrics(metrics, experiments=None)

    # for each group we create a new row in the output dataframe with the mean of the means of the experiments in that group
    group_means = []
    for group_name, experiment_names in episode_groups.items():
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]
        group_mean = (
            mean_metrics[mean_metrics["experiment"].isin(experiment_names)]
            .groupby("algorithm")
            .mean(numeric_only=True)
            .reset_index()
        )
        group_mean["experiment"] = group_name
        group_means.append(group_mean)
    return pd.concat(group_means, ignore_index=True)


def get_group_std_dev_metrics(
    metrics: pd.DataFrame,
    episode_groups: Mapping[str, Union[str, List[str], Tuple[str, ...]]],
) -> pd.DataFrame:
    """Calculate the standard deviation aggregated over episode groups.
    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        episode_groups (Mapping[str, Union[str, List[str], Tuple[str, ...]]]): Mapping of group names to experiment names (each definition can be a string or an iterable of strings).
    Returns:
        pd.DataFrame: DataFrame containing the mean metrics grouped by episode_group and algorithm.
    """

    # compute the mean metrics by experiment and algorithm
    mean_metrics = get_mean_metrics(metrics, experiments=None)
    # compute the std deviation metrics by experiment and algorithm
    std_dev_metrics = get_std_dev_metrics(metrics, experiments=None)

    # for each group we create a new row in the output dataframe with
    # the mean of the std deviations of the experiments in that group (std deviation within the group)
    # plus the std deviation of the means of the experiments in that group (std deviation between experiments in the group)

    group_std_devs = []
    for group_name, experiment_names in episode_groups.items():
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]
        group_std_dev_within = (
            std_dev_metrics[std_dev_metrics["experiment"].isin(experiment_names)]
            .groupby("algorithm")
            .mean(numeric_only=True)
            .reset_index()
        )
        group_std_dev_between = (
            mean_metrics[mean_metrics["experiment"].isin(experiment_names)]
            .groupby("algorithm")
            .std(numeric_only=True)
            .reset_index()
        )
        group_std_dev = group_std_dev_within.copy()
        for col in group_std_dev.columns:
            if col not in ["algorithm", "experiment"]:
                group_std_dev[col] = np.sqrt(
                    group_std_dev_within[col] ** 2 + group_std_dev_between[col] ** 2
                )
        group_std_dev["experiment"] = group_name
        group_std_devs.append(group_std_dev)
    return pd.concat(group_std_devs, ignore_index=True)


def get_group_max_metrics(
    metrics: pd.DataFrame,
    episode_groups: Mapping[str, Union[str, List[str], Tuple[str, ...]]],
    experiments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate the maximum aggregated over episode groups."""

    # first get the max metrics by experiment and algorithm
    max_metrics = get_max_metrics(metrics, experiments=experiments)

    # for each group we create a new row in the output dataframe with the max of the maxs of the experiments in that group
    group_maxs = []
    for group_name, experiment_names in episode_groups.items():
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]
        group_max = (
            max_metrics[max_metrics["experiment"].isin(experiment_names)]
            .groupby("algorithm")
            .max(numeric_only=True)
            .reset_index()
        )
        group_max["experiment"] = group_name
        group_maxs.append(group_max)
    return pd.concat(group_maxs, ignore_index=True)


def get_group_min_metrics(
    metrics: pd.DataFrame,
    episode_groups: Mapping[str, Union[str, List[str], Tuple[str, ...]]],
    experiments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calculate the minimum aggregated over episode groups."""

    # first get the min metrics by experiment and algorithm
    min_metrics = get_min_metrics(metrics, experiments=experiments)

    # for each group we create a new row in the output dataframe with the min of the mins of the experiments in that group
    group_mins = []
    for group_name, experiment_names in episode_groups.items():
        if isinstance(experiment_names, str):
            experiment_names = [experiment_names]
        group_min = (
            min_metrics[min_metrics["experiment"].isin(experiment_names)]
            .groupby("algorithm")
            .min(numeric_only=True)
            .reset_index()
        )
        group_min["experiment"] = group_name
        group_mins.append(group_min)
    return pd.concat(group_mins, ignore_index=True)


def get_metrics_summary(
    metrics: pd.DataFrame,
    episode_groups: Optional[
        Mapping[str, Union[str, List[str], Tuple[str, ...]]]
    ] = None,
    experiments: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate a summary of metrics including mean, std deviation, median, max, and min.

    Args:
        metrics (pd.DataFrame): DataFrame containing the metrics.
        experiments (Optional[List[str]]): List of experiments to include in the summary. If None, include all experiments.
        episode_groups (Optional[Mapping[str, Union[str, List[str], Tuple[str, ...]]]]): Optional mapping of group names to experiment names for grouped summaries.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing summarized metrics.
    """
    summary: Dict[str, pd.DataFrame] = {
        "mean": get_mean_metrics(metrics, experiments),
        "std_dev": get_std_dev_metrics(metrics, experiments),
        "median": get_median_metrics(metrics, experiments),
        "max": get_max_metrics(metrics, experiments),
        "min": get_min_metrics(metrics, experiments),
    }
    if episode_groups:
        summary["group_mean"] = get_group_mean_metrics(metrics, episode_groups)
        summary["group_std_dev"] = get_group_std_dev_metrics(metrics, episode_groups)
        summary["group_max"] = get_group_max_metrics(metrics, episode_groups)
        summary["group_min"] = get_group_min_metrics(metrics, episode_groups)
    return summary


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

    # Keep exact success matching and make substring checks robust for NaN/float values.
    result_series = metrics_[result_column]
    result_text = result_series.fillna("").astype(str)

    metrics_["success"] = (result_series == "Success").astype(int)
    metrics_["timeout"] = result_text.str.contains("Timeout", na=False).astype(int)
    metrics_["collision"] = result_text.str.contains("Collision", na=False).astype(int)
    metrics_["navigation_failure"] = (
        result_text.str.contains("Navigation failure", na=False).astype(int)
    )

    return metrics_
