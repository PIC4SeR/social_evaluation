import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional


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

    latex_tabular = report_latex.to_latex(
        column_format=column_format,
        header=processed_header,
        float_format=float_format,
        bold_rows=bold_rows,
        escape=False,
        na_rep="-",
        multicolumn_format="c",
    )
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
    mean_df,
    cmap_name: str = "Blues",
    alpha: float = 0.7,
    std_dev_df: Optional[Union[float, pd.Series, pd.DataFrame]] = None,
):
    from matplotlib import cm

    cmap = cm.get_cmap(cmap_name)
    norms = []
    for i, col in enumerate(mean_df.columns):
        if "%" in col or "Success Rate" in col or "SR" in col:
            norms.append(plt.Normalize(0, 100))  # type: ignore
        else:
            norms.append(plt.Normalize(mean_df[col].min(), mean_df[col].max()))  # type: ignore

    def blend_with_white(rgb, alpha):
        return tuple(alpha * c + (1 - alpha) * 1.0 for c in rgb)

    styled_df = mean_df.copy().astype(object)
    for row in mean_df.index:
        for col in mean_df.columns:
            val = mean_df.loc[row, col]
            # handle NaNs gracefully for coloring
            try:
                if "down" in col.lower():
                    norm_val = 1 - norms[mean_df.columns.get_loc(col)](val)
                else:
                    norm_val = norms[mean_df.columns.get_loc(col)](val)
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
            if std_dev_df is not None:
                # determine the correct std dev value per-cell
                if isinstance(std_dev_df, pd.DataFrame):
                    std_cell = (
                        std_dev_df.loc[row, col]
                        if (row in std_dev_df.index and col in std_dev_df.columns)
                        else float("nan")
                    )
                    if row not in std_dev_df.index or col not in std_dev_df.columns:
                        continue
                elif isinstance(std_dev_df, pd.Series):
                    # series indexed by row (algorithm) or col; try row first then col
                    std_cell = (
                        std_dev_df.loc[row]
                        if row in std_dev_df.index
                        else (
                            std_dev_df.loc[col]
                            if col in std_dev_df.index
                            else float("nan")
                        )
                    )
                else:  # scalar
                    std_cell = std_dev_df
                try:
                    styled_std = rf"\pm {float(std_cell):.2f}" if std_cell != 0 else ""  # type: ignore
                except Exception:
                    styled_std = "-"

                styled_df.loc[row, col] = (
                    rf"\cellcolor[rgb]{{{color}}} ${styled_val} {styled_std}$"
                )

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
