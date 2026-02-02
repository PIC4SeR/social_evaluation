"""Helpers to generate LaTeX driver files for benchmark plots."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


TEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{amsmath}
\usepackage{array}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{caption}
\usepackage{float}
\usepackage{longtable}
\geometry{margin=1in}
"""

TEX_PREAMBLE_WITH_SUBCAPTION = TEX_PREAMBLE + r"\usepackage{subcaption}" + "\n"

Logger = Optional[Callable[[str], None]]

__all__ = [
    "MissingAssetError",
    "generate_reports_for_root",
    "generate_reports_for_scenario",
]


class MissingAssetError(RuntimeError):
    """Raised when required assets for a scenario are missing."""


def _emit(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


def _format_relative(path: Path, base: Optional[Path]) -> str:
    if base is not None:
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _find_first_existing(base: Path, items: Iterable[str]) -> Optional[Path]:
    for entry in items:
        candidate = base / entry
        if candidate.exists():
            return candidate
    return None


def _relative(asset: Path, root: Path) -> str:
    return asset.relative_to(root).as_posix()


def _require_asset(asset: Optional[Path], description: str, scenario: str) -> Path:
    if asset is None:
        raise MissingAssetError(
            f"Missing {description} for scenario '{scenario}'. "
            "Run benchmark_plots to regenerate the required tables or plots."
        )
    return asset


def _format_float(value: Any, *, precision: int = 2, scientific: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if scientific:
        return f"{number:.1e}"
    return f"{number:.{precision}f}"


def _latex_path(path: str) -> str:
    return path.replace("_", r"\_")


def _load_anova_summary(summary_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if not raw:
                continue
            record: Dict[str, Any] = dict(raw)
            record["metric_key"] = raw.get("metric_key", "")
            record["metric_label"] = raw.get("metric_label", record["metric_key"])
            record["f_statistic"] = raw.get("f_statistic")
            record["p_value"] = raw.get("p_value")
            record["df_between"] = raw.get("df_between")
            record["df_within"] = raw.get("df_within")
            record["sample_size"] = raw.get("sample_size")
            record["test_type"] = raw.get("test_type", "")
            record["significant"] = (
                str(raw.get("significant", "False")).strip().lower() == "true"
            )
            record["heteroscedastic"] = (
                str(raw.get("heteroscedastic", "False")).strip().lower() == "true"
            )
            rows.append(record)
    return rows


def _render_anova_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        r"    \small",
        r"    \begin{tabular}{llcccccc}",
        r"        \toprule",
        r"        \textbf{Metric} & \textbf{Test} & $\mathbf{F}$ & $\mathbf{p}$ & \textbf{DoF} & \textbf{N} & \textbf{Sig.} & \textbf{Hetero.} \\",
        r"        \midrule",
    ]
    for row in rows:
        metric_label = row.get("metric_label", row.get("metric_key", ""))
        test_type = row.get("test_type", "")
        test_label = test_type.replace("_", " ").title() if test_type else "-"
        f_value = _format_float(row.get("f_statistic"), precision=2)
        p_value = _format_float(row.get("p_value"), scientific=True)
        try:
            df_between = float(row.get("df_between"))
            df_within = float(row.get("df_within"))
            df_label = f"{df_between:.1f}/{df_within:.1f}"
        except (TypeError, ValueError):
            df_label = "-/-"
        try:
            sample_size = int(float(row.get("sample_size")))
        except (TypeError, ValueError):
            sample_size = 0
        sig_label = "Yes" if row.get("significant") else "No"
        hetero_label = "Yes" if row.get("heteroscedastic") else "No"
        lines.append(
            f"        {metric_label} & {test_label} & {f_value} & {p_value} & {df_label} & {sample_size} & {sig_label} & {hetero_label} \\\\"
        )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \caption{Summary of one-way ANOVA tests across navigation metrics.}",
            r"    \label{tab:anova_summary}",
            r"\end{table}",
        ]
    )
    return lines


def _append_anova_sections(
    lines: List[str],
    scenario_dir: Path,
    anova_assets: Dict[str, Any],
) -> None:
    summary_rows: List[Dict[str, Any]] = anova_assets.get("summary_rows", [])
    pairwise_csv: Optional[Path] = anova_assets.get("pairwise_csv")
    plots: Dict[str, Path] = anova_assets.get("plots", {})
    label_map = {
        row.get("metric_key"): row.get("metric_label", row.get("metric_key"))
        for row in summary_rows
    }

    if summary_rows:
        lines.append(r"\section{ANOVA Summary}")
        lines.append(
            r"One-way ANOVA (Welch adjusted when heteroscedastic) highlights "
            r"which metrics exhibit statistically different behaviour across "
            r"navigators."
        )
        lines.extend(_render_anova_table(summary_rows))
        lines.append("")

    if plots:
        lines.append(r"\section{Tukey Pairwise Analysis}")
        description = (
            r"Tukey HSD post-hoc tests identify significant algorithm pairs. "
            r"The heatmaps display adjusted $p$-values."
        )
        if pairwise_csv:
            rel_csv = _relative(pairwise_csv, scenario_dir)
            description += (
                r" Complete numeric results are provided in "
                rf"\texttt{{{_latex_path(rel_csv)}}}."
            )
        lines.append(description)
        lines.append("")
        for metric_key, plot_path in sorted(plots.items()):
            rel_plot = _relative(plot_path, scenario_dir)
            metric_label = label_map.get(
                metric_key, metric_key.replace("_", " ").title()
            )
            lines.append(r"\begin{figure}[htbp]")
            lines.append(r"    \centering")
            lines.append(rf"    \includegraphics[width=\textwidth]{{{rel_plot}}}")
            lines.append(
                rf"    \caption{{Pairwise Tukey HSD adjusted $p$-values for {metric_label}.}}"
            )
            lines.append(rf"    \label{{fig:{metric_key}_pairwise}}")
            lines.append(r"\end{figure}")
            lines.append("")


def _build_standard_document(scenario_dir: Path, assets: Dict[str, Any]) -> str:
    nav_table = _relative(assets["nav_table"], scenario_dir)
    social_table = _relative(assets["social_table"], scenario_dir)
    success_plot = _relative(assets["success_plot"], scenario_dir)
    nav_box_plot = _relative(assets["nav_box_plot"], scenario_dir)
    nav_violin_plot = _relative(assets["nav_violin_plot"], scenario_dir)
    social_box_plot = _relative(assets["social_box_plot"], scenario_dir)
    social_violin_plot = _relative(assets["social_violin_plot"], scenario_dir)
    proxemic_plot = _relative(assets["proxemic_plot"], scenario_dir)

    lines: List[str] = []
    lines.append(TEX_PREAMBLE)
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(rf"\input{{{nav_table}}}")
    lines.append(rf"\input{{{social_table}}}")
    lines.append("")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.9\textwidth]{{{success_plot}}}")
    lines.append(
        r"    \caption{Success rate of each algorithm across benchmark environments.}"
    )
    lines.append(r"    \label{fig:success_rate_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.9\textwidth]{{{nav_box_plot}}}")
    lines.append(
        r"    \caption{Boxplots of navigation metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_nav_boxplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.9\textwidth]{{{nav_violin_plot}}}")
    lines.append(
        r"    \caption{Violin plots of navigation metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_nav_violinplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.85\textwidth]{{{social_box_plot}}}")
    lines.append(
        r"    \caption{Boxplots of social metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_social_boxplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.85\textwidth]{{{social_violin_plot}}}")
    lines.append(
        r"    \caption{Violin plots of social metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_social_violinplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append(
        r"%     \item \textit{Proxemics occupancy}: intimate space ($PR_I$), "
        r"personal space ($PR_{PE}$), social space ($PR_S$), public space "
        r"($PR_{PU}$)."
    )
    lines.append("")
    lines.append(r"\begin{figure}")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=0.9\textwidth]{{{proxemic_plot}}}")
    lines.append(
        r"    \caption{Proxemic distances and intrusions by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:proxemic_distances_intrusions}")
    lines.append(r"\end{figure}")
    lines.append("")

    anova_assets = assets.get("anova")
    if anova_assets:
        _append_anova_sections(lines, scenario_dir, anova_assets)

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def _build_weighted_document(
    scenario_dir: Path,
    assets: Dict[str, Any],
    include_paths: Dict[str, Optional[Path]],
) -> str:
    nav_table = _relative(assets["nav_weighted"], scenario_dir)
    social_table = _relative(assets["social_weighted"], scenario_dir)
    success_plot = _relative(assets["success_plot"], scenario_dir)
    proxemic_plot = _relative(assets["proxemic_plot"], scenario_dir)
    nav_box_plot = _relative(assets["nav_box_plot"], scenario_dir)
    nav_violin_plot = _relative(assets["nav_violin_plot"], scenario_dir)
    social_box_plot = _relative(assets["social_box_plot"], scenario_dir)
    social_violin_plot = _relative(assets["social_violin_plot"], scenario_dir)

    lines: List[str] = []
    lines.append(TEX_PREAMBLE_WITH_SUBCAPTION)
    lines.append(r"\begin{document}")
    lines.append("")
    lines.append(r"\section{Navigation Performance Summary}")
    lines.append(
        r"This section reports the expected cost (ENC) of each navigation metric "
        r"after integrating success-rate penalties, showing how often a navigator "
        r"finishes and how costly it is when it does."
    )
    lines.append(rf"\input{{{nav_table}}}")
    lines.append("")
    lines.append(r"\section{Social Interaction Summary}")
    lines.append(
        r"We apply the same success-aware penalty scheme to social compliance "
        r"metrics so poor or inconsistent behaviour is reflected even when "
        r"outright failures are rare."
    )
    lines.append(rf"\input{{{social_table}}}")
    lines.append("")
    lines.append(r"\section{Success Rate and Proxemics Overview}")
    lines.append(
        r"These plots summarise the raw outcome frequencies and proxemic intrusions "
        r"so the ENC/EPU tables above can be interpreted in the context of absolute "
        r"success behaviour."
    )
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\linewidth]{{{success_plot}}}")
    lines.append(
        r"    \caption{Success rate histogram for each navigator across benchmark "
        r"environments.}"
    )
    lines.append(r"    \label{fig:success_rate_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\linewidth]{{{proxemic_plot}}}")
    lines.append(
        r"    \caption{Distribution of proxemic intrusions per navigator; "
        r"consistent colour ordering matches the success-rate plot.}"
    )
    lines.append(r"    \label{fig:proxemics_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\section{Metrics Plots}")
    lines.append("")
    lines.append(r"\subsection{Navigation Metrics}")
    lines.append("")
    lines.append(
        r"We visualise the distribution of navigation metrics across benchmark "
        r"environments to illustrate how each navigator performs in different "
        r"scenarios."
    )
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\textwidth]{{{nav_box_plot}}}")
    lines.append(
        r"    \caption{Boxplots of navigation metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_nav_boxplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\textwidth]{{{nav_violin_plot}}}")
    lines.append(
        r"    \caption{Violin plots of navigation metrics by algorithm across "
        r"benchmark environments.}"
    )
    lines.append(r"    \label{fig:metrics_nav_violinplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\subsection{Social Metrics}")
    lines.append("")
    lines.append(
        r"We visualise the distribution of social metrics across benchmark "
        r"environments to illustrate how each navigator performs in different "
        r"scenarios."
    )
    lines.append("")

    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\textwidth]{{{social_box_plot}}}")
    lines.append(
        r"    \caption{Boxplots of social metrics by algorithm across benchmark "
        r"environments.}"
    )
    lines.append(r"    \label{fig:metrics_social_boxplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")
    lines.append(r"\begin{figure}[htbp]")
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width=\textwidth]{{{social_violin_plot}}}")
    lines.append(
        r"    \caption{Violin plots of social metrics by algorithm across benchmark "
        r"environments.}"
    )
    lines.append(r"    \label{fig:metrics_social_violinplots_by_algorithm}")
    lines.append(r"\end{figure}")
    lines.append("")

    interval_include = include_paths.get("interval")
    if interval_include:
        lines.append(r"\section{Distributional Reliability Analysis}")
        lines.append(
            r"We use rliable’s interval estimates to capture how each navigation "
            r"policy behaves across trials, highlighting variability and the "
            r"expected shortfall relative to the benchmark frontier."
        )
        rel_path = _relative(interval_include, scenario_dir)
        lines.append(rf"\input{{{rel_path}}}")
        lines.append("")

    performance_include = include_paths.get("performance")
    if performance_include:
        lines.append(r"\section{Performance Profiles}")
        lines.append(
            r"Performance profiles expose how much of the task set each navigator "
            r"solves within a factor of the best observed performance, making it "
            r"easy to spot robust options."
        )
        rel_path = _relative(performance_include, scenario_dir)
        lines.append(rf"\input{{{rel_path}}}")
        lines.append("")

    prob_include = include_paths.get("probability")
    if prob_include:
        lines.append(r"\section{Probability of Improvement}")
        lines.append(
            r"Here we quantify how likely each navigator is to outperform the MPC "
            r"baseline on the same tasks, showing where social-cost variants add "
            r"value."
        )
        rel_path = _relative(prob_include, scenario_dir)
        lines.append(rf"\input{{{rel_path}}}")
        lines.append("")

    anova_assets = assets.get("anova")
    if anova_assets:
        _append_anova_sections(lines, scenario_dir, anova_assets)

    lines.append(r"\end{document}")
    lines.append("")
    return "\n".join(lines)


def _collect_assets(scenario_dir: Path) -> Dict[str, Any]:
    tables = scenario_dir / "tables"
    plots = scenario_dir / "plots"
    scenario_name = scenario_dir.name

    nav_table = _require_asset(
        _find_first_existing(
            tables,
            [
                "metrics_nav_summary_by_navigator.tex",
                "metrics_nav_summary_by_algorithm.tex",
            ],
        ),
        "navigation summary table",
        scenario_name,
    )
    social_table = _require_asset(
        _find_first_existing(
            tables,
            [
                "metrics_social_summary_by_navigator.tex",
                "metrics_social_summary_by_algorithm.tex",
            ],
        ),
        "social summary table",
        scenario_name,
    )
    success_plot = _require_asset(
        _find_first_existing(
            plots,
            ["success_rate_by_navigator.png", "success_rate_by_algorithm.png"],
        ),
        "success rate figure",
        scenario_name,
    )
    nav_box_plot = _require_asset(
        _find_first_existing(
            plots,
            [
                "metrics_nav_boxplots_by_navigator.png",
                "metrics_nav_boxplots_by_algorithm.png",
            ],
        ),
        "navigation boxplot figure",
        scenario_name,
    )
    nav_violin_plot = _require_asset(
        _find_first_existing(
            plots,
            [
                "metrics_nav_violinplots_by_navigator.png",
                "metrics_nav_violinplots_by_algorithm.png",
            ],
        ),
        "navigation violin plot figure",
        scenario_name,
    )
    social_box_plot = _require_asset(
        _find_first_existing(
            plots,
            [
                "metrics_social_boxplots_by_navigator.png",
                "metrics_social_boxplots_by_algorithm.png",
            ],
        ),
        "social boxplot figure",
        scenario_name,
    )
    social_violin_plot = _require_asset(
        _find_first_existing(
            plots,
            [
                "metrics_social_violinplots_by_navigator.png",
                "metrics_social_violinplots_by_algorithm.png",
            ],
        ),
        "social violin plot figure",
        scenario_name,
    )
    proxemic_plot = _require_asset(
        _find_first_existing(plots, ["proxemic_distances_intrusions.png"]),
        "proxemic intrusion figure",
        scenario_name,
    )

    nav_weighted = _find_first_existing(
        tables,
        [
            "metrics_nav_weighted_summary_by_navigator.tex",
            "metrics_nav_weighted_summary_by_algorithm.tex",
        ],
    )
    social_weighted = _find_first_existing(
        tables,
        [
            "metrics_social_weighted_summary_by_navigator.tex",
            "metrics_social_weighted_summary_by_algorithm.tex",
        ],
    )

    assets: Dict[str, Any] = {
        "nav_table": nav_table,
        "social_table": social_table,
        "success_plot": success_plot,
        "nav_box_plot": nav_box_plot,
        "nav_violin_plot": nav_violin_plot,
        "social_box_plot": social_box_plot,
        "social_violin_plot": social_violin_plot,
        "proxemic_plot": proxemic_plot,
    }

    if nav_weighted and social_weighted:
        assets["nav_weighted"] = nav_weighted
        assets["social_weighted"] = social_weighted

    anova_assets = _gather_anova_assets(tables, plots)
    if anova_assets:
        assets["anova"] = anova_assets

    return assets


def _write_file(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --overwrite to allow replacing it."
        )
    path.write_text(content, encoding="utf-8")


def _build_include_paths(scenario_dir: Path) -> Dict[str, Optional[Path]]:
    includes = {
        "interval": scenario_dir / "rliable_interval_estimates_nav.tex",
        "performance": scenario_dir / "rliable_performance_profiles.tex",
        "probability": scenario_dir / "rliable_prob_of_improvement.tex",
    }
    resolved: Dict[str, Optional[Path]] = {}
    for key, candidate in includes.items():
        resolved[key] = candidate if candidate.exists() else None
    return resolved


def _gather_anova_assets(
    tables_dir: Path, plots_dir: Path
) -> Optional[Dict[str, Any]]:
    custom_stats = tables_dir / "custom_stats"
    summary_csv = custom_stats / "anova_summary.csv"
    pairwise_csv = custom_stats / "anova_pairwise.csv"

    summary_rows: List[Dict[str, Any]] = []
    if summary_csv.exists():
        summary_rows = _load_anova_summary(summary_csv)

    pairwise_path = pairwise_csv if pairwise_csv.exists() else None

    plots_map: Dict[str, Path] = {}
    anova_plot_dir = plots_dir / "anova"
    if anova_plot_dir.exists():
        for png_path in sorted(
            anova_plot_dir.glob("*_pairwise_pvalues.png"), key=lambda path: path.name
        ):
            metric_key = png_path.stem.replace("_pairwise_pvalues", "")
            plots_map[metric_key] = png_path

    if summary_rows or pairwise_path or plots_map:
        return {
            "summary_rows": summary_rows,
            "pairwise_csv": pairwise_path,
            "plots": plots_map,
        }
    return None


def generate_reports_for_scenario(
    scenario_dir: Path,
    report_prefix: str,
    *,
    overwrite: bool = False,
    reports_root: Optional[Path] = None,
    logger: Logger = None,
    announce: Optional[str] = None,
) -> List[Path]:
    """Generate LaTeX drivers for a single scenario directory."""

    if announce is None:
        message = f"Processing scenario '{scenario_dir.name}'"
    else:
        message = announce
    if message:
        _emit(logger, message)

    assets = _collect_assets(scenario_dir)
    created: List[Path] = []
    filename_prefix = f"{report_prefix}_{scenario_dir.name}"

    base_tex = scenario_dir / f"{filename_prefix}.tex"
    standard_document = _build_standard_document(scenario_dir, assets)
    _write_file(base_tex, standard_document, overwrite=overwrite)
    created.append(base_tex)
    _emit(logger, f"  wrote {_format_relative(base_tex, reports_root)}")

    if "nav_weighted" in assets and "social_weighted" in assets:
        include_paths = _build_include_paths(scenario_dir)
        weighted_tex = scenario_dir / f"{filename_prefix}_weighted.tex"
        weighted_document = _build_weighted_document(
            scenario_dir, assets, include_paths
        )
        _write_file(weighted_tex, weighted_document, overwrite=overwrite)
        created.append(weighted_tex)
        _emit(logger, f"  wrote {_format_relative(weighted_tex, reports_root)}")
    else:
        _emit(
            logger,
            "  skipping weighted report (weighted summary tables not found for this scenario)",
        )

    return created


def generate_reports_for_root(
    reports_root: Path,
    report_prefix: str,
    *,
    overwrite: bool = False,
    logger: Logger = None,
) -> Dict[str, List[Path]]:
    """Generate LaTeX reports for every scenario directory in ``reports_root``."""

    reports_root = reports_root.expanduser().resolve()
    if not reports_root.exists():
        raise ValueError(f"Reports root directory does not exist: {reports_root}")

    scenario_dirs = [
        entry
        for entry in reports_root.iterdir()
        if entry.is_dir()
        and not entry.name.startswith(".")
        and entry.name != "__pycache__"
    ]
    if not scenario_dirs:
        raise ValueError(
            f"No scenario directories found under {reports_root}. "
            "Have you run run_benchmark_report.py?"
        )

    generated: Dict[str, List[Path]] = {}
    for scenario_dir in sorted(scenario_dirs):
        try:
            created = generate_reports_for_scenario(
                scenario_dir,
                report_prefix,
                overwrite=overwrite,
                reports_root=reports_root,
                logger=logger,
            )
        except MissingAssetError as exc:
            _emit(logger, str(exc))
            continue
        except FileExistsError as exc:
            _emit(logger, str(exc))
            continue
        generated[scenario_dir.name] = created

    return generated
