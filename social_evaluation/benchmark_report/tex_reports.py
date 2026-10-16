"""Helpers to generate LaTeX driver files for benchmark plots."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


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


def _page_sort_key(path: Path) -> int:
    match = re.search(r"_page(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _find_existing_or_paginated(base: Path, items: Iterable[str]) -> List[Path]:
    for entry in items:
        candidate = base / entry
        if candidate.exists():
            return [candidate]
        stem = candidate.stem
        suffix = candidate.suffix
        paginated = sorted(
            base.glob(f"{stem}_page*{suffix}"),
            key=_page_sort_key,
        )
        if paginated:
            return paginated
    return []


def _relative(asset: Path, root: Path) -> str:
    return asset.relative_to(root).as_posix()


def _require_asset(asset: Optional[Path], description: str, scenario: str) -> Path:
    if asset is None:
        raise MissingAssetError(
            f"Missing {description} for scenario '{scenario}'. "
            "Run benchmark_plots to regenerate the required tables or plots."
        )
    return asset


def _require_assets(assets: List[Path], description: str, scenario: str) -> List[Path]:
    if not assets:
        raise MissingAssetError(
            f"Missing {description} for scenario '{scenario}'. "
            "Run benchmark_plots to regenerate the required tables or plots."
        )
    return assets


def _humanize_group_key(group_key: str) -> str:
    text = group_key.strip()
    if text.startswith("metrics_"):
        text = text[len("metrics_") :]
    text = text.replace("_", " ").strip()
    if not text:
        return group_key
    return text.title()


def _build_metric_group_order(
    tables_dir: Path,
    plots_dir: Path,
    preferred_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    discovered: List[str] = []
    seen: set[str] = set()

    def _add(group_key: str) -> None:
        if group_key and group_key not in seen:
            discovered.append(group_key)
            seen.add(group_key)

    for key in preferred_keys or []:
        _add(str(key))

    if tables_dir.exists():
        for path in sorted(tables_dir.glob("*_summary_by_*.tex")):
            match = re.match(
                r"^(?P<group>.+?)_(?:weighted_)?summary_by_(?:navigator|algorithm)\.tex$",
                path.name,
            )
            if match:
                _add(match.group("group"))

    if plots_dir.exists():
        for path in sorted(plots_dir.glob("*plots_by_*.png")):
            match = re.match(
                r"^(?P<group>.+?)_(?:box|violin)plots_by_(?:navigator|algorithm)(?:_page\d+)?\.png$",
                path.name,
            )
            if match:
                _add(match.group("group"))

    return discovered


def _append_metric_group_summary_sections(
    lines: List[str],
    scenario_dir: Path,
    metric_groups: List[Dict[str, Any]],
    *,
    weighted: bool,
) -> None:
    groups_with_tables = []
    for group in metric_groups:
        table = group["weighted_table"] if weighted else group["summary_table"]
        if table:
            groups_with_tables.append(group)
    if not groups_with_tables:
        return

    section_title = "Success-Weighted Metric Summary" if weighted else "Metric Summary"
    lines.append(rf"\section{{{section_title}}}")
    lines.append("")
    for group in groups_with_tables:
        title = group["title"]
        table_path = group["weighted_table"] if weighted else group["summary_table"]
        assert table_path is not None
        lines.append(rf"\subsection{{{title}}}")
        lines.append(
            (
                rf"This table summarises the success-weighted metrics for the {title.lower()} group."
                if weighted
                else rf"This table summarises the reported metrics for the {title.lower()} group."
            )
        )
        lines.append(rf"\input{{{_relative(table_path, scenario_dir)}}}")
        lines.append("")


def _append_metric_group_plot_sections(
    lines: List[str],
    scenario_dir: Path,
    metric_groups: List[Dict[str, Any]],
    *,
    width: str,
    placement: Optional[str] = None,
) -> None:
    groups_with_plots = [
        group for group in metric_groups if group["box_plots"] or group["violin_plots"]
    ]
    if not groups_with_plots:
        return

    lines.append(r"\section{Metric Distributions}")
    lines.append("")
    for group in groups_with_plots:
        title = group["title"]
        label_prefix = group["key"]
        lines.append(rf"\subsection{{{title}}}")
        lines.append(
            rf"These plots show the distribution of the {title.lower()} metrics across the benchmark environments."
        )
        lines.append("")
        if group["box_plots"]:
            _append_figure_series(
                lines,
                scenario_dir=scenario_dir,
                assets=group["box_plots"],
                width=width,
                caption=(
                    rf"Boxplots of {title.lower()} metrics by algorithm across benchmark environments."
                ),
                label=rf"fig:{label_prefix}_boxplots_by_algorithm",
                placement=placement,
            )
        if group["violin_plots"]:
            _append_figure_series(
                lines,
                scenario_dir=scenario_dir,
                assets=group["violin_plots"],
                width=width,
                caption=(
                    rf"Violin plots of {title.lower()} metrics by algorithm across benchmark environments."
                ),
                label=rf"fig:{label_prefix}_violinplots_by_algorithm",
                placement=placement,
            )
        lines.append("")


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


def _append_figure(
    lines: List[str],
    *,
    image_path: str,
    width: str,
    caption: str,
    label: str,
    placement: Optional[str] = None,
) -> None:
    begin = r"\begin{figure}"
    if placement:
        begin = rf"\begin{{figure}}[{placement}]"
    lines.append(begin)
    lines.append(r"    \centering")
    lines.append(rf"    \includegraphics[width={width}]{{{image_path}}}")
    lines.append(rf"    \caption{{{caption}}}")
    lines.append(rf"    \label{{{label}}}")
    lines.append(r"\end{figure}")


def _append_figure_series(
    lines: List[str],
    *,
    scenario_dir: Path,
    assets: List[Path],
    width: str,
    caption: str,
    label: str,
    placement: Optional[str] = None,
) -> None:
    total = len(assets)
    for idx, asset in enumerate(assets, start=1):
        rel_asset = _relative(asset, scenario_dir)
        page_caption = caption
        page_label = label
        if total > 1:
            page_caption = f"{caption.rstrip('.')} (page {idx}/{total})."
            page_label = f"{label}_page{idx}"
        _append_figure(
            lines,
            image_path=rel_asset,
            width=width,
            caption=page_caption,
            label=page_label,
            placement=placement,
        )
        lines.append("")


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
    lines: List[str] = []
    lines.append(TEX_PREAMBLE)
    lines.append(r"\begin{document}")
    lines.append("")
    _append_metric_group_summary_sections(
        lines,
        scenario_dir,
        assets["metric_groups"],
        weighted=False,
    )

    success_plot = assets.get("success_plot")
    proxemic_plot = assets.get("proxemic_plot")
    if success_plot or proxemic_plot:
        lines.append(r"\section{Overview Figures}")
        lines.append("")
    if success_plot:
        lines.append(r"\begin{figure}")
        lines.append(r"    \centering")
        lines.append(
            rf"    \includegraphics[width=0.9\textwidth]{{{_relative(success_plot, scenario_dir)}}}"
        )
        lines.append(
            r"    \caption{Success rate of each algorithm across benchmark environments.}"
        )
        lines.append(r"    \label{fig:success_rate_by_algorithm}")
        lines.append(r"\end{figure}")
        lines.append("")
    _append_metric_group_plot_sections(
        lines,
        scenario_dir,
        assets["metric_groups"],
        width=r"0.9\textwidth",
    )
    if proxemic_plot:
        lines.append(r"\begin{figure}")
        lines.append(r"    \centering")
        lines.append(
            rf"    \includegraphics[width=0.9\textwidth]{{{_relative(proxemic_plot, scenario_dir)}}}"
        )
        lines.append(
            r"    \caption{Proxemic distances and intrusions by algorithm across benchmark environments.}"
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
    lines: List[str] = []
    lines.append(TEX_PREAMBLE_WITH_SUBCAPTION)
    lines.append(r"\begin{document}")
    lines.append("")
    _append_metric_group_summary_sections(
        lines,
        scenario_dir,
        assets["metric_groups"],
        weighted=True,
    )

    success_plot = assets.get("success_plot")
    proxemic_plot = assets.get("proxemic_plot")
    if success_plot or proxemic_plot:
        lines.append(r"\section{Success Rate and Proxemics Overview}")
        lines.append(
            r"These plots summarise the raw outcome frequencies and proxemic intrusions so the weighted tables can be interpreted in the context of absolute success behaviour."
        )
        lines.append("")
    if success_plot:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"    \centering")
        lines.append(
            rf"    \includegraphics[width=\linewidth]{{{_relative(success_plot, scenario_dir)}}}"
        )
        lines.append(
            r"    \caption{Success rate histogram for each navigator across benchmark environments.}"
        )
        lines.append(r"    \label{fig:success_rate_by_algorithm}")
        lines.append(r"\end{figure}")
        lines.append("")
    if proxemic_plot:
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"    \centering")
        lines.append(
            rf"    \includegraphics[width=\linewidth]{{{_relative(proxemic_plot, scenario_dir)}}}"
        )
        lines.append(
            r"    \caption{Distribution of proxemic intrusions per navigator; consistent colour ordering matches the success-rate plot.}"
        )
        lines.append(r"    \label{fig:proxemics_by_algorithm}")
        lines.append(r"\end{figure}")
        lines.append("")

    _append_metric_group_plot_sections(
        lines,
        scenario_dir,
        assets["metric_groups"],
        width=r"\textwidth",
        placement="htbp",
    )

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
def _collect_assets(
    scenario_dir: Path,
    *,
    metric_group_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    tables = scenario_dir / "tables"
    plots = scenario_dir / "plots"
    scenario_name = scenario_dir.name

    metric_groups: List[Dict[str, Any]] = []
    for group_key in _build_metric_group_order(
        tables,
        plots,
        preferred_keys=metric_group_keys,
    ):
        summary_table = _find_first_existing(
            tables,
            [
                f"{group_key}_summary_by_navigator.tex",
                f"{group_key}_summary_by_algorithm.tex",
            ],
        )
        weighted_table = _find_first_existing(
            tables,
            [
                f"{group_key}_weighted_summary_by_navigator.tex",
                f"{group_key}_weighted_summary_by_algorithm.tex",
            ],
        )
        box_plots = _find_existing_or_paginated(
            plots,
            [
                f"{group_key}_boxplots_by_navigator.png",
                f"{group_key}_boxplots_by_algorithm.png",
            ],
        )
        violin_plots = _find_existing_or_paginated(
            plots,
            [
                f"{group_key}_violinplots_by_navigator.png",
                f"{group_key}_violinplots_by_algorithm.png",
            ],
        )
        if not any([summary_table, weighted_table, box_plots, violin_plots]):
            continue
        metric_groups.append(
            {
                "key": group_key,
                "title": _humanize_group_key(group_key),
                "summary_table": summary_table,
                "weighted_table": weighted_table,
                "box_plots": box_plots,
                "violin_plots": violin_plots,
            }
        )

    success_plot = _find_first_existing(
        plots,
        ["success_rate_by_navigator.png", "success_rate_by_algorithm.png"],
    )
    proxemic_plot = _find_first_existing(plots, ["proxemic_distances_intrusions.png"])

    assets: Dict[str, Any] = {
        "metric_groups": metric_groups,
        "success_plot": success_plot,
        "proxemic_plot": proxemic_plot,
    }

    anova_assets = _gather_anova_assets(tables, plots)
    if anova_assets:
        assets["anova"] = anova_assets

    has_reportable_assets = any(
        [
            metric_groups,
            success_plot is not None,
            proxemic_plot is not None,
            anova_assets is not None,
        ]
    )
    if not has_reportable_assets:
        raise MissingAssetError(
            f"Missing reportable tables or plots for scenario '{scenario_name}'. "
            "Run benchmark_plots to regenerate the required tables or plots."
        )

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
    metric_group_keys: Optional[Sequence[str]] = None,
    logger: Logger = None,
    weighted: bool = False,
    announce: Optional[str] = None,
) -> List[Path]:
    """Generate LaTeX drivers for a single scenario directory."""

    if announce is None:
        message = f"Processing scenario '{scenario_dir.name}'"
    else:
        message = announce
    if message:
        _emit(logger, message)

    assets = _collect_assets(
        scenario_dir,
        metric_group_keys=metric_group_keys,
    )
    created: List[Path] = []
    filename_prefix = f"{report_prefix}_{scenario_dir.name}"

    base_tex = scenario_dir / f"{filename_prefix}.tex"
    standard_document = _build_standard_document(scenario_dir, assets)
    _write_file(base_tex, standard_document, overwrite=overwrite)
    created.append(base_tex)
    _emit(logger, f"  wrote {_format_relative(base_tex, reports_root)}")

    if not weighted:
        pass                      # success-weighted document only on request (weighted: true)
    elif any(group["weighted_table"] for group in assets["metric_groups"]):
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
    weighted: bool = False,
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
                weighted=weighted,
            )
        except MissingAssetError as exc:
            _emit(logger, str(exc))
            continue
        except FileExistsError as exc:
            _emit(logger, str(exc))
            continue
        generated[scenario_dir.name] = created

    return generated
