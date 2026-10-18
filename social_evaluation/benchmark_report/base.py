"""Core implementation for benchmark reporting."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from social_evaluation.benchmark_report.utils import (
    filter_metrics_for_success,
    load_metrics_from_dir,
    parse_experiment_result,
    rename_metrics_columns,
)

from .anova import AnovaMixin
from .config import BenchmarkConfig
from .plots import PlottingMixin
from .rliable import RliableMixin
from .tables import TableExportMixin


def _apply_benchmark_report_rcparams() -> None:
    """Enforce consistent LaTeX styling for benchmark figures/tables."""
    plt.rcParams.update({"font.size": 20})
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman","Regular","Tinos"]
    plt.rcParams["font.weight"] = "bold"


class BenchmarkReportBase(AnovaMixin, RliableMixin, PlottingMixin, TableExportMixin):
    sns.set_context("paper")
    _apply_benchmark_report_rcparams()
    """Encapsulate data loading, formatting, and shared helpers."""

    results = ["success", "timeout", "collision", "navigation_failure"]

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        _apply_benchmark_report_rcparams()
        plt.rcParams["text.usetex"] = self.config.use_tex
        # Use display_episodes when explicitly provided (non-empty).  An empty
        # list means "show no individual episodes" (groups-only mode).
        display_keys = (
            config.display_episodes if config.display_episodes else config.episodes
        )
        # When only episode_groups are configured (no explicit episodes list),
        # skip individual episode rows so that only group aggregations appear.
        if not config.display_episodes and config.episode_groups:
            display_keys = []
        self.ordered_experiments = [
            config.episode_names[key]
            for key in display_keys
            if key in config.episode_names
        ]
        self.ordered_episode_groups = list(config.episode_groups.keys())
        self.ordered_navigators = [
            config.navigator_names.get(navigator, navigator)
            for navigator in config.navigators
        ]
        self._category_cmap_name = "Paired"
        self.metrics_results = self._load_metrics()
        self.metrics_results_parsed = parse_experiment_result(self.metrics_results)
        if self.config.discard_failed_metrics:
            self.success_results = filter_metrics_for_success(self.metrics_results)
        else:
            self.success_results = self.metrics_results.copy()
        # Pre-compute per-metric result sets when overrides exist
        self._success_only = filter_metrics_for_success(self.metrics_results)
        self._all_results = self.metrics_results
        self._derived_metric_results: Dict[str, pd.DataFrame] = {}
        self.aggregate_metric_messages: List[str] = []
        self.aggregate_metric_warnings: List[str] = []
        self._derived_metric_results = self._build_metric_group_aggregate_results()
        available = self.metrics_results["algorithm"].unique()
        self.available_navigators = [
            navigator for navigator in self.ordered_navigators if navigator in available
        ]
        self.navigator_palette = self.get_navigator_palette()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def results_for_metric(self, metric_key: str) -> pd.DataFrame:
        """Return the appropriate results DataFrame for *metric_key*.

        When a per-metric ``discard_failed`` override is set in the
        config it takes priority; otherwise the global
        ``discard_failed_metrics`` flag is used.
        """
        if metric_key in self._derived_metric_results:
            return self._derived_metric_results[metric_key]
        overrides = self.config.metric_discard_failed
        if metric_key in overrides:
            discard = overrides[metric_key]
        else:
            discard = self.config.discard_failed_metrics
        return self._success_only if discard else self._all_results

    def _build_metric_group_aggregate_results(self) -> Dict[str, pd.DataFrame]:
        """Build optional per-run aggregate metrics derived from metric groups."""

        aggregate_results: Dict[str, pd.DataFrame] = {}
        index_cols = ["experiment", "algorithm", "run"]
        for aggregate_key, spec in self.config.metric_group_aggregates.items():
            normalize_mode = str(spec.get("normalize", "minmax")).lower()
            source_metrics = [
                str(metric_key)
                for metric_key in spec.get("metrics", [])
                if str(metric_key) not in self.config.metric_group_aggregates
            ]
            if not source_metrics:
                continue
            normalized_frames: List[pd.DataFrame] = []
            present_metrics: List[str] = []
            limits_spec = spec.get("limits")
            for metric_key in source_metrics:
                source = self.results_for_metric(metric_key)
                if metric_key not in source.columns:
                    continue
                metric_df = source[index_cols + [metric_key]].copy()
                metric_df[metric_key] = pd.to_numeric(
                    metric_df[metric_key], errors="coerce"
                )
                metric_df = metric_df.dropna(subset=index_cols + [metric_key])
                if metric_df.empty:
                    continue
                values = metric_df[metric_key].astype(float)
                if self.config.metric_directions.get(metric_key, "UP").upper() == "DOWN":
                    values = -values
                norm_col = f"__norm_{metric_key}"
                if normalize_mode == "none":
                    metric_df[norm_col] = values
                else:
                    explicit_limits: Optional[Tuple[float, float]] = None
                    if isinstance(limits_spec, (list, tuple)) and len(limits_spec) == 2:
                        try:
                            explicit_limits = (
                                float(limits_spec[0]),
                                float(limits_spec[1]),
                            )
                        except (TypeError, ValueError):
                            explicit_limits = None
                    elif isinstance(limits_spec, dict):
                        raw_limits = limits_spec.get(
                            metric_key, limits_spec.get("default")
                        )
                        if isinstance(raw_limits, (list, tuple)) and len(raw_limits) == 2:
                            try:
                                explicit_limits = (
                                    float(raw_limits[0]),
                                    float(raw_limits[1]),
                                )
                            except (TypeError, ValueError):
                                explicit_limits = None
                    if explicit_limits is None:
                        experiment_stats = metric_df.groupby("experiment")[metric_key].agg(
                            ["min", "max"]
                        )
                        mins = metric_df["experiment"].map(
                            experiment_stats["min"]
                        ).astype(float)
                        maxs = metric_df["experiment"].map(
                            experiment_stats["max"]
                        ).astype(float)
                    else:
                        mins = pd.Series(explicit_limits[0], index=metric_df.index)
                        maxs = pd.Series(explicit_limits[1], index=metric_df.index)
                    if self.config.metric_directions.get(metric_key, "UP").upper() == "DOWN":
                        orig_mins = mins.copy()
                        mins = -maxs
                        maxs = -orig_mins
                    denom = (maxs - mins).replace(0.0, 1.0)
                    metric_df[norm_col] = ((values - mins) / denom).clip(0.0, 1.0)
                normalized_frames.append(metric_df[index_cols + [norm_col]])
                present_metrics.append(metric_key)
            if not normalized_frames:
                missing_metrics = [
                    metric_key
                    for metric_key in source_metrics
                    if metric_key not in present_metrics
                ]
                if missing_metrics:
                    self.aggregate_metric_warnings.append(
                        "Aggregate metric "
                        f"'{aggregate_key}' skipped because its source metrics "
                        f"are missing from the loaded results: {', '.join(missing_metrics)}"
                    )
                continue
            merged = normalized_frames[0]
            for frame in normalized_frames[1:]:
                merged = merged.merge(frame, on=index_cols, how="outer")
            norm_cols = [f"__norm_{metric_key}" for metric_key in present_metrics]
            weights_raw = spec.get("weights", {})
            weight_values = pd.Series(
                {
                    f"__norm_{metric_key}": float(weights_raw.get(metric_key, 1.0))
                    for metric_key in present_metrics
                }
            )
            if np.isclose(weight_values.sum(), 0.0):
                weight_values[:] = 1.0
            normalized = merged[norm_cols].astype(float)
            require_complete = bool(spec.get("require_complete", True))
            exclusions_raw = spec.get("exclude_metrics_by_experiment", {})
            exclusions_by_experiment: Dict[str, set[str]] = {}
            if isinstance(exclusions_raw, dict):
                exclusions_by_experiment = {
                    str(experiment_name): {
                        f"__norm_{metric_key}"
                        for metric_key in metric_keys
                        if f"__norm_{metric_key}" in norm_cols
                    }
                    for experiment_name, metric_keys in exclusions_raw.items()
                }
            if exclusions_by_experiment:
                exclusion_parts = []
                for experiment_name in sorted(exclusions_by_experiment):
                    excluded_cols = sorted(exclusions_by_experiment[experiment_name])
                    if not excluded_cols:
                        continue
                    excluded_metrics = [
                        norm_col.removeprefix("__norm_") for norm_col in excluded_cols
                    ]
                    exclusion_parts.append(
                        f"{experiment_name}: {', '.join(excluded_metrics)}"
                    )
                if exclusion_parts:
                    self.aggregate_metric_messages.append(
                        "Aggregate metric "
                        f"'{aggregate_key}' applying experiment-specific exclusions: "
                        + "; ".join(exclusion_parts)
                    )
            if exclusions_by_experiment:
                def _score_row(row: pd.Series) -> float:
                    excluded_cols = exclusions_by_experiment.get(
                        str(row["experiment"]), set()
                    )
                    applicable_cols = [
                        norm_col for norm_col in norm_cols if norm_col not in excluded_cols
                    ]
                    if not applicable_cols:
                        return np.nan
                    row_values = row[applicable_cols].astype(float)
                    row_weights = weight_values.reindex(applicable_cols).astype(float)
                    if require_complete:
                        if row_values.isna().any():
                            return np.nan
                        return float(
                            row_values.mul(row_weights).sum() / float(row_weights.sum())
                        )
                    valid_mask = row_values.notna()
                    if not valid_mask.any():
                        return np.nan
                    valid_weights = row_weights.loc[valid_mask]
                    denom = float(valid_weights.sum())
                    if np.isclose(denom, 0.0):
                        return np.nan
                    return float(
                        row_values.loc[valid_mask].mul(valid_weights).sum() / denom
                    )

                score = merged.apply(_score_row, axis=1)
            elif require_complete:
                valid = normalized.notna().all(axis=1)
                score = normalized.mul(weight_values, axis=1).sum(axis=1)
                score = score.divide(float(weight_values.sum()))
                score = score.where(valid)
            else:
                weighted = normalized.mul(weight_values, axis=1)
                denom = normalized.notna().mul(weight_values, axis=1).sum(axis=1)
                score = weighted.sum(axis=1, skipna=True).divide(
                    denom.where(denom > 0, np.nan)
                )
            if str(spec.get("direction", "UP")).upper() == "DOWN":
                if normalize_mode == "none":
                    score = -score
                else:
                    score = 1.0 - score
            aggregate_df = merged[index_cols].copy()
            aggregate_df[aggregate_key] = score
            aggregate_df = aggregate_df.dropna(subset=[aggregate_key]).reset_index(
                drop=True
            )
            if not aggregate_df.empty:
                aggregate_results[aggregate_key] = aggregate_df
            else:
                self.aggregate_metric_warnings.append(
                    "Aggregate metric "
                    f"'{aggregate_key}' produced no plottable rows after filtering "
                    "and completeness checks"
                )
        return aggregate_results

    def _load_metrics(self) -> pd.DataFrame:
        metrics_results = load_metrics_from_dir(
            str(self.config.root_dir), recursive=True
        )
        experiment_column = next(
            (col for col in ("experiment_tag", "experiment") if col in metrics_results.columns),
            None,
        )
        if experiment_column is None:
            raise KeyError(
                "Metrics files must include either 'experiment_tag' or 'experiment' column"
            )
        # The env is only encoded in the folder layout (<root>/<algo>/<env>/...),
        # while the CSVs tag rows with the bare episode name. Rebuild
        # "<env>/<episode>" so per-env episodes (as used in scenario
        # episode_groups) can be told apart.
        if "source_dir" in metrics_results.columns:
            env = metrics_results["source_dir"].str.split("/").str[1]
            tags = metrics_results[experiment_column].astype(str)
            needs_prefix = env.notna() & ~tags.str.contains("/")
            metrics_results.loc[needs_prefix, experiment_column] = (
                env[needs_prefix] + "/" + tags[needs_prefix]
            )
            metrics_results = metrics_results.drop(columns=["source_dir"])
        algorithm_column = next(
            (col for col in ("algorithm", "navigator") if col in metrics_results.columns),
            None,
        )
        if algorithm_column is None:
            raise KeyError(
                "Metrics files must include an 'algorithm' column to identify navigators"
            )
        mask_experiments = metrics_results[experiment_column].isin(self.config.episodes)
        mask_algorithms = metrics_results[algorithm_column].isin(self.config.navigators)
        mask = mask_experiments & mask_algorithms
        df = metrics_results.loc[mask].copy()
        if df.empty:
            df = metrics_results.loc[mask_algorithms].copy()
        df = df.dropna(axis=1, how="all")
        df["scenario"] = self.config.scenario
        rename_map: Dict[str, str] = {}
        if experiment_column != "experiment":
            rename_map[experiment_column] = "experiment"
        if algorithm_column != "algorithm":
            rename_map[algorithm_column] = "algorithm"
        if "run_id" in df.columns and "run" not in df.columns:
            rename_map["run_id"] = "run"
        df = rename_metrics_columns(df, rename_map)
        if "run" not in df.columns:
            df["run"] = np.arange(len(df))
        df["algorithm"] = df["algorithm"].replace(self.config.navigator_names)
        df["experiment"] = df["experiment"].replace(self.config.episode_names)
        if {
            "social_space_intrusions",
            "personal_space_intrusions",
            "intimate_space_intrusions",
        }.issubset(df.columns):
            df["public_space_intrusions"] = (
                100
                - df["social_space_intrusions"]
                - df["personal_space_intrusions"]
                - df["intimate_space_intrusions"]
            )
        if {
            "group_social_space_intrusions",
            "group_personal_space_intrusions",
            "group_intimate_space_intrusions",
        }.issubset(df.columns):
            df["group_public_space_intrusions"] = (
                100
                - df["group_social_space_intrusions"]
                - df["group_personal_space_intrusions"]
                - df["group_intimate_space_intrusions"]
            )
        if {"social_work", "time_to_reach_goal"}.issubset(df.columns):
            df["social_work_per_second"] = df["social_work"] / df["time_to_reach_goal"]
        if "result" not in df.columns:
            if "completed" in df.columns:
                df["result"] = np.where(df["completed"], "Success", "Failure")
            else:
                df["result"] = "Success"
        return df

    def output_dir(self) -> Path:
        if self.config.output_dir is not None:
            return self.config.output_dir
        return self.config.root_dir / "reports" / self.config.scenario

    def output_dir_tables(self) -> Path:
        return self.output_dir() / "tables"

    def output_dir_figures(self) -> Path:
        return self.output_dir() / "plots"

    @staticmethod
    def _ensure_parent_directory(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_label(text: str) -> str:
        cleaned = text.replace("\\\\", "\\")
        patterns = [
            (r"\\textbf\{([^}]*)\}", r"\1"),
            (r"\\mathbf\{([^}]*)\}", r"\1"),
            (r"\\mathrm\{([^}]*)\}", r"\1"),
            (r"\\textrm\{([^}]*)\}", r"\1"),
        ]
        for pattern, repl in patterns:
            cleaned = re.sub(pattern, repl, cleaned)
        replacements = {
            r"\%": "%",
            r"\_": "_",
            "$": "",
            "{": "",
            "}": "",
            r"\uparrow": "↑",
            r"\downarrow": "↓",
            r"\tau": "τ",
            r"\psi": "ψ",
        }
        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)
        cleaned = re.sub(r"\\[A-Za-z]+", "", cleaned)
        cleaned = cleaned.replace("_", " ")
        cleaned = re.sub("[ \t\n\r\f\v]+", " ", cleaned)
        return cleaned.strip()

    def _format_text(self, text: str) -> str:
        return text if self.config.use_tex else self._sanitize_label(text)

    def _resolve_navigator_label(self, label: str) -> Optional[str]:
        if label in self.ordered_navigators:
            return label
        sanitized = self._sanitize_label(label)
        for navigator_label in self.ordered_navigators:
            if self._sanitize_label(navigator_label) == sanitized:
                return navigator_label
        for raw_key, mapped_label in self.config.navigator_names.items():
            if raw_key == label:
                return mapped_label
            if mapped_label == label:
                return mapped_label
            if self._sanitize_label(mapped_label) == sanitized:
                return mapped_label
            if self._sanitize_label(raw_key) == sanitized:
                return self.config.navigator_names.get(raw_key, raw_key)
        return None

    def _style_axes(
        self,
        axes_iter: Iterable[Any],
        *,
        title_size: int = 18,
        label_size: int = 16,
        tick_size: int = 13,
    ) -> None:
        for ax in axes_iter:
            if not hasattr(ax, "set_title") or not ax.get_visible():
                continue
            title = ax.get_title()
            if title:
                ax.set_title(self._format_text(title), fontsize=title_size, pad=10)
            xlabel = ax.get_xlabel()
            if xlabel:
                ax.set_xlabel(self._format_text(xlabel), fontsize=label_size)
            ylabel = ax.get_ylabel()
            if ylabel:
                ax.set_ylabel(self._format_text(ylabel), fontsize=label_size)
            ax.tick_params(axis="both", labelsize=tick_size)
            for tick in ax.get_xticklabels():
                text = tick.get_text()
                if text:
                    tick.set_text(self._format_text(text))
                tick.set_fontsize(tick_size)
            for tick in ax.get_yticklabels():
                text = tick.get_text()
                if text:
                    tick.set_text(self._format_text(text))
                tick.set_fontsize(tick_size)

    def get_navigator_palette(
        self, cmap_name: str = "Paired"
    ) -> Dict[str, Tuple[float, float, float, float]]:
        cmap = plt.get_cmap(cmap_name, len(self.ordered_navigators))
        return {label: cmap(i) for i, label in enumerate(self.ordered_navigators)}

    def get_result_palette(self) -> Dict[str, Tuple[float, float, float, float]]:
        cmap = plt.get_cmap(self.config.quality_result_cmap)
        return {result: cmap(idx) for idx, result in enumerate(self.results)}

    def get_intrusion_palette(self) -> Dict[str, Tuple[float, float, float, float]]:
        intrusion_labels = list(self.config.intrusion_labels.values())
        cmap = plt.get_cmap(self.config.quality_result_cmap)
        return {label: cmap(idx) for idx, label in enumerate(intrusion_labels)}

    def _experiment_order_with_groups(
        self, available: Optional[Iterable[str]] = None
    ) -> List[str]:
        """Return experiments followed by configured episode groups."""

        available_set = set(available) if available is not None else None
        ordered: List[str] = []

        def _extend(items: Iterable[str]) -> None:
            for item in items:
                if not item:
                    continue
                if available_set is not None and item not in available_set:
                    continue
                if item not in ordered:
                    ordered.append(item)

        _extend(self.ordered_experiments)
        if self.ordered_episode_groups:
            _extend(self.ordered_episode_groups)
        return ordered

    # ------------------------------------------------------------------
    # Data summaries
    # ------------------------------------------------------------------
    @property
    def success_rate(self) -> pd.DataFrame:
        if not hasattr(self, "_success_rate"):
            base_sr = (
                self.metrics_results.groupby(["experiment", "algorithm"])["result"]
                .apply(lambda values: (values == "Success").mean() * 100)
                .reset_index()
            )
            sr_frames = [base_sr]
            if self.config.episode_groups:
                group_frames = []
                for group_name, members in self.config.episode_groups.items():
                    subset = self.metrics_results[
                        self.metrics_results["experiment"].isin(members)
                    ]
                    if subset.empty:
                        continue
                    group_sr = (
                        subset.groupby(["algorithm"])["result"]
                        .apply(lambda values: (values == "Success").mean() * 100)
                        .reset_index()
                    )
                    group_sr["experiment"] = group_name
                    group_frames.append(group_sr)
                if group_frames:
                    sr_frames.append(pd.concat(group_frames, ignore_index=True))
            sr = pd.concat(sr_frames, ignore_index=True)
            sr = sr.rename(
                columns={
                    "experiment": r"\textbf{Experiment}",
                    "algorithm": r"\textbf{Navigator}",
                    "result": r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$",
                }
            )
            pair_cols = [r"\textbf{Experiment}", r"\textbf{Navigator}"]
            sr = sr.drop_duplicates(subset=pair_cols, keep="first")
            sr = sr.set_index(pair_cols)
            experiment_order = self._experiment_order_with_groups(
                sr.index.get_level_values(r"\textbf{Experiment}").unique()
            )
            sr = sr.reindex(experiment_order, level=0)
            sr = sr.reindex(self.ordered_navigators, level=1)
            # Missing experiment/navigator combinations become NaN after the
            # reindex above. Kept as float: an int cast truncated 29/30 to 96.
            sr[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"] = (
                sr[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"]
                .round(2)
                .fillna(0)
            )
            self._success_rate = sr
        return self._success_rate


__all__ = ["BenchmarkReportBase"]
