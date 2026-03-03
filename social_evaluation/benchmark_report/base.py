"""Core implementation for benchmark reporting."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    plt.rcParams["font.serif"] = ["Times New Roman"]
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
        overrides = self.config.metric_discard_failed
        if metric_key in overrides:
            discard = overrides[metric_key]
        else:
            discard = self.config.discard_failed_metrics
        return self._success_only if discard else self._all_results

    def _load_metrics(self) -> pd.DataFrame:
        metrics_results = load_metrics_from_dir(
            str(self.config.root_dir), recursive=True
        )
        mask = metrics_results["experiment_tag"].isin(
            self.config.episodes
        ) & metrics_results["algorithm"].isin(self.config.navigators)
        df = metrics_results.loc[mask].copy()
        df = df.dropna(axis=1, how="all")
        df["scenario"] = self.config.scenario
        df = rename_metrics_columns(
            df, {"experiment_tag": "experiment", "run_id": "run"}
        )
        df["algorithm"] = df["algorithm"].replace(self.config.navigator_names)
        df["experiment"] = df["experiment"].replace(self.config.episode_names)
        df["public_space_intrusions"] = (
            100
            - df["social_space_intrusions"]
            - df["personal_space_intrusions"]
            - df["intimate_space_intrusions"]
        )
        df["group_public_space_intrusions"] = (
            100
            - df["group_social_space_intrusions"]
            - df["group_personal_space_intrusions"]
            - df["group_intimate_space_intrusions"]
        )
        df["social_work_per_second"] = df["social_work"] / df["time_to_reach_goal"]
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
        cmap = plt.cm.get_cmap(cmap_name, len(self.ordered_navigators))
        return {label: cmap(i) for i, label in enumerate(self.ordered_navigators)}

    def get_result_palette(self) -> Dict[str, Tuple[float, float, float, float]]:
        cmap = plt.cm.get_cmap(self.config.quality_result_cmap)
        return {result: cmap(idx) for idx, result in enumerate(self.results)}

    def get_intrusion_palette(self) -> Dict[str, Tuple[float, float, float, float]]:
        intrusion_labels = list(self.config.intrusion_labels.values())
        cmap = plt.cm.get_cmap(self.config.quality_result_cmap)
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
            sr = sr.set_index([r"\textbf{Experiment}", r"\textbf{Navigator}"])
            experiment_order = self._experiment_order_with_groups(
                sr.index.get_level_values(r"\textbf{Experiment}").unique()
            )
            sr = sr.reindex(experiment_order, level=r"\textbf{Experiment}")
            sr = sr.reindex(self.ordered_navigators, level=r"\textbf{Navigator}")
            sr[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"] = (
                sr[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"]
                .round(2)
                .astype(int)
            )
            self._success_rate = sr
        return self._success_rate


__all__ = ["BenchmarkReportBase"]
