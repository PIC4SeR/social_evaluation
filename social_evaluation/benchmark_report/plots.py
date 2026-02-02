"""Plotting mixins for benchmark reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from social_evaluation.benchmark_report.utils import (
    get_metrics_summary,
    select_metrics_by_columns,
)

from .config import BenchmarkConfig

if TYPE_CHECKING:  # pragma: no cover - linting aid only

    class _BaseProtocol:
        config: BenchmarkConfig
        metrics_results_parsed: pd.DataFrame
        metrics_results: pd.DataFrame
        success_results: pd.DataFrame
        navigator_palette: Dict[str, Tuple[float, float, float, float]]
        available_navigators: List[str]
        ordered_experiments: List[str]
        ordered_navigators: List[str]
        results: List[str]

        def output_dir_figures(self) -> Path: ...

        def get_result_palette(
            self,
        ) -> Dict[str, Tuple[float, float, float, float]]: ...

        def get_intrusion_palette(
            self,
        ) -> Dict[str, Tuple[float, float, float, float]]: ...

        def _format_text(self, text: str) -> str: ...

        def _ensure_parent_directory(self, path: Path) -> None: ...


class PlottingMixin:
    """Provide the high-level plotting helpers for benchmark reports."""

    if TYPE_CHECKING:  # pragma: no cover - linting aid only
        config: BenchmarkConfig
        metrics_results_parsed: pd.DataFrame
        metrics_results: pd.DataFrame
        success_results: pd.DataFrame
        navigator_palette: Dict[str, Tuple[float, float, float, float]]
        available_navigators: List[str]
        ordered_experiments: List[str]
        ordered_navigators: List[str]
        results: List[str]
        output_dir_figures: _BaseProtocol.output_dir_figures  # type: ignore[assignment]
        get_result_palette: _BaseProtocol.get_result_palette  # type: ignore[assignment]
        get_intrusion_palette: _BaseProtocol.get_intrusion_palette  # type: ignore[assignment]
        _format_text: _BaseProtocol._format_text  # type: ignore[assignment]
        _ensure_parent_directory: _BaseProtocol._ensure_parent_directory  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # High-level plotting helpers
    # ------------------------------------------------------------------
    def plot_success_rate(self) -> None:
        """Plot stacked bar charts of success metrics for each experiment."""

        output_path = self.output_dir_figures() / "success_rate_by_navigator.png"
        self._ensure_parent_directory(output_path)
        n_cols = 3
        n_rows = math.ceil(len(self.ordered_experiments) / n_cols)
        env_metrics_cache: Dict[str, pd.DataFrame] = {}
        env_navigator_map: Dict[str, List[str]] = {}
        max_nav_count = 1
        for environment in self.ordered_experiments:
            env_metrics = self.metrics_results_parsed[
                self.metrics_results_parsed["experiment"] == environment
            ]
            env_metrics_cache[environment] = env_metrics
            env_navigators = [
                navigator
                for navigator in self.available_navigators
                if navigator in env_metrics["algorithm"].unique()
            ]
            env_navigator_map[environment] = env_navigators
            if env_navigators:
                max_nav_count = max(max_nav_count, len(env_navigators))

        subplot_width, subplot_height, tick_font_size, title_font_size = (
            self._layout_params_for_navigators(max_nav_count)
        )
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * subplot_width, n_rows * subplot_height),
            sharey=True,
            sharex=True,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)
        result_palette = self.get_result_palette()
        for idx, environment in enumerate(self.ordered_experiments):
            row_idx, col_idx = divmod(idx, n_cols)
            target_ax = axes[row_idx, col_idx]
            env_metrics = env_metrics_cache.get(environment, pd.DataFrame())
            env_navigators = env_navigator_map.get(environment, [])
            if not env_navigators:
                target_ax.axis("off")
                continue
            means = (
                env_metrics.groupby("algorithm")[self.results]
                .mean()
                .reindex(env_navigators)
            )
            means = means.reindex(columns=self.results).fillna(0.0)
            means.plot.bar(
                stacked=True,
                color=[tuple(result_palette[result]) for result in self.results],
                ax=target_ax,
                alpha=self.config.alpha,
            )
            target_ax.set_title(
                self._format_text(environment), fontsize=title_font_size
            )
            target_ax.set_ylim(0, 1)
            target_ax.set_yticks(
                [0, 0.2, 0.4, 0.6, 0.8, 1],
                ["0%", "20%", "40%", "60%", "80%", "100%"],
            )
            display_navigators = [self._format_text(label) for label in env_navigators]
            target_ax.set_xticklabels(display_navigators, rotation=45, ha="right")
            target_ax.set_xlabel("")
            target_ax.tick_params(axis="x", labelsize=tick_font_size)
            target_ax.tick_params(axis="y", labelsize=tick_font_size)
            for label in target_ax.get_xticklabels():
                label.set_weight("bold")
            for label in target_ax.get_yticklabels():
                label.set_weight("bold")
            target_ax.legend().remove()
        for idx in range(len(self.ordered_experiments), n_rows * n_cols):
            row_idx, col_idx = divmod(idx, n_cols)
            axes[row_idx, col_idx].axis("off")
        fig.supylabel(
            self._format_text(r"\textbf{Success Rate (\%)}"), fontsize=title_font_size
        )
        result_display_labels = [
            r"\textbf{Success}",
            r"\textbf{Timeout}",
            r"\textbf{Collision}",
            r"\textbf{Navigation Failure}",
        ]
        legend_handles: List[Patch] = []
        legend_labels: List[str] = []
        for result, display_label in zip(self.results, result_display_labels):
            formatted_label = self._format_text(display_label)
            legend_handles.append(
                Patch(
                    facecolor=cast(Any, result_palette[result]),
                    edgecolor="black",
                    label=formatted_label,
                )
            )
            legend_labels.append(formatted_label)
        legend_font_size = max(tick_font_size - 2, 12)
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=4,
            fontsize=legend_font_size,
            frameon=False,
            labelspacing=1.5,
        )
        fig.tight_layout(rect=(0.001, 0.05, 1.0, 1.0))
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def plot_metric_distributions(self, kind: str, cmap_name: str = "Paired") -> None:
        """Plot distribution summaries (box or violin) for each metric group."""

        output_path = Path(self.output_dir_figures())
        output_path.mkdir(parents=True, exist_ok=True)
        palette = self.navigator_palette
        legend_handles = [
            Patch(
                facecolor=cast(Any, palette[label]),
                edgecolor="black",
                label=self._format_text(label),
            )
            for label in palette
        ]
        rows = len(self.ordered_experiments)
        for group_key, metric_labels in self.config.metric_groups.items():
            cols = len(metric_labels)
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=(cols * 12, rows * 10),
                sharex=True,
            )
            axes = np.array(axes).reshape(rows, cols)
            for row_idx, environment in enumerate(self.ordered_experiments):
                env_metrics = self.success_results[
                    self.success_results["experiment"] == environment
                ].copy()

                for col_idx, metric_key in enumerate(metric_labels.keys()):
                    target_ax = axes[row_idx, col_idx]
                    if kind == "box":
                        sns.boxplot(
                            data=env_metrics,
                            x="algorithm",
                            order=self.ordered_navigators,
                            y=metric_key,
                            hue="algorithm",
                            hue_order=self.ordered_navigators,
                            palette=cast(Any, palette),
                            ax=target_ax,
                            showfliers=False,
                        )
                    else:
                        sns.violinplot(
                            data=env_metrics,
                            x="algorithm",
                            order=self.ordered_navigators,
                            y=metric_key,
                            hue="algorithm",
                            hue_order=self.ordered_navigators,
                            palette=cast(Any, palette),
                            ax=target_ax,
                        )
                    if row_idx == 0:
                        target_ax.set_title(
                            self._format_text(metric_labels[metric_key]),
                            fontsize=37,
                        )
                    if col_idx == 0:
                        target_ax.set_ylabel(environment, fontsize=37)
                    else:
                        target_ax.set_ylabel("")
                    target_ax.set_xlabel("")
                    target_ax.set_xticklabels([])
                    legend = target_ax.get_legend()
                    if legend is not None:
                        legend.remove()
                    for item in list(target_ax.get_yticklabels()) + [
                        target_ax.yaxis.label
                    ]:
                        item.set_fontsize(37)
            output_file = output_path / f"{group_key}_{kind}plots_by_navigator.png"
            legend_labels = [
                self._format_text(str(handle.get_label())) for handle in legend_handles
            ]
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=8,
                fontsize=37,
                labelspacing=1.5,
            )
            fig.tight_layout(rect=(0.02, 0.1, 0.99, 0.98))
            fig.savefig(output_file, bbox_inches="tight", format="png")
            plt.close(fig)

    def plot_proxemic_intrusions(self) -> None:
        """Plot stacked bar charts for proxemic intrusion statistics."""

        output_path = self.output_dir_figures() / "proxemic_distances_intrusions.png"
        self._ensure_parent_directory(output_path)
        metrics_df = select_metrics_by_columns(
            self.success_results, list(self.config.intrusion_labels.keys())
        )
        metrics_df = metrics_df.rename(columns=self.config.intrusion_labels)
        summary = get_metrics_summary(metrics_df)["mean"].reset_index()
        summary = summary.rename(
            columns={
                "experiment": r"\textbf{Experiment}",
                "algorithm": r"\textbf{Navigator}",
            }
        )
        intrusion_columns: List[str] = [
            str(label) for label in self.config.intrusion_labels.values()
        ]
        intrusion_palette = self.get_intrusion_palette()
        rows = 2
        cols = math.ceil(len(self.ordered_experiments) / rows)
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 12, rows * 17),
            sharey=True,
            sharex=True,
        )
        axes = np.array(axes).reshape(rows, cols)
        for idx, environment in enumerate(self.ordered_experiments):
            row_idx, col_idx = divmod(idx, cols)
            target_ax = axes[row_idx, col_idx]
            env_summary = summary[summary[r"\textbf{Experiment}"] == environment]
            env_navigators = [
                navigator
                for navigator in self.available_navigators
                if navigator in env_summary[r"\textbf{Navigator}"].unique()
            ]
            if not env_navigators:
                target_ax.axis("off")
                continue
            env_means = (
                env_summary.set_index(r"\textbf{Navigator}")
                .reindex(env_navigators)
                .reset_index()
            )
            env_means[intrusion_columns] = env_means[intrusion_columns].fillna(0.0)
            env_means.plot.bar(
                ax=target_ax,
                x=r"\textbf{Navigator}",
                y=cast(Sequence, intrusion_columns),
                stacked=True,
                color=[tuple(intrusion_palette[label]) for label in intrusion_columns],
                alpha=self.config.alpha,
            )
            target_ax.set_xticklabels(
                [
                    self._format_text(label.get_text())
                    for label in target_ax.get_xticklabels()
                ],
                rotation=45,
                ha="right",
            )
            for label in target_ax.get_xticklabels():
                label.set_fontsize(32)
            target_ax.set_title(environment, fontsize=40)
            target_ax.legend().remove()
            target_ax.set_xlabel("")
            target_ax.set_ylim(0, 100)
            target_ax.set_yticks([25, 50, 75, 100])
            for item in target_ax.get_yticklabels():
                item.set_fontsize(40)
            for item in [target_ax.title, target_ax.xaxis.label, target_ax.yaxis.label]:
                item.set_fontsize(40)
        for idx in range(len(self.ordered_experiments), rows * cols):
            row_idx, col_idx = divmod(idx, cols)
            axes[row_idx, col_idx].axis("off")
        fig.supylabel(self._format_text(r"Intrusions (\%)"), fontsize=45, y=0.62)
        intrusion_handles: List[Patch] = []
        intrusion_labels_formatted: List[str] = []
        for label in intrusion_columns:
            formatted_label = self._format_text(label)
            intrusion_handles.append(
                Patch(
                    facecolor=cast(Any, intrusion_palette[label]),
                    edgecolor="black",
                    label=formatted_label,
                )
            )
            intrusion_labels_formatted.append(formatted_label)
        fig.legend(
            intrusion_handles,
            intrusion_labels_formatted,
            loc="lower center",
            ncol=4,
            fontsize=45,
            frameon=False,
        )
        fig.tight_layout(rect=(0.02, 0.1, 0.99, 0.98))
        fig.savefig(output_path, bbox_inches="tight", format="png")
        plt.close(fig)

    @staticmethod
    def _layout_params_for_navigators(
        max_navigators: int,
    ) -> Tuple[float, float, int, int]:
        """Return subplot sizing and font scales tuned to navigator counts."""

        safe_count = max(1, max_navigators)
        subplot_width = max(4.5, 3.0 + 0.3 * safe_count)
        subplot_height = max(4.0, 4.0 + 0.04 * safe_count)
        if safe_count <= 6:
            tick_font = 22
        elif safe_count <= 10:
            tick_font = 20
        elif safe_count <= 14:
            tick_font = 18
        elif safe_count <= 18:
            tick_font = 16
        else:
            tick_font = 14
        title_font = tick_font + 4
        return subplot_width, subplot_height, tick_font, title_font


__all__ = ["PlottingMixin"]
