"""rliable plotting helpers for benchmark reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from .config import BenchmarkConfig

if TYPE_CHECKING:  # pragma: no cover - linting aid only
    import pandas as pd

    class _BaseProtocol:
        config: BenchmarkConfig
        success_results: pd.DataFrame
        navigator_palette: Dict[str, Tuple[float, float, float, float]]
        ordered_experiments: List[str]
        ordered_navigators: List[str]
        available_navigators: List[str]

        def _style_axes(
            self,
            axes_iter: Iterable[Any],
            *,
            title_size: int,
            label_size: int,
            tick_size: int,
        ) -> None: ...

        def output_dir_figures(self) -> Path: ...

        def _ensure_parent_directory(self, path: Path) -> None: ...

        def metric_direction(self, metric_key: str) -> str: ...

        def metric_plain_label(self, metric_key: str) -> str: ...

        def _format_text(self, text: str) -> str: ...

        def _resolve_navigator_label(self, label: str) -> Optional[str]: ...

        def _sanitize_label(self, text: str) -> str: ...

        def get_navigator_palette(
            self, cmap_name: str = "Paired"
        ) -> Dict[str, Tuple[float, float, float, float]]: ...


class RliableMixin:
    """Provide probability-of-improvement and related rliable plots."""

    if TYPE_CHECKING:  # pragma: no cover - linting aid only
        config: BenchmarkConfig
        success_results: pd.DataFrame
        navigator_palette: Dict[str, Tuple[float, float, float, float]]
        ordered_experiments: List[str]
        ordered_navigators: List[str]
        available_navigators: List[str]
        _style_axes: _BaseProtocol._style_axes  # type: ignore[assignment]
        output_dir_figures: _BaseProtocol.output_dir_figures  # type: ignore[assignment]
        _ensure_parent_directory: _BaseProtocol._ensure_parent_directory  # type: ignore[assignment]
        _format_text: _BaseProtocol._format_text  # type: ignore[assignment]
        _resolve_navigator_label: _BaseProtocol._resolve_navigator_label  # type: ignore[assignment]
    _sanitize_label: _BaseProtocol._sanitize_label  # type: ignore[assignment]
    get_navigator_palette: _BaseProtocol.get_navigator_palette  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def metric_plain_label(self, metric_key: str) -> str:
        for labels in self.config.metric_groups.values():
            if metric_key in labels:
                return self._sanitize_label(labels[metric_key])
        return self._sanitize_label(metric_key)

    def metric_direction(self, metric_key: str) -> str:
        return self.config.metric_directions.get(metric_key, "UP").upper()

    def metric_score_matrices(
        self, metric_key: str, normalize: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, float]]]:
        columns = ["experiment", "algorithm", "run", metric_key]
        df = self.success_results[columns].dropna(
            subset=["experiment", "algorithm", "run"]
        )
        if df.empty:
            return {}, None
        experiment_stats = df.groupby("experiment")[metric_key].agg(["min", "max"])
        matrices: Dict[str, np.ndarray] = {}
        direction = self.metric_direction(metric_key)
        used_experiments: Set[str] = set()
        for navigator_label in self.ordered_navigators:
            alg_df = df[df["algorithm"] == navigator_label]
            if alg_df.empty:
                continue
            pivot = (
                alg_df.pivot(index="run", columns="experiment", values=metric_key)
                .reindex(columns=self.ordered_experiments)
                .dropna(axis=1, how="all")
            )
            if pivot.empty:
                continue
            pivot = pivot.dropna(axis=0, how="any")
            if pivot.empty:
                continue
            pivot = pivot.sort_index()
            used_experiments.update(pivot.columns.tolist())

            if direction == "DOWN":
                pivot = -pivot
                column_mins = -experiment_stats.loc[pivot.columns, "max"].astype(float)
                column_maxs = -experiment_stats.loc[pivot.columns, "min"].astype(float)
            else:
                column_mins = experiment_stats.loc[pivot.columns, "min"].astype(float)
                column_maxs = experiment_stats.loc[pivot.columns, "max"].astype(float)

            if normalize and not column_mins.empty and not column_maxs.empty:
                denom = column_maxs - column_mins
                denom = denom.replace(0, 1.0)
                pivot = pivot.subtract(column_mins, axis=1)
                pivot = pivot.divide(denom, axis=1)

            pivot_values = pivot.to_numpy(dtype=float)
            if pivot_values.size == 0:
                continue
            matrices[navigator_label] = pivot_values

        stats = None
        if normalize and used_experiments:
            used_list = sorted(used_experiments)
            min_vals = experiment_stats.loc[used_list, "min"].to_numpy(dtype=float)
            max_vals = experiment_stats.loc[used_list, "max"].to_numpy(dtype=float)
            if not (np.all(np.isnan(min_vals)) or np.all(np.isnan(max_vals))):
                stats = {
                    "min": float(np.nanmin(min_vals)),
                    "max": float(np.nanmax(max_vals)),
                }
        return matrices, stats

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plot_rliable_interval_estimates(
        self,
        metric_key: str,
        normalize: bool = True,
        reps: int = 5000,
    ) -> None:
        matrices, stats = self.metric_score_matrices(metric_key, normalize=normalize)
        if not matrices:
            raise ValueError(f"No metric data available for {metric_key!r}")
        aggregate_fn = lambda scores: np.array(
            [
                metrics.aggregate_median(scores),
                metrics.aggregate_iqm(scores),
                metrics.aggregate_mean(scores),
                metrics.aggregate_optimality_gap(scores, 1),
            ]
        )
        aggregate_scores, aggregate_cis = rly.get_interval_estimates(
            matrices, aggregate_fn, reps=reps
        )
        navigators = list(aggregate_scores.keys())
        palette = self.get_navigator_palette()
        colors = {
            navigator: palette.get(navigator, (0.2, 0.2, 0.2, 1.0))
            for navigator in navigators
        }
        metric_names = ["Median", "IQM", "Mean", "Optimality Gap"]
        plot_result = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_cis,
            metric_names=metric_names,
            algorithms=navigators,
            colors=colors,
        )
        if isinstance(plot_result, tuple):
            fig, axes = plot_result
        else:
            fig = plot_result
            axes = fig.axes
        axes_iter: Iterable[Any] = []
        if isinstance(axes, np.ndarray):
            axes_iter = list(axes.flat)
        elif isinstance(axes, (list, tuple)):
            axes_iter = list(axes)
        elif axes is not None:
            axes_iter = [axes]
        title_suffix = ""
        if normalize and stats:
            title_suffix = f" (normalized {stats['min']:.2f}–{stats['max']:.2f})"
        axes_iter = list(axes_iter)
        fig.suptitle(
            f"{self.metric_plain_label(metric_key)}{title_suffix}", fontsize=18, y=0.95
        )
        display_navigators = [self._format_text(name) for name in navigators]
        for ax in axes_iter:
            if not hasattr(ax, "set_yticklabels"):
                continue
            yticks = ax.get_yticks()
            if len(yticks) != len(display_navigators):
                ax.set_yticks(np.arange(len(display_navigators)))
            ax.set_yticklabels(display_navigators)
        fig_height = max(3.5, 1.2 + 0.5 * len(navigators))
        fig.set_size_inches(12.0, fig_height)
        self._style_axes(axes_iter, title_size=18, label_size=16, tick_size=12)
        output_path = (
            self.output_dir_figures()
            / "rliable"
            / f"{metric_key}_interval_estimates.png"
        )
        self._ensure_parent_directory(output_path)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def plot_probability_of_improvement(
        self,
        metric_key: str,
        baseline_algorithm: str,
        normalize: bool = True,
        reps: int = 2000,
    ) -> None:
        matrices, _ = self.metric_score_matrices(metric_key, normalize=normalize)
        if not matrices:
            raise ValueError(f"No metric data available for {metric_key!r}")
        baseline_label = self._resolve_navigator_label(baseline_algorithm)
        if baseline_label is None:
            available = ", ".join(sorted(matrices.keys()))
            raise ValueError(
                f"Baseline navigator {baseline_algorithm!r} missing from metric data. "
                f"Available navigators: {available}"
            )
        if baseline_label not in matrices:
            raise ValueError(
                f"Baseline navigator {baseline_algorithm!r} missing from metric data"
            )
        pair_dict = {}
        baseline_scores = matrices[baseline_label]
        for navigator_label, scores in matrices.items():
            if navigator_label == baseline_label:
                continue
            tasks = min(baseline_scores.shape[1], scores.shape[1])
            if tasks == 0:
                continue
            aligned_baseline = baseline_scores[:, :tasks]
            aligned_scores = scores[:, :tasks]
            pair_dict[navigator_label] = (aligned_baseline, aligned_scores)
        if not pair_dict:
            raise ValueError(
                "No comparison algorithms found for probability of improvement plot"
            )
        probability_fn = cast(Any, metrics.probability_of_improvement)
        probabilities, probability_cis = rly.get_interval_estimates(
            pair_dict, probability_fn, reps=reps
        )
        colors = self.navigator_palette
        fig_height = max(3.0, 1.2 + 0.5 * len(pair_dict))
        fig, ax = plt.subplots(figsize=(8.0, fig_height))
        for idx, (navigator_label, prob) in enumerate(probabilities.items()):
            lower, upper = probability_cis[navigator_label]
            ax.barh(
                y=idx,
                width=upper - lower,
                height=0.6,
                left=lower,
                color=cast(Any, colors.get(navigator_label, (0.2, 0.2, 0.2, 1.0))),
                alpha=0.75,
            )
            ax.vlines(prob, idx - 0.25, idx + 0.25, color="k", linewidth=2)
        ax.set_yticks(range(len(pair_dict)))
        ax.set_yticklabels(
            [self._format_text(label) for label in list(pair_dict.keys())]
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            self._format_text(
                f"Probability of Improvement over {self._resolve_navigator_label(baseline_algorithm) or baseline_algorithm}"
            )
        )
        ax.set_title(
            f"{self.metric_plain_label(metric_key)} Probability of Improvement"
        )
        ax.grid(axis="x", alpha=0.2)
        self._style_axes([ax], title_size=18, label_size=16, tick_size=12)
        fig.tight_layout()
        output_path = (
            self.output_dir_figures()
            / "rliable"
            / f"{metric_key}_probability_of_improvement.png"
        )
        self._ensure_parent_directory(output_path)
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def plot_metric_performance_profiles(
        self,
        metric_keys: Optional[Union[str, List[str]]] = None,
        normalize: bool = True,
        reps: int = 2000,
        tau: Optional[np.ndarray] = None,
        use_average_distribution: bool = False,
        output_path: Optional[Path] = None,
        legend_columns: int = 4,
        figsize_scale: float = 6.5,
    ) -> None:
        if metric_keys is None:
            metric_sequence: List[str] = []
            for group_metrics in self.config.metric_groups.values():
                metric_sequence.extend(group_metrics.keys())
            metric_sequence = list(dict.fromkeys(metric_sequence))
        elif isinstance(metric_keys, str):
            if metric_keys in self.config.metric_groups:
                metric_sequence = list(self.config.metric_groups[metric_keys].keys())
            else:
                metric_sequence = [metric_keys]
        else:
            metric_sequence = list(metric_keys)
        if not metric_sequence:
            raise ValueError("No metrics specified for performance profile plotting")

        palette = {
            label: self.navigator_palette[label]
            for label in self.available_navigators
            if label in self.navigator_palette
        }
        num_metrics = len(metric_sequence)
        cols = 2
        rows = math.ceil(num_metrics / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_scale, rows * 4.5))
        axes = np.atleast_1d(axes).ravel()

        plotted_metrics = 0
        for metric_key in metric_sequence:
            matrices, _ = self.metric_score_matrices(metric_key, normalize=normalize)
            if not matrices:
                continue
            matrices = {
                alg: values for alg, values in matrices.items() if alg in palette
            }
            if not matrices:
                continue
            flattened_values: List[np.ndarray] = []
            for matrix in matrices.values():
                matrix_np = np.asarray(matrix, dtype=float)
                if matrix_np.size:
                    flattened_values.append(matrix_np.ravel())
            if not flattened_values:
                continue
            flattened_scores = np.concatenate(flattened_values)
            with np.errstate(all="ignore"):
                flattened_scores = flattened_scores[np.isfinite(flattened_scores)]
            if flattened_scores.size == 0:
                continue
            if tau is None:
                low = float(flattened_scores.min())
                high = float(flattened_scores.max())
                if math.isclose(low, high, rel_tol=1e-9, abs_tol=1e-9):
                    high = low + 1.0
                span = high - low
                margin = 0.05 * span if span > 0 else 0.05
                metric_tau = np.linspace(low - margin, high + margin, 201)
            else:
                metric_tau = tau
            profile, profile_cis = rly.create_performance_profile(
                matrices,
                metric_tau,
                use_score_distribution=not use_average_distribution,
                reps=reps,
            )
            ax = axes[plotted_metrics]
            plot_kwargs = {
                "performance_profile_cis": profile_cis,
                "colors": palette,
                "xlabel": "Normalized Score" if normalize else "Score",
                "labelsize": 12,
                "ax": ax,
            }
            if plotted_metrics % cols == 0:
                ylabel_tex = r"Fraction of runs with score $> \tau$"
                ylabel_plain = "Fraction of runs with score > τ"
                plot_kwargs["ylabel"] = (
                    ylabel_tex if self.config.use_tex else ylabel_plain
                )
            plot_utils.plot_performance_profiles(profile, metric_tau, **plot_kwargs)
            ax.set_title(self.metric_plain_label(metric_key), fontsize=18)
            ax.axhline(0.5, linestyle="--", color="k", alpha=0.3)
            plotted_metrics += 1

        if plotted_metrics == 0:
            plt.close(fig)
            raise ValueError("No metric data available to plot performance profiles")

        for ax in axes[plotted_metrics:]:
            ax.axis("off")

        legend_handles = [
            Patch(
                facecolor=cast(Any, palette[label]),
                edgecolor="black",
                label=self._format_text(label),
            )
            for label in self.ordered_navigators
            if label in palette
        ]
        if legend_handles:
            legend_labels = [str(handle.get_label()) for handle in legend_handles]
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=min(len(legend_handles), legend_columns),
                fontsize=16,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )

        self._style_axes(
            axes,
            title_size=18,
            label_size=16,
            tick_size=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        if output_path is not None:
            self._ensure_parent_directory(output_path)
            fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)


__all__ = ["RliableMixin"]
