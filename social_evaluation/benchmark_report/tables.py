"""Table construction and export helpers for benchmark reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize

from social_evaluation.benchmark_report.utils import (
    get_metrics_summary,
    select_metrics_by_columns,
)

from .utils import export_latex_table, style_report_latex
from .config import BenchmarkConfig

if TYPE_CHECKING:  # pragma: no cover - linting aid only
    import pandas as pd

    class _BaseProtocol:
        config: BenchmarkConfig
        success_results: pd.DataFrame
        metrics_results: pd.DataFrame
        metrics_results_parsed: pd.DataFrame
        results: List[str]
        ordered_experiments: List[str]
        ordered_navigators: List[str]

        def metric_direction(self, metric_key: str) -> str: ...

        @property
        def success_rate(self) -> pd.DataFrame: ...

        def output_dir_tables(self) -> Path: ...

        def _ensure_parent_directory(self, path: Path) -> None: ...


class TableExportMixin:
    """Provide metric table generation and export capabilities."""

    if TYPE_CHECKING:  # pragma: no cover - linting aid only
        config: BenchmarkConfig
        success_results: pd.DataFrame
        metrics_results: pd.DataFrame
        metrics_results_parsed: pd.DataFrame
        results: List[str]
        ordered_experiments: List[str]
        ordered_navigators: List[str]
        metric_direction: _BaseProtocol.metric_direction  # type: ignore[assignment]
        success_rate: _BaseProtocol.success_rate  # type: ignore[assignment]
        output_dir_tables: _BaseProtocol.output_dir_tables  # type: ignore[assignment]
        _ensure_parent_directory: _BaseProtocol._ensure_parent_directory  # type: ignore[assignment]

    def _metric_summary_tables(
        self, metric_labels: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        subset = select_metrics_by_columns(
            self.success_results, list(metric_labels.keys())
        )
        subset = subset.rename(columns=metric_labels)
        summary = get_metrics_summary(subset)
        mean_table = summary["mean"].rename(
            columns={
                "experiment": r"\textbf{Experiment}",
                "algorithm": r"\textbf{Navigator}",
            }
        )
        std_table = summary["std_dev"].rename(
            columns={
                "experiment": r"\textbf{Experiment}",
                "algorithm": r"\textbf{Navigator}",
            }
        )
        return mean_table, std_table

    def _failure_penalty_table(self, metric_labels: Dict[str, str]) -> pd.DataFrame:
        if not metric_labels:
            return pd.DataFrame()
        metric_keys = [key for key in metric_labels if key in self.metrics_results]
        if not metric_keys:
            return pd.DataFrame()
        data = self.metrics_results[["experiment"] + metric_keys].copy()
        penalties: Dict[str, pd.Series] = {}
        for metric_key, label in metric_labels.items():
            if metric_key not in data:
                continue
            direction = self.metric_direction(metric_key)
            group = data.groupby("experiment")[metric_key]
            series = group.max() if direction == "DOWN" else group.min()
            penalties[label] = series
        penalty_df = pd.DataFrame(penalties)
        if penalty_df.empty:
            return penalty_df
        penalty_df = penalty_df.reindex(self.ordered_experiments)
        return penalty_df

    def _build_styled_metric_table(
        self,
        metric_labels: Dict[str, str],
        *,
        weight_by_success: bool = False,
    ) -> pd.DataFrame:
        mean_table, std_table = self._metric_summary_tables(metric_labels)
        mean_df = mean_table.set_index([r"\textbf{Experiment}", r"\textbf{Navigator}"])
        std_df = std_table.set_index([r"\textbf{Experiment}", r"\textbf{Navigator}"])
        mean_df = mean_df.reindex(
            self.ordered_experiments, level=r"\textbf{Experiment}"
        )
        mean_df = mean_df.reindex(self.ordered_navigators, level=r"\textbf{Navigator}")
        std_df = std_df.reindex(self.ordered_experiments, level=r"\textbf{Experiment}")
        std_df = std_df.reindex(self.ordered_navigators, level=r"\textbf{Navigator}")
        mean_df = self.success_rate.join(mean_df, how="inner")
        std_df = self.success_rate.join(std_df, how="inner")
        success_col = r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"
        std_df[success_col] = 0
        column_order = [success_col] + list(metric_labels.values())
        mean_df = mean_df[column_order]
        std_df = std_df[column_order]

        if weight_by_success:
            success_fraction = (self.success_rate[success_col] / 100.0).clip(
                lower=0, upper=1
            )
            penalty_table = self._failure_penalty_table(metric_labels)

            def _to_float(value: Any, default: float = 0.0) -> float:
                if value is None or pd.isna(value):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            exp_level_index = 0
            if mean_df.index.names:
                exp_level_index = mean_df.index.names.index(r"\textbf{Experiment}")
            for idx in mean_df.index:
                sr_value = success_fraction.loc[idx] if idx in success_fraction else 0.0
                sr = _to_float(sr_value, default=0.0)
                sr = max(0.0, min(sr, 1.0))
                experiment_name = idx[exp_level_index]
                for metric_label in metric_labels.values():
                    penalty_value = None
                    if (not penalty_table.empty) and (
                        metric_label in penalty_table.columns
                    ):
                        if experiment_name in penalty_table.index:
                            penalty_value = penalty_table.at[
                                experiment_name, metric_label
                            ]
                    current_mean = mean_df.at[idx, metric_label]
                    current_mean_numeric = _to_float(current_mean, default=0.0)
                    penalty_numeric = _to_float(
                        penalty_value, default=current_mean_numeric
                    )
                    new_mean = sr * current_mean_numeric + (1.0 - sr) * penalty_numeric
                    mean_df.at[idx, metric_label] = new_mean
                    std_value = std_df.at[idx, metric_label]
                    std_numeric = _to_float(std_value, default=0.0)
                    std_df.at[idx, metric_label] = sr * std_numeric

        styled_df = mean_df.copy().astype(object)
        for experiment in self.ordered_experiments:
            styled_section = style_report_latex(
                mean_df=mean_df.loc[experiment],
                std_dev_df=std_df.loc[experiment],
                cmap_name="viridis",
                alpha=0.3,
            )
            styled_df.loc[[experiment]] = styled_section.values
        return styled_df

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def get_styled_metric_table(self, group_key: str) -> pd.DataFrame:
        metric_labels = self.config.metric_groups[group_key]
        return self._build_styled_metric_table(metric_labels)

    def get_weighted_metric_table(self, group_key: str) -> pd.DataFrame:
        metric_labels = self.config.metric_groups[group_key]
        return self._build_styled_metric_table(metric_labels, weight_by_success=True)

    # ------------------------------------------------------------------
    # Export routines
    # ------------------------------------------------------------------
    def export_metric_tables(
        self, scale: float = 0.71, cmap_name: str = "viridis"
    ) -> None:
        output_path = self.output_dir_tables()
        output_path.mkdir(parents=True, exist_ok=True)
        for group_key, metric_labels in self.config.metric_groups.items():
            styled_df = self._build_styled_metric_table(metric_labels)
            colorbar_file = output_path / f"{group_key}_colormap.pdf"
            self._ensure_parent_directory(colorbar_file)
            fig, ax = plt.subplots(figsize=(3.2, 0.35))
            ax.axis("off")
            norm = Normalize(0.0, 1.0)
            fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.45)
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap(cmap_name)),
                ax=ax,
                orientation="horizontal",
                fraction=0.8,
                alpha=self.config.alpha,
            )
            ax.tick_params(axis="both", which="both", length=0)
            fig.savefig(colorbar_file, bbox_inches="tight")
            plt.close(fig)
            export_latex_table(
                report_latex=styled_df,
                filename=str(output_path / f"{group_key}_summary_by_navigator.tex"),
                column_format="c" * (len(metric_labels) + 1),
                header=[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"]
                + list(metric_labels.values()),
                float_format="%.2f",
                bold_rows=True,
                caption=self.config.metric_captions.get(group_key, ""),
                label=f"tab:algorithm_{group_key}_metrics",
                scale=scale,
                multirow=True,
                colorbar_path=colorbar_file,
                colorbar_caption="Color scale (low to high)",
            )
            weighted_styled_df = self.get_weighted_metric_table(group_key)
            weighted_caption = self.config.metric_captions.get(group_key, "").strip()
            if weighted_caption:
                weighted_caption = f"{weighted_caption} (success-weighted)"
            else:
                weighted_caption = "Success-weighted metrics"
            export_latex_table(
                report_latex=weighted_styled_df,
                filename=str(
                    output_path / f"{group_key}_weighted_summary_by_navigator.tex"
                ),
                column_format="c" * (len(metric_labels) + 1),
                header=[r"\textbf{Success Rate (\%)} $\mathbf{\uparrow}$"]
                + list(metric_labels.values()),
                float_format="%.2f",
                bold_rows=True,
                caption=weighted_caption,
                label=f"tab:algorithm_{group_key}_metrics_weighted",
                scale=scale,
                multirow=True,
                colorbar_path=colorbar_file,
                colorbar_caption="Color scale (low to high)",
            )

    def export_success_rate_tables(self) -> None:
        output_path = self.output_dir_tables()
        output_path.mkdir(parents=True, exist_ok=True)
        summary = (
            self.metrics_results_parsed.groupby(["experiment", "algorithm"])[
                self.results
            ]
            .mean()
            .reindex(
                index=pd.MultiIndex.from_product(
                    [self.ordered_experiments, self.ordered_navigators]
                )
            )
            .round(2)
        )
        summary = summary.drop(columns=["timeout", "collision", "navigation_failure"])
        summary = summary.rename(columns={"success": r"Goal [\%]"})
        summary = (summary * 100).round(0).fillna(0).astype(int)
        chunk_size = max(1, math.ceil(len(self.ordered_experiments) / 2))
        for idx, start in enumerate(
            range(0, len(self.ordered_experiments), chunk_size), start=1
        ):
            experiments_slice = self.ordered_experiments[start : start + chunk_size]
            if not experiments_slice:
                continue
            sub_table = summary.loc[experiments_slice]
            with open(
                output_path / f"success_rate_by_navigator{idx}.tex",
                "w",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    sub_table.to_latex(
                        multicolumn=True,
                        multirow=True,
                        bold_rows=True,
                        float_format="%.0f",
                        na_rep="0",
                    )
                )


__all__ = ["TableExportMixin"]
