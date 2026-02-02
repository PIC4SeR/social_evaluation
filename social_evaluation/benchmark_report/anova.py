"""Statistical helpers for analysing algorithm effects on metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

try:  # pragma: no cover - optional dependency guard
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency guard
    stats = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except ImportError:  # pragma: no cover - optional dependency guard
    pairwise_tukeyhsd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency guard
    from statsmodels.stats.oneway import anova_oneway
except ImportError:  # pragma: no cover - optional dependency guard
    anova_oneway = None  # type: ignore[assignment]

from .config import BenchmarkConfig

if TYPE_CHECKING:  # pragma: no cover - linting aid only

    class _BaseProtocol:
        config: BenchmarkConfig
        ordered_navigators: List[str]
        success_results: pd.DataFrame

        def output_dir_tables(self) -> Path: ...
        def output_dir_figures(self) -> Path: ...

        def _ensure_parent_directory(self, path: Path) -> None: ...
        def _format_text(self, text: str) -> str: ...

        def metric_plain_label(self, metric_key: str) -> str: ...


@dataclass
class AnovaSummary:
    """Container for ANOVA summary details."""

    metric_key: str
    metric_label: str
    f_statistic: float
    p_value: float
    df_between: float
    df_within: float
    sample_size: int
    algorithms: Tuple[str, ...]
    alpha: float
    significant: bool
    test_type: str
    heteroscedastic: bool


@dataclass
class PostHocComparison:
    """Container for Tukey + Hedge's g pairwise results."""

    metric_key: str
    metric_label: str
    group_a: str
    group_b: str
    mean_diff: float
    lower_ci: float
    upper_ci: float
    p_adjusted: float
    reject: bool
    hedges_g: float


class AnovaMixin:
    """Provide ANOVA/Tukey helpers to quantify algorithm effects per metric."""

    if TYPE_CHECKING:  # pragma: no cover - linting aid only
        config: BenchmarkConfig
        ordered_navigators: List[str]
        success_results: pd.DataFrame
        output_dir_tables: _BaseProtocol.output_dir_tables  # type: ignore[assignment]
        output_dir_figures: _BaseProtocol.output_dir_figures  # type: ignore[assignment]
        _ensure_parent_directory: _BaseProtocol._ensure_parent_directory  # type: ignore[assignment]
        _format_text: _BaseProtocol._format_text  # type: ignore[assignment]
        metric_plain_label: _BaseProtocol.metric_plain_label  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _metric_algorithm_samples(
        self, metric_key: str, *, min_group_size: int
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
        """Return per-algorithm samples for ``metric_key``."""

        if metric_key not in self.success_results:
            return pd.DataFrame(), {}
        subset = self.success_results[["algorithm", metric_key]].dropna()
        if subset.empty:
            return pd.DataFrame(), {}
        subset = subset.copy()
        subset[metric_key] = pd.to_numeric(subset[metric_key], errors="coerce")
        subset = subset.dropna(subset=[metric_key])
        subset = subset[subset["algorithm"].isin(self.ordered_navigators)]
        if subset.empty:
            return pd.DataFrame(), {}
        subset = subset.rename(columns={metric_key: "value"})
        groups: Dict[str, np.ndarray] = {}
        for algorithm in self.ordered_navigators:
            values = subset.loc[subset["algorithm"] == algorithm, "value"].to_numpy(
                dtype=float
            )
            if values.size >= min_group_size:
                groups[algorithm] = values
        return subset, groups

    @staticmethod
    def _has_heteroscedasticity(
        sample_arrays: Sequence[np.ndarray], alpha: float
    ) -> bool:
        """Return True when Levene's test detects unequal variances."""

        if stats is None:
            return False
        valid_samples = [arr for arr in sample_arrays if arr.size >= 2]
        if len(valid_samples) < 2:
            return False
        try:
            _, p_value = stats.levene(*valid_samples, center="median")
        except ValueError:
            return False
        return bool(np.isfinite(p_value) and p_value <= alpha)

    @staticmethod
    def _hedges_g(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
        """Compute Hedge's g between two samples."""

        n_a = sample_a.size
        n_b = sample_b.size
        if n_a < 2 or n_b < 2:
            return float("nan")
        var_a = np.var(sample_a, ddof=1)
        var_b = np.var(sample_b, ddof=1)
        df = n_a + n_b - 2
        if df <= 0:
            return float("nan")
        pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
        if pooled <= 0:
            return 0.0
        effect = (np.mean(sample_a) - np.mean(sample_b)) / np.sqrt(pooled)
        correction = 1.0 - (3.0 / (4.0 * df - 1.0)) if df > 1 else 1.0
        return float(effect * correction)

    @staticmethod
    def _build_pairwise_value_matrices(
        metric_pairs: pd.DataFrame,
        algorithm_order: List[str],
        value_specs: Sequence[Tuple[str, Optional[float]]],
    ) -> Dict[str, pd.DataFrame]:
        """Construct symmetric matrices for requested pairwise statistics."""

        matrices: Dict[str, pd.DataFrame] = {}
        for column_name, diagonal_value in value_specs:
            matrix = pd.DataFrame(
                np.nan,
                index=algorithm_order,
                columns=algorithm_order,
                dtype=float,
            )
            if diagonal_value is not None:
                np.fill_diagonal(matrix.values, diagonal_value)
            matrices[column_name] = matrix

        for _, row in metric_pairs.iterrows():
            group_a = str(row["group_a"])
            group_b = str(row["group_b"])
            for column_name, _ in value_specs:
                value = row.get(column_name)
                try:
                    value_float = float(value)
                except (TypeError, ValueError):
                    continue
                matrix = matrices[column_name]
                if group_a in matrix.index and group_b in matrix.columns:
                    matrix.at[group_a, group_b] = value_float
                if group_b in matrix.index and group_a in matrix.columns:
                    matrix.at[group_b, group_a] = value_float
        return matrices

    def _tukey_pairwise(
        self,
        subset: pd.DataFrame,
        metric_key: str,
        metric_label: str,
        groups: Dict[str, np.ndarray],
        *,
        alpha: float,
    ) -> List[PostHocComparison]:
        """Compute Tukey HSD post-hoc test for the provided subset."""

        if pairwise_tukeyhsd is None:
            raise RuntimeError(
                "statsmodels is required for Tukey HSD comparisons. "
                "Install it with 'pip install statsmodels'."
            )
        if subset.empty:
            return []
        tukey = pairwise_tukeyhsd(
            endog=subset["value"].to_numpy(dtype=float),
            groups=subset["algorithm"].astype(str).to_numpy(),
            alpha=alpha,
        )
        table = tukey.summary()
        if not hasattr(table, "data"):
            return []
        records: List[PostHocComparison] = []
        for row in table.data[1:]:
            try:
                group_a = str(row[0])
                group_b = str(row[1])
                mean_diff = float(row[2])
                p_adj = float(row[3])
                lower = float(row[4])
                upper = float(row[5])
                reject = str(row[6]).strip().lower() == "true"
            except (TypeError, ValueError):
                continue
            hedges = self._hedges_g(
                groups.get(group_a, np.array([])),
                groups.get(group_b, np.array([])),
            )
            records.append(
                PostHocComparison(
                    metric_key=metric_key,
                    metric_label=metric_label,
                    group_a=group_a,
                    group_b=group_b,
                    mean_diff=mean_diff,
                    lower_ci=lower,
                    upper_ci=upper,
                    p_adjusted=p_adj,
                    reject=reject,
                    hedges_g=hedges,
                )
            )
        return records

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def analyze_algorithm_effects(
        self,
        metric_keys: Optional[Sequence[str]] = None,
        *,
        alpha: float = 0.05,
        min_group_size: int = 2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run ANOVA/Tukey analysis for the provided metrics.

        Returns a tuple of (ANOVA summary DataFrame, Tukey pairwise comparisons DataFrame).
            ANOVA summary DataFrame is indexed by metric_key and contains:
                - metric_label: str
                - f_statistic: the F statistic value from the ANOVA test (float)
                - p_value: the p-value from the ANOVA test (float)
                - df_between: the degrees of freedom between groups (float)
                - df_within: the degrees of freedom within groups (float)
                - sample_size: total number of samples across all groups (int)
                - algorithms: tuple of algorithm names involved in the test (Tuple[str, ...])
                - alpha: significance level used (float)
                - significant: bool indicating if the result is significant (bool)
                - test_type: str indicating the type of ANOVA test used
                  ("anova", "welch", or "anova_fallback")
                - heteroscedastic: bool indicating if heteroscedasticity was detected (bool)
            Tukey pairwise comparisons DataFrame is indexed by (metric_key, group_a, group_b).
                - metric_label: the plain label for the metric (str)
                - mean_diff: the mean difference between the two groups (float)
                - lower_ci: the lower bound of the confidence interval (float)
                - upper_ci: the upper bound of the confidence interval (float)
                - p_adjusted: the adjusted p-value from Tukey's test (float)
                - reject: bool indicating if the null hypothesis is rejected (bool)
                - hedges_g: Hedge's g effect size between the two groups (float)
        """

        if stats is None:
            raise RuntimeError(
                "scipy is required for ANOVA analysis. Install it with 'pip install scipy'."
            )
        keys: Iterable[str]
        if metric_keys is None:
            keys = tuple(self.config.metric_directions.keys())
        else:
            keys = tuple(metric_keys)
        summary_records: List[AnovaSummary] = []
        pairwise_records: List[PostHocComparison] = []

        for metric_key in keys:
            subset, groups = self._metric_algorithm_samples(
                metric_key, min_group_size=min_group_size
            )
            if len(groups) < 2:
                continue
            metric_label = self.metric_plain_label(metric_key)
            sample_arrays = list(groups.values())
            heteroscedastic = self._has_heteroscedasticity(sample_arrays, alpha=alpha)
            use_welch = bool(heteroscedastic and anova_oneway is not None)
            try:
                if use_welch:
                    welch_result = anova_oneway(  # type: ignore[operator]
                        sample_arrays,
                        use_var="unequal",
                        welch_correction=True,
                    )
                    f_stat = float(welch_result.statistic)
                    p_value = float(welch_result.pvalue)
                    df_between = float(welch_result.df_num)
                    df_within = float(welch_result.df_denom)
                    test_type = "welch"
                else:
                    if heteroscedastic and anova_oneway is None:
                        warnings.warn(
                            "statsmodels is required for Welch's ANOVA; "
                            "falling back to standard one-way ANOVA.",
                            RuntimeWarning,
                        )
                    f_stat, p_value = stats.f_oneway(*sample_arrays)  # type: ignore[operator]
                    df_between = float(len(sample_arrays) - 1)
                    df_within = float(sum(len(arr) - 1 for arr in sample_arrays))
                    test_type = "anova" if not heteroscedastic else "anova_fallback"
            except ValueError:
                continue
            sample_size = int(sum(len(arr) for arr in sample_arrays))
            significant = bool(np.isfinite(p_value) and p_value <= alpha)
            summary_records.append(
                AnovaSummary(
                    metric_key=metric_key,
                    metric_label=metric_label,
                    f_statistic=float(f_stat),
                    p_value=float(p_value),
                    df_between=df_between,
                    df_within=df_within,
                    sample_size=sample_size,
                    algorithms=tuple(groups.keys()),
                    alpha=alpha,
                    significant=significant,
                    test_type=test_type,
                    heteroscedastic=heteroscedastic,
                )
            )
            if significant:
                try:
                    pairwise_records.extend(
                        self._tukey_pairwise(
                            subset,
                            metric_key,
                            metric_label,
                            groups,
                            alpha=alpha,
                        )
                    )
                except RuntimeError:
                    # Surfacing absence of statsmodels through a dedicated row keeps the API predictable.
                    pass

        summary_df = pd.DataFrame([s.__dict__ for s in summary_records])
        pairwise_df = pd.DataFrame([p.__dict__ for p in pairwise_records])
        if not summary_df.empty:
            summary_df = summary_df.set_index("metric_key")
        if not pairwise_df.empty:
            pairwise_df = pairwise_df.set_index(["metric_key", "group_a", "group_b"])
        return summary_df, pairwise_df

    def export_anova_analysis(
        self,
        metric_keys: Optional[Sequence[str]] = None,
        *,
        alpha: float = 0.05,
        min_group_size: int = 2,
        output_dir: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run and persist ANOVA/Tukey results for the provided metrics.
        Returns a tuple of (ANOVA summary DataFrame, Tukey pairwise comparisons DataFrame).
        See `analyze_algorithm_effects` for details on the DataFrame structures.
        """

        summary_df, pairwise_df = self.analyze_algorithm_effects(
            metric_keys, alpha=alpha, min_group_size=min_group_size
        )
        target_dir = (
            Path(output_dir) if output_dir else self.output_dir_tables() / "anova"
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        summary_path = target_dir / "anova_summary.csv"
        pairwise_path = target_dir / "anova_pairwise.csv"
        self._ensure_parent_directory(summary_path)
        self._ensure_parent_directory(pairwise_path)
        summary_df.to_csv(summary_path)
        pairwise_df.to_csv(pairwise_path)
        return summary_df, pairwise_df

    def plot_pairwise_pvalue_heatmaps(
        self,
        metric_keys: Optional[Sequence[str]] = None,
        *,
        alpha: float = 0.05,
        min_group_size: int = 2,
        cmap: str = "mako",
        annotate: Optional[bool] = True,
        fmt: str = ".3f",
        label_size: int = 12,
        title_size: int = 14,
        colorbar_label_size: int = 12,
        annot_font_size: int = 9,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Plot heatmaps of pairwise adjusted p-values for the requested metrics.

        Arguments:
            metric_keys: Optional sequence of metric keys to analyze. If None, all metrics are analyzed.
            alpha: Significance level for ANOVA and Tukey tests. Default is 0.05.
            min_group_size: Minimum number of samples required per algorithm group. Default is 2.
            cmap: Matplotlib colormap name for the heatmap. Default is "mako".
            annotate: Whether to annotate heatmap cells with p-value numbers. Default is True.
            fmt: String format for annotating p-values. Default is ".3f".
            label_size: Font size for axis labels. Default is 12.
            title_size: Font size for the plot title. Default is 14.
            colorbar_label_size: Font size for the colorbar label. Default is 12.
            annot_font_size: Font size for annotations in heatmap cells. Default is 9.
            output_dir: Optional directory to save the heatmap figures. If None, uses default output directory.

        Returns a mapping of metric keys to saved figure paths
            - metric_key: Path to the saved heatmap figure (PNG format).
            - Raises ValueError if no pairwise comparisons are available
              for the requested metrics.

        """

        summary_df, pairwise_df = self.analyze_algorithm_effects(
            metric_keys, alpha=alpha, min_group_size=min_group_size
        )
        if pairwise_df.empty:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        pairwise_reset = pairwise_df.reset_index()
        metrics_to_plot: Iterable[str]
        if metric_keys is None:
            metrics_to_plot = pairwise_reset["metric_key"].unique().tolist()
        else:
            metrics_to_plot = metric_keys

        saved_paths: Dict[str, Path] = {}
        base_dir = (
            Path(output_dir) if output_dir else self.output_dir_figures() / "anova"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        for metric_key in metrics_to_plot:
            metric_pairs = pairwise_reset[pairwise_reset["metric_key"] == metric_key]
            if metric_pairs.empty:
                continue
            algorithms = list(
                dict.fromkeys(
                    metric_pairs["group_a"].tolist() + metric_pairs["group_b"].tolist()
                )
            )
            algorithm_order = [
                label for label in self.ordered_navigators if label in algorithms
            ]
            for label in algorithms:
                if label not in algorithm_order:
                    algorithm_order.append(label)
            if len(algorithm_order) < 2:
                continue
            matrices = self._build_pairwise_value_matrices(
                metric_pairs,
                algorithm_order,
                [("p_adjusted", float("nan"))],
            )
            matrix = (
                matrices["p_adjusted"]
                .apply(pd.to_numeric, errors="coerce")
                .astype(float)
            )
            mask = matrix.isna()
            fig_size = max(6.0, 0.6 * len(algorithm_order))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            colorbar_label = self._format_text(r"Adjusted $p$-value")

            significance_matrix = matrix.copy()

            def significance_level(p: float) -> float:
                if p <= 0.0001:
                    return 1.0
                elif p <= 0.001:
                    return 0.75
                elif p <= 0.01:
                    return 0.5
                elif p <= 0.05:
                    return 0.25
                else:
                    return 0.0

            significance_matrix = significance_matrix.map(
                lambda p: significance_level(p) if pd.notna(p) else np.nan
            )

            def _format_annotation(p_val: float) -> str:
                if pd.isna(p_val):
                    return ""
                if p_val <= 0.0001:
                    return "****"
                if p_val <= 0.001:
                    return "***"
                if p_val <= 0.01:
                    return "**"
                if p_val <= 0.05:
                    return "*"
                return "ns"

            annot_kws = {"size": annot_font_size} if annotate else None
            annot_arg: Any = False
            fmt_arg = fmt
            if annotate:
                annotation_table = matrix.map(_format_annotation)
                np.fill_diagonal(annotation_table.values, "")
                annot_arg = annotation_table.to_numpy(dtype=object)
                fmt_arg = ""

            sns.heatmap(
                significance_matrix,
                mask=mask,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                annot=annot_arg,
                annot_kws=annot_kws,
                fmt=fmt_arg,
                cbar=True,
                square=True,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": colorbar_label},
                ax=ax,
            )
            ax.set_xticklabels(
                [self._format_text(label) for label in matrix.columns],
                rotation=45,
                ha="right",
                fontsize=label_size,
            )
            ax.set_yticklabels(
                [self._format_text(label) for label in matrix.index],
                rotation=0,
                fontsize=label_size,
            )
            metric_label: str
            if not summary_df.empty and metric_key in summary_df.index:
                metric_label = str(summary_df.at[metric_key, "metric_label"])
            else:
                metric_label = self.metric_plain_label(metric_key)
            ax.set_title(
                self._format_text(f"{metric_label} Pairwise p-values"),
                pad=12,
                fontsize=title_size,
            )
            ax.set_xlabel(self._format_text("Algorithm"), fontsize=label_size)
            ax.set_ylabel(self._format_text("Algorithm"), fontsize=label_size)
            cbar = ax.collections[0].colorbar if ax.collections else None
            if cbar is not None:
                cbar.ax.tick_params(labelsize=colorbar_label_size)
                cbar.set_label(colorbar_label, size=colorbar_label_size)
            fig.tight_layout()
            output_path = base_dir / f"{metric_key}_pairwise_pvalues.png"
            self._ensure_parent_directory(output_path)
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths[metric_key] = output_path

        if not saved_paths:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        return saved_paths

    def plot_pairwise_mean_diff_heatmaps(
        self,
        metric_keys: Optional[Sequence[str]] = None,
        *,
        alpha: float = 0.05,
        min_group_size: int = 2,
        cmap: str = "coolwarm",
        annotate: Optional[bool] = True,
        fmt: str = ".2f",
        label_size: int = 12,
        title_size: int = 14,
        colorbar_label_size: int = 12,
        annot_font_size: int = 9,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Plot heatmaps visualising mean differences between algorithms."""

        summary_df, pairwise_df = self.analyze_algorithm_effects(
            metric_keys, alpha=alpha, min_group_size=min_group_size
        )
        if pairwise_df.empty:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        pairwise_reset = pairwise_df.reset_index()
        metrics_to_plot: Iterable[str]
        if metric_keys is None:
            metrics_to_plot = pairwise_reset["metric_key"].unique().tolist()
        else:
            metrics_to_plot = metric_keys

        saved_paths: Dict[str, Path] = {}
        base_dir = (
            Path(output_dir) if output_dir else self.output_dir_figures() / "anova"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        for metric_key in metrics_to_plot:
            metric_pairs = pairwise_reset[pairwise_reset["metric_key"] == metric_key]
            if metric_pairs.empty:
                continue
            algorithms = list(
                dict.fromkeys(
                    metric_pairs["group_a"].tolist() + metric_pairs["group_b"].tolist()
                )
            )
            algorithm_order = [
                label for label in self.ordered_navigators if label in algorithms
            ]
            for label in algorithms:
                if label not in algorithm_order:
                    algorithm_order.append(label)
            if len(algorithm_order) < 2:
                continue
            matrices = self._build_pairwise_value_matrices(
                metric_pairs,
                algorithm_order,
                [("mean_diff", 0.0)],
            )
            matrix = (
                matrices["mean_diff"]
                .apply(pd.to_numeric, errors="coerce")
                .astype(float)
            )
            if matrix.isna().all().all():
                continue
            mask = matrix.isna()
            fig_size = max(6.0, 0.6 * len(algorithm_order))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            annot_kws = {"size": annot_font_size} if annotate else None
            annot_arg: Any = False
            fmt_arg = fmt
            if annotate:
                annot_arg = matrix.to_numpy(dtype=float)

            sns.heatmap(
                matrix,
                mask=mask,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                center=0.0,
                annot=annot_arg,
                annot_kws=annot_kws,
                fmt=fmt_arg,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": self._format_text("Mean difference")},
                ax=ax,
            )
            ax.set_xticklabels(
                [self._format_text(label) for label in matrix.columns],
                rotation=45,
                ha="right",
                fontsize=label_size,
            )
            ax.set_yticklabels(
                [self._format_text(label) for label in matrix.index],
                rotation=0,
                fontsize=label_size,
            )
            metric_label: str
            if not summary_df.empty and metric_key in summary_df.index:
                metric_label = str(summary_df.at[metric_key, "metric_label"])
            else:
                metric_label = self.metric_plain_label(metric_key)
            ax.set_title(
                self._format_text(f"{metric_label} Mean differences"),
                pad=12,
                fontsize=title_size,
            )
            ax.set_xlabel(self._format_text("Algorithm"), fontsize=label_size)
            ax.set_ylabel(self._format_text("Algorithm"), fontsize=label_size)
            cbar = ax.collections[0].colorbar if ax.collections else None
            if cbar is not None:
                cbar.ax.tick_params(labelsize=colorbar_label_size)
                cbar.set_label(
                    self._format_text("Mean difference"), size=colorbar_label_size
                )
            fig.tight_layout()
            output_path = base_dir / f"{metric_key}_mean_diff.png"
            self._ensure_parent_directory(output_path)
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths[metric_key] = output_path

        if not saved_paths:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        return saved_paths

    def plot_pairwise_hedges_heatmaps(
        self,
        metric_keys: Optional[Sequence[str]] = None,
        *,
        alpha: float = 0.05,
        min_group_size: int = 2,
        cmap: str = "coolwarm",
        annotate: Optional[bool] = True,
        fmt: str = ".2f",
        label_size: int = 12,
        title_size: int = 14,
        colorbar_label_size: int = 12,
        annot_font_size: int = 9,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Plot heatmaps visualising Hedge's g pairwise effect sizes."""

        summary_df, pairwise_df = self.analyze_algorithm_effects(
            metric_keys, alpha=alpha, min_group_size=min_group_size
        )
        if pairwise_df.empty:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        pairwise_reset = pairwise_df.reset_index()
        metrics_to_plot: Iterable[str]
        if metric_keys is None:
            metrics_to_plot = pairwise_reset["metric_key"].unique().tolist()
        else:
            metrics_to_plot = metric_keys

        saved_paths: Dict[str, Path] = {}
        base_dir = (
            Path(output_dir) if output_dir else self.output_dir_figures() / "anova"
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        for metric_key in metrics_to_plot:
            metric_pairs = pairwise_reset[pairwise_reset["metric_key"] == metric_key]
            if metric_pairs.empty:
                continue
            algorithms = list(
                dict.fromkeys(
                    metric_pairs["group_a"].tolist() + metric_pairs["group_b"].tolist()
                )
            )
            algorithm_order = [
                label for label in self.ordered_navigators if label in algorithms
            ]
            for label in algorithms:
                if label not in algorithm_order:
                    algorithm_order.append(label)
            if len(algorithm_order) < 2:
                continue
            matrices = self._build_pairwise_value_matrices(
                metric_pairs,
                algorithm_order,
                [("hedges_g", float("nan"))],
            )
            matrix = (
                matrices["hedges_g"].apply(pd.to_numeric, errors="coerce").astype(float)
            )
            if matrix.isna().all().all():
                continue
            mask = matrix.isna()
            fig_size = max(6.0, 0.6 * len(algorithm_order))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            vmax = float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))
            if not np.isfinite(vmax) or vmax == 0.0:
                vmax = 1.0
            annot_kws = {"size": annot_font_size} if annotate else None
            annot_arg: Any = False
            fmt_arg = fmt
            if annotate:
                annot_arg = matrix.to_numpy(dtype=float)

            sns.heatmap(
                matrix,
                mask=mask,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                center=0.0,
                annot=annot_arg,
                annot_kws=annot_kws,
                fmt=fmt_arg,
                linewidths=0.5,
                linecolor="white",
                cbar_kws={"label": self._format_text("Hedge's g")},
                ax=ax,
            )
            ax.set_xticklabels(
                [self._format_text(label) for label in matrix.columns],
                rotation=45,
                ha="right",
                fontsize=label_size,
            )
            ax.set_yticklabels(
                [self._format_text(label) for label in matrix.index],
                rotation=0,
                fontsize=label_size,
            )
            metric_label: str
            if not summary_df.empty and metric_key in summary_df.index:
                metric_label = str(summary_df.at[metric_key, "metric_label"])
            else:
                metric_label = self.metric_plain_label(metric_key)
            ax.set_title(
                self._format_text(f"{metric_label} Hedge's g"),
                pad=12,
                fontsize=title_size,
            )
            ax.set_xlabel(self._format_text("Algorithm"), fontsize=label_size)
            ax.set_ylabel(self._format_text("Algorithm"), fontsize=label_size)
            cbar = ax.collections[0].colorbar if ax.collections else None
            if cbar is not None:
                cbar.ax.tick_params(labelsize=colorbar_label_size)
                cbar.set_label(self._format_text("Hedge's g"), size=colorbar_label_size)
            fig.tight_layout()
            output_path = base_dir / f"{metric_key}_hedges_g.png"
            self._ensure_parent_directory(output_path)
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            saved_paths[metric_key] = output_path

        if not saved_paths:
            raise ValueError(
                "No pairwise comparisons available for the requested metrics."
            )
        return saved_paths


__all__ = ["AnovaMixin"]
