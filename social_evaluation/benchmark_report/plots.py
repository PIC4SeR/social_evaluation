"""Plotting mixins for benchmark reports."""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)
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

try:  # pragma: no cover - optional dependency guard
    from statannotations.Annotator import Annotator as _Annotator
except ImportError:  # pragma: no cover - optional dependency guard
    _Annotator = None  # type: ignore[assignment, misc]


class PlottingMixin:
    """Provide the high-level plotting helpers for benchmark reports."""

    config: BenchmarkConfig
    metrics_results_parsed: pd.DataFrame
    metrics_results: pd.DataFrame
    success_results: pd.DataFrame
    navigator_palette: Dict[str, Tuple[float, float, float, float]]
    available_navigators: List[str]
    ordered_experiments: List[str]
    ordered_episode_groups: List[str]
    ordered_navigators: List[str]
    results: List[str]
    output_dir_figures: _BaseProtocol.output_dir_figures  # type: ignore[assignment]
    get_result_palette: _BaseProtocol.get_result_palette  # type: ignore[assignment]
    get_intrusion_palette: _BaseProtocol.get_intrusion_palette  # type: ignore[assignment]
    _experiment_order_with_groups: _BaseProtocol._experiment_order_with_groups  # type: ignore[assignment]
    _format_text: _BaseProtocol._format_text  # type: ignore[assignment]
    _ensure_parent_directory: _BaseProtocol._ensure_parent_directory  # type: ignore[assignment]

    def _prepare_distribution_plot_data(
        self,
        data: pd.DataFrame,
        metric_key: str,
    ) -> pd.DataFrame:
        """Return only rows that can produce a drawable distribution plot."""

        if data.empty or metric_key not in data.columns:
            return data.iloc[0:0].copy()
        plot_data = data.loc[data["algorithm"].isin(self.ordered_navigators)].copy()
        if plot_data.empty:
            return plot_data
        metric_values = pd.to_numeric(plot_data[metric_key], errors="coerce")
        plot_data = plot_data.loc[metric_values.notna()].copy()
        if plot_data.empty:
            return plot_data
        plot_data[metric_key] = metric_values.loc[plot_data.index]
        return plot_data

    @staticmethod
    def _apply_box_alpha(ax: Any, alpha: float) -> None:
        """Apply alpha to box patches after seaborn renders them."""

        for patch in ax.patches:
            patch.set_alpha(alpha)

    # ------------------------------------------------------------------
    # High-level plotting helpers
    # ------------------------------------------------------------------
    def plot_success_rate(
        self,
        font_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        """Plot stacked bar charts of success metrics for each experiment.

        Parameters
        ----------
        font_sizes : dict, optional
            Override default font sizes.  Recognised keys:
            ``title``, ``axis_label``, ``tick_label``, ``legend``,
            ``sup_label``, ``legend_ncol``.
        """

        fs = {**self._DEFAULT_SUCCESS_RATE_FONT_SIZES, **(font_sizes or {})}
        output_path = self.output_dir_figures() / "success_rate_by_navigator.png"
        self._ensure_parent_directory(output_path)
        n_cols = self.config.success_rate_n_cols
        env_metrics_cache: Dict[str, pd.DataFrame] = {}
        for environment in self.ordered_experiments:
            env_metrics = self.metrics_results_parsed[
                self.metrics_results_parsed["experiment"] == environment
            ]
            env_metrics_cache[environment] = env_metrics
        if self.config.episode_groups:
            for group_name, members in self.config.episode_groups.items():
                if not members:
                    continue
                group_metrics = self.metrics_results_parsed[
                    self.metrics_results_parsed["experiment"].isin(members)
                ]
                env_metrics_cache[group_name] = group_metrics

        env_navigator_map: Dict[str, List[str]] = {}
        max_nav_count = 1
        for environment, env_metrics in env_metrics_cache.items():
            available = set(env_metrics["algorithm"].unique().tolist())
            env_navigators = [
                navigator
                for navigator in self.available_navigators
                if navigator in available
            ]
            env_navigator_map[environment] = env_navigators
            if env_navigators:
                max_nav_count = max(max_nav_count, len(env_navigators))

        experiment_order = self._experiment_order_with_groups(env_metrics_cache.keys())
        if not experiment_order:
            return
        n_cols = min(n_cols, len(experiment_order))
        n_rows = max(1, math.ceil(len(experiment_order) / n_cols))

        tick_font_size = fs["tick_label"]
        title_font_size = fs["title"]
        legend_font_size = fs["legend"]
        sup_font_size = fs["sup_label"]
        legend_ncol = fs["legend_ncol"]

        subplot_width = max(4.5, 3.0 + 0.3 * max_nav_count)
        subplot_height = max(4.0, 4.0 + 0.04 * max_nav_count)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * subplot_width, n_rows * subplot_height),
            sharey=True,
            sharex=True,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)
        result_palette = self.get_result_palette()
        for idx, environment in enumerate(experiment_order):
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
        for idx in range(len(experiment_order), n_rows * n_cols):
            row_idx, col_idx = divmod(idx, n_cols)
            axes[row_idx, col_idx].axis("off")
        fig.supylabel(
            self._format_text(r"\textbf{Success Rate (\%)}"), fontsize=sup_font_size
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
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=legend_ncol,
            fontsize=legend_font_size,
            frameon=False,
            labelspacing=1.5,
        )
        # Dynamically compute tight_layout rect from font sizes
        # and figure dimensions so margins scale automatically.
        fig_w, fig_h = fig.get_size_inches()
        pts_to_inches = 1.0 / 72.0
        # Bottom: legend rows
        n_legend_rows_sr = max(1, math.ceil(len(legend_handles) / legend_ncol))
        legend_h = n_legend_rows_sr * legend_font_size * pts_to_inches * 1.8 + 0.3
        rect_bottom = min(legend_h / fig_h, 0.25)
        # Left: supylabel
        sup_w = sup_font_size * pts_to_inches * 1.5
        rect_left = min(sup_w / fig_w, 0.08)
        fig.tight_layout(rect=(rect_left, rect_bottom, 1.0, 1.0))
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    _DEFAULT_SUCCESS_RATE_FONT_SIZES: Dict[str, int] = {
        "title": 37,
        "axis_label": 37,
        "tick_label": 37,
        "legend": 37,
        "sup_label": 37,
        "legend_ncol": 4,
    }

    _DEFAULT_DIST_FONT_SIZES: Dict[str, int] = {
        "title": 37,
        "axis_label": 37,
        "tick_label": 37,
        "legend": 37,
        "legend_ncol": 8,
    }

    _DEFAULT_PROXEMIC_FONT_SIZES: Dict[str, int] = {
        "title": 60,
        "axis_label": 60,
        "tick_label": 60,
        "xtick_label": 40,
        "legend": 60,
        "sup_label": 60,
        "legend_ncol": 4,
    }

    def plot_metric_distributions(
        self,
        kind: str,
        cmap_name: str = "Paired",
        rows: int = 0,
        n_cols: int = 0,
        font_sizes: Optional[Dict[str, int]] = None,
        show_episode_ylabel: bool = True,
        connect_medians: bool = False,
        annotate_stats: bool = False,
        stat_test: str = "t-test_ind",
        stat_comparisons: Optional[List[Tuple[str, str]]] = None,
        stat_correction: Optional[str] = "bonferroni",
        stat_display_format: str = "star",
        stat_alpha: float = 0.05,
        stat_min_group_size: int = 2,
        stat_use_anova: bool = True,
        stat_font_size: Optional[int] = None,
        stat_line_height: float = 0.04,
        stat_annotation_mode: str = "brackets",
        stat_baseline: Optional[str] = None,
        highlight_baseline: bool = False,
        highlight_baseline_label: str = "Ours",
    ) -> None:
        """Plot distribution summaries (box or violin) for each metric group.

        Parameters
        ----------
        kind : str
            Plot type: ``"box"`` or ``"violin"``.
        cmap_name : str
            Colour-map name (unused but kept for API compat).
        rows : int
            Maximum number of episode rows per figure.  ``0`` (default)
            means all episodes in a single figure.
        n_cols : int
            Maximum number of metric columns per episode row.  ``0``
            (default) means all metrics in a single row.  When the
            number of metrics exceeds *n_cols*, each episode is spread
            across multiple grid rows.
        font_sizes : dict, optional
            Override default font sizes.  Recognised keys:
            ``title``, ``axis_label``, ``tick_label``, ``legend``.
        show_episode_ylabel : bool
            When ``True`` (default), the episode/group name is shown
            as the y-axis label on the first column of each row.
            Set to ``False`` to hide it.
        connect_medians : bool
            Connect navigator medians with a polyline when plotting
            box plots.  Ignored for violin plots.
        annotate_stats : bool
            When ``True``, add statistical annotations to each
            subplot.  Default is ``False``.
        stat_test : str
            Statistical test to use with ``statannotations`` when
            ``stat_use_anova`` is ``False``.  Common values:
            ``"t-test_ind"``, ``"Mann-Whitney"``, ``"Kruskal"``.
            Ignored when ``stat_use_anova`` is ``True`` (the pre-
            computed Tukey HSD adjusted p-values are used instead).
        stat_comparisons : list of tuple, optional
            Explicit list of ``(group_a, group_b)`` pairs to annotate.
            When ``None`` (default), all pairs with significant results
            from the ANOVA/Tukey analysis are used (if
            ``stat_use_anova`` is ``True``), or all possible pairs are
            tested (if ``stat_use_anova`` is ``False``).
        stat_correction : str or None
            Multiple-testing correction applied by ``statannotations``
            when ``stat_use_anova`` is ``False``.
            Set to ``None`` to disable correction.  Ignored when
            ``stat_use_anova`` is ``True``.
        stat_display_format : str
            Annotation display format: ``"star"`` (default) for
            asterisks (``*``, ``**``, …) or ``"simple"`` for
            textual p-value descriptions.
        stat_alpha : float
            Significance level.  Default is ``0.05``.
        stat_min_group_size : int
            Minimum number of observations per algorithm group
            required to include a pair in the annotation.
            Default is ``2``.
        stat_use_anova : bool
            When ``True`` (default), use pre-computed ANOVA/Tukey HSD
            results from ``analyze_algorithm_effects`` and annotate
            with the adjusted p-values.  When ``False``, let
            ``statannotations`` run the test specified by
            ``stat_test`` directly.
        stat_font_size : int, optional
            Font size for the annotation text.  Defaults to
            ``tick_label`` font size from ``font_sizes``.
        stat_line_height : float
            Proportion of the y-axis range used to separate
            annotation levels.  Default is ``0.04``.
        stat_annotation_mode : str
            Annotation mode.  ``"brackets"`` (default) draws
            ``statannotations`` brackets between pairs.
            ``"baseline_stars"`` places compact star symbols
            above each non-baseline box/violin comparing it to
            the *stat_baseline* algorithm, keeping the plot clean.
            A caption text file with full statistical details is
            saved alongside each figure.
        stat_baseline : str, optional
            Navigator name used as the reference when
            ``stat_annotation_mode="baseline_stars"``.  Required
            for that mode.
        highlight_baseline : bool
            When ``True`` and ``stat_annotation_mode="baseline_stars"``,
            the baseline algorithm's legend label is rendered **bold**
            and a label (see *highlight_baseline_label*) is placed
            above the baseline's box/violin instead of leaving it
            blank.  Default is ``False``.
        highlight_baseline_label : str
            Text shown above the baseline box/violin when
            *highlight_baseline* is ``True``.  Default is ``"Ours"``.
        """

        fs = {**self._DEFAULT_DIST_FONT_SIZES, **(font_sizes or {})}

        output_path = Path(self.output_dir_figures())
        output_path.mkdir(parents=True, exist_ok=True)
        palette = self.navigator_palette
        use_baseline_stars = annotate_stats and stat_annotation_mode == "baseline_stars"
        _do_highlight = highlight_baseline and use_baseline_stars and stat_baseline
        legend_handles = [
            Patch(
                facecolor=cast(Any, palette[label]),
                edgecolor="black",
                label=self._format_text(
                    rf"\textbf{{{label}}}"
                    if _do_highlight and label == stat_baseline
                    else label
                ),
            )
            for label in palette
        ]
        # Pre-compute ANOVA/Tukey pairwise results when annotation is
        # requested and `stat_use_anova` is True.
        _pairwise_df: Optional[pd.DataFrame] = None
        _anova_summary: Optional[pd.DataFrame] = None
        if annotate_stats and stat_use_anova:
            if not use_baseline_stars and _Annotator is None:
                raise RuntimeError(
                    "statannotations is required for bracket annotations. "
                    "Install it with 'pip install statannotations'."
                )
            # `analyze_algorithm_effects` lives on the AnovaMixin which
            # is mixed into the same concrete class.
            _anova_summary, _pairwise_df = self.analyze_algorithm_effects(  # type: ignore[attr-defined]
                alpha=stat_alpha, min_group_size=stat_min_group_size
            )
            if _pairwise_df is not None and not _pairwise_df.empty:
                _pairwise_df = _pairwise_df.reset_index()
            else:
                _pairwise_df = None
        # Accumulate caption lines when using baseline_stars mode.
        _caption_lines: List[str] = []
        env_metrics_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        # Build per-metric, per-environment caches so each metric uses
        # its own filtered source (success-only vs all results).
        all_metric_keys: list[str] = []
        for group_key, metric_labels in self.config.metric_groups.items():
            all_metric_keys.extend(
                self.config.metric_group_plot_metrics.get(
                    group_key, list(metric_labels.keys())
                )
            )
        for metric_key in all_metric_keys:
            source = self.results_for_metric(metric_key)
            cache: Dict[str, pd.DataFrame] = {}
            for environment in self.ordered_experiments:
                cache[environment] = source[source["experiment"] == environment].copy()
            if self.config.episode_groups:
                for group_name, members in self.config.episode_groups.items():
                    if not members:
                        continue
                    cache[group_name] = source[
                        source["experiment"].isin(members)
                    ].copy()
            env_metrics_cache[metric_key] = cache

        # Determine experiment order from any metric's cache
        first_cache = next(iter(env_metrics_cache.values()), {})
        experiment_order = self._experiment_order_with_groups(first_cache.keys())
        if not experiment_order:
            return
        if all(
            df.empty for cache in env_metrics_cache.values() for df in cache.values()
        ):
            return

        # Determine rows-per-figure: 0 means all in one figure
        rows_per_fig = rows if rows > 0 else len(experiment_order)
        # Split experiment_order into pages
        pages: list[list[str]] = [
            experiment_order[i : i + rows_per_fig]
            for i in range(0, len(experiment_order), rows_per_fig)
        ]

        for group_key, metric_labels in self.config.metric_groups.items():
            metric_keys = self.config.metric_group_plot_metrics.get(
                group_key, list(metric_labels.keys())
            )
            total_metrics = len(metric_keys)
            if total_metrics == 0:
                continue
            # Determine actual columns per grid row
            cols = n_cols if n_cols > 0 else total_metrics
            # Number of grid rows each episode occupies
            metric_rows_per_ep = math.ceil(total_metrics / cols)

            for page_idx, page_experiments in enumerate(pages):
                n_ep = len(page_experiments)
                grid_rows = n_ep * metric_rows_per_ep
                fig, axes = plt.subplots(
                    grid_rows,
                    cols,
                    figsize=(cols * 15, grid_rows * 15),
                    sharex=False,
                )
                axes = np.array(axes).reshape(grid_rows, cols)
                for ep_idx, environment in enumerate(page_experiments):
                    for m_idx, metric_key in enumerate(metric_keys):
                        env_metrics = env_metrics_cache.get(metric_key, {}).get(
                            environment, pd.DataFrame()
                        )
                        local_row = m_idx // cols
                        local_col = m_idx % cols
                        grid_row = ep_idx * metric_rows_per_ep + local_row
                        target_ax = axes[grid_row, local_col]
                        plot_data = self._prepare_distribution_plot_data(
                            env_metrics, metric_key
                        )
                        if plot_data.empty:
                            target_ax.axis("off")
                            continue
                        plot_order = [
                            nav
                            for nav in self.ordered_navigators
                            if nav in set(plot_data["algorithm"].unique())
                        ]
                        if not plot_order:
                            target_ax.axis("off")
                            continue
                        if kind == "box":
                            sns.boxplot(
                                data=plot_data,
                                x="algorithm",
                                order=plot_order,
                                y=metric_key,
                                hue="algorithm",
                                hue_order=plot_order,
                                palette=cast(Any, palette),
                                ax=target_ax,
                                showfliers=False,
                            )
                            self._apply_box_alpha(target_ax, self.config.alpha)
                            if connect_medians:
                                sns.pointplot(
                                    data=plot_data,
                                    x="algorithm",
                                    order=plot_order,
                                    y=metric_key,
                                    estimator=np.median,
                                    errorbar=None,
                                    color="grey",  # single color
                                    markers="x",
                                    linestyles="--",
                                    markersize=20,
                                    markeredgecolor="black",
                                    zorder=10,
                                    ax=target_ax,
                                )

                        else:
                            sns.violinplot(
                                data=plot_data,
                                x="algorithm",
                                order=plot_order,
                                y=metric_key,
                                hue="algorithm",
                                hue_order=plot_order,
                                palette=cast(Any, palette),
                                ax=target_ax,
                            )
                        # -- Statistical annotations --
                        if annotate_stats:
                            if use_baseline_stars:
                                cap = self._annotate_baseline_stars(
                                    ax=target_ax,
                                    data=plot_data,
                                    x="algorithm",
                                    y=metric_key,
                                    order=plot_order,
                                    baseline=stat_baseline or "",
                                    pairwise_df=_pairwise_df,
                                    anova_summary=_anova_summary,
                                    alpha=stat_alpha,
                                    font_size=(
                                        stat_font_size or fs.get("tick_label", 14)
                                    ),
                                    environment=environment,
                                    highlight_label=(
                                        highlight_baseline_label
                                        if highlight_baseline
                                        else None
                                    ),
                                )
                                if cap:
                                    _caption_lines.append(cap)
                            else:
                                self._annotate_subplot(
                                    ax=target_ax,
                                    data=plot_data,
                                    x="algorithm",
                                    y=metric_key,
                                    order=plot_order,
                                    kind=kind,
                                    palette=palette,
                                    pairwise_df=_pairwise_df,
                                    comparisons=stat_comparisons,
                                    stat_test=stat_test,
                                    stat_use_anova=stat_use_anova,
                                    correction=stat_correction,
                                    display_format=stat_display_format,
                                    alpha=stat_alpha,
                                    min_group_size=stat_min_group_size,
                                    font_size=(
                                        stat_font_size or fs.get("tick_label", 14)
                                    ),
                                    line_height=stat_line_height,
                                )

                        # Title: show metric name on every subplot
                        target_ax.set_title(
                            self._format_text(metric_labels[metric_key]),
                            fontsize=fs["title"],
                        )
                        # Episode label on the first column of every
                        # grid row belonging to this episode
                        if local_col == 0 and show_episode_ylabel:
                            target_ax.set_ylabel(environment, fontsize=fs["axis_label"])
                        else:
                            target_ax.set_ylabel("")
                        target_ax.set_xlabel("")
                        target_ax.set_xticks([])
                        target_ax.set_xticklabels([])
                        legend = target_ax.get_legend()
                        if legend is not None:
                            legend.remove()
                        for item in list(target_ax.get_yticklabels()) + [
                            target_ax.yaxis.label
                        ]:
                            item.set_fontsize(fs["tick_label"])

                    # Hide any leftover axes in the last metric-row
                    # of this episode (when metrics don't fill a full row)
                    last_metric_row = (
                        ep_idx * metric_rows_per_ep + metric_rows_per_ep - 1
                    )
                    filled_in_last = total_metrics % cols or cols
                    for empty_col in range(filled_in_last, cols):
                        axes[last_metric_row, empty_col].axis("off")

                suffix = f"_page{page_idx + 1}" if len(pages) > 1 else ""
                output_file = (
                    output_path / f"{group_key}_{kind}plots_by_navigator{suffix}.png"
                )
                legend_labels = [
                    self._format_text(str(handle.get_label()))
                    for handle in legend_handles
                ]
                fig.legend(
                    legend_handles,
                    legend_labels,
                    loc="lower center",
                    ncol=fs["legend_ncol"],
                    fontsize=fs["legend"],
                    frameon=False,
                    labelspacing=1.5,
                )
                # Dynamically compute tight_layout rect from font sizes
                # and figure dimensions so margins scale automatically.
                fig_w, fig_h = fig.get_size_inches()
                pts_to_inches = 1.0 / 72.0
                # Bottom: reserve space for the legend (rows × line-height
                # + spacing).  Each legend row is ~1.8× the font height.
                n_legend_rows = max(
                    1, math.ceil(len(legend_handles) / max(1, fs["legend_ncol"]))
                )
                legend_height_in = (
                    n_legend_rows * fs["legend"] * pts_to_inches * 1.8
                    + 0.3  # extra padding below legend
                )
                rect_bottom = min(legend_height_in / fig_h, 0.25)
                # Left: reserve space for y-axis label when shown.
                if show_episode_ylabel:
                    ylabel_width_in = fs["axis_label"] * pts_to_inches * 1.5
                    rect_left = min(ylabel_width_in / fig_w, 0.08)
                else:
                    rect_left = 0.01
                # Top: small margin for titles.
                title_height_in = fs["title"] * pts_to_inches * 1.5
                rect_top = 1.0 - min(title_height_in / fig_h, 0.05)
                fig.tight_layout(rect=(rect_left, rect_bottom, 0.99, rect_top))
                fig.savefig(output_file, bbox_inches="tight", format="png")
                plt.close(fig)
                # Persist caption alongside the figure when using
                # baseline_stars mode.
                if use_baseline_stars and _caption_lines:
                    caption_file = output_file.with_suffix(".caption.txt")
                    caption_file.write_text(
                        "\n".join(_caption_lines) + "\n",
                        encoding="utf-8",
                    )
                    _caption_lines.clear()

    def plot_proxemic_intrusions(
        self,
        rows: int = 2,
        font_sizes: Optional[Dict[str, int]] = None,
    ) -> None:
        """Plot stacked bar charts for proxemic intrusion statistics.

        Parameters
        ----------
        rows : int
            Number of subplot rows.
        font_sizes : dict, optional
            Override default font sizes.  Recognised keys:
            ``title``, ``axis_label``, ``tick_label``, ``xtick_label``,
            ``legend``, ``sup_label``.
        """

        fs = {**self._DEFAULT_PROXEMIC_FONT_SIZES, **(font_sizes or {})}

        output_path = self.output_dir_figures() / "proxemic_distances_intrusions.png"
        self._ensure_parent_directory(output_path)
        intrusion_keys = list(self.config.intrusion_labels.keys())
        index_cols = ["experiment", "algorithm", "run"]
        # Build a merged DataFrame where each intrusion column comes
        # from its own filtered source (success-only vs all results).
        base_source = self.results_for_metric(intrusion_keys[0])
        merged = base_source[index_cols].drop_duplicates().copy()
        for col_key in intrusion_keys:
            src = self.results_for_metric(col_key)
            merged = merged.merge(
                src[index_cols + [col_key]],
                on=index_cols,
                how="left",
            )
        metrics_df = merged.reset_index(drop=True)
        metrics_df = metrics_df.rename(columns=self.config.intrusion_labels)
        summary_dict = get_metrics_summary(
            metrics_df,
            episode_groups=(
                self.config.episode_groups if self.config.episode_groups else None
            ),
        )
        summary = summary_dict["mean"].reset_index()
        summary = summary.rename(
            columns={
                "experiment": r"\textbf{Experiment}",
                "algorithm": r"\textbf{Navigator}",
            }
        )
        group_summary = summary_dict.get("group_mean")
        if group_summary is not None and not group_summary.empty:
            rename_map: Dict[str, str] = {}
            if "episode_group" in group_summary.columns:
                rename_map["episode_group"] = r"\textbf{Experiment}"
            elif "experiment_group" in group_summary.columns:
                rename_map["experiment_group"] = r"\textbf{Experiment}"
            elif "experiment" in group_summary.columns:
                rename_map["experiment"] = r"\textbf{Experiment}"
            if "algorithm" in group_summary.columns:
                rename_map["algorithm"] = r"\textbf{Navigator}"
            group_summary = group_summary.rename(columns=rename_map).reset_index(
                drop=True
            )
            summary = pd.concat([summary, group_summary], ignore_index=True, sort=False)
        pair_cols = [r"\textbf{Experiment}", r"\textbf{Navigator}"]
        summary = summary.drop_duplicates(subset=pair_cols, keep="first")
        intrusion_columns: List[str] = [
            str(label) for label in self.config.intrusion_labels.values()
        ]
        intrusion_palette = self.get_intrusion_palette()
        available_labels = summary[r"\textbf{Experiment}"].unique().tolist()
        experiment_order = self._experiment_order_with_groups(available_labels)
        if not experiment_order:
            return
        n_items = len(experiment_order)
        cols = max(1, math.ceil(n_items / rows))
        rows = max(1, math.ceil(n_items / cols))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(cols * 15, rows * 15),
            sharey=True,
            sharex=True,
        )
        axes = np.array(axes).reshape(rows, cols)
        for idx, environment in enumerate(experiment_order):
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
                label.set_fontsize(fs["xtick_label"])
            target_ax.set_title(environment, fontsize=fs["title"])
            target_ax.legend().remove()
            target_ax.set_xlabel("")
            target_ax.set_ylim(0, 100)
            target_ax.set_yticks([25, 50, 75, 100])
            for item in target_ax.get_yticklabels():
                item.set_fontsize(fs["tick_label"])
            for item in [target_ax.title, target_ax.xaxis.label, target_ax.yaxis.label]:
                item.set_fontsize(fs["axis_label"])
        for idx in range(len(experiment_order), rows * cols):
            row_idx, col_idx = divmod(idx, cols)
            axes[row_idx, col_idx].axis("off")
        fig.supylabel(
            self._format_text(r"Intrusion (Time \%)"),
            fontsize=fs["sup_label"],
            y=0.6,
        )
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
            ncol=fs["legend_ncol"],
            fontsize=fs["legend"],
            frameon=False,
            labelspacing=1.5,
        )
        # Dynamically compute tight_layout rect from font sizes
        # and figure dimensions so margins scale automatically.
        fig_w, fig_h = fig.get_size_inches()
        pts_to_inches = 1.0 / 72.0
        # Bottom: legend rows
        n_legend_rows_pi = max(
            1, math.ceil(len(intrusion_handles) / max(1, fs["legend_ncol"]))
        )
        legend_h = n_legend_rows_pi * fs["legend"] * pts_to_inches * 1.8 + 0.3
        rect_bottom = min(legend_h / fig_h, 0.35)
        # Left: supylabel
        sup_w = fs["sup_label"] * pts_to_inches 
        rect_left = min(sup_w / fig_w, 0.08)
        # Top: title
        title_h = fs["title"] * pts_to_inches * 1.5
        rect_top = 1.0 - min(title_h / fig_h, 0.05)
        fig.tight_layout(rect=(rect_left, rect_bottom, 0.99, rect_top))
        fig.savefig(output_path, bbox_inches="tight", format="png")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Baseline-stars annotation helper
    # ------------------------------------------------------------------
    @staticmethod
    def _pvalue_to_stars(p: float) -> str:
        """Convert a p-value to a compact star string."""
        if p <= 0.0001:
            return "****"
        if p <= 0.001:
            return "***"
        if p <= 0.01:
            return "**"
        if p <= 0.05:
            return "*"
        return "ns"

    def _annotate_baseline_stars(
        self,
        *,
        ax: Any,
        data: pd.DataFrame,
        x: str,
        y: str,
        order: List[str],
        baseline: str,
        pairwise_df: Optional[pd.DataFrame],
        anova_summary: Optional[pd.DataFrame],
        alpha: float,
        font_size: int,
        environment: str,
        highlight_label: Optional[str] = None,
    ) -> str:
        """Place compact stars above each non-baseline box/violin.

        Returns a caption string summarising the statistical details
        for this metric × environment combination.  The caller can
        aggregate these captions and save them alongside the figure.

        When *highlight_label* is not ``None``, the given text (e.g.
        ``"Ours"``) is placed above the baseline position so it
        visually stands out from the ``*``/``ns`` markers on the
        other algorithms.
        """

        present = set(data[x].unique())
        plot_order = [nav for nav in order if nav in present]
        if baseline not in present or len(plot_order) < 2:
            return ""

        # Look up x-positions (integer index in plot_order)
        xpos = {label: idx for idx, label in enumerate(plot_order)}

        # Collect (navigator, p-value, stars) for each comparison vs
        # baseline from the Tukey pairwise results.
        annotations: List[Tuple[str, float, str]] = []

        if pairwise_df is not None and not pairwise_df.empty:
            metric_rows = pairwise_df[pairwise_df["metric_key"] == y]
            for nav in plot_order:
                if nav == baseline:
                    continue
                # Look up in either direction (a,b) or (b,a)
                mask = (
                    (metric_rows["group_a"] == baseline)
                    & (metric_rows["group_b"] == nav)
                ) | (
                    (metric_rows["group_a"] == nav)
                    & (metric_rows["group_b"] == baseline)
                )
                match = metric_rows[mask]
                if match.empty:
                    continue
                p_val = float(match.iloc[0]["p_adjusted"])
                stars = self._pvalue_to_stars(p_val)
                annotations.append((nav, p_val, stars))
        else:
            # No pairwise data available – nothing to annotate
            return ""

        if not annotations:
            return ""

        # Determine y-position: place stars just above the data range
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min
        star_y = y_max + y_range * 0.02
        # Extend axis to make room for the stars
        ax.set_ylim(y_min, star_y + y_range * 0.06)

        for nav, p_val, stars in annotations:
            xi = xpos[nav]
            ax.text(
                xi,
                star_y,
                stars,
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold",
                color="black" if stars != "ns" else "gray",
            )

        # Place the highlight label above the baseline box/violin.
        if highlight_label and baseline in xpos:
            ax.text(
                xpos[baseline],
                star_y,
                highlight_label,
                ha="center",
                va="bottom",
                fontsize=font_size,
                fontweight="bold",
                fontstyle="italic",
                color="black",
            )

        # --- Build caption text ---
        # ANOVA omnibus line (if available)
        caption_parts: List[str] = []
        if anova_summary is not None and not anova_summary.empty:
            if y in anova_summary.index:
                row = anova_summary.loc[y]
                test_name = str(row.get("test_type", "ANOVA")).replace("_", " ").title()
                f_val = row.get("f_statistic", float("nan"))
                p_omni = row.get("p_value", float("nan"))
                df_b = row.get("df_between", "?")
                df_w = row.get("df_within", "?")
                caption_parts.append(
                    f"[{environment} | {y}] {test_name}: "
                    f"F({df_b},{df_w})={f_val:.2f}, p={p_omni:.4f}."
                )

        # Pairwise details
        pair_strs: List[str] = []
        for nav, p_val, stars in annotations:
            pair_strs.append(f"{nav} vs {baseline}: p={p_val:.4f} ({stars})")
        caption_parts.append(
            f"  Pairwise (Tukey HSD, alpha={alpha}): " + "; ".join(pair_strs) + "."
        )
        return " ".join(caption_parts)

    def _annotate_subplot(
        self,
        *,
        ax: Any,
        data: pd.DataFrame,
        x: str,
        y: str,
        order: List[str],
        kind: str,
        palette: Any,
        pairwise_df: Optional[pd.DataFrame],
        comparisons: Optional[List[Tuple[str, str]]],
        stat_test: str,
        stat_use_anova: bool,
        correction: Optional[str],
        display_format: str,
        alpha: float,
        min_group_size: int,
        font_size: int,
        line_height: float,
    ) -> None:
        """Add ``statannotations`` brackets to a single subplot.

        When *stat_use_anova* is ``True`` the pre-computed Tukey HSD
        adjusted p-values are injected directly into the annotator so
        that the displayed significance matches the ANOVA analysis.
        Otherwise ``statannotations`` runs *stat_test* on its own.
        """

        if _Annotator is None:
            return  # silently skip when not installed

        # Determine which navigators actually appear in the data
        present = set(data[x].unique())
        plot_order = [nav for nav in order if nav in present]
        if len(plot_order) < 2:
            return

        # --- Build the list of pairs to annotate -----------------------
        if stat_use_anova and pairwise_df is not None and not pairwise_df.empty:
            metric_rows = pairwise_df[pairwise_df["metric_key"] == y]
            if metric_rows.empty:
                return
            if comparisons is not None:
                # Keep only user-specified pairs that exist in results
                pairs = [
                    (a, b) for a, b in comparisons if a in present and b in present
                ]
            else:
                # Take all significant pairs from Tukey results
                sig_rows = metric_rows[metric_rows["reject"] == True]  # noqa: E712
                pairs = []
                for _, row in sig_rows.iterrows():
                    a, b = str(row["group_a"]), str(row["group_b"])
                    if a in present and b in present:
                        pairs.append((a, b))
            if not pairs:
                return

            # Map each pair to its adjusted p-value
            pair_pvalues: Dict[Tuple[str, str], float] = {}
            for _, row in metric_rows.iterrows():
                key = (str(row["group_a"]), str(row["group_b"]))
                pair_pvalues[key] = float(row["p_adjusted"])
                # Also store the reverse for easy look-up
                pair_pvalues[(key[1], key[0])] = float(row["p_adjusted"])

            # Re-plot into Annotator with matching plot params
            plot_params: Dict[str, Any] = {
                "data": data,
                "x": x,
                "y": y,
                "order": plot_order,
                "hue": x,
                "hue_order": plot_order,
                "palette": palette,
            }
            if kind == "box":
                plot_params["showfliers"] = False

            annotator = _Annotator(
                ax=ax,
                pairs=pairs,
                plot=f"{kind}plot",
                **plot_params,
            )
            # Inject pre-computed p-values so statannotations only
            # formats / draws — no test is run.
            p_values = [pair_pvalues.get(pair, 1.0) for pair in pairs]
            annotator.configure(
                text_format=display_format,
                loc="inside",
                fontsize=font_size,
                line_height=line_height,
                line_offset=line_height,
                hide_non_significant=True,
            )
            annotator.set_pvalues(p_values)
            annotator.annotate()
            if kind == "box":
                self._apply_box_alpha(ax, self.config.alpha)

        else:
            # Let statannotations run the test itself
            if comparisons is not None:
                pairs = [
                    (a, b) for a, b in comparisons if a in present and b in present
                ]
            else:
                from itertools import combinations

                pairs = list(combinations(plot_order, 2))
            if not pairs:
                return

            plot_params = {
                "data": data,
                "x": x,
                "y": y,
                "order": plot_order,
                "hue": x,
                "hue_order": plot_order,
                "palette": palette,
            }
            if kind == "box":
                plot_params["showfliers"] = False

            annotator = _Annotator(
                ax=ax,
                pairs=pairs,
                plot=f"{kind}plot",
                **plot_params,
            )
            annotator.configure(
                test=stat_test,
                text_format=display_format,
                loc="inside",
                fontsize=font_size,
                line_height=line_height,
                line_offset=line_height,
                hide_non_significant=True,
                comparisons_correction=correction,
            )
            annotator.apply_and_annotate()
            if kind == "box":
                self._apply_box_alpha(ax, self.config.alpha)

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
