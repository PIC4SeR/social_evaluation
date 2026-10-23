#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""Command line utility to generate benchmark reports from a YAML config."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import yaml
import numpy as np

from social_evaluation.benchmark_report import (
    BenchmarkConfig,
    BenchmarkReport,
    StatsSettings,
    format_intrusion_labels,
    format_metric_groups,
    generate_reports_for_scenario,
    resolve_path,
)
from social_evaluation.benchmark_report.tex_reports import MissingAssetError


def _log(message: str) -> None:
    print(f"[benchmark-report] {message}")


def _is_enabled(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        if "enabled" in value:
            return bool(value.get("enabled"))
        return True
    return bool(value)


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("Top-level YAML structure must be a mapping")
    return dict(data)


def _to_camel_case(snake_str: str) -> str:
    components = snake_str.split("_")
    # Capitalize the first letter of each component and separate them by spaces
    return " ".join(components).title()


def _resolve_required_path(value: Optional[str], base: Path, key: str) -> Path:
    if value is None:
        raise ValueError(f"Missing required path configuration: '{key}'")
    resolved = resolve_path(value, base)
    if resolved is None:
        raise ValueError(f"Could not resolve path for '{key}'")
    return resolved


def _as_scenario_entries(value: Any) -> List[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _to_numpy_tau(values: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    if not isinstance(values, Sequence):
        raise ValueError("tau must be a sequence of numeric values")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("tau must be a one-dimensional sequence")
    return array


def _parse_font_sizes(cfg: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """Extract a ``font_sizes`` mapping from a config entry, if present."""
    raw = cfg.get("font_sizes")
    if not isinstance(raw, Mapping):
        return None
    return {str(k): int(v) for k, v in raw.items()}


def _parse_stat_comparisons(
    cfg: Mapping[str, Any],
) -> Optional[List[Tuple[str, str]]]:
    """Parse ``stat_comparisons`` from a config entry.

    Accepts a list of two-element lists/tuples, e.g.::

        stat_comparisons:
          - ["AlgoA", "AlgoB"]
          - ["AlgoA", "AlgoC"]

    Returns ``None`` when the key is absent or empty, signalling that
    the default behaviour (all significant pairs) should be used.
    """
    raw = cfg.get("stat_comparisons")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    pairs: List[Tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs if pairs else None


def _parse_stat_annotation_kwargs(
    entry: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract statistical-annotation parameters from a config entry.

    Returns a dict of keyword arguments ready to be unpacked into
    ``plot_metric_distributions``.
    """
    kwargs: Dict[str, Any] = {}
    if "annotate_stats" in entry:
        kwargs["annotate_stats"] = bool(entry["annotate_stats"])
    if "stat_test" in entry:
        kwargs["stat_test"] = str(entry["stat_test"])
    comparisons = _parse_stat_comparisons(entry)
    if comparisons is not None:
        kwargs["stat_comparisons"] = comparisons
    if "stat_correction" in entry:
        val = entry["stat_correction"]
        kwargs["stat_correction"] = None if val is None else str(val)
    if "stat_display_format" in entry:
        kwargs["stat_display_format"] = str(entry["stat_display_format"])
    if "stat_alpha" in entry:
        kwargs["stat_alpha"] = float(entry["stat_alpha"])
    if "stat_min_group_size" in entry:
        kwargs["stat_min_group_size"] = int(entry["stat_min_group_size"])
    # stat_use_anova is the older name, kept for existing configs
    for key in ("stat_use_anova", "stat_use_statistics"):
        if key in entry:
            kwargs["stat_use_statistics"] = bool(entry[key])
    if "stat_font_size" in entry:
        kwargs["stat_font_size"] = int(entry["stat_font_size"])
    if "stat_line_height" in entry:
        kwargs["stat_line_height"] = float(entry["stat_line_height"])
    if "stat_annotation_mode" in entry:
        kwargs["stat_annotation_mode"] = str(entry["stat_annotation_mode"])
    if "stat_baseline" in entry:
        kwargs["stat_baseline"] = str(entry["stat_baseline"])
    if "highlight_baseline" in entry:
        kwargs["highlight_baseline"] = bool(entry["highlight_baseline"])
    if "highlight_baseline_label" in entry:
        kwargs["highlight_baseline_label"] = str(entry["highlight_baseline_label"])
    return kwargs


def _statistics_config(outputs_cfg: Mapping[str, Any]) -> Any:
    """The ``statistics`` block; the older ``export_anova_analysis`` key means test: auto."""
    if "statistics" in outputs_cfg:
        return outputs_cfg["statistics"]
    return outputs_cfg.get("export_anova_analysis")


def _run_exports(report: BenchmarkReport, outputs_cfg: Mapping[str, Any]) -> None:
    if _is_enabled(outputs_cfg.get("export_success_rate_tables"), False):
        _log("Exporting success rate tables")
        report.export_success_rate_tables()

    metric_tables_cfg = outputs_cfg.get("export_metric_tables")
    if _is_enabled(metric_tables_cfg, False):
        scale = 0.71
        cmap_name: Optional[str] = None
        show_std = True
        group_keys: Optional[List[str]] = None
        weighted_tables = False               # success-weighted tables only on request
        if isinstance(metric_tables_cfg, Mapping):
            weighted_tables = bool(metric_tables_cfg.get("weighted", weighted_tables))
            scale = float(metric_tables_cfg.get("scale", scale))
            raw_cmap = metric_tables_cfg.get("cmap_name")
            if raw_cmap is not None:
                cmap_name = str(raw_cmap).strip() or None
            show_std_val = metric_tables_cfg.get("show_std")
            if show_std_val is not None:
                show_std = bool(show_std_val)
            raw_groups = metric_tables_cfg.get("groups")
            if raw_groups is not None:
                if isinstance(raw_groups, (str, bytes)):
                    group_keys = [str(raw_groups)]
                elif isinstance(raw_groups, Iterable):
                    group_keys = [str(g) for g in raw_groups]
        _log("Exporting metric summary tables")
        report.export_metric_tables(
            scale=scale,
            cmap_name=cmap_name,
            show_std=show_std,
            group_keys=group_keys,
            weighted=weighted_tables,
        )

    stats_cfg = _statistics_config(outputs_cfg)
    if _is_enabled(stats_cfg, False) and (
        not isinstance(stats_cfg, Mapping) or stats_cfg.get("export", True)
    ):
        subdir = stats_cfg.get("output_subdir") if isinstance(stats_cfg, Mapping) else None
        _log("Exporting statistics")
        report.export_statistics(report.output_dir_tables() / subdir if subdir else None)


def _run_plots(
    report: BenchmarkReport, outputs_cfg: Mapping[str, Any], base: Path
) -> None:
    sr_cfg = outputs_cfg.get("plot_success_rate")
    if _is_enabled(sr_cfg, False):
        sr_font_sizes: Optional[Dict[str, int]] = None
        if isinstance(sr_cfg, Mapping):
            if "n_cols" in sr_cfg:
                report.config.success_rate_n_cols = int(sr_cfg["n_cols"])
            sr_font_sizes = _parse_font_sizes(sr_cfg)
        _log("Plotting success rate")
        report.plot_success_rate(font_sizes=sr_font_sizes)

    distributions_cfg = outputs_cfg.get("plot_metric_distributions")
    if isinstance(distributions_cfg, Iterable) and not isinstance(
        distributions_cfg, (str, Mapping)
    ):
        for entry in distributions_cfg:
            if not isinstance(entry, Mapping):
                continue
            kind = str(entry.get("kind", "box"))
            cmap_name = str(entry.get("cmap_name", "Paired"))
            n_rows = int(entry.get("n_rows", 0))
            n_cols = int(entry.get("n_cols", 0))
            show_episode_ylabel = bool(entry.get("show_episode_ylabel", True))
            connect_medians = bool(entry.get("connect_medians", False))
            font_sizes = _parse_font_sizes(entry)
            stat_kwargs = _parse_stat_annotation_kwargs(entry)
            _log(f"Plotting metric distributions ({kind})")
            report.plot_metric_distributions(
                kind=kind,
                cmap_name=cmap_name,
                rows=n_rows,
                n_cols=n_cols,
                font_sizes=font_sizes,
                show_episode_ylabel=show_episode_ylabel,
                connect_medians=connect_medians,
                **stat_kwargs,
            )
    elif _is_enabled(distributions_cfg, False):
        _log("Plotting metric distributions (box)")
        report.plot_metric_distributions(kind="box", cmap_name="Paired")

    proxemic_cfg = outputs_cfg.get("plot_proxemic_intrusions")
    if _is_enabled(proxemic_cfg, False):
        _log("Plotting proxemic intrusions")
        n_rows = 2
        font_sizes_prox: Optional[Dict[str, int]] = None
        if isinstance(proxemic_cfg, Mapping):
            n_rows = int(proxemic_cfg.get("n_rows", n_rows))
            font_sizes_prox = _parse_font_sizes(proxemic_cfg)
        report.plot_proxemic_intrusions(rows=n_rows, font_sizes=font_sizes_prox)

    heatmaps = (
        ("plot_pairwise_pvalue_heatmap", "p_adjusted"),
        ("plot_pairwise_mean_diff_heatmap", "mean_diff"),
        ("plot_pairwise_hedges_heatmap", "hedges_g"),
    )
    style_keys = ("cmap", "annotate", "fmt", "label_size", "title_size",
                  "colorbar_label_size", "annot_font_size")
    for key, value in heatmaps:
        heatmap_cfg = outputs_cfg.get(key)
        if not _is_enabled(heatmap_cfg, False):
            continue
        heatmap_cfg = heatmap_cfg if isinstance(heatmap_cfg, Mapping) else {}
        subdir = heatmap_cfg.get("output_subdir")
        _log(f"Plotting pairwise {value} heatmaps")
        try:
            report.plot_pairwise_heatmaps(
                value,
                heatmap_cfg.get("metrics"),
                output_dir=report.output_dir_figures() / subdir if subdir else None,
                alpha=heatmap_cfg.get("alpha"),
                min_group_size=heatmap_cfg.get("min_group_size"),
                **{k: heatmap_cfg[k] for k in style_keys if k in heatmap_cfg},
            )
        except (RuntimeError, ValueError) as exc:
            _log(f"Skipping {key}: {exc}")

    perf_cfg = outputs_cfg.get("plot_metric_performance_profiles")
    if _is_enabled(perf_cfg, False):
        normalize = True
        reps = 2000
        legend_columns = 4
        figsize_scale = 6.5
        use_average = False
        tau_values: Optional[np.ndarray] = None
        output_subdir = report.output_dir_figures() / "rliable"
        groups: Optional[Sequence[str]] = None
        metrics: Optional[Sequence[str]] = None

        if isinstance(perf_cfg, Mapping):
            normalize = bool(perf_cfg.get("normalize", normalize))
            reps = int(perf_cfg.get("reps", reps))
            legend_columns = int(perf_cfg.get("legend_columns", legend_columns))
            figsize_scale = float(perf_cfg.get("figsize_scale", figsize_scale))
            use_average = bool(perf_cfg.get("use_average_distribution", use_average))
            tau_values = _to_numpy_tau(perf_cfg.get("tau"))
            groups = perf_cfg.get("groups")
            metrics = perf_cfg.get("metrics")
            output_dir_cfg = perf_cfg.get("output_subdir")
            if output_dir_cfg is not None:
                output_subdir = _resolve_required_path(
                    output_dir_cfg, base, "output_subdir"
                )
        if groups is None and metrics is None:
            groups = list(report.config.metric_groups.keys())

        for group_key in groups or []:
            output_path = output_subdir / f"{group_key}_performance_profiles.png"
            _log(f"Plotting performance profiles for '{group_key}'")
            report.plot_metric_performance_profiles(
                metric_keys=group_key,
                normalize=normalize,
                reps=reps,
                tau=tau_values,
                use_average_distribution=use_average,
                output_path=output_path,
                legend_columns=legend_columns,
                figsize_scale=figsize_scale,
            )

        if metrics:
            output_path = output_subdir / "selected_metrics_performance_profiles.png"
            _log("Plotting performance profiles for custom metrics")
            report.plot_metric_performance_profiles(
                metric_keys=list(metrics),
                normalize=normalize,
                reps=reps,
                tau=tau_values,
                use_average_distribution=use_average,
                output_path=output_path,
                legend_columns=legend_columns,
                figsize_scale=figsize_scale,
            )

    interval_cfg = outputs_cfg.get("plot_rliable_interval_estimates")
    if _is_enabled(interval_cfg, False):
        metrics_list: Sequence[str] = ()
        normalize = True
        reps = 5000
        if isinstance(interval_cfg, Mapping):
            metrics_list = interval_cfg.get("metrics", metrics_list)
            normalize = bool(interval_cfg.get("normalize", normalize))
            reps = int(interval_cfg.get("reps", reps))
        if not metrics_list:
            metrics_list = tuple(report.config.metric_directions.keys())
        for metric_key in metrics_list:
            _log(f"Plotting rliable interval estimates for '{metric_key}'")
            report.plot_rliable_interval_estimates(
                metric_key=str(metric_key),
                normalize=normalize,
                reps=reps,
            )

    prob_cfg = outputs_cfg.get("plot_probability_of_improvement")
    if _is_enabled(prob_cfg, False):
        if not isinstance(prob_cfg, Mapping):
            raise ValueError(
                "Probability of improvement configuration must be a mapping"
            )
        baseline = prob_cfg.get("baseline_algorithm")
        if not baseline:
            raise ValueError(
                "'baseline_algorithm' is required for probability of improvement plots"
            )
        metrics_list: Sequence[str] = prob_cfg.get("metrics", ())
        normalize = bool(prob_cfg.get("normalize", True))
        reps = int(prob_cfg.get("reps", 2000))
        if not metrics_list:
            raise ValueError(
                "Specify at least one metric for probability of improvement plots"
            )
        for metric_key in metrics_list:
            _log(
                "Plotting probability of improvement for "
                f"'{metric_key}' vs baseline '{baseline}'"
            )
            try:
                report.plot_probability_of_improvement(
                    metric_key=str(metric_key),
                    baseline_algorithm=str(baseline),
                    normalize=normalize,
                    reps=reps,
                )
            except ValueError as exc:
                _log(
                    f"Skipping probability of improvement plot for '{metric_key}': {exc}"
                )


def _generate_tex_reports(
    report: BenchmarkReport, outputs_cfg: Mapping[str, Any]
) -> None:
    tex_cfg = outputs_cfg.get("generate_latex_reports")
    if not _is_enabled(tex_cfg, False):
        return

    prefix_override: Optional[str] = None
    overwrite = False
    weighted = False                          # <prefix>_weighted.tex only on request
    if isinstance(tex_cfg, Mapping):
        prefix_override = tex_cfg.get("prefix")
        overwrite = bool(tex_cfg.get("overwrite", overwrite))
        weighted = bool(tex_cfg.get("weighted", weighted))

    scenario_dir = report.output_dir()
    scenario_dir.mkdir(parents=True, exist_ok=True)
    parent_name = scenario_dir.parent.name
    default_prefix = (
        parent_name if parent_name and parent_name != scenario_dir.name else "report"
    )
    report_prefix = prefix_override or default_prefix

    _log(
        "Generating LaTeX report drivers for "
        f"'{report.config.scenario}' (prefix '{report_prefix}')"
    )
    try:
        generate_reports_for_scenario(
            scenario_dir=scenario_dir,
            report_prefix=report_prefix,
            overwrite=overwrite,
            reports_root=scenario_dir.parent,
            metric_group_keys=list(report.config.metric_groups.keys()),
            logger=_log,
            weighted=weighted,
        )
    except MissingAssetError as exc:
        _log(f"Skipping LaTeX reports for '{report.config.scenario}': {exc}")
    except FileExistsError as exc:
        _log(f"Skipping LaTeX reports for '{report.config.scenario}': {exc}")


def _build_benchmark_config(
    benchmark_cfg: Mapping[str, Any],
    base: Path,
    scenario_entry: Any,
    *,
    multi_scenario: bool = False,
) -> BenchmarkConfig:
    required_keys = [
        "root_dir",
        "navigators",
        "navigator_names",
        "metric_groups",
        "intrusion_labels",
    ]
    for key in required_keys:
        if key not in benchmark_cfg:
            raise ValueError(f"Missing required benchmark configuration key: '{key}'")

    def _resolve_bool_config(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    scenario_name: Optional[str] = None
    scenario_episodes_value: Any = None
    scenario_navigators_value: Any = None
    scenario_groups_value: Any = None
    scenario_discard_failed_value: Any = None
    if isinstance(scenario_entry, Mapping):
        scenario_name = scenario_entry.get("name") or scenario_entry.get("scenario")
        scenario_episodes_value = scenario_entry.get("episodes")
        scenario_navigators_value = scenario_entry.get("navigators")
        scenario_groups_value = scenario_entry.get("episode_groups")
        if scenario_groups_value is None:
            scenario_groups_value = scenario_entry.get("experiment_groups")
        scenario_discard_failed_value = scenario_entry.get("discard_failed_metrics")
    else:
        scenario_name = str(scenario_entry)

    if not scenario_name:
        raise ValueError("Each scenario entry must provide a non-empty 'name'")

    episodes_value = scenario_episodes_value
    if episodes_value is None:
        episodes_value = benchmark_cfg.get("episodes")
    episodes: List[str] = []
    if episodes_value is not None:
        if isinstance(episodes_value, (str, bytes)):
            raise ValueError(
                f"Episodes for scenario '{scenario_name}' must be a sequence, not a string"
            )
        if not isinstance(episodes_value, Iterable):
            raise ValueError(
                f"Episodes for scenario '{scenario_name}' must be iterable"
            )
        episodes = [str(item) for item in episodes_value]
    display_episode_order = list(episodes)
    seen_episodes = set()
    duplicate_episodes: List[str] = []
    for episode in episodes:
        if episode in seen_episodes:
            duplicate_episodes.append(episode)
        else:
            seen_episodes.add(episode)
    if duplicate_episodes:
        duplicates_sorted = ", ".join(sorted(set(duplicate_episodes)))
        raise ValueError(
            f"Duplicate episodes found for scenario '{scenario_name}': {duplicates_sorted}"
        )

    navigators_value = (
        scenario_navigators_value
        if scenario_navigators_value is not None
        else benchmark_cfg.get("navigators")
    )
    if navigators_value is None:
        raise ValueError(
            f"Missing navigators configuration for scenario '{scenario_name}'"
        )
    if isinstance(navigators_value, (str, bytes)):
        navigators = [str(navigators_value)]
    else:
        if not isinstance(navigators_value, Iterable):
            raise ValueError(
                f"Navigators for scenario '{scenario_name}' must be iterable"
            )
        navigators = [str(item) for item in navigators_value]
    if not navigators:
        raise ValueError(
            f"Scenario '{scenario_name}' must specify at least one navigator"
        )
    seen_navigators = set()
    duplicate_navigators: List[str] = []
    for navigator in navigators:
        if navigator in seen_navigators:
            duplicate_navigators.append(navigator)
        else:
            seen_navigators.add(navigator)
    if duplicate_navigators:
        duplicates_sorted = ", ".join(sorted(set(duplicate_navigators)))
        raise ValueError(
            f"Duplicate navigators found for scenario '{scenario_name}': {duplicates_sorted}"
        )

    metric_groups_spec = benchmark_cfg.get("metric_groups", {})
    (
        formatted_groups,
        metric_dirs,
        discard_overrides,
        metric_group_aggregates,
        metric_group_plot_metrics,
    ) = format_metric_groups(metric_groups_spec)

    intrusion_spec = benchmark_cfg.get("intrusion_labels", {})
    intrusion_labels_formatted, intrusion_discard_overrides = format_intrusion_labels(
        intrusion_spec
    )

    output_dir = base / scenario_name
    output_dir = output_dir.resolve()
    # check if episode names are provided; if not, generate from episode keys
    episode_names_cfg = dict(benchmark_cfg.get("episode_names", {}))
    scenario_episode_names: Dict[str, str] = {}

    def _ensure_episode_entry(episode_key: str) -> str:
        if episode_key not in episode_names_cfg:
            episode_names_cfg[episode_key] = _to_camel_case(episode_key)
        scenario_episode_names.setdefault(
            episode_key, str(episode_names_cfg[episode_key])
        )
        return scenario_episode_names[episode_key]

    def _append_episode_if_missing(episode_key: str) -> None:
        if episode_key not in episodes:
            episodes.append(episode_key)

    for episode in episodes:
        _ensure_episode_entry(episode)

    navigator_names_cfg = dict(benchmark_cfg.get("navigator_names", {}))
    scenario_navigator_names: Dict[str, str] = {}
    missing_navigator_labels: List[str] = []
    for navigator in navigators:
        mapped_label = navigator_names_cfg.get(navigator)
        if mapped_label is None:
            missing_navigator_labels.append(navigator)
            mapped_label = str(navigator)
        scenario_navigator_names[navigator] = str(mapped_label)
    if missing_navigator_labels:
        missing_formatted = ", ".join(sorted(missing_navigator_labels))
        _log(
            "Navigator name mappings missing for: "
            f"{missing_formatted}. Using raw keys as labels."
        )

    episode_groups_value = (
        scenario_groups_value
        if scenario_groups_value is not None
        else benchmark_cfg.get("episode_groups", benchmark_cfg.get("experiment_groups"))
    )

    def _normalize_episode_groups(
        raw_groups: Any,
    ) -> Dict[str, List[str]]:
        if raw_groups is None:
            return {}
        if not isinstance(raw_groups, Mapping):
            raise ValueError(
                f"'episode_groups' for scenario '{scenario_name}' must be a mapping"
            )
        normalized: Dict[str, List[str]] = {}

        def _resolve_member(label: str) -> Optional[str]:
            if not label:
                return None
            if label in scenario_episode_names:
                return scenario_episode_names[label]
            for key, display_name in scenario_episode_names.items():
                if display_name == label:
                    return display_name
            # Treat as a raw episode key not yet listed; include for data loading.
            _append_episode_if_missing(label)
            return _ensure_episode_entry(label)

        for raw_group_name, members in raw_groups.items():
            group_name = str(raw_group_name)
            if not group_name or members is None:
                continue
            if isinstance(members, (str, bytes)):
                member_iterable = [members]
            elif isinstance(members, Iterable):
                member_iterable = list(members)
            else:
                raise ValueError(
                    "Episode group definitions must be strings or iterables "
                    f"of strings (group '{group_name}' in scenario '{scenario_name}')"
                )
            resolved_members: List[str] = []
            for member in member_iterable:
                resolved_label = _resolve_member(str(member))
                if resolved_label:
                    resolved_members.append(resolved_label)
            if resolved_members:
                normalized[group_name] = resolved_members
        return normalized

    scenario_episode_groups = _normalize_episode_groups(episode_groups_value)

    for aggregate_spec in metric_group_aggregates.values():
        raw_exclusions = aggregate_spec.get("exclude_metrics_by_experiment")
        if not isinstance(raw_exclusions, Mapping):
            continue
        normalized_exclusions: Dict[str, List[str]] = {}
        for raw_experiment_name, raw_metric_keys in raw_exclusions.items():
            experiment_name = str(raw_experiment_name)
            normalized_experiment_name = scenario_episode_names.get(
                experiment_name, experiment_name
            )
            if normalized_experiment_name in scenario_episode_groups:
                target_name = normalized_experiment_name
            elif experiment_name in scenario_episode_groups:
                target_name = experiment_name
            else:
                target_name = normalized_experiment_name
            if isinstance(raw_metric_keys, (str, bytes)):
                metric_keys = [str(raw_metric_keys)]
            elif isinstance(raw_metric_keys, Iterable):
                metric_keys = [str(metric_key) for metric_key in raw_metric_keys]
            else:
                continue
            if metric_keys:
                normalized_exclusions[target_name] = metric_keys
        aggregate_spec["exclude_metrics_by_experiment"] = normalized_exclusions

    discard_failed_default = _resolve_bool_config(
        benchmark_cfg.get("discard_failed_metrics"), True
    )

    discard_failed_metrics = _resolve_bool_config(
        scenario_discard_failed_value, discard_failed_default
    )

    # Ensure episodes used for data loading include group members while keeping display order separate.
    ordered_episodes: List[str] = []
    seen_eps: set[str] = set()
    for episode in episodes:
        if episode not in seen_eps:
            ordered_episodes.append(episode)
            seen_eps.add(episode)
    if not ordered_episodes:
        raise ValueError(
            f"Scenario '{scenario_name}' must specify at least one episode via "
            "'episodes' or 'episode_groups'"
        )

    config = BenchmarkConfig(
        root_dir=_resolve_required_path(
            benchmark_cfg.get("root_dir"), base, "root_dir"
        ),
        scenario=str(scenario_name),
        navigators=navigators,
        episodes=ordered_episodes,
        display_episodes=display_episode_order,
        navigator_names=scenario_navigator_names,
        episode_names=scenario_episode_names,
        metric_groups=formatted_groups,
        metric_directions=metric_dirs,
        metric_captions=dict(benchmark_cfg.get("metric_captions", {})),
        intrusion_labels=intrusion_labels_formatted,
        episode_groups=scenario_episode_groups,
        alpha=float(benchmark_cfg.get("alpha", 0.8)),
        output_dir=output_dir,
        use_tex=bool(benchmark_cfg.get("use_tex", True)),
        discard_failed_metrics=discard_failed_metrics,
        quality_result_cmap=str(benchmark_cfg.get("intrusion_cmap", "Paired")),
        metric_discard_failed={**discard_overrides, **intrusion_discard_overrides},
        metric_group_aggregates=metric_group_aggregates,
        metric_group_plot_metrics=metric_group_plot_metrics,
    )
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("report_config.yaml"),
        help="Path to the YAML configuration file",
    )

    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=None,
        help="Override the base output directory for all scenarios",
    )

    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise SystemExit(f"Configuration file not found: {config_path}")

    base_dir = (
        config_path.parent
        if args.base_output_dir is None
        else args.base_output_dir.expanduser().resolve()
    )
    raw_config = _load_yaml_config(config_path)

    benchmark_cfg = raw_config.get("benchmark")
    if not isinstance(benchmark_cfg, Mapping):
        raise SystemExit("The YAML file must contain a 'benchmark' mapping")

    outputs_cfg = raw_config.get("outputs", {})
    if outputs_cfg and not isinstance(outputs_cfg, Mapping):
        raise SystemExit("The 'outputs' section must be a mapping if provided")

    scenario_spec = benchmark_cfg.get("scenario")
    if scenario_spec is None:
        raise SystemExit("The benchmark configuration must include a 'scenario' entry")
    scenario_entries = _as_scenario_entries(scenario_spec)
    configs: List[BenchmarkConfig] = []
    for entry in scenario_entries:
        config = _build_benchmark_config(
            benchmark_cfg,
            base_dir,
            entry,
            multi_scenario=len(scenario_entries) > 1,
        )
        configs.append(config)

    seen_scenarios = set()
    duplicate_scenarios: List[str] = []
    for config in configs:
        if config.scenario in seen_scenarios:
            duplicate_scenarios.append(config.scenario)
        else:
            seen_scenarios.add(config.scenario)
    if duplicate_scenarios:
        duplicates_sorted = ", ".join(sorted(set(duplicate_scenarios)))
        raise SystemExit(
            f"Duplicate scenario names detected: {duplicates_sorted}. "
            "Ensure each scenario has a unique name."
        )

    for config in configs:
        if not config.root_dir.exists():
            _log(f"Warning: metrics root directory does not exist: {config.root_dir}")
        if config.output_dir and not config.output_dir.exists():
            _log(f"Creating output directory at {config.output_dir}")
            config.output_dir.mkdir(parents=True, exist_ok=True)
        _log(f"Initialising benchmark report for scenario '{config.scenario}'")
        report = BenchmarkReport(config=config)
        report.stats_settings = StatsSettings.from_config(_statistics_config(outputs_cfg))
        for message in getattr(report, "aggregate_metric_messages", []):
            _log(message)
        for warning in getattr(report, "aggregate_metric_warnings", []):
            _log(warning)
        for warning in report.load_warnings:
            _log(f"Warning: {warning}")
        # Drop assets of earlier runs: the LaTeX drivers include whatever they find.
        if _is_enabled(outputs_cfg.get("clean"), True):
            for stale_dir in (report.output_dir_tables(), report.output_dir_figures()):
                if stale_dir.is_dir():
                    _log(f"Removing previous outputs in {stale_dir}")
                    shutil.rmtree(stale_dir)

        _run_exports(report, outputs_cfg)
        _run_plots(report, outputs_cfg, base_dir)
        _generate_tex_reports(report, outputs_cfg)
    _log("Benchmark report generation completed")


if __name__ == "__main__":
    main(sys.argv[1:])
