"""Configuration helpers for benchmark reporting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class BenchmarkConfig:
    """Container for benchmark configuration values."""

    root_dir: Path
    scenario: str
    navigators: List[str]
    episodes: List[str]
    display_episodes: List[str]
    navigator_names: Dict[str, str]
    episode_names: Dict[str, str]
    metric_groups: Dict[str, Dict[str, str]]
    metric_directions: Dict[str, str]
    metric_captions: Dict[str, str]
    intrusion_labels: Dict[str, str]
    episode_groups: Dict[str, List[str]] = field(default_factory=dict)
    success_rate_n_cols: int = 3
    alpha: float = 0.8
    output_dir: Optional[Path] = None
    use_tex: bool = True
    discard_failed_metrics: bool = True
    quality_result_cmap: str = "Paired"
    metric_discard_failed: Dict[str, bool] = field(default_factory=dict)
    metric_group_aggregates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_group_plot_metrics: Dict[str, List[str]] = field(default_factory=dict)
    """Per-metric override for *discard_failed_metrics*.

    Keys are metric column names (e.g. ``path_length``).  When a metric
    is present here its value takes priority over the global
    ``discard_failed_metrics`` flag.
    """


def resolve_path(
    path_value: Union[str, os.PathLike, None], base: Path
) -> Optional[Path]:
    """Resolve ``path_value`` relative to ``base`` when needed."""

    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def format_metric_groups(
    metric_groups: Dict[str, Dict[str, Dict[str, str]]],
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, bool],
    Dict[str, Dict[str, Any]],
    Dict[str, List[str]],
]:
    """Normalise metric labels and extract metric directions.

    Returns
    -------
    formatted
        ``{group_key: {metric_key: display_label}}``
    directions
        ``{metric_key: "UP" | "DOWN"}``
    discard_overrides
        ``{metric_key: bool}`` — per-metric *discard_failed* overrides,
        only for metrics where the flag was explicitly set.
    """

    formatted: Dict[str, Dict[str, str]] = {}
    directions: Dict[str, str] = {}
    discard_overrides: Dict[str, bool] = {}
    aggregate_specs: Dict[str, Dict[str, Any]] = {}
    plot_metric_keys: Dict[str, List[str]] = {}
    arrow_map = {"UP": r" $\mathbf{\uparrow}$", "DOWN": r" $\mathbf{\downarrow}$"}

    def _humanize_group_key(group_key: str) -> str:
        text = group_key.strip()
        if text.startswith("metrics_"):
            text = text[len("metrics_") :]
        text = text.replace("_", " ").strip()
        return text.title() if text else group_key

    reserved_group_keys = {"aggregate_metric", "plot_metrics"}

    for group_key, metrics_spec in metric_groups.items():
        formatted[group_key] = {}
        aggregate_spec_raw: Optional[Dict[str, Any]] = None
        group_plot_metrics_raw: Optional[Any] = None
        for metric_key, spec in metrics_spec.items():
            if metric_key == "aggregate_metric":
                if isinstance(spec, dict):
                    aggregate_spec_raw = spec
                continue
            if metric_key == "plot_metrics":
                group_plot_metrics_raw = spec
                continue
            direction_key = spec.get("direction", "UP").upper()
            name = spec.get("name", metric_key)
            label = f"{name}{arrow_map.get(direction_key, '')}"
            label = re.sub(r"_(?!\{)", " ", label)
            label = label.replace("\\\\", "\\")
            formatted[group_key][metric_key] = label
            directions[metric_key] = direction_key
            if "discard_failed" in spec:
                raw = spec["discard_failed"]
                if isinstance(raw, str):
                    discard_overrides[metric_key] = raw.lower() in (
                        "true",
                        "1",
                        "yes",
                    )
                else:
                    discard_overrides[metric_key] = bool(raw)
        if not aggregate_spec_raw:
            if isinstance(group_plot_metrics_raw, (list, tuple)):
                plot_metric_keys[group_key] = [
                    str(metric_key)
                    for metric_key in group_plot_metrics_raw
                    if str(metric_key) in formatted[group_key]
                ]
            continue
        if bool(aggregate_spec_raw.get("enabled", False)):
            source_metrics_raw = aggregate_spec_raw.get("metrics")
            if source_metrics_raw is None:
                source_metrics = list(formatted[group_key].keys())
            else:
                source_metrics = [
                    str(metric_key)
                    for metric_key in source_metrics_raw
                    if str(metric_key) in formatted[group_key]
                    and str(metric_key) not in reserved_group_keys
                ]
            if source_metrics:
                aggregate_key = str(
                    aggregate_spec_raw.get("key", f"{group_key}_aggregate")
                )
                direction_key = str(aggregate_spec_raw.get("direction", "UP")).upper()
                name = str(
                    aggregate_spec_raw.get(
                        "name", f"\\textbf{{{_humanize_group_key(group_key)} Aggregate}}"
                    )
                )
                aggregate_label = f"{name}{arrow_map.get(direction_key, '')}"
                aggregate_label = re.sub(r"_(?!\{)", " ", aggregate_label)
                aggregate_label = aggregate_label.replace("\\\\", "\\")
                formatted[group_key][aggregate_key] = aggregate_label
                directions[aggregate_key] = direction_key
                weights_raw = aggregate_spec_raw.get("weights", {})
                weights: Dict[str, float] = {}
                if isinstance(weights_raw, dict):
                    for metric_key in source_metrics:
                        raw = weights_raw.get(metric_key)
                        if raw is None:
                            continue
                        try:
                            weights[metric_key] = float(raw)
                        except (TypeError, ValueError):
                            continue
                limits_raw = aggregate_spec_raw.get("limits")
                exclude_metrics_raw = aggregate_spec_raw.get(
                    "exclude_metrics_by_experiment", {}
                )
                exclude_metrics_by_experiment: Dict[str, List[str]] = {}
                if isinstance(exclude_metrics_raw, dict):
                    exclude_metrics_by_experiment = {
                        str(experiment_key): [
                            str(metric_key)
                            for metric_key in raw_metric_keys
                            if str(metric_key) in source_metrics
                        ]
                        for experiment_key, raw_metric_keys in exclude_metrics_raw.items()
                    }
                aggregate_specs[aggregate_key] = {
                    "group_key": group_key,
                    "metrics": source_metrics,
                    "weights": weights,
                    "direction": direction_key,
                    "normalize": str(
                        aggregate_spec_raw.get("normalize", "minmax")
                    ).lower(),
                    "require_complete": bool(
                        aggregate_spec_raw.get("require_complete", True)
                    ),
                    "limits": limits_raw,
                    "exclude_metrics_by_experiment": exclude_metrics_by_experiment,
                    "plot_only": bool(aggregate_spec_raw.get("plot_only", False)),
                }
                if aggregate_specs[aggregate_key]["plot_only"]:
                    plot_metric_keys[group_key] = [aggregate_key]
        if isinstance(group_plot_metrics_raw, (list, tuple)):
            selected_metric_keys = [
                str(metric_key)
                for metric_key in group_plot_metrics_raw
                if str(metric_key) in formatted[group_key]
            ]
            if selected_metric_keys:
                plot_metric_keys[group_key] = selected_metric_keys
    return formatted, directions, discard_overrides, aggregate_specs, plot_metric_keys


def format_intrusion_labels(
    intrusion_spec: Dict[str, Union[str, Dict[str, str]]],
) -> Tuple[Dict[str, str], Dict[str, bool]]:
    """Normalise intrusion labels supporting both simple and extended formats.

    Simple format::

        intimate_space_intrusions: Intimate Space Intrusions

    Extended format::

        intimate_space_intrusions:
            name: Intimate Space Intrusions
            discard_failed: false

    Returns
    -------
    labels
        ``{column_key: display_label}``
    discard_overrides
        ``{column_key: bool}`` — only for entries where ``discard_failed``
        was explicitly set.
    """

    labels: Dict[str, str] = {}
    discard_overrides: Dict[str, bool] = {}
    for col_key, spec in intrusion_spec.items():
        if isinstance(spec, dict):
            labels[col_key] = str(spec.get("name", col_key))
            if "discard_failed" in spec:
                raw = spec["discard_failed"]
                if isinstance(raw, str):
                    discard_overrides[col_key] = raw.lower() in (
                        "true",
                        "1",
                        "yes",
                    )
                else:
                    discard_overrides[col_key] = bool(raw)
        else:
            labels[col_key] = str(spec)
    return labels, discard_overrides


__all__ = [
    "BenchmarkConfig",
    "format_intrusion_labels",
    "format_metric_groups",
    "resolve_path",
]
