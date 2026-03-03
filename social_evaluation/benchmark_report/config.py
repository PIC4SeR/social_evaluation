"""Configuration helpers for benchmark reporting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


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
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, bool]]:
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
    arrow_map = {"UP": r" $\mathbf{\uparrow}$", "DOWN": r" $\mathbf{\downarrow}$"}

    for group_key, metrics_spec in metric_groups.items():
        formatted[group_key] = {}
        for metric_key, spec in metrics_spec.items():
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
    return formatted, directions, discard_overrides


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
