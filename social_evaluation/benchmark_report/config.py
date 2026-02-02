"""Configuration helpers for benchmark reporting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class BenchmarkConfig:
    """Container for benchmark configuration values."""

    root_dir: Path
    scenario: str
    navigators: List[str]
    episodes: List[str]
    navigator_names: Dict[str, str]
    episode_names: Dict[str, str]
    metric_groups: Dict[str, Dict[str, str]]
    metric_directions: Dict[str, str]
    metric_captions: Dict[str, str]
    intrusion_labels: Dict[str, str]
    alpha: float = 0.8
    output_dir: Optional[Path] = None
    use_tex: bool = True


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
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
    """Normalise metric labels and extract metric directions."""

    formatted: Dict[str, Dict[str, str]] = {}
    directions: Dict[str, str] = {}
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
    return formatted, directions


__all__ = [
    "BenchmarkConfig",
    "format_metric_groups",
    "resolve_path",
]
