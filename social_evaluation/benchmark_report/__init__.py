"""Benchmark report package (refactored).

This package contains smaller, focused modules used by the top-level shim
`benchmark_report.py` so existing imports remain backward compatible.
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib")
)

import matplotlib

matplotlib.use("Agg")

from .anova import AnovaMixin
from .config import BenchmarkConfig, format_intrusion_labels, format_metric_groups, resolve_path
from .base import BenchmarkReportBase as BenchmarkReport
from .plots import PlottingMixin
from .rliable import RliableMixin
from .tables import TableExportMixin
from .tex_reports import generate_reports_for_root, generate_reports_for_scenario

__all__ = [
    "BenchmarkConfig",
    "BenchmarkReport",
    "AnovaMixin",
    "TableExportMixin",
    "PlottingMixin",
    "RliableMixin",
    "format_intrusion_labels",
    "format_metric_groups",
    "resolve_path",
    "generate_reports_for_root",
    "generate_reports_for_scenario",
]
