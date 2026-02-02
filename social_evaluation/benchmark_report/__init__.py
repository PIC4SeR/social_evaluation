"""Benchmark report package (refactored).

This package contains smaller, focused modules used by the top-level shim
`benchmark_report.py` so existing imports remain backward compatible.
"""

from .anova import AnovaMixin
from .config import BenchmarkConfig, format_metric_groups, resolve_path
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
    "format_metric_groups",
    "resolve_path",
    "generate_reports_for_root",
    "generate_reports_for_scenario",
]
