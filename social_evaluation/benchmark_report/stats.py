"""Statistics for benchmark reports, one test per episode group and metric.

Omnibus tests (ANOVA, Welch, Kruskal-Wallis), post-hoc comparisons and a permutation test of
the success rates, from statsmodels and scikit-posthocs (scipy comes with both). Every consumer
(CSV/LaTeX export, box-plot stars, heatmaps) reads the same ``StatsSettings``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from .config import BenchmarkConfig

OMNIBUS_TESTS = ("auto", "anova", "welch", "kruskal", "permutation")
# Every post-hoc returns the square matrix of pairwise p-values.
POSTHOC_TESTS = {
    "tukey": sp.posthoc_tukey,  # family-wise adjusted by construction
    "tamhane": sp.posthoc_tamhane,  # unequal variances, adjusted by construction
    "dunn": sp.posthoc_dunn,
    "conover": sp.posthoc_conover,
    "mannwhitney": sp.posthoc_mannwhitney,
    "ttest": sp.posthoc_ttest,
}
SELF_ADJUSTED = {"tukey", "tamhane"}
DEFAULT_POSTHOC = {"anova": "tukey", "welch": "tamhane", "kruskal": "dunn", "permutation": "dunn"}


@dataclass
class StatsSettings:
    """What the report tests, from the ``outputs.statistics`` config block."""

    test: str = "auto"  # auto: Levene's test picks welch or anova
    posthoc: Optional[str] = None  # default follows the test (DEFAULT_POSTHOC)
    p_adjust: str = "holm"  # pairwise correction (statsmodels method name)
    correct_omnibus: str = "holm"  # across episode groups x metrics; "none" skips it
    alpha: float = 0.05
    min_group_size: int = 2
    metrics: Optional[List[str]] = None  # default: every configured metric
    pairs: Optional[List[Tuple[str, str]]] = None  # pairs listed in the LaTeX table
    success_resamples: int = 9999  # 0 skips the success-rate test
    permutations: int = 9999  # label shuffles for test: permutation
    seed: int = 0

    @classmethod
    def from_config(cls, cfg: Any) -> "StatsSettings":
        if not isinstance(cfg, Mapping):
            return cls()
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in cfg.items() if key in known}
        if values.get("pairs"):
            values["pairs"] = [(str(a), str(b)) for a, b in values["pairs"]]
        success = cfg.get("success_test", True)
        if isinstance(success, Mapping):
            values["success_resamples"] = int(success.get("n_resamples", cls.success_resamples))
            values["seed"] = int(success.get("seed", cls.seed))
        elif not success:
            values["success_resamples"] = 0
        settings = cls(**values)
        if settings.test not in OMNIBUS_TESTS:
            raise ValueError(f"statistics.test must be one of {OMNIBUS_TESTS}, not {settings.test!r}")
        if settings.posthoc is not None and settings.posthoc not in POSTHOC_TESTS:
            raise ValueError(f"statistics.posthoc must be one of {sorted(POSTHOC_TESTS)}, not {settings.posthoc!r}")
        return settings


def omnibus(samples: Sequence[np.ndarray], test: str, alpha: float,
            permutations: int = 9999, seed: int = 0) -> Dict[str, Any]:
    """One omnibus test over ``samples`` (one array per navigator); ValueError if untestable.

    ``permutation`` is Kruskal-Wallis with its p-value from shuffling the navigator labels
    instead of the chi-square approximation (rough with ~5 runs per navigator)."""
    if test == "auto":
        try:
            heteroscedastic = stats.levene(*samples, center="median").pvalue <= alpha
        except ValueError:
            heteroscedastic = False
        test = "welch" if heteroscedastic else "anova"
    n, k = sum(len(s) for s in samples), len(samples)
    if test in ("kruskal", "permutation"):
        h, p = stats.kruskal(*samples)
        if test == "permutation":
            p = stats.permutation_test(
                samples, lambda *s, axis: stats.kruskal(*s, axis=axis).statistic,
                vectorized=True, n_resamples=permutations, alternative="greater",
                random_state=seed).pvalue
        return dict(test=test, statistic=float(h), df1=k - 1, df2=np.nan, p_value=float(p),
                    effect_size=(h - k + 1) / (n - k), effect_name="eta2_H", n=n, k=k)
    result = anova_oneway(samples, use_var="unequal" if test == "welch" else "equal",
                          welch_correction=True)
    f, df1, df2 = float(result.statistic), float(result.df_num), float(result.df_denom)
    return dict(test=test, statistic=f, df1=df1, df2=df2, p_value=float(result.pvalue),
                effect_size=f * df1 / (f * df1 + df2), effect_name="eta2", n=n, k=k)


def posthoc(data: pd.DataFrame, name: str, p_adjust: str) -> pd.DataFrame:
    """Matrix of pairwise p-values for ``data`` (columns ``algorithm`` and ``value``)."""
    kwargs = {} if name in SELF_ADJUSTED else {"p_adjust": p_adjust}
    return POSTHOC_TESTS[name](data, val_col="value", group_col="algorithm", **kwargs)


def adjust(p_values: pd.Series, method: str) -> pd.Series:
    """Correct the finite p-values together ("none" returns them unchanged)."""
    adjusted = p_values.astype(float).copy()
    finite = adjusted.notna()
    if method != "none" and finite.any():
        adjusted[finite] = multipletests(adjusted[finite], method=method)[1]
    return adjusted


def success_permutation_test(algorithms: Sequence[str], success: Sequence[bool],
                             resamples: int, seed: int) -> Tuple[float, float]:
    """Permutation test of independence of the navigator x success table (chi-square)."""
    codes, labels = pd.factorize(np.asarray(algorithms))
    success = np.asarray(success, dtype=int)
    if len(labels) < 2 or np.unique(success).size < 2:
        return 0.0, 1.0  # one navigator, or every run with the same outcome

    def chi2(s: np.ndarray) -> float:
        table = np.bincount(codes * 2 + s, minlength=2 * len(labels)).reshape(-1, 2)
        return stats.chi2_contingency(table, correction=False)[0]

    result = stats.permutation_test((success,), chi2, permutation_type="pairings",
                                    n_resamples=resamples, alternative="greater",
                                    random_state=seed)
    return float(result.statistic), float(result.pvalue)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Hedges' g between two samples (bias-corrected pooled-SD effect size)."""
    df = a.size + b.size - 2
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / df
    if pooled <= 0:
        return 0.0
    return float((a.mean() - b.mean()) / np.sqrt(pooled) * (1.0 - 3.0 / (4.0 * df - 1.0)))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "all"


HEATMAPS = {  # value column -> (title, colour-bar label, default cmap, file suffix)
    "p_adjusted": ("Pairwise p-values", r"Adjusted $p$-value", "mako", "pairwise_pvalues"),
    "mean_diff": ("Mean differences", "Mean difference", "coolwarm", "mean_diff"),
    "hedges_g": ("Hedge's g", "Hedge's g", "coolwarm", "hedges_g"),
}


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    for limit, mark in ((1e-4, "****"), (1e-3, "***"), (1e-2, "**"), (0.05, "*")):
        if p <= limit:
            return mark
    return "ns"


class StatsMixin:
    """Per-episode-group statistics for the benchmark report."""

    stats_settings: StatsSettings = StatsSettings()

    if TYPE_CHECKING:  # pragma: no cover - provided by BenchmarkReportBase
        config: BenchmarkConfig
        ordered_navigators: List[str]
        ordered_experiments: List[str]
        _all_results: pd.DataFrame

        def results_for_metric(self, metric_key: str) -> pd.DataFrame: ...
        def metric_plain_label(self, metric_key: str) -> str: ...
        def output_dir_tables(self) -> Path: ...
        def output_dir_figures(self) -> Path: ...
        def _format_text(self, text: str) -> str: ...
        def _experiment_order_with_groups(self, available: Any = None) -> List[str]: ...

    def environment_frames(self, source: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Rows of ``source`` per displayed experiment / episode group, in display order."""
        frames = {e: source[source["experiment"] == e] for e in self.ordered_experiments}
        for group, members in self.config.episode_groups.items():
            if members:
                frames[group] = source[source["experiment"].isin(members)]
        return {e: frames[e] for e in self._experiment_order_with_groups(frames)}

    def _settings(self, settings: Optional[StatsSettings], overrides: Mapping[str, Any]) -> StatsSettings:
        return replace(settings or self.stats_settings,
                       **{k: v for k, v in overrides.items() if v is not None})

    def analyze_algorithm_effects(
        self, metric_keys: Optional[Sequence[str]] = None, *,
        settings: Optional[StatsSettings] = None, **overrides: Any,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Omnibus test per (episode group, metric), then post-hocs where it is significant.

        Returns ``(summary, pairwise)``: ``summary`` indexed by (episode_group, metric_key) with
        test, statistic, df1, df2, p_value, p_adjusted (across every test, ``correct_omnibus``),
        effect_size/effect_name, n, k, significant and a ``note`` for untestable cells;
        ``pairwise`` indexed by (episode_group, metric_key, group_a, group_b) with posthoc,
        p_adjusted, reject, mean_diff, hedges_g, median_a and median_b.
        """
        s = self._settings(settings, overrides)
        keys = list(metric_keys or s.metrics or self.config.metric_directions)
        rows, samples = [], {}
        for metric_key in keys:
            source = self.results_for_metric(metric_key)
            if metric_key not in source:
                continue
            for env, frame in self.environment_frames(source).items():
                data = frame[["algorithm", metric_key]].rename(columns={metric_key: "value"})
                data = data.assign(value=pd.to_numeric(data["value"], errors="coerce")).dropna()
                counts = data["algorithm"].value_counts()
                navigators = [n for n in self.ordered_navigators if counts.get(n, 0) >= s.min_group_size]
                data = data[data["algorithm"].isin(navigators)]
                groups = {n: data.loc[data["algorithm"] == n, "value"].to_numpy(float) for n in navigators}
                row = dict(episode_group=env, metric_key=metric_key,
                           metric_label=self.metric_plain_label(metric_key), n=len(data), k=len(groups))
                try:
                    if len(groups) < 2:
                        raise ValueError(f"fewer than 2 navigators with {s.min_group_size}+ runs")
                    row.update(omnibus(list(groups.values()), s.test, s.alpha, s.permutations, s.seed))
                except ValueError as exc:
                    row["note"] = str(exc)
                rows.append(row)
                samples[env, metric_key] = (data, groups)

        summary = pd.DataFrame(rows).reindex(columns=[
            "episode_group", "metric_key", "metric_label", "test", "statistic", "df1", "df2",
            "p_value", "effect_size", "effect_name", "n", "k", "note"])
        summary["p_adjusted"] = adjust(summary["p_value"], s.correct_omnibus)
        summary["significant"] = summary["p_adjusted"] <= s.alpha

        pair_rows = []
        for row in summary[summary["significant"]].itertuples():
            data, groups = samples[row.episode_group, row.metric_key]
            name = s.posthoc or DEFAULT_POSTHOC[row.test]
            matrix = posthoc(data, name, s.p_adjust)
            for a, b in combinations(groups, 2):
                p = float(matrix.loc[a, b])
                pair_rows.append(dict(
                    episode_group=row.episode_group, metric_key=row.metric_key,
                    metric_label=row.metric_label, group_a=a, group_b=b, posthoc=name,
                    p_adjusted=p, reject=p <= s.alpha,
                    mean_diff=groups[a].mean() - groups[b].mean(),
                    hedges_g=hedges_g(groups[a], groups[b]),
                    median_a=np.median(groups[a]), median_b=np.median(groups[b])))
        pairwise = pd.DataFrame(pair_rows)
        if not pairwise.empty:
            pairwise = pairwise.set_index(["episode_group", "metric_key", "group_a", "group_b"])
        return summary.set_index(["episode_group", "metric_key"]), pairwise

    def analyze_success_rates(self, settings: Optional[StatsSettings] = None) -> pd.DataFrame:
        """Permutation chi-square test of navigator x success per episode group."""
        s = settings or self.stats_settings
        rows = []
        for env, frame in self.environment_frames(self._all_results).items():
            frame = frame[frame["algorithm"].isin(self.ordered_navigators)]
            success = frame["result"] == "Success"
            chi2, p = success_permutation_test(frame["algorithm"], success, s.success_resamples, s.seed)
            rows.append(dict(episode_group=env, n=len(frame), successes=int(success.sum()),
                             chi2=chi2, p_value=p))
        result = pd.DataFrame(rows)
        result["p_adjusted"] = adjust(result["p_value"], s.correct_omnibus)
        return result.set_index("episode_group")

    def export_statistics(self, output_dir: Optional[Path] = None, *,
                          settings: Optional[StatsSettings] = None, **overrides: Any) -> Path:
        """Write stats_summary/pairwise/success/runs CSVs; returns their directory."""
        s = self._settings(settings, overrides)
        target = Path(output_dir) if output_dir else self.output_dir_tables() / "stats"
        target.mkdir(parents=True, exist_ok=True)
        summary, pairwise = self.analyze_algorithm_effects(settings=s)
        summary.to_csv(target / "stats_summary.csv")
        if not pairwise.empty:
            pairs = {(self.config.navigator_names.get(a, a), self.config.navigator_names.get(b, b))
                     for a, b in (s.pairs or [])}
            ab = list(zip(pairwise.index.get_level_values("group_a"),
                          pairwise.index.get_level_values("group_b")))
            # the LaTeX table lists the configured pairs, else the significant ones
            pairwise["listed"] = ([(a, b) in pairs or (b, a) in pairs for a, b in ab]
                                  if pairs else pairwise["reject"].tolist())
        pairwise.to_csv(target / "stats_pairwise.csv")
        if s.success_resamples > 0:
            self.analyze_success_rates(s).to_csv(target / "stats_success.csv")
        metrics = [m for m in (s.metrics or self.config.metric_directions) if m in self._all_results]
        runs = pd.concat({env: frame[["experiment", "algorithm", "run", "result", *metrics]]
                          for env, frame in self.environment_frames(self._all_results).items()},
                         names=["episode_group"]).reset_index(level=0)
        runs.to_csv(target / "stats_runs.csv", index=False)
        return target

    def plot_pairwise_heatmaps(
        self, value: str = "p_adjusted", metric_keys: Optional[Sequence[str]] = None, *,
        cmap: Optional[str] = None, annotate: bool = True, fmt: str = ".2f",
        label_size: int = 12, title_size: int = 14, colorbar_label_size: int = 12,
        annot_font_size: int = 9, output_dir: Optional[Path] = None,
        settings: Optional[StatsSettings] = None, **overrides: Any,
    ) -> Dict[Tuple[str, str], Path]:
        """One heatmap of ``value`` (p_adjusted, mean_diff or hedges_g) per episode group and
        metric with significant post-hocs, saved as <dir>/<group>/<metric>_<suffix>.png."""
        title, cbar_label, default_cmap, suffix = HEATMAPS[value]
        _, pairwise = self.analyze_algorithm_effects(metric_keys, settings=settings, **overrides)
        if pairwise.empty:
            raise ValueError("No pairwise comparisons available for the requested metrics.")
        base_dir = Path(output_dir) if output_dir else self.output_dir_figures() / "stats"
        saved: Dict[Tuple[str, str], Path] = {}
        for (env, metric_key), pairs in pairwise.reset_index().groupby(["episode_group", "metric_key"], sort=False):
            names = set(pairs["group_a"]) | set(pairs["group_b"])
            order = [n for n in self.ordered_navigators if n in names]
            matrix = pd.DataFrame(np.nan, index=order, columns=order)
            for r in pairs.itertuples():
                v = getattr(r, value)
                # mean_diff/hedges_g are a - b: the mirrored cell is its opposite
                matrix.at[r.group_a, r.group_b] = v
                matrix.at[r.group_b, r.group_a] = v if value == "p_adjusted" else -v
            if value == "p_adjusted":
                colours = matrix.map(lambda p: np.nan if pd.isna(p) else
                                     next((c for lim, c in ((1e-4, 1), (1e-3, .75), (1e-2, .5), (.05, .25)) if p <= lim), 0.0))
                annot = matrix.map(_stars).to_numpy(object) if annotate else False
                kwargs = dict(vmin=0.0, vmax=1.0, fmt="")
            else:
                np.fill_diagonal(matrix.values, 0.0)
                colours = matrix
                vmax = float(np.nanmax(np.abs(matrix.to_numpy()))) or 1.0
                annot = matrix.to_numpy() if annotate else False
                kwargs = dict(vmin=-vmax, vmax=vmax, center=0.0, fmt=fmt)
            size = max(6.0, 0.6 * len(order))
            fig, ax = plt.subplots(figsize=(size, size))
            sns.heatmap(colours, mask=matrix.isna(), cmap=cmap or default_cmap, annot=annot,
                        annot_kws={"size": annot_font_size}, square=True, linewidths=0.5,
                        linecolor="white", ax=ax, **kwargs)
            ticks = [self._format_text(n) for n in order]
            ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=label_size)
            ax.set_yticklabels(ticks, rotation=0, fontsize=label_size)
            label = pairs["metric_label"].iloc[0]
            ax.set_title(self._format_text(f"{env}: {label} {title}"), pad=12, fontsize=title_size)
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=colorbar_label_size)
            cbar.set_label(self._format_text(cbar_label), size=colorbar_label_size)
            path = base_dir / _slug(env) / f"{metric_key}_{suffix}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
            saved[env, metric_key] = path
        return saved


__all__ = ["StatsMixin", "StatsSettings", "omnibus", "posthoc", "adjust",
           "success_permutation_test", "hedges_g", "POSTHOC_TESTS", "DEFAULT_POSTHOC"]
