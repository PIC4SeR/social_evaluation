"""Statistics for benchmark reports, one test per episode group and metric.

Omnibus tests, post-hoc comparisons and a permutation test of the success rates, from
statsmodels and scikit-posthocs (scipy comes with both). Every consumer (CSV/LaTeX export,
box-plot stars, heatmaps) reads the same ``StatsSettings``.

Independent-sample tests (ANOVA, Welch, Kruskal-Wallis, permutation) treat every run as its own
observation. Repeated-measures tests (Friedman, Cochran's Q) and models (LMM, GEE) match runs of
different navigators through the ``block`` column (run id / seed, or subject), with an optional
``cohort`` column the blocks are nested in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from .config import BenchmarkConfig

INDEPENDENT_TESTS = ("auto", "anova", "welch", "kruskal", "permutation")
PAIRED_TESTS = ("friedman", "cochran")  # one value per block and navigator, complete blocks
MODEL_TESTS = ("lmm", "gee_binomial", "gee_poisson")  # random/clustered effect on the block
OMNIBUS_TESTS = INDEPENDENT_TESTS + PAIRED_TESTS + MODEL_TESTS
BINARY_TESTS = ("cochran", "gee_binomial")
# Every post-hoc returns the square matrix of pairwise p-values.
POSTHOC_TESTS = {
    "tukey": sp.posthoc_tukey,  # family-wise adjusted by construction
    "tamhane": sp.posthoc_tamhane,  # unequal variances, adjusted by construction
    "dunn": sp.posthoc_dunn,
    "conover": sp.posthoc_conover,
    "mannwhitney": sp.posthoc_mannwhitney,
    "ttest": sp.posthoc_ttest,
}
PAIRED_POSTHOCS = ("wilcoxon", "mcnemar")  # on the matched blocks, corrected with p_adjust
MODEL_POSTHOCS = ("emm",)  # Tukey-adjusted contrasts of the model's marginal means
ALL_POSTHOCS = (*POSTHOC_TESTS, *PAIRED_POSTHOCS, *MODEL_POSTHOCS)
SELF_ADJUSTED = {"tukey", "tamhane", "emm"}
DEFAULT_POSTHOC = {
    "anova": "tukey", "welch": "tamhane", "kruskal": "dunn", "permutation": "dunn",
    "friedman": "wilcoxon", "wilcoxon": "wilcoxon", "cochran": "mcnemar", "mcnemar": "mcnemar",
    "lmm": "emm", "gee_binomial": "emm", "gee_poisson": "emm",
}
TRANSFORMS = {"log": np.log, "log1p": np.log1p}
# Run outcomes (from the result column) that can be tested as binary metrics.
OUTCOME_METRICS = ("success", "timeout", "collision", "navigation_failure")
# Overridable per metric (statistics.per_metric) and per episode group (statistics.per_group).
PER_METRIC_KEYS = ("test", "posthoc", "p_adjust", "transform", "block", "cohort", "min_group_size")


def _check_override_keys(where: str, overrides: Mapping[str, Any], allowed: Sequence[str]) -> None:
    unknown = set(overrides) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed: {tuple(allowed)}")


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
    block: Optional[str] = "run"  # column matching runs across navigators (paired tests, models)
    cohort: Optional[str] = None  # column the blocks are nested in (models only)
    transform: Optional[str] = None  # log | log1p, applied before testing
    per_metric: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # episode / episode-group display label -> overrides, plus an optional nested per_metric
    per_group: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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
        per_metric = values.get("per_metric") or {}
        for metric_key, overrides in per_metric.items():
            _check_override_keys(f"statistics.per_metric.{metric_key}", overrides, PER_METRIC_KEYS)
        values["per_metric"] = {str(k): dict(v) for k, v in per_metric.items()}
        per_group = values.get("per_group") or {}
        for group, overrides in per_group.items():
            _check_override_keys(f"statistics.per_group.{group}", overrides,
                                 (*PER_METRIC_KEYS, "per_metric"))
            for metric_key, nested in (overrides.get("per_metric") or {}).items():
                _check_override_keys(f"statistics.per_group.{group}.per_metric.{metric_key}",
                                     nested, PER_METRIC_KEYS)
        values["per_group"] = {str(g): {**o, "per_metric": {str(k): dict(v) for k, v in
                                                            (o.get("per_metric") or {}).items()}}
                               for g, o in per_group.items()}
        settings = cls(**values)
        settings.validate("statistics")
        # every combination a cell can end up with
        for group in (None, *settings.per_group):
            nested = settings.per_group.get(group, {}).get("per_metric", {}) if group else {}
            for metric_key in (None, *settings.per_metric, *nested):
                where = "statistics" + (f".per_group.{group}" if group else "") + \
                        (f".per_metric.{metric_key}" if metric_key else "")
                settings.for_cell(group, metric_key).validate(where)
        return settings

    def for_cell(self, group: Optional[str], metric_key: Optional[str]) -> "StatsSettings":
        """The settings of one (episode group, metric) cell. Most specific last:
        base, per_group[group], per_metric[metric], per_group[group].per_metric[metric]."""
        group_overrides = dict(self.per_group.get(group, {})) if group is not None else {}
        nested = group_overrides.pop("per_metric", {}) or {}
        s = replace(self, **group_overrides)
        if metric_key is not None:
            s = replace(s, **self.per_metric.get(metric_key, {}))
            s = replace(s, **nested.get(metric_key, {}))
        return s

    def validate(self, where: str) -> None:
        if self.test not in OMNIBUS_TESTS:
            raise ValueError(f"{where}.test must be one of {OMNIBUS_TESTS}, not {self.test!r}")
        if self.posthoc is not None and self.posthoc not in ALL_POSTHOCS:
            raise ValueError(f"{where}.posthoc must be one of {ALL_POSTHOCS}, not {self.posthoc!r}")
        if self.posthoc == "emm" and self.test not in MODEL_TESTS:
            raise ValueError(f"{where}.posthoc: emm needs a model test {MODEL_TESTS}, not {self.test!r}")
        if self.transform is not None and self.transform not in TRANSFORMS:
            raise ValueError(f"{where}.transform must be one of {sorted(TRANSFORMS)}, not {self.transform!r}")
        paired = self.test in PAIRED_TESTS + MODEL_TESTS or self.posthoc in PAIRED_POSTHOCS
        if paired and not self.block:
            raise ValueError(f"{where}: test {self.test!r} / posthoc {self.posthoc!r} needs a block column")


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


def _check_values(values: pd.Series, test: str) -> None:
    if test in BINARY_TESTS and not values.isin([0, 1]).all():
        raise ValueError(f"{test} needs 0/1 values")
    if test == "gee_poisson" and (values < 0).any():
        raise ValueError("gee_poisson needs non-negative counts")


def blocks_wide(data: pd.DataFrame, navigators: Sequence[str], binary: bool) -> pd.DataFrame:
    """Blocks x navigators table of ``value`` keeping complete blocks; ValueError if too few.

    Several runs of one navigator in a block (e.g. repetitions per subject) are averaged, which
    a binary test cannot do, so there they are an error."""
    if "block" not in data:
        raise ValueError("block column missing from the metrics")
    if binary and data.duplicated(["block", "algorithm"]).any():
        raise ValueError("binary paired tests need one run per block and navigator")
    wide = data.pivot_table(index="block", columns="algorithm", values="value", aggfunc="mean")
    wide = wide.reindex(columns=list(navigators)).dropna()
    if len(wide) < 2:
        raise ValueError(f"{len(wide)} block(s) with a run of every navigator, need 2+")
    return wide


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if np.allclose(a, b):
        return 0.0, 1.0  # no difference in any block
    result = stats.wilcoxon(a, b)
    return float(result.statistic), float(result.pvalue)


def _mcnemar_p(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    table = [[np.sum((a == 1) & (b == 1)), np.sum((a == 1) & (b == 0))],
             [np.sum((a == 0) & (b == 1)), np.sum((a == 0) & (b == 0))]]
    result = mcnemar(table, exact=True)
    return float(result.statistic), float(result.pvalue)


def paired_omnibus(wide: pd.DataFrame, test: str) -> Dict[str, Any]:
    """Friedman or Cochran's Q over complete blocks (Wilcoxon / McNemar for 2 navigators)."""
    x = wide.to_numpy(float)
    n, k = x.shape
    base = dict(df2=np.nan, effect_size=np.nan, effect_name="", n=n * k, k=k, blocks=n)
    if k == 2:
        name, pair_test = ("wilcoxon", _wilcoxon_p) if test == "friedman" else ("mcnemar", _mcnemar_p)
        statistic, p = pair_test(x[:, 0], x[:, 1])
        return dict(base, test=name, statistic=statistic, df1=np.nan, p_value=p)
    if test == "friedman":
        if np.all(x == x[:, :1]):
            raise ValueError("no differences within any block")
        chi2, p = stats.friedmanchisquare(*x.T)
        return dict(base, test=test, statistic=float(chi2), df1=k - 1, p_value=float(p),
                    effect_size=chi2 / (n * (k - 1)), effect_name="kendall_W")
    result = cochrans_q(x)
    if not np.isfinite(result.statistic):
        raise ValueError("no differences within any block")
    return dict(base, test=test, statistic=float(result.statistic), df1=int(result.df),
                p_value=float(result.pvalue))


def fit_model(data: pd.DataFrame, test: str, navigators: Sequence[str]) -> Any:
    """LMM (random intercept per block, nested in cohort when given) or GEE clustered on the
    block (or cohort) with exchangeable correlation; episodes enter as a fixed effect."""
    df = data.copy()
    codes = {name: f"L{i}" for i, name in enumerate(navigators)}  # patsy-safe level names
    df["algorithm"] = pd.Categorical(df["algorithm"].map(codes), categories=list(codes.values()))
    formula = "value ~ C(algorithm)" + (" + C(experiment)" if df["experiment"].nunique() > 1 else "")
    nested = "cohort" in df
    if test == "lmm":
        if nested:
            model = smf.mixedlm(formula, df, groups=df["cohort"], re_formula="1",
                                vc_formula={"block": "0 + C(block)"})
        else:
            model = smf.mixedlm(formula, df, groups=df["block"])
        return model.fit(reml=True)
    family = sm.families.Binomial() if test == "gee_binomial" else sm.families.Poisson()
    return smf.gee(formula, groups=df["cohort" if nested else "block"], data=df, family=family,
                   cov_struct=sm.cov_struct.Exchangeable()).fit()


def _algorithm_terms(navigators: Sequence[str]) -> List[str]:
    return [f"C(algorithm)[T.L{i}]" for i in range(1, len(navigators))]


def model_omnibus(fit: Any, test: str, navigators: Sequence[str], n: int) -> Dict[str, Any]:
    """Wald chi-square test that every navigator coefficient is zero."""
    names = _algorithm_terms(navigators)
    beta = fit.params[names].to_numpy(float)
    cov = fit.cov_params().loc[names, names].to_numpy(float)
    chi2 = float(beta @ np.linalg.solve(cov, beta))
    return dict(test=test, statistic=chi2, df1=len(names), df2=np.nan,
                p_value=float(stats.chi2.sf(chi2, len(names))), effect_size=np.nan,
                effect_name="", n=n, k=len(navigators))


def emm_contrasts(fit: Any, navigators: Sequence[str]) -> pd.DataFrame:
    """Matrix of Tukey-adjusted p-values of the pairwise differences of the model's marginal
    means (on the link scale), from the studentized range with infinite degrees of freedom."""
    names = _algorithm_terms(navigators)
    params, cov = fit.params[names].to_numpy(float), fit.cov_params().loc[names, names].to_numpy(float)
    k = len(navigators)
    matrix = pd.DataFrame(1.0, index=list(navigators), columns=list(navigators))
    for i, j in combinations(range(k), 2):
        c = np.zeros(k - 1)
        if i:
            c[i - 1] += 1.0
        if j:
            c[j - 1] -= 1.0
        z = abs(c @ params) / np.sqrt(c @ cov @ c)
        p = float(stats.studentized_range.sf(z * np.sqrt(2.0), k, np.inf)) if k > 2 else float(2 * stats.norm.sf(z))
        matrix.iat[i, j] = matrix.iat[j, i] = p
    return matrix


def posthoc(data: pd.DataFrame, name: str, p_adjust: str, *, fit: Any = None,
            navigators: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Matrix of pairwise p-values for ``data`` (columns ``algorithm`` and ``value``, plus
    ``block`` for the paired post-hocs; ``fit`` is the model for ``emm``)."""
    navigators = list(navigators or pd.unique(data["algorithm"]))
    if name == "emm":
        return emm_contrasts(fit, navigators)
    if name in PAIRED_POSTHOCS:
        if name == "mcnemar" and not data["value"].isin([0, 1]).all():
            raise ValueError("posthoc mcnemar needs 0/1 values")
        wide = blocks_wide(data, navigators, binary=name == "mcnemar")
        pair_test = _wilcoxon_p if name == "wilcoxon" else _mcnemar_p
        pairs = list(combinations(navigators, 2))
        raw = pd.Series([pair_test(wide[a].to_numpy(), wide[b].to_numpy())[1] for a, b in pairs])
        matrix = pd.DataFrame(1.0, index=navigators, columns=navigators)
        for (a, b), p in zip(pairs, adjust(raw, p_adjust)):
            matrix.at[a, b] = matrix.at[b, a] = p
        return matrix
    kwargs = {} if name in SELF_ADJUSTED else {"p_adjust": p_adjust}
    return POSTHOC_TESTS[name](data, val_col="value", group_col="algorithm", **kwargs)


def stat_symbol(test: str) -> Tuple[str, bool]:
    """Symbol of a test's statistic and whether it is reported with (df1, df2)."""
    symbols = {"kruskal": "H", "permutation": "H", "friedman": "chi2", "cochran": "Q",
               "lmm": "Wald chi2", "gee_binomial": "Wald chi2", "gee_poisson": "Wald chi2",
               "wilcoxon": "W", "mcnemar": "M"}
    return symbols.get(test, "F"), test in ("anova", "welch")


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

    def _stats_source(self, metric_key: str) -> pd.DataFrame:
        """Rows to test ``metric_key`` on; run outcomes are 0/1 columns over every run."""
        if metric_key in OUTCOME_METRICS:
            from .utils import parse_experiment_result

            return parse_experiment_result(self._all_results)
        return self.results_for_metric(metric_key)

    @staticmethod
    def _test_data(frame: pd.DataFrame, metric_key: str, s: StatsSettings) -> pd.DataFrame:
        """Columns experiment, algorithm, value (transformed), raw, and block/cohort when set.

        Blocks are per episode (``<episode>/<block>``) so pooled episode groups never match runs
        of different episodes; cohorts are kept as they are (a subject cohort spans episodes)."""
        data = pd.DataFrame({
            "experiment": frame["experiment"].astype(str),
            "algorithm": frame["algorithm"],
            "raw": pd.to_numeric(frame[metric_key], errors="coerce").astype(float),
        })
        if s.block and s.block in frame:
            data["block"] = data["experiment"] + "/" + frame[s.block].astype(str)
        if s.cohort and s.cohort in frame:
            data["cohort"] = frame[s.cohort].astype(str)
        elif s.cohort:
            raise ValueError(f"cohort column {s.cohort!r} missing from the metrics")
        data = data.dropna(subset=["raw"])
        data["value"] = data["raw"]
        if s.transform:
            if s.transform == "log" and (data["raw"] <= 0).any():
                raise ValueError(f"log transform of {metric_key} needs positive values; use log1p")
            data["value"] = TRANSFORMS[s.transform](data["raw"])
        return data.reset_index(drop=True)

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
            source = self._stats_source(metric_key)
            if metric_key not in source:
                continue
            for env, frame in self.environment_frames(source).items():
                ms = s.for_cell(env, metric_key)
                try:
                    data = self._test_data(frame, metric_key, ms)
                except ValueError as exc:
                    rows.append(dict(episode_group=env, metric_key=metric_key, note=str(exc),
                                     metric_label=self.metric_plain_label(metric_key)))
                    continue
                counts = data["algorithm"].value_counts()
                navigators = [n for n in self.ordered_navigators if counts.get(n, 0) >= ms.min_group_size]
                data = data[data["algorithm"].isin(navigators)]
                groups = {n: data.loc[data["algorithm"] == n, "value"].to_numpy(float) for n in navigators}
                row = dict(episode_group=env, metric_key=metric_key,
                           metric_label=self.metric_plain_label(metric_key), n=len(data), k=len(groups))
                fit = None
                try:
                    if len(groups) < 2:
                        raise ValueError(f"fewer than 2 navigators with {ms.min_group_size}+ runs")
                    _check_values(data["value"], ms.test)
                    if ms.test in PAIRED_TESTS:
                        wide = blocks_wide(data, navigators, binary=ms.test in BINARY_TESTS)
                        row.update(paired_omnibus(wide, ms.test))
                    elif ms.test in MODEL_TESTS:
                        fit = fit_model(data, ms.test, navigators)
                        row.update(model_omnibus(fit, ms.test, navigators, len(data)))
                        if not getattr(fit, "converged", True):
                            row["note"] = "model did not converge; check the estimates"
                    else:
                        row.update(omnibus(list(groups.values()), ms.test, ms.alpha, ms.permutations, ms.seed))
                    row["transform"] = ms.transform or ""
                except (ValueError, np.linalg.LinAlgError) as exc:
                    row["note"] = str(exc)
                rows.append(row)
                samples[env, metric_key] = (data, groups, ms, fit)

        summary = pd.DataFrame(rows).reindex(columns=[
            "episode_group", "metric_key", "metric_label", "test", "statistic", "df1", "df2",
            "p_value", "effect_size", "effect_name", "n", "k", "blocks", "transform", "note"])
        summary["p_adjusted"] = adjust(summary["p_value"], s.correct_omnibus)
        summary["significant"] = summary["p_adjusted"] <= s.alpha

        pair_rows = []
        for row in summary[summary["significant"]].itertuples():
            data, groups, ms, fit = samples[row.episode_group, row.metric_key]
            name = ms.posthoc or DEFAULT_POSTHOC[row.test]
            if name == "emm" and fit is None:
                raise ValueError(f"posthoc emm needs a model test, not {row.test!r} ({row.metric_key})")
            matrix = posthoc(data, name, ms.p_adjust, fit=fit, navigators=list(groups))
            raw = {n: data.loc[data["algorithm"] == n, "raw"].to_numpy(float) for n in groups}
            for a, b in combinations(groups, 2):
                p = float(matrix.loc[a, b])
                pair_rows.append(dict(
                    episode_group=row.episode_group, metric_key=row.metric_key,
                    metric_label=row.metric_label, group_a=a, group_b=b, posthoc=name,
                    p_adjusted=p, reject=p <= s.alpha,
                    # descriptives on the metric's own scale, before any transform
                    mean_diff=raw[a].mean() - raw[b].mean(),
                    hedges_g=hedges_g(raw[a], raw[b]),
                    median_a=np.median(raw[a]), median_b=np.median(raw[b])))
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
        id_columns = list(dict.fromkeys(
            c for c in ("experiment", "algorithm", "run", s.block, s.cohort, "result")
            if c and c in self._all_results))
        runs = pd.concat({env: frame[[*id_columns, *metrics]]
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
           "success_permutation_test", "hedges_g", "POSTHOC_TESTS", "DEFAULT_POSTHOC",
           "paired_omnibus", "blocks_wide", "fit_model", "model_omnibus", "emm_contrasts",
           "stat_symbol"]
