"""benchmark_report statistics: dispatch to the libraries, per-group testing, corrections."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scikit_posthocs as sp
from scipy import stats as ss
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

from social_evaluation.benchmark_report.stats import (
    StatsMixin,
    StatsSettings,
    omnibus,
    posthoc,
    success_permutation_test,
)

RNG = np.random.default_rng(0)
A, B, C = RNG.normal(0, 1, 12), RNG.normal(1.5, 1, 12), RNG.normal(0.3, 3, 12)


def test_omnibus_dispatches_to_the_libraries():
    h, p = ss.kruskal(A, B, C)
    kw = omnibus([A, B, C], "kruskal", 0.05)
    assert (kw["statistic"], kw["p_value"]) == pytest.approx((h, p))
    assert kw["effect_size"] == pytest.approx((h - 3 + 1) / (36 - 3))
    assert omnibus([A, B, C], "anova", 0.05)["p_value"] == pytest.approx(ss.f_oneway(A, B, C).pvalue)
    welch = anova_oneway([A, B, C], use_var="unequal", welch_correction=True)
    assert omnibus([A, B, C], "welch", 0.05)["p_value"] == pytest.approx(welch.pvalue)
    # same H, p from shuffles: close to the chi-square p at n = 12 per group
    perm = omnibus([A, B, C], "permutation", 0.05, permutations=4999, seed=1)
    assert perm["statistic"] == pytest.approx(h)
    assert perm["p_value"] == pytest.approx(p, abs=0.02)


def test_posthoc_dispatches_to_scikit_posthocs():
    data = pd.DataFrame({"value": np.r_[A, B, C], "algorithm": np.repeat(list("abc"), 12)})
    expected = sp.posthoc_dunn(data, val_col="value", group_col="algorithm", p_adjust="holm")
    pd.testing.assert_frame_equal(posthoc(data, "dunn", "holm"), expected)


class _Report(StatsMixin):
    """The attributes the mixin reads from BenchmarkReportBase."""

    def __init__(self, results: pd.DataFrame):
        self.config = SimpleNamespace(
            episode_groups={"G1": ["e1"], "G2": ["e2"]},
            metric_directions={"m": "UP"},
            navigator_names={},
        )
        self.ordered_navigators = ["A", "B"]
        self.ordered_experiments = []
        self._all_results = results

    def results_for_metric(self, metric_key):
        return self._all_results

    def metric_plain_label(self, metric_key):
        return metric_key

    def _experiment_order_with_groups(self, available=None):
        return list(available)


def _runs(experiment, a, b):
    return pd.DataFrame({"experiment": experiment, "algorithm": ["A"] * len(a) + ["B"] * len(b),
                         "m": np.r_[a, b], "result": "Success"})


def test_each_episode_group_is_tested_on_its_own_runs():
    # G1: B far above A; G2: no difference. Pooled, G2's runs would look different too.
    results = pd.concat([_runs("e1", RNG.normal(0, 1, 10), RNG.normal(6, 1, 10)),
                         _runs("e2", RNG.normal(0, 1, 10), RNG.normal(0, 1, 10))])
    report = _Report(results)
    summary, pairwise = report.analyze_algorithm_effects(settings=StatsSettings(test="kruskal"))
    assert summary.loc[("G1", "m"), "significant"] and not summary.loc[("G2", "m"), "significant"]
    assert list(pairwise.index.get_level_values("episode_group")) == ["G1"]
    holm = multipletests(summary["p_value"], method="holm")[1]
    np.testing.assert_allclose(summary["p_adjusted"], holm)


def test_success_permutation_test():
    algorithms = ["a"] * 10 + ["b"] * 10
    _, p_equal = success_permutation_test(algorithms, [1, 0] * 10, 2000, 0)
    _, p_split = success_permutation_test(algorithms, [1] * 10 + [0] * 10, 2000, 0)
    assert p_equal > 0.9 and p_split < 0.01


# --- repeated-measures tests and models -------------------------------------------------------

from statsmodels.stats.contingency_tables import cochrans_q, mcnemar  # noqa: E402

from social_evaluation.benchmark_report.stats import emm_contrasts, fit_model  # noqa: E402


def _blocked(values: dict, experiment="e1", **extra) -> pd.DataFrame:
    """One row per (navigator, run): values[navigator][run]."""
    rows = [dict(experiment=experiment, algorithm=nav, run=i, m=v, result="Success", **extra)
            for nav, vs in values.items() for i, v in enumerate(vs)]
    return pd.DataFrame(rows)


def _report3(results):
    report = _Report(results)
    report.ordered_navigators = ["A", "B", "C"]
    report.config.episode_groups = {"G1": ["e1"]}
    return report


def test_friedman_uses_complete_blocks_and_wilcoxon_posthocs():
    base = RNG.normal(0, 1, 15)  # shared block effect
    vals = {"A": base, "B": base + 1.0 + RNG.normal(0, .3, 15), "C": base + RNG.normal(0, .3, 15)}
    results = _blocked(vals)
    results = results[~((results["algorithm"] == "C") & (results["run"] == 0))]  # block 0 incomplete
    summary, pairwise = _report3(results).analyze_algorithm_effects(
        settings=StatsSettings(test="friedman"))
    row = summary.loc[("G1", "m")]
    chi2, p = ss.friedmanchisquare(*(v[1:] for v in vals.values()))
    assert row["test"] == "friedman" and row["blocks"] == 14
    assert (row["statistic"], row["p_value"]) == pytest.approx((chi2, p))
    assert row["effect_size"] == pytest.approx(chi2 / (14 * 2))  # Kendall's W
    raw = [ss.wilcoxon(vals[a][1:], vals[b][1:]).pvalue for a, b in (("A", "B"), ("A", "C"), ("B", "C"))]
    expected = multipletests(raw, method="holm")[1]
    np.testing.assert_allclose(pairwise["p_adjusted"], expected)
    assert set(pairwise["posthoc"]) == {"wilcoxon"}


def test_friedman_with_two_navigators_is_wilcoxon():
    a = RNG.normal(0, 1, 12)
    b = a + 1 + RNG.normal(0, .2, 12)
    report = _Report(_blocked({"A": a, "B": b}))
    report.config.episode_groups = {"G1": ["e1"]}
    row = report.analyze_algorithm_effects(settings=StatsSettings(test="friedman"))[0].loc[("G1", "m")]
    assert row["test"] == "wilcoxon"
    assert row["p_value"] == pytest.approx(ss.wilcoxon(a, b).pvalue)


def test_cochran_q_and_mcnemar_on_binary_outcomes():
    a = np.r_[np.ones(18), np.zeros(2)]
    b = np.r_[np.ones(6), np.zeros(14)]
    c = np.r_[np.ones(15), np.zeros(5)]
    summary, pairwise = _report3(_blocked({"A": a, "B": b, "C": c})).analyze_algorithm_effects(
        settings=StatsSettings(test="cochran"))
    row = summary.loc[("G1", "m")]
    expected = cochrans_q(np.c_[a, b, c])
    assert (row["statistic"], row["p_value"]) == pytest.approx((expected.statistic, expected.pvalue))

    def mc(x, y):
        return mcnemar([[np.sum((x == 1) & (y == 1)), np.sum((x == 1) & (y == 0))],
                        [np.sum((x == 0) & (y == 1)), np.sum((x == 0) & (y == 0))]], exact=True).pvalue

    np.testing.assert_allclose(pairwise["p_adjusted"],
                               multipletests([mc(a, b), mc(a, c), mc(b, c)], method="holm")[1])


def test_cochran_rejects_non_binary_values():
    summary, _ = _report3(_blocked({n: RNG.normal(0, 1, 8) for n in "ABC"})).analyze_algorithm_effects(
        settings=StatsSettings(test="cochran"))
    assert "0/1" in summary.loc[("G1", "m"), "note"]


def test_lmm_wald_test_and_emm_contrasts():
    base = RNG.normal(0, 2, 20)
    vals = {"A": base + RNG.normal(0, .5, 20), "B": base + 1.5 + RNG.normal(0, .5, 20),
            "C": base + RNG.normal(0, .5, 20)}
    report = _report3(_blocked(vals))
    summary, pairwise = report.analyze_algorithm_effects(settings=StatsSettings(test="lmm"))
    row = summary.loc[("G1", "m")]
    assert row["test"] == "lmm" and row["df1"] == 2 and row["p_value"] < 1e-6
    p = pairwise["p_adjusted"].droplevel([0, 1])
    assert p[("A", "B")] < 1e-3 and p[("B", "C")] < 1e-3 and p[("A", "C")] > 0.05
    # two navigators: the Tukey contrast is the coefficient's two-sided z test
    data = report._test_data(_blocked({"A": vals["A"], "B": vals["B"]}), "m", StatsSettings())
    fit = fit_model(data, "lmm", ["A", "B"])
    assert emm_contrasts(fit, ["A", "B"]).loc["A", "B"] == pytest.approx(fit.pvalues["C(algorithm)[T.L1]"])


def test_lmm_with_nested_cohort_and_gee_models_run():
    base = RNG.normal(0, 1, 24)
    results = _blocked({"A": base, "B": base + 1 + RNG.normal(0, .3, 24), "C": base + RNG.normal(0, .3, 24)})
    results["cohort"] = results["run"] // 3
    summary, _ = _report3(results).analyze_algorithm_effects(
        settings=StatsSettings(test="lmm", cohort="cohort"))
    assert summary.loc[("G1", "m"), "p_value"] < 1e-3
    counts = _blocked({"A": RNG.poisson(1, 30), "B": RNG.poisson(4, 30), "C": RNG.poisson(1, 30)})
    summary, pairwise = _report3(counts).analyze_algorithm_effects(settings=StatsSettings(test="gee_poisson"))
    assert summary.loc[("G1", "m"), "p_value"] < 1e-3 and set(pairwise["posthoc"]) == {"emm"}


def test_per_metric_overrides_transform_and_outcome_metrics():
    vals = {n: np.exp(RNG.normal(mu, .3, 12)) for n, mu in (("A", 0), ("B", 1), ("C", 0))}
    results = _blocked(vals)
    results.loc[results["algorithm"] == "B", "result"] = ["Collision"] * 9 + ["Success"] * 3
    report = _report3(results)
    settings = StatsSettings.from_config({
        "metrics": ["m", "collision"],
        "per_metric": {"m": {"test": "lmm", "transform": "log"}, "collision": {"test": "cochran"}}})
    summary, pairwise = report.analyze_algorithm_effects(settings=settings)
    assert summary.loc[("G1", "m"), "test"] == "lmm" and summary.loc[("G1", "m"), "transform"] == "log"
    assert summary.loc[("G1", "collision"), "test"] == "cochran"
    ab = pairwise.loc[("G1", "m", "A", "B")]
    assert ab["median_a"] == pytest.approx(np.median(vals["A"]))  # descriptives untransformed


def test_settings_validation():
    with pytest.raises(ValueError, match="emm needs a model"):
        StatsSettings.from_config({"test": "anova", "posthoc": "emm"})
    with pytest.raises(ValueError, match="unknown keys"):
        StatsSettings.from_config({"per_metric": {"m": {"alpha": 0.1}}})
    with pytest.raises(ValueError, match="needs a block"):
        StatsSettings.from_config({"test": "friedman", "block": None})
    assert StatsSettings.from_config({"per_metric": {"m": {"test": "friedman"}}}).for_cell("G1", "m").test == "friedman"


def test_per_group_overrides_and_precedence():
    settings = StatsSettings.from_config({
        "test": "kruskal",
        "per_group": {"G1": {"test": "friedman",
                             "per_metric": {"collision": {"test": "cochran"}}}},
        "per_metric": {"collision": {"test": "permutation"}, "d": {"transform": "log"}}})
    assert settings.for_cell("G2", "m").test == "kruskal"
    assert settings.for_cell("G1", "m").test == "friedman"              # group over base
    assert settings.for_cell("G2", "collision").test == "permutation"   # metric over base
    assert settings.for_cell("G1", "collision").test == "cochran"       # nested wins
    assert settings.for_cell("G1", "d").transform == "log" and settings.for_cell("G1", "d").test == "friedman"


def test_per_group_validation():
    with pytest.raises(ValueError, match="per_group.G1: unknown keys"):
        StatsSettings.from_config({"per_group": {"G1": {"alpha": 0.1}}})
    with pytest.raises(ValueError, match="per_group.G1.per_metric.m: unknown keys"):
        StatsSettings.from_config({"per_group": {"G1": {"per_metric": {"m": {"alpha": 0.1}}}}})
    with pytest.raises(ValueError, match="per_group.G1.*emm needs a model"):
        StatsSettings.from_config({"per_group": {"G1": {"posthoc": "emm"}}})


def test_each_group_uses_its_own_test():
    base = RNG.normal(0, 1, 12)
    paired = _blocked({"A": base, "B": base + 1 + RNG.normal(0, .2, 12), "C": base}, experiment="e1")
    independent = _blocked({n: RNG.normal(0, 1, 12) for n in "ABC"}, experiment="e2")
    report = _report3(pd.concat([paired, independent]))
    report.config.episode_groups = {"G1": ["e1"], "G2": ["e2"]}
    settings = StatsSettings.from_config({"test": "kruskal", "per_group": {"G1": {"test": "friedman"}}})
    summary, _ = report.analyze_algorithm_effects(settings=settings)
    assert summary.loc[("G1", "m"), "test"] == "friedman" and summary.loc[("G2", "m"), "test"] == "kruskal"
