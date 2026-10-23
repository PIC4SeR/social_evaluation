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
