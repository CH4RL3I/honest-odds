import numpy as np
import pytest

from pmcal.metrics import (
    bootstrap_ci,
    brier,
    calibration_table,
    expit,
    fit_recalibration,
    holm,
    log_loss,
    logit,
    murphy_decomposition,
    wilson_interval,
)


def test_wilson_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2  # upper bound ~0.161
    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0


def test_wilson_coverage_simulation():
    rng = np.random.default_rng(1)
    n, p, reps = 50, 0.2, 4000
    k = rng.binomial(n, p, reps)
    lo, hi = wilson_interval(k, np.full(reps, n))
    cover = np.mean((lo <= p) & (p <= hi))
    assert 0.92 < cover < 0.98


def test_calibration_table_counts_and_bins():
    price = np.array([0.05, 0.05, 0.15, 0.95, 1.0, 0.0])
    y = np.array([0, 1, 0, 1, 1, 0])
    t = calibration_table(price, y)
    assert t["n"].sum() == len(price)
    first = t.iloc[0]  # [0,0.1): 0.05, 0.05, 0.0
    assert first["n"] == 3 and first["freq"] == pytest.approx(1 / 3)
    last = t.iloc[-1]  # [0.9,1.0]: 0.95 and 1.0 (right edge included)
    assert last["n"] == 2
    assert (t["ci_lo"] <= t["freq"]).all() and (t["freq"] <= t["ci_hi"]).all()


def test_brier_and_logloss():
    assert brier([1, 0], [1, 0]) == 0
    assert brier([0.5, 0.5], [1, 0]) == 0.25
    assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(np.log(2))
    assert np.isfinite(log_loss([0.0, 1.0], [1, 0]))  # clipped, so finite


def test_murphy_identity_exact_for_discrete_forecasts():
    rng = np.random.default_rng(0)
    price = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=5000)
    y = (rng.random(5000) < price * 0.9 + 0.03).astype(float)
    m = murphy_decomposition(price, y)  # bins = unique forecast values
    assert m.reliability - m.resolution + m.uncertainty == pytest.approx(m.brier, abs=1e-12)
    assert abs(m.identity_gap) < 1e-12


def test_murphy_identity_with_within_bin_term_for_continuous_prices():
    rng = np.random.default_rng(2)
    price = rng.beta(0.6, 0.6, 20000)
    y = (rng.random(20000) < price).astype(float)
    m = murphy_decomposition(price, y, edges=np.linspace(0, 1, 11))
    assert m.identity_gap == pytest.approx(m.within_bin)
    assert abs(m.identity_gap) < 0.005  # small with 10 bins
    assert m.reliability < 0.002  # simulated as perfectly calibrated
    assert m.uncertainty == pytest.approx(y.mean() * (1 - y.mean()))


def test_logit_expit_roundtrip():
    p = np.array([0.02, 0.5, 0.97])
    assert expit(logit(p)) == pytest.approx(p)


def test_recalibration_recovers_known_slope():
    rng = np.random.default_rng(3)
    price = rng.uniform(0.03, 0.97, 40000)
    true_p = expit(0.2 + 0.7 * logit(price))
    y = (rng.random(len(price)) < true_p).astype(float)
    fit = fit_recalibration(price, y)
    assert fit.slope == pytest.approx(0.7, abs=4 * fit.se_slope)
    assert fit.intercept == pytest.approx(0.2, abs=4 * fit.se_intercept)
    assert fit.p_slope_eq_1 < 1e-6 and fit.p_joint < 1e-6


def test_recalibration_calibrated_data_not_rejected():
    rng = np.random.default_rng(4)
    price = rng.uniform(0.03, 0.97, 20000)
    y = (rng.random(len(price)) < price).astype(float)
    fit = fit_recalibration(price, y, groups=np.arange(len(price)))
    assert abs(fit.slope - 1) < 4 * fit.se_slope
    assert fit.se_slope > 0


def test_holm_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert (adj >= np.array([0.01, 0.04, 0.03, 0.5])).all() and (adj <= 1).all()


def test_cluster_bootstrap_widens_with_clustering():
    rng = np.random.default_rng(5)
    g = np.repeat(np.arange(50), 20)
    v = rng.normal(size=50)[g]  # perfectly correlated within cluster
    lo_i, hi_i = bootstrap_ci(v, n_boot=500)
    lo_c, hi_c = bootstrap_ci(v, groups=g, n_boot=500)
    assert (hi_c - lo_c) > 1.5 * (hi_i - lo_i)
