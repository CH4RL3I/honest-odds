"""Calibration statistics: binning with CIs, proper scores, Murphy decomposition,
and a logistic recalibration regression with robust standard errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

EPS = 0.01  # prices are clipped to [EPS, 1-EPS] before taking logits / log loss


def clip(p, eps: float = EPS):
    return np.clip(np.asarray(p, dtype=float), eps, 1 - eps)


def logit(p, eps: float = EPS):
    p = clip(p, eps)
    return np.log(p / (1 - p))


def expit(x):
    return special.expit(np.asarray(x, dtype=float))


def wilson_interval(k, n, alpha: float = 0.05):
    """Wilson score interval for a binomial proportion. Vectorised."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    z = stats.norm.ppf(1 - alpha / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        phat = k / n
        denom = 1 + z**2 / n
        centre = (phat + z**2 / (2 * n)) / denom
        half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def make_bins(n_bins: int = 10) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def calibration_table(price, outcome, edges=None) -> pd.DataFrame:
    """Per-bin mean price, realised frequency and Wilson 95% CI. Bins are
    [lo, hi) except the last, which includes 1.0."""
    price = np.asarray(price, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    edges = make_bins(10) if edges is None else np.asarray(edges)
    idx = np.clip(np.digitize(price, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        k = float(outcome[m].sum())
        lo, hi = wilson_interval(k, n)
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_price": float(price[m].mean()),
                "freq": k / n,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


def brier(price, outcome) -> float:
    price = np.asarray(price, dtype=float)
    return float(np.mean((price - np.asarray(outcome, dtype=float)) ** 2))


def log_loss(price, outcome, eps: float = EPS) -> float:
    p = clip(price, eps)
    y = np.asarray(outcome, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


@dataclass(frozen=True)
class Murphy:
    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin: float  # BS - (REL - RES + UNC); zero if forecasts are constant within bins

    @property
    def identity_gap(self) -> float:
        return self.brier - (self.reliability - self.resolution + self.uncertainty)


def murphy_decomposition(price, outcome, edges=None) -> Murphy:
    """Murphy (1973) decomposition BS = REL - RES + UNC.

    With bin means f_k, base rate obar and bin frequencies o_k:
      REL = sum n_k/N (f_k - o_k)^2,  RES = sum n_k/N (o_k - obar)^2,  UNC = obar(1-obar).
    The identity is exact when forecasts are identical within a bin. For continuous
    prices there is an additional within-bin term (variance of forecasts minus twice
    their covariance with outcomes, not sign-definite), returned as `within_bin` so
    nothing is hidden. Pass edges=None to use the unique forecast
    values as bins (exact identity).
    """
    price = np.asarray(price, dtype=float)
    y = np.asarray(outcome, dtype=float)
    n = len(price)
    obar = y.mean()
    if edges is None:
        _, idx = np.unique(price, return_inverse=True)
    else:
        edges = np.asarray(edges)
        idx = np.clip(np.digitize(price, edges[1:-1]), 0, len(edges) - 2)
    rel = res = 0.0
    for b in np.unique(idx):
        m = idx == b
        w = m.sum() / n
        fk, ok = price[m].mean(), y[m].mean()
        rel += w * (fk - ok) ** 2
        res += w * (ok - obar) ** 2
    unc = obar * (1 - obar)
    bs = brier(price, y)
    return Murphy(bs, rel, res, unc, bs - (rel - res + unc))


@dataclass(frozen=True)
class LogitFit:
    intercept: float
    slope: float
    se_intercept: float
    se_slope: float
    p_slope_eq_1: float  # Wald test, H0: slope = 1 (robust SE)
    p_joint: float  # Wald chi2(2), H0: intercept = 0 and slope = 1
    n: int

    def predict(self, price):
        return expit(self.intercept + self.slope * logit(price))


def fit_recalibration(price, outcome, groups=None, max_iter: int = 100) -> LogitFit:
    """Logistic regression  P(Y=1) = expit(a + b * logit(price)).

    A perfectly calibrated market has a=0, b=1. b<1: prices too extreme (overconfident);
    b>1: prices too moderate. Newton-Raphson (IRLS); standard errors are the
    Huber-White sandwich, clustered on `groups` when given (e.g. market id).
    """
    x = logit(price)
    y = np.asarray(outcome, dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0])

    def loglik(b):
        z = X @ b
        return float(np.sum(y * z - np.logaddexp(0.0, z)))

    for _ in range(max_iter):
        mu = expit(X @ beta)
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-10 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        t, ll0 = 1.0, loglik(beta)
        while loglik(beta + t * step) < ll0 - 1e-12 and t > 1e-8:  # backtracking: never overshoot
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-10:
            break
    mu = expit(X @ beta)
    W = mu * (1 - mu)
    bread = np.linalg.inv(X.T @ (X * W[:, None]) + 1e-10 * np.eye(2))
    score = X * (y - mu)[:, None]
    if groups is not None:
        score = pd.DataFrame(score).groupby(np.asarray(groups)).sum().to_numpy()
    meat = score.T @ score
    cov = bread @ meat @ bread
    se = np.sqrt(np.diag(cov))
    z = (beta[1] - 1.0) / se[1]
    p1 = 2 * stats.norm.sf(abs(z))
    d = beta - np.array([0.0, 1.0])
    wald = float(d @ np.linalg.solve(cov, d))
    pj = float(stats.chi2.sf(wald, 2))
    return LogitFit(float(beta[0]), float(beta[1]), float(se[0]), float(se[1]), p1, pj, len(y))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bootstrap_ci(values, groups=None, stat=np.mean, n_boot: int = 2000, seed: int = 0):
    """Percentile bootstrap CI for `stat`, resampling whole groups (clusters) if given."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    if groups is None:
        idx = rng.integers(0, len(v), size=(n_boot, len(v)))
        draws = np.array([stat(v[i]) for i in idx])
    else:
        g = pd.Series(v).groupby(np.asarray(groups)).apply(lambda s: s.to_numpy())
        arrs = list(g.values)
        k = len(arrs)
        draws = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, k, size=k)
            draws[b] = stat(np.concatenate([arrs[i] for i in pick]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
