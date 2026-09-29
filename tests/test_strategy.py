import numpy as np
import pandas as pd
import pytest

from pmcal.metrics import expit, logit
from pmcal.strategy import Costs, expected_and_realised_pnl, run_strategy, time_split


def _frame(n=600, seed=0, slope=0.6):
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.95, n)
    y = (rng.random(n) < expit(slope * logit(price))).astype(int)
    t_res = np.sort(rng.integers(1_700_000_000, 1_760_000_000, n))
    lead = rng.integers(3600, 20 * 86400, n)
    return pd.DataFrame(
        {
            "market_id": [str(i) for i in range(n)],
            "price": price,
            "outcome": y,
            "t_res": t_res,
            "price_ts": t_res - lead,
            "horizon_days": 7,
        }
    )


def test_time_split_no_training_data_after_split():
    df = _frame()
    train, test, split = time_split(df, 0.6)
    assert (train["t_res"] < split).all()  # outcomes all public before the split
    assert (test["t_res"] >= split).all()
    assert (test["price_ts"] >= split).all()  # bets placed after the split as well
    assert set(train["market_id"]).isdisjoint(test["market_id"])
    assert len(train) > 0 and len(test) > 0


def test_time_split_drops_straddling_markets():
    df = _frame()
    train, test, split = time_split(df, 0.6)
    straddle = df[(df["t_res"] >= split) & (df["price_ts"] < split)]
    assert len(straddle) > 0
    assert set(straddle["market_id"]).isdisjoint(test["market_id"])


def test_fit_uses_training_rows_only():
    """Changing test outcomes must not change the fitted mapping; changing train must."""
    df = _frame()
    train, test, _ = time_split(df)
    base = run_strategy(train, test, 0.02, Costs())
    flipped = test.assign(outcome=1 - test["outcome"])
    assert run_strategy(train, flipped, 0.02, Costs())["fit_slope"] == base["fit_slope"]
    train2 = train.assign(outcome=1 - train["outcome"])
    assert run_strategy(train2, test, 0.02, Costs())["fit_slope"] != base["fit_slope"]


def test_pnl_arithmetic():
    c = Costs(half_spread=0.01, fee_on_profit=0.02)
    ev_y, ev_n, pnl_y, pnl_n = expected_and_realised_pnl([0.40], [1], np.array([0.5]), c)
    ask = 0.41
    assert pnl_y[0] == pytest.approx((1 - ask) * 0.98)
    assert pnl_n[0] == pytest.approx(-(0.60 + 0.01))
    assert ev_y[0] == pytest.approx(0.5 * (1 - ask) * 0.98 - 0.5 * ask)


def test_no_edge_no_bets_when_model_matches_price():
    rng = np.random.default_rng(1234)  # distinct from the seed that generated prices
    df = _frame(n=8000, slope=1.0, seed=9)
    df["outcome"] = (rng.random(len(df)) < df["price"].to_numpy()).astype(int)
    train, test, _ = time_split(df)
    r = run_strategy(train, test, 0.05, Costs())
    assert r["n_bets"] <= 0.05 * r["n_test"]


def test_miscalibrated_market_frictionless_positive():
    df = _frame(n=6000, seed=1, slope=0.5)
    train, test, _ = time_split(df)
    r = run_strategy(train, test, 0.02, Costs(0.0, 0.0))
    assert r["n_bets"] > 0 and r["mean_pnl"] > 0
