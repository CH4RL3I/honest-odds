"""Out-of-sample check of a naive recalibration strategy. Not trading advice."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, fit_recalibration


@dataclass(frozen=True)
class Costs:
    half_spread: float = 0.01  # paid on entry: buy YES at price+hs, buy NO at (1-price)+hs
    fee_on_profit: float = 0.02  # proportional fee charged on winnings only


def time_split(df: pd.DataFrame, train_frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Strict temporal split on market resolution time.

    Train: markets that resolved strictly before the split timestamp S.
    Test:  markets that resolve at or after S AND whose price snapshot was also taken
           at or after S, so at bet time every training outcome was already public.
    Markets straddling S (snapshot before S, resolution after) are dropped.
    """
    per_market = df.drop_duplicates("market_id")
    split = int(per_market["t_res"].quantile(train_frac))
    train = df[df["t_res"] < split]
    test = df[(df["t_res"] >= split) & (df["price_ts"] >= split)]
    return train, test, split


def expected_and_realised_pnl(price, outcome, p_hat, costs: Costs):
    """Per-contract expected profit (under the model) and realised profit for both
    sides; returns arrays (ev_yes, ev_no, pnl_yes, pnl_no)."""
    price = np.asarray(price, dtype=float)
    y = np.asarray(outcome, dtype=float)
    ask_yes = np.clip(price + costs.half_spread, 0.001, 0.999)
    ask_no = np.clip(1 - price + costs.half_spread, 0.001, 0.999)
    f = costs.fee_on_profit

    def pnl(win, ask):
        return np.where(win, (1 - ask) * (1 - f), -ask)

    ev_yes = p_hat * (1 - ask_yes) * (1 - f) - (1 - p_hat) * ask_yes
    ev_no = (1 - p_hat) * (1 - ask_no) * (1 - f) - p_hat * ask_no
    return ev_yes, ev_no, pnl(y == 1, ask_yes), pnl(y == 0, ask_no)


def run_strategy(
    train: pd.DataFrame, test: pd.DataFrame, threshold: float, costs: Costs, seed: int = 0
) -> dict:
    """Fit the logistic recalibration on `train` only, then bet on each test contract
    whose model-implied expected profit (after costs) exceeds `threshold`."""
    fit = fit_recalibration(train["price"], train["outcome"], groups=train["market_id"])
    p_hat = fit.predict(test["price"].to_numpy())
    ev_y, ev_n, pnl_y, pnl_n = expected_and_realised_pnl(
        test["price"], test["outcome"], p_hat, costs
    )
    side_yes = ev_y >= ev_n
    ev = np.where(side_yes, ev_y, ev_n)
    pnl = np.where(side_yes, pnl_y, pnl_n)
    stake = np.where(
        side_yes,
        np.clip(test["price"].to_numpy() + costs.half_spread, 0.001, 0.999),
        np.clip(1 - test["price"].to_numpy() + costs.half_spread, 0.001, 0.999),
    )
    bet = ev > threshold
    n_bets = int(bet.sum())
    out = {
        "threshold": threshold,
        "n_test": len(test),
        "n_bets": n_bets,
        "fit_slope": fit.slope,
        "fit_intercept": fit.intercept,
    }
    if n_bets == 0:
        return out | {
            "mean_pnl": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
            "roi": np.nan,
            "hit_rate": np.nan,
            "total_pnl": 0.0,
            "pnl": pnl[bet],
            "bet_rows": test[bet],
        }
    lo, hi = bootstrap_ci(pnl[bet], groups=test["market_id"].to_numpy()[bet], seed=seed)
    return out | {
        "mean_pnl": float(pnl[bet].mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "roi": float(pnl[bet].sum() / stake[bet].sum()),
        "hit_rate": float((pnl[bet] > 0).mean()),
        "total_pnl": float(pnl[bet].sum()),
        "pnl": pnl[bet],
        "bet_rows": test[bet],
    }
