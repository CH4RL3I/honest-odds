import numpy as np

from pmcal.parse import DAY, Market, parse_history, price_at_horizon, snapshot_rows


def _series():
    # one observation per day for 10 days, price = day/10
    t = np.arange(0, 10) * DAY
    p = np.arange(0, 10) / 10
    return t, p


def test_uses_last_observation_at_or_before_cutoff():
    t, p = _series()
    price, ts = price_at_horizon(t, p, t_res=9 * DAY, horizon_days=3)  # cutoff = day 6
    assert ts == 6 * DAY and price == 0.6


def test_observation_exactly_at_cutoff_is_allowed():
    t, p = _series()
    _, ts = price_at_horizon(t, p, t_res=9 * DAY, horizon_days=4)
    assert ts == 5 * DAY


def test_no_lookahead_future_price_never_used():
    """Poison every observation after the cutoff; the result must not change."""
    t, p = _series()
    t_res, h = 9 * DAY, 3
    cutoff = t_res - h * DAY
    clean = price_at_horizon(t, p, t_res, h)
    p_poison = np.where(t > cutoff, 0.999, p)
    assert price_at_horizon(t, p_poison, t_res, h) == clean
    # and the returned timestamp is never after the cutoff, over a grid of horizons
    for hd in (0.5, 1, 2, 3.7, 8):
        got = price_at_horizon(t, p, t_res, hd)
        assert got is None or got[1] <= t_res - int(hd * DAY)


def test_one_second_after_cutoff_is_excluded():
    t = np.array([0, 5 * DAY + 1])
    p = np.array([0.3, 0.9])
    price, ts = price_at_horizon(
        t, p, t_res=6 * DAY, horizon_days=1, max_stale_days=10
    )  # cutoff = 5*DAY
    assert (price, ts) == (0.3, 0)


def test_stale_or_missing_returns_none():
    t, p = _series()
    assert price_at_horizon(t, p, t_res=9 * DAY, horizon_days=30) is None  # nothing that early
    t2, p2 = np.array([0]), np.array([0.4])
    assert price_at_horizon(t2, p2, t_res=20 * DAY, horizon_days=7) is None  # 13 days stale
    assert price_at_horizon(np.array([], dtype=int), np.array([]), 1, 1) is None


def test_snapshot_rows_on_recorded_history(history_fx):
    t, p = parse_history(history_fx)
    m = Market("1", "q", "Sports", 5e4, int(t[-1]) + 3600, "tok", 1)
    rows = snapshot_rows(m, t, p, horizons=(1, 7))
    assert rows, "expected at least one horizon to be available"
    for r in rows:
        assert r["price_ts"] <= m.t_res - r["horizon_days"] * DAY
        assert r["outcome"] == 1 and 0 <= r["price"] <= 1
