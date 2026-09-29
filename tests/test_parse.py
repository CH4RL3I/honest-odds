import numpy as np

from pmcal.parse import (
    normalise_category,
    parse_history,
    parse_market,
    parse_ts,
    resolution_time,
)


def test_yes_market_parsed(markets_fx):
    m = parse_market(markets_fx["yes"])
    assert m is not None
    assert m.outcome == 1
    assert (
        m.yes_token
        == "45491661479368191262580562547363246014236215722706966480661285338190329062638"
    )
    assert m.volume > 30_000


def test_no_market_parsed(markets_fx):
    assert parse_market(markets_fx["no"]).outcome == 0


def test_legacy_zero_zero_prices_rejected(markets_fx):
    assert parse_market(markets_fx["amm_zero_zero"]) is None


def test_non_yes_no_rejected(markets_fx):
    assert parse_market(markets_fx["non_yes_no"]) is None


def test_void_5050_rejected(markets_fx):
    assert parse_market(markets_fx["void_5050"]) is None


def test_min_volume_filter(markets_fx):
    raw = markets_fx["yes"]
    assert parse_market(raw, min_volume=raw["volumeNum"] - 1) is not None
    assert parse_market(raw, min_volume=raw["volumeNum"] + 1) is None


def test_near_clean_resolution_accepted():
    raw = {
        "id": "1",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.9999", "0.0001"]',
        "clobTokenIds": '["a", "b"]',
        "volumeNum": 1e5,
        "endDate": "2025-01-02T00:00:00Z",
    }
    assert parse_market(raw).outcome == 1
    raw["outcomePrices"] = '["0.9", "0.1"]'  # not resolved
    assert parse_market(raw) is None


def test_parse_ts_formats():
    assert parse_ts("2020-11-02 16:31:01+00") == parse_ts("2020-11-02T16:31:01Z")
    assert parse_ts("2020-11-04T00:00:00Z") == 1604448000
    assert parse_ts(None) is None and parse_ts("garbage") is None


def test_resolution_time_is_earlier_of_end_and_close():
    early = {"endDate": "2025-01-10T00:00:00Z", "closedTime": "2025-01-05 00:00:00+00"}
    late = {"endDate": "2025-01-10T00:00:00Z", "closedTime": "2025-01-20 00:00:00+00"}
    assert resolution_time(early) == parse_ts("2025-01-05T00:00:00Z")
    assert resolution_time(late) == parse_ts("2025-01-10T00:00:00Z")
    assert resolution_time({}) is None


def test_category_normalisation():
    assert normalise_category(["Sports", "EFL Championship"]) == "Sports"
    assert normalise_category("US-current-affairs") == "Politics"
    assert normalise_category(["Up or Down", "Crypto", "Bitcoin"]) == "Crypto"
    assert normalise_category([None]) == "Uncategorised"
    assert normalise_category(["Something odd"]) == "Other"


def test_history_shape_and_sorting(history_fx):
    t, p = parse_history(history_fx)
    assert len(t) == len(p) > 10
    assert np.all(np.diff(t) >= 0)
    assert np.all((p >= 0) & (p <= 1))
    # unsorted input gets sorted
    t2, p2 = parse_history({"history": [{"t": 20, "p": 0.2}, {"t": 10, "p": 0.1}]})
    assert list(t2) == [10, 20] and list(p2) == [0.1, 0.2]


def test_empty_history(history_empty_fx):
    t, p = parse_history(history_empty_fx)
    assert len(t) == 0 and len(p) == 0
