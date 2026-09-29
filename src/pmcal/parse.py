"""Parsing and filtering of Gamma / CLOB API payloads. Pure functions, no I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

DAY = 86400
HORIZONS_DAYS = (30, 7, 1)
# A snapshot is only used if the last observation at or before the horizon is
# at most this old. Prevents quoting a stale price from weeks earlier.
MAX_STALE_DAYS = 2.0
# A market counts as cleanly resolved if the Yes price ended within this of 0 or 1.
CLEAN_TOL = 0.01


@dataclass(frozen=True)
class Market:
    market_id: str
    question: str
    category: str
    volume: float
    t_res: int  # unix seconds, see resolution_time()
    yes_token: str
    outcome: int  # 1 if Yes won


def _loads(x):
    """Gamma returns several list fields as JSON-encoded strings."""
    if isinstance(x, str):
        return json.loads(x)
    return x


def parse_ts(s: str | None) -> int | None:
    """Parse '2020-11-02 16:31:01+00' or '2020-11-04T00:00:00Z' into unix seconds."""
    if not s:
        return None
    s = s.strip().replace("Z", "+00:00").replace(" ", "T")
    if s.endswith("+00"):
        s += ":00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def resolution_time(raw: dict) -> int | None:
    """Reference time for horizons: the earlier of endDate and closedTime.

    endDate is the nominal deadline; closedTime is when the market was closed.
    Markets that resolve early ("before Jan 1") close before endDate, and markets
    whose resolution was delayed close after it, so the minimum is the best
    available proxy for when the outcome became knowable.
    """
    cands = [t for t in (parse_ts(raw.get("endDate")), parse_ts(raw.get("closedTime"))) if t]
    return min(cands) if cands else None


# Gamma exposes a free-text `category` only for legacy markets; newer ones carry tags
# (fetched with include_tag=true). Both are mapped onto a small fixed set of buckets,
# first match wins.
_CATEGORY_RULES = (
    ("Sports", ("sports", "esports", "games", "olympics", "nba", "nfl", "soccer", "tennis")),
    ("Crypto", ("crypto", "bitcoin", "ethereum", "solana", "crypto prices")),
    (
        "Politics",
        ("politics", "us-current-affairs", "elections", "election", "world", "geopolitics"),
    ),
    ("Economics", ("economy", "finance", "business", "fed", "economics", "stocks", "commodities")),
    ("Culture", ("pop-culture", "culture", "entertainment", "music", "movies", "celebrities")),
    ("Science/Tech", ("tech", "science", "ai", "coronavirus", "weather")),
)


def normalise_category(labels) -> str:
    """Map a category string and/or list of tag labels to one coarse bucket."""
    if isinstance(labels, str):
        labels = [labels]
    low = {str(x).strip().lower() for x in (labels or []) if x}
    for bucket, keys in _CATEGORY_RULES:
        if low & set(keys):
            return bucket
    return "Other" if low else "Uncategorised"


def parse_market(raw: dict, min_volume: float = 0.0) -> Market | None:
    """Return a Market if `raw` is a binary Yes/No market with a clean resolution."""
    try:
        outcomes = _loads(raw.get("outcomes"))
        prices = [float(p) for p in _loads(raw.get("outcomePrices"))]
        tokens = _loads(raw.get("clobTokenIds"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not outcomes or [str(o).strip().lower() for o in outcomes] != ["yes", "no"]:
        return None
    if len(prices) != 2 or not tokens or len(tokens) != 2:
        return None
    p_yes, p_no = prices
    if p_yes >= 1 - CLEAN_TOL and p_no <= CLEAN_TOL:
        outcome = 1
    elif p_yes <= CLEAN_TOL and p_no >= 1 - CLEAN_TOL:
        outcome = 0
    else:  # 50/50 voids, "0,0" placeholders on legacy AMM markets, unresolved
        return None
    volume = float(raw.get("volumeNum") or raw.get("volume") or 0.0)
    if volume < min_volume:
        return None
    t_res = resolution_time(raw)
    if t_res is None:
        return None
    return Market(
        market_id=str(raw["id"]),
        question=str(raw.get("question", "")),
        category=normalise_category(
            [raw.get("category")] + [t.get("label") for t in (raw.get("tags") or [])]
        ),
        volume=volume,
        t_res=t_res,
        yes_token=str(tokens[0]),
        outcome=outcome,
    )


def parse_history(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    """CLOB prices-history -> (timestamps, prices), sorted by time."""
    h = payload.get("history") or []
    if not h:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    t = np.array([int(x["t"]) for x in h], dtype=np.int64)
    p = np.array([float(x["p"]) for x in h], dtype=float)
    order = np.argsort(t, kind="stable")
    return t[order], p[order]


def price_at_horizon(
    t: np.ndarray,
    p: np.ndarray,
    t_res: int,
    horizon_days: float,
    max_stale_days: float = MAX_STALE_DAYS,
) -> tuple[float, int] | None:
    """Last price observed at or before t_res - horizon (no lookahead).

    Returns (price, observation_timestamp), or None when no observation exists
    or the latest one is older than `max_stale_days`.
    """
    cutoff = t_res - int(horizon_days * DAY)
    i = int(np.searchsorted(t, cutoff, side="right")) - 1  # last index with t <= cutoff
    if i < 0:
        return None
    if cutoff - int(t[i]) > max_stale_days * DAY:
        return None
    return float(p[i]), int(t[i])


def snapshot_rows(m: Market, t: np.ndarray, p: np.ndarray, horizons=HORIZONS_DAYS) -> list[dict]:
    rows = []
    for h in horizons:
        got = price_at_horizon(t, p, m.t_res, h)
        if got is None:
            continue
        rows.append(
            {
                "market_id": m.market_id,
                "category": m.category,
                "volume": m.volume,
                "t_res": m.t_res,
                "horizon_days": h,
                "price": got[0],
                "price_ts": got[1],
                "outcome": m.outcome,
            }
        )
    return rows


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["t_res_dt"] = pd.to_datetime(df["t_res"], unit="s", utc=True)
    return df.sort_values(["t_res", "market_id", "horizon_days"]).reset_index(drop=True)
