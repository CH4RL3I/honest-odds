"""Build the dataset: list resolved markets (Gamma), pull price history (CLOB)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .api import CLOB, GAMMA, Client
from .parse import (
    Market,
    normalise_category,
    parse_history,
    parse_market,
    rows_to_frame,
    snapshot_rows,
)

DEFAULT_MIN_VOLUME = 20_000.0  # USD, lifetime traded volume


def iter_market_pages(client: Client, min_volume: float, max_pages: int | None = None):
    """Walk /markets/keyset (offset pagination is capped by the API). Pages are cached
    by index so an interrupted run resumes without re-requesting."""
    cursor = None
    page = 0
    while max_pages is None or page < max_pages:
        params = {"closed": "true", "limit": 100, "volume_num_min": int(min_volume)}
        if cursor:
            params["after_cursor"] = cursor
        cache = client.raw_dir / "markets" / f"page_{page:05d}.json"
        if cache.exists():
            data = json.loads(cache.read_text())
        else:
            data = client.get_json(f"{GAMMA}/markets/keyset", params)
            if data is None:
                raise RuntimeError(f"Gamma request failed at page {page}")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        yield data.get("markets", [])
        cursor = data.get("next_cursor")
        page += 1
        if not cursor or not data.get("markets"):
            return


def collect_markets(
    client, min_volume, max_pages=None, log=print, min_lifetime_days=0.0
) -> list[Market]:
    out: list[Market] = []
    seen = n_raw = 0
    for page in iter_market_pages(client, min_volume, max_pages):
        for raw in page:
            n_raw += 1
            m = parse_market(raw, min_volume, min_lifetime_days)
            if m:
                out.append(m)
        seen += 1
        if seen % 50 == 0:
            log(f"  pages={seen} raw={n_raw} kept={len(out)}")
    log(f"markets: raw={n_raw} binary/clean/volume-filtered={len(out)}")
    return out


# The CLOB API answers 200 with an empty history when `fidelity` is too fine for the requested
# range (observed: fidelity=60 with interval=max works for short-lived markets, returns nothing
# for long-lived ones; 720 works for all of them). Try fine first, fall back to coarse.
FIDELITIES = (60, 720)


def fill_categories(client: Client, markets: list[Market], log=print) -> list[Market]:
    """Newer Gamma markets have no `category`; look up their tags one by one (cached)."""
    out = []
    for i, m in enumerate(markets, 1):
        if m.category == "Uncategorised":
            raw = client.get_json(
                f"{GAMMA}/markets/{m.market_id}",
                {"include_tag": "true"},
                cache_key="tags/{h}.json",
            )
            labels = [t.get("label") for t in (raw or {}).get("tags") or []]
            m = replace(m, category=normalise_category(labels))
        out.append(m)
        if i % 500 == 0:
            log(f"  tags {i}/{len(markets)}")
    return out


def fetch_history(client: Client, token: str, fidelities=FIDELITIES):
    for fid in fidelities:
        payload = client.get_json(
            f"{CLOB}/prices-history",
            {"market": token, "interval": "max", "fidelity": fid},
            cache_key="history/{h}.json",
        )
        t, p = parse_history(payload or {})
        if len(t):
            return t, p
    return t, p


def build_dataset(
    raw_dir: Path,
    min_volume: float = DEFAULT_MIN_VOLUME,
    max_pages: int | None = None,
    min_lifetime_days: float = 0.0,
    sample: int | None = None,
    seed: int = 0,
    log=print,
) -> pd.DataFrame:
    client = Client(raw_dir)
    markets = collect_markets(client, min_volume, max_pages, log, min_lifetime_days)
    if sample and len(markets) > sample:
        # seeded uniform random subsample of eligible markets (reproducible given the same listing)
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(len(markets), size=sample, replace=False))
        markets = [markets[i] for i in keep]
        log(f"subsampled to {len(markets)} markets (seed={seed})")
    markets = fill_categories(client, markets, log)
    rows: list[dict] = []
    no_hist = 0
    for i, m in enumerate(markets, 1):
        t, p = fetch_history(client, m.yes_token)
        if len(t) == 0:
            no_hist += 1
        else:
            rows += snapshot_rows(m, t, p)
        if i % 200 == 0:
            log(f"  history {i}/{len(markets)} rows={len(rows)} empty={no_hist}")
    log(f"markets with empty history: {no_hist}/{len(markets)}")
    return rows_to_frame(rows)
