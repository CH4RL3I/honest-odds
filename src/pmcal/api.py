"""Polite HTTP client: throttled, retried with backoff, raw responses cached on disk."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
UA = "pmcal-research/0.1 (academic calibration study; contact emiliogappa@gmail.com)"


class Client:
    def __init__(self, raw_dir: Path, min_interval: float = 0.15, retries: int = 6):
        self.raw_dir = Path(raw_dir)
        self.min_interval = min_interval
        self.retries = retries
        self._last = 0.0
        self._http = httpx.Client(timeout=30, headers={"User-Agent": UA})

    def _sleep_for_throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get_json(self, url: str, params: dict, cache_key: str | None = None):
        """GET with cache. Empty-but-valid responses are cached too."""
        path = None
        if cache_key:
            h = hashlib.sha1(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:10]
            path = self.raw_dir / cache_key.format(h=h)
            if path.exists():
                return json.loads(path.read_text())
        delay = 1.0
        for _ in range(self.retries):
            self._sleep_for_throttle()
            try:
                r = self._http.get(url, params=params)
                if r.status_code == 200:
                    data = r.json()
                    if path:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(data))
                    return data
                if r.status_code not in (429, 500, 502, 503, 504):
                    return None  # permanent failure for this request (e.g. 400/404)
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(delay)
            delay = min(delay * 2, 60)
        return None
