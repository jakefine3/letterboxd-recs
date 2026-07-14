"""BeautifulSoup scraper for public Letterboxd profiles.

Letterboxd has no public API, but ratings and watchlists are visible on
profile pages. Plain requests get blocked by Cloudflare's TLS fingerprinting,
so we use curl_cffi's browser impersonation instead.

Page structure (as of mid-2026):
    li.griditem
        [data-item-name] = "Film Title (2019)"
        span.rating.rated-N  where N = stars * 2  (rated-9 = 4.5 stars)

Usage:
    scraper = LetterboxdScraper()
    ratings = scraper.fetch_ratings("someuser")    # name, year, rating
    watchlist = scraper.fetch_watchlist("someuser")  # name, year
"""
import re
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as stdlib_requests
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

_BASE = "https://letterboxd.com"
_MAX_PAGES = 100  # safety cap: 100 pages * 72 films = 7200 films

# "Film Title (2019)" -> ("Film Title", 2019)
_NAME_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)$")
# rating classes look like "rated-9" (stars * 2)
_RATED_RE = re.compile(r"rated-(\d+)")


class ScrapeError(Exception):
    """Raised when a profile can't be fetched (bad username, blocked, etc.)."""


class LetterboxdScraper:
    def __init__(self, scraperapi_key: str | None = None):
        self._scraperapi_key = scraperapi_key
        self._session = None if scraperapi_key else cffi_requests.Session(impersonate="chrome")

    def _get_page(self, url: str) -> BeautifulSoup | None:
        if self._scraperapi_key:
            proxy_url = (
                "http://api.scraperapi.com/"
                f"?api_key={self._scraperapi_key}"
                f"&url={urllib.parse.quote(url, safe='')}"
            )
            resp = stdlib_requests.get(proxy_url, timeout=30)
        else:
            resp = self._session.get(url, timeout=15)

        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise ScrapeError(f"Letterboxd returned {resp.status_code} for {url}")
        return BeautifulSoup(resp.text, "lxml")

    @staticmethod
    def _parse_items(soup: BeautifulSoup, with_rating: bool) -> list[dict]:
        rows = []
        for item in soup.select("li.griditem"):
            named = item.select_one("[data-item-name]")
            if named is None:
                continue
            m = _NAME_YEAR_RE.match(named["data-item-name"])
            if not m:
                continue  # films without a year can't be TMDB-matched reliably
            row = {"name": m.group(1), "year": int(m.group(2))}

            if with_rating:
                rating_span = item.select_one("span.rating")
                if rating_span is None:
                    continue
                rated = _RATED_RE.search(" ".join(rating_span.get("class", [])))
                if not rated:
                    continue
                row["rating"] = int(rated.group(1)) / 2

            rows.append(row)
        return rows

    def _scrape_paginated(self, base_url: str, with_rating: bool) -> pd.DataFrame:
        # Page 1 first — validates the profile exists before firing parallel work.
        first = self._get_page(base_url)
        if first is None:
            raise ScrapeError(f"Profile not found: {base_url}")
        first_rows = self._parse_items(first, with_rating)
        if not first_rows:
            return pd.DataFrame()

        all_rows: list[dict] = list(first_rows)

        # Fetch remaining pages in parallel batches. Stop as soon as a batch
        # contains an empty page (signals end of the list).
        _BATCH = 5
        for batch_start in range(2, _MAX_PAGES + 1, _BATCH):
            pages = range(batch_start, min(batch_start + _BATCH, _MAX_PAGES + 1))
            with ThreadPoolExecutor(max_workers=_BATCH) as ex:
                futures = {
                    ex.submit(self._get_page, f"{base_url}page/{p}/"): p
                    for p in pages
                }
                batch: dict[int, list[dict]] = {}
                for fut in as_completed(futures):
                    p = futures[fut]
                    soup = fut.result()
                    batch[p] = self._parse_items(soup, with_rating) if soup else []

            done = False
            for p in sorted(batch):
                if not batch[p]:
                    done = True
                    break
                all_rows.extend(batch[p])
            if done:
                break

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=["name", "year"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ratings(self, username: str) -> pd.DataFrame:
        """All rated films for a user. Columns: name, year, rating.

        Matches the schema of csv_loader.load_ratings so the two input
        paths are interchangeable downstream.
        """
        df = self._scrape_paginated(f"{_BASE}/{username}/films/ratings/", with_rating=True)
        if df.empty:
            raise ScrapeError(f"No rated films found for user '{username}'")
        logger.info("Scraped %d rated films for %s", len(df), username)
        return df

    def fetch_watchlist(self, username: str) -> pd.DataFrame:
        """A user's watchlist. Columns: name, year. Empty DataFrame if none."""
        try:
            df = self._scrape_paginated(f"{_BASE}/{username}/watchlist/", with_rating=False)
        except ScrapeError:
            return pd.DataFrame(columns=["name", "year"])
        logger.info("Scraped %d watchlist films for %s", len(df), username)
        return df
