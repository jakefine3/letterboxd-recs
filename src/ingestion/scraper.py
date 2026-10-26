"""BeautifulSoup scraper for public Letterboxd profiles.

Letterboxd has no public API, but ratings and watchlists are visible on
profile pages. Plain requests get blocked by Cloudflare's TLS fingerprinting,
so we use curl_cffi's browser impersonation instead. Cloudflare also blocks
datacenter IPs outright (Streamlit Cloud, GitHub runners), so when a direct
request is refused we retry through ZenRows' residential proxies.

Page structure (as of mid-2026):
    li.griditem
        [data-item-name] = "Film Title (2019)"
        span.rating.rated-N  where N = stars * 2  (rated-9 = 4.5 stars)
    .paginate-pages li     last one holds the page count

Usage:
    scraper = LetterboxdScraper()
    ratings = scraper.fetch_ratings("someuser")    # name, year, rating
    watchlist = scraper.fetch_watchlist("someuser")  # name, year
"""
import re
import logging
from concurrent.futures import ThreadPoolExecutor

import requests as stdlib_requests
import pandas as pd
from bs4 import BeautifulSoup
from curl_cffi import requests as cffi_requests

logger = logging.getLogger(__name__)

_BASE = "https://letterboxd.com"
_ZENROWS = "https://api.zenrows.com/v1/"
# ratings stay effectively uncapped: anything we skip isn't marked as seen and
# can come back as a recommendation. The watchlist only drives badges, so it
# gets a tight cap (28 films a page, so it's the pricier list to fetch).
_MAX_RATING_PAGES = 100
_MAX_WATCHLIST_PAGES = 5
_WORKERS = 5  # ZenRows free plan allows 5 concurrent requests

# "Film Title (2019)" -> ("Film Title", 2019)
_NAME_YEAR_RE = re.compile(r"^(.*?)\s*\((\d{4})\)$")
# rating classes look like "rated-9" (stars * 2)
_RATED_RE = re.compile(r"rated-(\d+)")


class ScrapeError(Exception):
    """Raised when a profile can't be fetched (bad username, blocked, etc.)."""


class BlockedError(ScrapeError):
    """Letterboxd (Cloudflare) refused the request and no proxy could get through."""


class ProxyQuotaError(ScrapeError):
    """ZenRows refused the request: monthly credits used up or key invalid."""


# statuses Cloudflare uses when it blocks a datacenter IP
_BLOCKED = {403, 429, 503}


class LetterboxdScraper:
    def __init__(self, zenrows_key: str | None = None):
        self._zenrows_key = zenrows_key
        self._session = cffi_requests.Session(impersonate="chrome")
        # cheapest route first: direct (free), then ZenRows' standard proxies
        # (1 credit), then residential (10). Each flag flips once per scrape
        # the first time a route gets blocked.
        self._use_proxy = False
        self._premium = False

    def _get_page(self, url: str) -> BeautifulSoup | None:
        resp = None
        if not self._use_proxy:
            try:
                resp = self._session.get(url, timeout=15)
            except cffi_requests.RequestsError as e:
                logger.warning("direct fetch failed for %s: %s", url, e)
            if resp is None or resp.status_code in _BLOCKED:
                if not self._zenrows_key:
                    raise BlockedError(f"Letterboxd returned {getattr(resp, 'status_code', 'no response')} for {url}")
                logger.info("direct fetch blocked, switching to ZenRows")
                self._use_proxy = True
                resp = None
        if resp is None:
            resp = self._get_via_proxy(url)

        if resp.status_code == 404:
            return None
        if resp.status_code in _BLOCKED:
            raise BlockedError(f"Letterboxd returned {resp.status_code} for {url}")
        if resp.status_code != 200:
            raise ScrapeError(f"Letterboxd returned {resp.status_code} for {url}")
        return BeautifulSoup(resp.text, "lxml")

    def _get_via_proxy(self, url: str, attempts: int = 5):
        if not self._premium:
            # one try only: if ZenRows can't get through on standard proxies
            # (422) or Cloudflare blocks them, residential is the fix, not a retry
            try:
                resp = self._zenrows(url, premium=False, attempts=1)
                if resp.status_code not in _BLOCKED:
                    return resp
            except ProxyQuotaError:
                raise
            except ScrapeError:
                pass
            logger.info("standard proxy blocked, switching to residential")
            self._premium = True
        return self._zenrows(url, premium=True, attempts=attempts)

    def _zenrows(self, url: str, premium: bool, attempts: int):
        """Fetch through ZenRows: standard proxies cost 1 credit, residential 10.

        original_status passes Letterboxd's own status through, so a 404 still
        means "no such user". ZenRows' own errors: 401 = bad key, 402 = credits
        used up, 429 = over the concurrency limit, 422/5xx = couldn't get the
        page this time (not billed, worth retrying). With original_status, a
        RESP001 body means ZenRows' exit IP got challenged — also unbilled, and
        residential IPs rotate, so a retry usually lands (~half fail)."""
        params = {
            "apikey": self._zenrows_key,
            "url": url,
            "original_status": "true",
        }
        if premium:
            params["premium_proxy"] = "true"
        last_err = None
        for attempt in range(attempts):
            try:
                resp = stdlib_requests.get(_ZENROWS, params=params, timeout=70)
            except (stdlib_requests.Timeout, stdlib_requests.ConnectionError) as e:
                last_err = e
            else:
                if resp.status_code in (401, 402):
                    raise ProxyQuotaError(f"ZenRows returned {resp.status_code}: {resp.text[:200]}")
                if (resp.status_code not in (422, 429) and resp.status_code < 500
                        and "RESP001" not in resp.text[:200]):
                    return resp
                last_err = f"ZenRows returned {resp.status_code}: {resp.text[:200]}"
            logger.warning("ZenRows attempt %d/%d failed for %s: %s",
                           attempt + 1, attempts, url, last_err)
        raise ScrapeError(
            f"Letterboxd is responding slowly right now — please try again in a minute. ({last_err})"
        )

    @staticmethod
    def _page_count(soup: BeautifulSoup) -> int:
        nums = [int(t) for li in soup.select(".paginate-pages li")
                if (t := li.get_text(strip=True)).isdigit()]
        return max(nums, default=1)

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

    def _scrape_paginated(self, base_url: str, with_rating: bool, max_pages: int) -> pd.DataFrame:
        # Page 1 first — validates the profile exists and tells us how many
        # pages there are, so we never pay for pages past the end.
        first = self._get_page(base_url)
        if first is None:
            raise ScrapeError(f"Profile not found: {base_url}")
        all_rows: list[dict] = self._parse_items(first, with_rating)
        if not all_rows:
            return pd.DataFrame()

        n_pages = min(self._page_count(first), max_pages)
        urls = [f"{base_url}page/{p}/" for p in range(2, n_pages + 1)]
        with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
            for soup in ex.map(self._get_page, urls):
                if soup is not None:
                    all_rows.extend(self._parse_items(soup, with_rating))

        df = pd.DataFrame(all_rows)
        if df.empty:
            return df
        return df.drop_duplicates(subset=["name", "year"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ratings(self, username: str) -> pd.DataFrame:
        """A user's rated films, newest first (capped at the 720 most recent).
        Columns: name, year, rating.

        Matches the schema of csv_loader.load_ratings so the two input
        paths are interchangeable downstream.
        """
        df = self._scrape_paginated(f"{_BASE}/{username}/films/ratings/", with_rating=True,
                                     max_pages=_MAX_RATING_PAGES)
        if df.empty:
            raise ScrapeError(f"No rated films found for user '{username}'")
        logger.info("Scraped %d rated films for %s", len(df), username)
        return df

    def fetch_watchlist(self, username: str) -> pd.DataFrame:
        """A user's watchlist. Columns: name, year. Empty DataFrame if none."""
        try:
            df = self._scrape_paginated(f"{_BASE}/{username}/watchlist/", with_rating=False,
                                         max_pages=_MAX_WATCHLIST_PAGES)
        except ScrapeError:
            return pd.DataFrame(columns=["name", "year"])
        logger.info("Scraped %d watchlist films for %s", len(df), username)
        return df
