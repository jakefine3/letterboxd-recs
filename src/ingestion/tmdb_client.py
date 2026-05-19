import json
import time
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.themoviedb.org/3"
_DEFAULT_CACHE = Path("data/processed/tmdb_cache.json")
_TOP_CAST = 5
_REQUEST_DELAY = 0.1  # seconds between API calls


class TMDBClient:
    def __init__(self, api_key: str, cache_path: str | Path = _DEFAULT_CACHE):
        self._api_key = api_key
        self._cache_path = Path(cache_path)
        self._cache = self._load_cache()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {"search": {}, "details": {}, "credits": {}, "keywords": {}}

    def _save_cache(self) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f)

    # ------------------------------------------------------------------
    # Raw API calls
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> dict | None:
        url = f"{_BASE_URL}/{endpoint}"
        p = {"api_key": self._api_key, **(params or {})}
        try:
            resp = requests.get(url, params=p, timeout=10)
            resp.raise_for_status()
            time.sleep(_REQUEST_DELAY)
            return resp.json()
        except requests.RequestException as e:
            logger.warning("TMDB request failed (%s %s): %s", endpoint, params, e)
            return None

    def _search_id(self, name: str, year: int) -> int | None:
        key = f"{name}_{year}"
        if key in self._cache["search"]:
            return self._cache["search"][key]

        data = self._get("search/movie", {"query": name, "year": year})
        if not data or not data.get("results"):
            # retry without year constraint — handles year mismatches between
            # Letterboxd and TMDB (e.g. festival year vs release year)
            data = self._get("search/movie", {"query": name})

        tmdb_id = None
        if data and data.get("results"):
            tmdb_id = data["results"][0]["id"]

        self._cache["search"][key] = tmdb_id
        self._save_cache()
        return tmdb_id

    def _fetch_details(self, tmdb_id: int) -> dict | None:
        sid = str(tmdb_id)
        if sid in self._cache["details"]:
            return self._cache["details"][sid]

        data = self._get(f"movie/{tmdb_id}")
        self._cache["details"][sid] = data
        self._save_cache()
        return data

    def _fetch_credits(self, tmdb_id: int) -> dict | None:
        sid = str(tmdb_id)
        if sid in self._cache["credits"]:
            return self._cache["credits"][sid]

        data = self._get(f"movie/{tmdb_id}/credits")
        self._cache["credits"][sid] = data
        self._save_cache()
        return data

    def _fetch_keywords(self, tmdb_id: int) -> dict | None:
        sid = str(tmdb_id)
        if sid in self._cache["keywords"]:
            return self._cache["keywords"][sid]

        data = self._get(f"movie/{tmdb_id}/keywords")
        self._cache["keywords"][sid] = data
        self._save_cache()
        return data

    # ------------------------------------------------------------------
    # Public enrichment
    # ------------------------------------------------------------------

    def enrich_film(self, name: str, year: int) -> dict:
        """Return a flat dict of TMDB metadata for a single film.

        Returns an empty dict (with only tmdb_id=None) if the film can't be
        found, so callers can still build a row for it.
        """
        base = {
            "tmdb_id": None,
            "genres": [],
            "director": None,
            "cast": [],
            "keywords": [],
            "runtime": None,
            "language": None,
            "country": None,
            "overview": None,
        }

        tmdb_id = self._search_id(name, year)
        if tmdb_id is None:
            logger.warning("No TMDB match for '%s' (%s)", name, year)
            return base

        base["tmdb_id"] = tmdb_id

        details = self._fetch_details(tmdb_id)
        if details:
            base["genres"] = [g["name"] for g in details.get("genres", [])]
            base["runtime"] = details.get("runtime")
            base["language"] = details.get("original_language")
            base["country"] = (details.get("production_countries") or [{}])[0].get("iso_3166_1")
            base["overview"] = details.get("overview")

        credits = self._fetch_credits(tmdb_id)
        if credits:
            directors = [
                p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"
            ]
            base["director"] = directors[0] if directors else None
            base["cast"] = [p["name"] for p in credits.get("cast", [])[:_TOP_CAST]]

        keywords = self._fetch_keywords(tmdb_id)
        if keywords:
            base["keywords"] = [k["name"] for k in keywords.get("keywords", [])]

        return base

    def enrich_dataframe(self, df, verbose: bool = True) -> "pd.DataFrame":
        """Add TMDB columns to a ratings DataFrame in-place.

        Expects columns: name, year (as produced by csv_loader.load_ratings).
        Skips rows that already have a tmdb_id populated so reruns are cheap.
        """
        import pandas as pd

        tmdb_cols = ["tmdb_id", "genres", "director", "cast", "keywords",
                     "runtime", "language", "country", "overview"]
        for col in tmdb_cols:
            if col not in df.columns:
                df[col] = None

        total = len(df)
        for i, row in df.iterrows():
            if pd.notna(df.at[i, "tmdb_id"]):
                continue  # already enriched

            if verbose:
                print(f"[{i + 1}/{total}] {row['name']} ({row['year']})", end="\r")

            enriched = self.enrich_film(str(row["name"]), int(row["year"]))
            for col in tmdb_cols:
                df.at[i, col] = enriched[col]

        if verbose:
            print()  # newline after progress output

        return df
