import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.themoviedb.org/3"
_DEFAULT_CACHE = Path("data/processed/tmdb_cache.json")
_TOP_CAST = 5
_MAX_WORKERS = 10  # safe against TMDB's 40 req/sec limit


class TMDBClient:
    def __init__(self, api_key: str, cache_path: str | Path = _DEFAULT_CACHE):
        self._api_key = api_key
        self._cache_path = Path(cache_path)
        self._cache = self._load_cache()
        self._lock = threading.Lock()  # protects cache writes across threads

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                return json.load(f)
        return {"search": {}, "combined": {}}

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
            return resp.json()
        except requests.RequestException as e:
            logger.warning("TMDB request failed (%s %s): %s", endpoint, params, e)
            return None

    def _search_id(self, name: str, year: int) -> int | None:
        key = f"{name}_{year}"

        with self._lock:
            if key in self._cache["search"]:
                return self._cache["search"][key]

        data = self._get("search/movie", {"query": name, "year": year})
        if not data or not data.get("results"):
            # retry without year — handles festival vs wide release year mismatches
            data = self._get("search/movie", {"query": name})

        tmdb_id = data["results"][0]["id"] if data and data.get("results") else None

        with self._lock:
            self._cache["search"][key] = tmdb_id
            self._save_cache()

        return tmdb_id

    def _fetch_combined(self, tmdb_id: int) -> dict | None:
        """Single request for details + credits + keywords via append_to_response."""
        sid = str(tmdb_id)

        with self._lock:
            if sid in self._cache["combined"]:
                return self._cache["combined"][sid]

        data = self._get(
            f"movie/{tmdb_id}",
            {"append_to_response": "credits,keywords"},
        )

        with self._lock:
            self._cache["combined"][sid] = data
            self._save_cache()

        return data

    # ------------------------------------------------------------------
    # Public enrichment
    # ------------------------------------------------------------------

    def enrich_film(self, name: str, year: int) -> dict:
        """Return a flat dict of TMDB metadata for a single film."""
        base = {
            "tmdb_id": None,
            "genres": [],
            "director": None,
            "cast": [],
            "keywords": [],
            "runtime": None,
            "language": None,
            "country": None,
        }

        tmdb_id = self._search_id(name, year)
        if tmdb_id is None:
            logger.warning("No TMDB match for '%s' (%s)", name, year)
            return base

        base["tmdb_id"] = tmdb_id

        data = self._fetch_combined(tmdb_id)
        if not data:
            return base

        base["genres"] = [g["name"] for g in data.get("genres", [])]
        base["runtime"] = data.get("runtime")
        base["language"] = data.get("original_language")
        base["country"] = (data.get("production_countries") or [{}])[0].get("iso_3166_1")

        credits = data.get("credits", {})
        directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
        base["director"] = directors[0] if directors else None
        base["cast"] = [p["name"] for p in credits.get("cast", [])[:_TOP_CAST]]

        base["keywords"] = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]

        return base

    def enrich_dataframe(self, df, verbose: bool = True) -> "pd.DataFrame":
        """Enrich a ratings DataFrame with TMDB metadata using parallel requests.

        Expects columns: name, year (as produced by csv_loader.load_ratings).
        Skips rows already in cache so reruns only fetch what's missing.
        """
        import pandas as pd

        tmdb_cols = ["tmdb_id", "genres", "director", "cast", "keywords",
                     "runtime", "language", "country"]
        for col in tmdb_cols:
            if col not in df.columns:
                df[col] = None

        # only fetch rows not yet enriched
        todo = [(i, row) for i, row in df.iterrows() if pd.isna(df.at[i, "tmdb_id"])]
        total = len(todo)

        if total == 0:
            print("All films already cached — nothing to fetch.")
            return df

        print(f"Fetching {total} films from TMDB ({_MAX_WORKERS} workers)...")
        completed = 0

        def fetch(i, row):
            return i, self.enrich_film(str(row["name"]), int(row["year"]))

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(fetch, i, row): i for i, row in todo}
            for future in as_completed(futures):
                i, enriched = future.result()
                for col in tmdb_cols:
                    df.at[i, col] = enriched[col]

                if verbose:
                    nonlocal_completed = futures  # just used as a counter proxy
                    completed += 1
                    print(f"  {completed}/{total}", end="\r")

        print(f"\nDone. {df['tmdb_id'].isna().sum()} film(s) not matched in TMDB.")
        return df
