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
                data = json.load(f)
            # migrate old cache format that stored details/credits/keywords separately
            if "combined" not in data and "details" in data:
                data = self._migrate_cache(data)
            return data
        return {"search": {}, "combined": {}}

    @staticmethod
    def _migrate_cache(old: dict) -> dict:
        new: dict = {"search": old.get("search", {}), "combined": {}}
        for sid, details in old.get("details", {}).items():
            combined = dict(details)
            combined["credits"] = old.get("credits", {}).get(sid, {"cast": [], "crew": []})
            combined["keywords"] = old.get("keywords", {}).get(sid, {"keywords": []})
            new["combined"][sid] = combined
        return new

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
            "overview": "",
            "vote_average": None,
            "vote_count": 0,
            "poster_path": None,
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
        base["overview"] = data.get("overview", "") or ""

        credits = data.get("credits", {})
        directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
        base["director"] = directors[0] if directors else None
        base["cast"] = [p["name"] for p in credits.get("cast", [])[:_TOP_CAST]]

        base["keywords"] = [k["name"] for k in data.get("keywords", {}).get("keywords", [])]
        base["vote_average"] = data.get("vote_average")
        base["vote_count"] = data.get("vote_count", 0)
        base["poster_path"] = data.get("poster_path")

        return base

    def _genre_name_to_id(self) -> dict:
        """Map TMDB genre names to IDs (needed for /discover filters)."""
        data = self._get("genre/movie/list")
        if not data:
            return {}
        return {g["name"]: g["id"] for g in data.get("genres", [])}

    def fetch_candidates(
        self,
        seen_ids: set,
        n_popular_pages: int = 5,
        top_films: list | None = None,
        taste_genres: list | None = None,
        n_niche_pages: int = 3,
    ) -> "pd.DataFrame":
        """Fetch unseen candidate films from TMDB.

        Four sources:
          - popular + trending (broad, well-known pool)
          - similar-to-favorites (personalised pool)
          - niche discovery: acclaimed films (vote_average >= 7) in the user's
            top genres with a capped vote_count, so the pool includes quality
            films outside the blockbuster mainstream

        Returns a DataFrame with the same columns as enrich_dataframe output,
        minus the rating field (candidates are unrated).
        """
        import pandas as pd

        candidate_ids: set = set()

        for page in range(1, n_popular_pages + 1):
            data = self._get("movie/popular", {"page": page})
            if data:
                candidate_ids.update(m["id"] for m in data.get("results", []))

        data = self._get("trending/movie/week")
        if data:
            candidate_ids.update(m["id"] for m in data.get("results", []))

        for film_id in (top_films or [])[:10]:
            data = self._get(f"movie/{film_id}/similar")
            if data:
                candidate_ids.update(m["id"] for m in data.get("results", []))

        if taste_genres:
            genre_map = self._genre_name_to_id()
            genre_ids = [str(genre_map[g]) for g in taste_genres if g in genre_map]
            if genre_ids:
                for page in range(1, n_niche_pages + 1):
                    data = self._get("discover/movie", {
                        "with_genres": "|".join(genre_ids),
                        "vote_average.gte": 7.0,
                        "vote_count.gte": 200,
                        "vote_count.lte": 3000,  # cap filters out blockbusters
                        "sort_by": "vote_average.desc",
                        "page": page,
                    })
                    if data:
                        candidate_ids.update(m["id"] for m in data.get("results", []))

        candidate_ids -= set(seen_ids)
        candidate_ids.discard(None)

        print(f"  Fetching details for {len(candidate_ids)} candidates...")

        rows = []
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(self._fetch_combined, tid): tid for tid in candidate_ids}
            for future in as_completed(futures):
                tid = futures[future]
                data = future.result()
                if not data:
                    continue
                name = data.get("title") or data.get("original_title")
                release = (data.get("release_date") or "")[:4]
                year = int(release) if release.isdigit() else None
                if not name or not year:
                    continue

                credits = data.get("credits", {})
                directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]

                rows.append({
                    "name": name,
                    "year": year,
                    "tmdb_id": tid,
                    "genres": [g["name"] for g in data.get("genres", [])],
                    "director": directors[0] if directors else None,
                    "cast": [p["name"] for p in credits.get("cast", [])[:_TOP_CAST]],
                    "keywords": [k["name"] for k in data.get("keywords", {}).get("keywords", [])],
                    "runtime": data.get("runtime"),
                    "language": data.get("original_language"),
                    "country": (data.get("production_countries") or [{}])[0].get("iso_3166_1"),
                    "overview": data.get("overview", "") or "",
                    "vote_average": data.get("vote_average"),
                    "vote_count": data.get("vote_count", 0),
                    "poster_path": data.get("poster_path"),
                })

        self._save_cache()
        return pd.DataFrame(rows)

    def fetch_film_db(self, n_pages: int = 200) -> "pd.DataFrame":
        """Build a large film corpus for offline use (popular + top-rated pages).

        Run once via scripts/build_film_db.py. The result is saved as a parquet
        file that the web app loads at startup for instant, API-free inference.
        """
        import pandas as pd

        film_ids: set = set()
        for endpoint in ("movie/popular", "movie/top_rated"):
            for page in range(1, n_pages + 1):
                data = self._get(endpoint, {"page": page})
                if data:
                    film_ids.update(m["id"] for m in data.get("results", []))
                if page % 50 == 0:
                    print(f"  {endpoint}: {page}/{n_pages} pages")

        for period in ("day", "week"):
            data = self._get(f"trending/movie/{period}")
            if data:
                film_ids.update(m["id"] for m in data.get("results", []))

        print(f"Found {len(film_ids)} unique IDs. Fetching details...")
        _BUILD_WORKERS = 20
        rows = []
        done = 0

        with ThreadPoolExecutor(max_workers=_BUILD_WORKERS) as executor:
            futures = {executor.submit(self._fetch_combined, tid): tid for tid in film_ids}
            for future in as_completed(futures):
                done += 1
                if done % 200 == 0:
                    print(f"  {done}/{len(film_ids)}", end="\r")
                data = future.result()
                if not data:
                    continue
                name = data.get("title") or data.get("original_title")
                release = (data.get("release_date") or "")[:4]
                year = int(release) if release.isdigit() else None
                if not name or not year:
                    continue
                credits = data.get("credits", {})
                directors = [p["name"] for p in credits.get("crew", []) if p.get("job") == "Director"]
                rows.append({
                    "tmdb_id": futures[future],
                    "name": name,
                    "year": year,
                    "name_key": f"{name.lower().strip()}_{year}",
                    "genres": [g["name"] for g in data.get("genres", [])],
                    "director": directors[0] if directors else None,
                    "cast": [p["name"] for p in credits.get("cast", [])[:_TOP_CAST]],
                    "keywords": [k["name"] for k in data.get("keywords", {}).get("keywords", [])],
                    "runtime": data.get("runtime"),
                    "language": data.get("original_language"),
                    "country": (data.get("production_countries") or [{}])[0].get("iso_3166_1"),
                    "overview": data.get("overview", "") or "",
                    "vote_average": data.get("vote_average"),
                    "vote_count": data.get("vote_count", 0),
                    "poster_path": data.get("poster_path"),
                })

        self._save_cache()
        print(f"\nBuilt DB: {len(rows)} films.")
        return pd.DataFrame(rows)

    def enrich_dataframe(self, df, verbose: bool = True) -> "pd.DataFrame":
        """Enrich a ratings DataFrame with TMDB metadata using parallel requests.

        Expects columns: name, year (as produced by csv_loader.load_ratings).
        Skips rows already in cache so reruns only fetch what's missing.
        """
        import pandas as pd

        tmdb_cols = ["tmdb_id", "genres", "director", "cast", "keywords",
                     "runtime", "language", "country", "overview", "vote_average",
                     "vote_count", "poster_path"]
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

        self._save_cache()
        print(f"\nDone. {df['tmdb_id'].isna().sum()} film(s) not matched in TMDB.")
        return df
