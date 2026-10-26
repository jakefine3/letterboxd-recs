"""Snapshot the demo profile for the app's "Try a demo profile" button.

Scrapes the profile (run locally — Letterboxd lets residential IPs in),
enriches it with TMDB metadata and fetches the candidate pool, then saves
everything to data/demo/ so the demo needs no scrape or API calls at runtime.

Run with:  uv run python scripts/build_demo.py [username]
"""
import ast
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingestion.scraper import LetterboxdScraper
from src.ingestion.tmdb_client import TMDBClient

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "demo"


def top_genres(enriched, n: int = 3) -> list[str]:
    # same as app.top_genres, so the demo's candidate pool matches a live lookup
    liked = enriched[enriched["rating"] >= 4.0]["genres"]
    liked = liked.apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
    return liked.explode().value_counts().head(n).index.tolist()


def main(username: str = "jakefine"):
    load_dotenv()
    client = TMDBClient(api_key=os.environ["TMDB_API_KEY"])
    scraper = LetterboxdScraper()

    ratings = scraper.fetch_ratings(username)
    watchlist = scraper.fetch_watchlist(username)
    enriched = client.enrich_dataframe(ratings.copy()).dropna(subset=["tmdb_id"]).reset_index(drop=True)
    wl_enriched = client.enrich_dataframe(watchlist.copy()).dropna(subset=["tmdb_id"]).reset_index(drop=True)
    candidates = client.fetch_candidates(
        set(enriched["tmdb_id"].astype(int)),
        top_films=enriched[enriched["rating"] >= 4.5]["tmdb_id"].astype(int).tolist(),
        taste_genres=top_genres(enriched),
    )

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(DEMO_DIR / f"{username}_ratings.csv", index=False)
    watchlist.to_csv(DEMO_DIR / f"{username}_watchlist.csv", index=False)
    enriched.to_pickle(DEMO_DIR / f"{username}_enriched.pkl")
    wl_enriched.to_pickle(DEMO_DIR / f"{username}_wl_enriched.pkl")
    candidates.to_pickle(DEMO_DIR / f"{username}_candidates.pkl")
    print(f"Saved {len(ratings)} ratings, {len(watchlist)} watchlist, {len(candidates)} candidates to {DEMO_DIR}")


if __name__ == "__main__":
    main(*sys.argv[1:])
