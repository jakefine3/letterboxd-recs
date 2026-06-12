"""CLI entry point: train the hybrid model on rated films and score TMDB candidates.

Usage:
    python scripts/recommend.py                     # uses data/processed/enriched_ratings.csv
    python scripts/recommend.py --n 30              # return top-30
    python scripts/recommend.py --ratings <path>    # enrich raw CSV first
"""
import os
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.csv_loader import load_ratings
from src.ingestion.tmdb_client import TMDBClient
from src.models.hybrid import HybridModel

load_dotenv()

DEFAULT_ENRICHED = Path("data/processed/enriched_ratings.csv")
DEFAULT_CACHE = Path("data/processed/tmdb_cache.json")


def main():
    parser = argparse.ArgumentParser(description="Generate movie recommendations from Letterboxd data.")
    parser.add_argument(
        "--enriched", type=Path, default=DEFAULT_ENRICHED,
        help="Path to enriched_ratings.csv (default: data/processed/enriched_ratings.csv)",
    )
    parser.add_argument(
        "--ratings", type=Path, default=None,
        help="Raw ratings.csv export — used only if --enriched doesn't exist",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--n", type=int, default=20, help="Number of recommendations to return")
    parser.add_argument(
        "--popular-pages", type=int, default=5,
        help="Pages of TMDB popular films to use as candidates (20 films per page)",
    )
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY not set — add it to .env")

    # load enriched ratings or enrich from raw CSV
    if args.enriched.exists():
        print(f"Loading {args.enriched}")
        df = pd.read_csv(args.enriched)
    elif args.ratings and args.ratings.exists():
        print(f"Enriching {args.ratings} with TMDB metadata...")
        df = load_ratings(args.ratings)
        client = TMDBClient(api_key=api_key, cache_path=args.cache)
        df = client.enrich_dataframe(df)
        args.enriched.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.enriched, index=False)
        print(f"Saved to {args.enriched}")
    else:
        raise SystemExit(
            f"No enriched file at {args.enriched} and no --ratings path provided.\n"
            "Run:  python scripts/run_recommendations.py  to enrich your ratings first."
        )

    df = df.dropna(subset=["rating"]).reset_index(drop=True)
    print(f"{len(df)} rated films")

    # fetch unseen candidates from TMDB
    client = TMDBClient(api_key=api_key, cache_path=args.cache)
    seen_ids = set(df["tmdb_id"].dropna().astype(int).tolist())
    top_films = df.nlargest(10, "rating")["tmdb_id"].dropna().astype(int).tolist()

    print("Fetching candidates from TMDB...")
    candidates = client.fetch_candidates(
        seen_ids=seen_ids,
        n_popular_pages=args.popular_pages,
        top_films=top_films,
    )

    if candidates.empty:
        raise SystemExit("No candidates returned — check your TMDB_API_KEY.")

    candidates = candidates.dropna(subset=["overview"]).reset_index(drop=True)
    print(f"{len(candidates)} candidate films\n")

    print("Training hybrid model...")
    model = HybridModel()
    model.fit(df)

    print("Scoring candidates...")
    recs = model.recommend(candidates, n=args.n)

    print(f"\nTop {args.n} recommendations")
    print("-" * 55)
    for _, row in recs.iterrows():
        print(f"  {row['predicted_rating']:.2f}  {row['name']} ({int(row['year'])})")


if __name__ == "__main__":
    main()
