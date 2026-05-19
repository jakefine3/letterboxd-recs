"""CLI entrypoint: load ratings, enrich with TMDB, save processed dataset."""
import os
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ingestion.csv_loader import load_ratings
from src.ingestion.tmdb_client import TMDBClient

load_dotenv()

DEFAULT_RATINGS = Path("data/raw/letterboxd-jakefine-2026-05-18-13-16-utc/ratings.csv")
DEFAULT_OUTPUT = Path("data/processed/enriched_ratings.csv")
DEFAULT_CACHE = Path("data/processed/tmdb_cache.json")


def main():
    parser = argparse.ArgumentParser(description="Enrich Letterboxd ratings with TMDB data.")
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise SystemExit("TMDB_API_KEY not set — copy .env.example to .env and add your key.")

    print(f"Loading ratings from {args.ratings}")
    df = load_ratings(args.ratings)
    print(f"  {len(df)} rated films found")

    client = TMDBClient(api_key=api_key, cache_path=args.cache)
    print("Enriching with TMDB data (cached requests skipped)...")
    df = client.enrich_dataframe(df)

    missing = df["tmdb_id"].isna().sum()
    if missing:
        print(f"  Warning: {missing} film(s) could not be matched in TMDB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved enriched dataset to {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
