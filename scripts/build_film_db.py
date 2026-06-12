"""One-time script to build the local film database from TMDB.

Run:
    python scripts/build_film_db.py [--pages 200]

Output: data/processed/film_db.parquet
200 pages × 20 films × 2 endpoints ≈ 8k unique films (~5 min on first run,
faster on reruns because combined-data hits the local cache).

The web app loads this parquet at startup for instant, API-free recommendations.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from src.ingestion.tmdb_client import TMDBClient

_OUT = Path("data/processed/film_db.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pages", type=int, default=200,
        help="Pages to fetch from popular + top_rated (20 films/page each, default 200)",
    )
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY", "")
    if not api_key:
        sys.exit("TMDB_API_KEY not set in .env")

    client = TMDBClient(api_key=api_key)
    df = client.fetch_film_db(n_pages=args.pages)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT, index=False)
    print(f"Saved {len(df)} films → {_OUT}")


if __name__ == "__main__":
    main()
