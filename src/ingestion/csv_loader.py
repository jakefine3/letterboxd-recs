import pandas as pd
from pathlib import Path


def load_ratings(path: str | Path) -> pd.DataFrame:
    """Load a Letterboxd ratings.csv export into a clean DataFrame.

    Returns columns: name, year, rating, letterboxd_uri
    """
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Name": "name",
        "Year": "year",
        "Rating": "rating",
        "Letterboxd URI": "letterboxd_uri",
    })

    df = df[["name", "year", "rating", "letterboxd_uri"]].copy()

    df["name"] = df["name"].str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # drop rows missing the essentials
    df = df.dropna(subset=["name", "year", "rating"]).reset_index(drop=True)

    return df
