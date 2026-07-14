"""Human-readable taste analysis from an enriched ratings DataFrame.

Unlike model feature importances (which surface internal features like
vote_avg_norm), everything here is computed directly from the user's ratings
and phrased for humans: favourite directors, genre leanings, recurring themes,
era and language preferences.
"""
import ast

import pandas as pd

from src.models.features import _KEYWORD_BLOCKLIST

_MIN_DIRECTOR_FILMS = 2
_MIN_GENRE_FILMS = 5
_MIN_THEME_FILMS = 4


def _as_list(series: pd.Series) -> pd.Series:
    def parse(val):
        if isinstance(val, list):
            return val
        if hasattr(val, "tolist"):
            return val.tolist()
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return []
    return series.apply(parse)


def top_directors(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Directors with >= 2 films, ranked by the user's average rating."""
    g = (
        df.dropna(subset=["director"])
        .groupby("director")["rating"]
        .agg(["mean", "count"])
        .reset_index()
    )
    g = g[g["count"] >= _MIN_DIRECTOR_FILMS]
    return g.sort_values(["mean", "count"], ascending=False).head(n)


def genre_leanings(df: pd.DataFrame, n: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genres the user rates above / below their personal average.

    Returns (loved, avoided) frames with columns: genre, mean, count, delta.
    """
    overall = df["rating"].mean()
    exploded = df.assign(genre=_as_list(df["genres"])).explode("genre")
    g = exploded.groupby("genre")["rating"].agg(["mean", "count"]).reset_index()
    g = g[g["count"] >= _MIN_GENRE_FILMS]
    g["delta"] = g["mean"] - overall
    g = g.sort_values("delta", ascending=False)
    return g.head(n), g.tail(n).iloc[::-1]


def recurring_themes(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Keywords that show up repeatedly in the user's 4+ star films."""
    high = df[df["rating"] >= 4.0]
    kws = _as_list(high["keywords"]).explode().dropna()
    kws = kws[~kws.isin(_KEYWORD_BLOCKLIST)]
    counts = kws.value_counts()
    counts = counts[counts >= _MIN_THEME_FILMS]
    return counts.head(n).reset_index().set_axis(["theme", "count"], axis=1)


def era_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average rating per decade (only decades with >= 5 films)."""
    d = df.copy()
    d["decade"] = (d["year"].astype(int) // 10) * 10
    g = d.groupby("decade")["rating"].agg(["mean", "count"]).reset_index()
    return g[g["count"] >= 5]


def language_split(df: pd.DataFrame) -> dict:
    """Compare english vs non-english average ratings."""
    eng = df[df["language"] == "en"]["rating"]
    non = df[df["language"] != "en"]["rating"]
    return {
        "english_mean": float(eng.mean()) if len(eng) else None,
        "foreign_mean": float(non.mean()) if len(non) else None,
        "foreign_count": int(len(non)),
        "foreign_lift": float(non.mean() - eng.mean()) if len(eng) and len(non) else 0.0,
    }


def persona(df: pd.DataFrame) -> str:
    """A short, fun label summarising the user's strongest taste signal."""
    lang = language_split(df)
    loved, _ = genre_leanings(df)
    top_genre = loved.iloc[0]["genre"] if not loved.empty else None

    if lang["foreign_lift"] > 0.3 and lang["foreign_count"] >= 10:
        return "World Cinema Devotee"

    labels = {
        "Horror": "Midnight Movie Fiend",
        "Documentary": "Truth Seeker",
        "Thriller": "Edge-of-Seat Enthusiast",
        "Crime": "Underworld Connoisseur",
        "Drama": "Serious Cinema Appreciator",
        "Science Fiction": "Future Dreamer",
        "Comedy": "Comedy Scholar",
        "Romance": "Hopeless Romantic",
        "Animation": "Animation Aficionado",
        "War": "History on Film Buff",
        "Western": "Frontier Wanderer",
        "Music": "Rhythm & Reels",
        "Mystery": "Puzzle Solver",
        "Fantasy": "Worldbuilding Wanderer",
    }
    if top_genre in labels:
        return labels[top_genre]
    return "Eclectic Watcher"


def contrarian_picks(
    df: pd.DataFrame, n: int = 5, min_votes: int = 1_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Films where the user's rating diverges most from TMDB consensus.

    Returns (loved_more, loved_less): films you rated above / below the crowd.
    Only considers films with enough TMDB votes to have a reliable consensus.
    """
    d = df.dropna(subset=["vote_average", "vote_count"]).copy()
    d = d[d["vote_count"] >= min_votes]
    d["tmdb_scaled"] = d["vote_average"] / 2.0
    d["delta"] = d["rating"] - d["tmdb_scaled"]
    cols = ["name", "year", "rating", "tmdb_scaled", "delta"]
    return d.nlargest(n, "delta")[cols], d.nsmallest(n, "delta")[cols]


def obscurity_profile(df: pd.DataFrame) -> dict:
    """How mainstream or underground is the user's taste?

    Measures the median vote_count of films the user rated 4+ stars.
    """
    loved = df[df["rating"] >= 4.0].dropna(subset=["vote_count"])
    if loved.empty:
        return {"label": "Unknown", "median_vote_count": 0, "pct_niche": 0.0}

    median_vc = float(loved["vote_count"].median())
    pct_niche = float((loved["vote_count"] < 10_000).mean() * 100)

    if median_vc < 5_000:
        label = "Underground Explorer"
    elif median_vc < 30_000:
        label = "Hidden Gem Hunter"
    elif median_vc < 100_000:
        label = "Discerning Viewer"
    elif median_vc < 500_000:
        label = "Mainstream Enjoyer"
    else:
        label = "Blockbuster Fan"

    return {"label": label, "median_vote_count": int(median_vc), "pct_niche": pct_niche}


def runtime_sweet_spot(df: pd.DataFrame) -> pd.DataFrame:
    """Average rating bucketed by runtime (buckets with < 5 films excluded)."""
    d = df.dropna(subset=["runtime"]).copy()
    bins = [0, 80, 100, 120, 150, float("inf")]
    labels = ["<80 min", "80–100", "100–120", "120–150", "150+ min"]
    d["bucket"] = pd.cut(d["runtime"], bins=bins, labels=labels)
    g = (
        d.groupby("bucket", observed=True)["rating"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"bucket": "length", "mean": "avg_rating"})
    )
    return g[g["count"] >= 5]


def critic_tendency(df: pd.DataFrame) -> dict:
    """How much harsher or easier the user rates vs. the TMDB aggregate."""
    d = df.dropna(subset=["vote_average"]).copy()
    d["tmdb_scaled"] = d["vote_average"] / 2.0
    delta = float((d["rating"] - d["tmdb_scaled"]).mean())
    return {"delta": delta, "n": len(d)}


def taste_summary(df: pd.DataFrame) -> list[str]:
    """Plain-English sentences describing the user's taste, strongest first."""
    lines = []

    directors = top_directors(df, n=3)
    if not directors.empty:
        names = [
            f"**{r['director']}** ({r['mean']:.1f}★ over {int(r['count'])} films)"
            for _, r in directors.iterrows()
        ]
        lines.append("Your most trusted directors: " + ", ".join(names) + ".")

    loved, avoided = genre_leanings(df)
    if not loved.empty:
        top = loved.iloc[0]
        lines.append(
            f"**{top['genre']}** is your sweet spot — you rate it "
            f"{top['delta']:+.2f}★ above your personal average."
        )
    if not avoided.empty and avoided.iloc[0]["delta"] < -0.2:
        bottom = avoided.iloc[0]
        lines.append(
            f"**{bottom['genre']}** rarely lands for you "
            f"({bottom['delta']:+.2f}★ vs your average)."
        )

    themes = recurring_themes(df, n=5)
    if not themes.empty:
        theme_list = ", ".join(f"*{t}*" for t in themes["theme"].head(4))
        lines.append(f"Themes that keep pulling you back: {theme_list}.")

    lang = language_split(df)
    if lang["foreign_lift"] > 0.2 and lang["foreign_count"] >= 5:
        lines.append(
            f"You rate non-English-language films {lang['foreign_lift']:+.2f}★ higher "
            f"than English-language ones — subtitles are no obstacle."
        )

    era = era_profile(df)
    if not era.empty:
        best = era.loc[era["mean"].idxmax()]
        lines.append(
            f"Your golden decade is the **{int(best['decade'])}s** "
            f"({best['mean']:.2f}★ average across {int(best['count'])} films)."
        )

    return lines
