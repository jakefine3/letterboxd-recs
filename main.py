"""FastAPI web interface for the Letterboxd recommendation system.

Run with:
    uvicorn main:app --reload

Loads data/processed/film_db.parquet at startup for instant, API-free inference.
Build the DB once with:
    python scripts/build_film_db.py

Falls back to live TMDB API calls (~30s first run) if the DB is not present.
"""
import io
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.ingestion.csv_loader import load_ratings
from src.ingestion.tmdb_client import TMDBClient
from src.models.content_based import ContentBasedModel

load_dotenv()

app = FastAPI(title="Letterboxd Recommender", docs_url=None, redoc_url=None)

_DEFAULT_CACHE = Path("data/processed/tmdb_cache.json")
_FILM_DB_PATH = Path("data/processed/film_db.parquet")
_POPULAR_PAGES = 5
_LANG_MIN_RATINGS = 3   # min rated films in a language to include it in candidates
_GENRE_MIN_RATINGS = 2  # min rated films in a genre to include it in candidates
_MIN_VOTE_COUNT = 25    # ignore films with fewer votes than this
_MIN_VOTE_AVERAGE = 5.0 # ignore films rated below this on TMDB (0–10 scale)

_film_db: pd.DataFrame | None = None
_film_db_lookup: dict | None = None  # name_key -> row dict


def _to_list(v) -> list:
    if isinstance(v, list):
        return v
    if hasattr(v, "tolist"):  # numpy ndarray
        return v.tolist()
    return []


@app.on_event("startup")
async def _load_film_db():
    global _film_db, _film_db_lookup
    if _FILM_DB_PATH.exists():
        df = pd.read_parquet(_FILM_DB_PATH)
        # pyarrow stores list columns as numpy arrays — convert to Python lists
        for col in ("genres", "keywords", "cast"):
            df[col] = df[col].apply(_to_list)
        # keep the most-voted entry when name+year collides (remakes, re-releases)
        df = df.sort_values("vote_count", ascending=False).drop_duplicates("name_key")
        _film_db = df.reset_index(drop=True)
        _film_db_lookup = _film_db.set_index("name_key").to_dict("index")
        print(f"Loaded film DB: {len(_film_db)} films")
    else:
        print("film_db.parquet not found — run scripts/build_film_db.py for instant inference")


def _require_api_key() -> str:
    key = os.getenv("TMDB_API_KEY", "")
    if not key:
        raise HTTPException(500, "TMDB_API_KEY not set in .env")
    return key


def _match_ratings_to_db(df: pd.DataFrame) -> pd.DataFrame:
    """Match user's rated films to the film DB by name+year key."""
    rows = []
    for _, r in df.iterrows():
        key = f"{str(r['name']).lower().strip()}_{int(r['year'])}"
        match = _film_db_lookup.get(key)
        if match is None:
            for dy in (1, -1):
                alt = f"{str(r['name']).lower().strip()}_{int(r['year']) + dy}"
                match = _film_db_lookup.get(alt)
                if match:
                    break
        if match:
            row = dict(match)
            row["rating"] = r["rating"]
            rows.append(row)
    return pd.DataFrame(rows)


def _filter_by_language(rated: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep only candidates in languages the user has meaningfully engaged with."""
    lang_counts = rated["language"].dropna().value_counts()
    allowed = set(lang_counts[lang_counts >= _LANG_MIN_RATINGS].index)
    if not allowed:
        return candidates
    return candidates[candidates["language"].isin(allowed)]


def _filter_by_genre(rated: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep candidates whose primary genre the user has meaningfully engaged with.

    Uses TMDB's first genre (most dominant) rather than any-match, so a kids film
    with genres [Animation, Adventure, Comedy] is rejected when the user hasn't
    rated Animation films, even if Adventure/Comedy are in their profile.
    """
    from collections import Counter
    all_genres: list[str] = [
        g for gs in rated["genres"] for g in (gs if isinstance(gs, list) else [])
    ]
    counts = Counter(all_genres)
    allowed = {g for g, c in counts.items() if c >= _GENRE_MIN_RATINGS}
    if not allowed:
        return candidates
    mask = candidates["genres"].apply(
        lambda gs: isinstance(gs, list) and bool(gs) and gs[0] in allowed
    )
    return candidates[mask]


def _filter_by_keyword_overlap(rated: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop candidates with zero keyword overlap with the user's highly-rated films.

    The check is genre-specific: an Animation-primary candidate must share keywords
    with the user's Animation-primary 4+ star films — not just any 4+ star film.
    This prevents generic keywords like 'friendship' or 'police' on unrelated
    high-rated films from vouching for kids movies.
    """
    high = rated[rated["rating"] >= 4.0] if "rating" in rated.columns else rated
    if high.empty:
        return candidates

    def primary(gs):
        return gs[0] if isinstance(gs, list) and gs else None

    high = high.copy()
    high["_pg"] = high["genres"].apply(primary)

    genre_kws: dict[str, set[str]] = {}
    for pg, grp in high.groupby("_pg"):
        genre_kws[pg] = {
            kw for kws in grp["keywords"] for kw in (kws if isinstance(kws, list) else [])
        }

    global_kws: set[str] = {kw for s in genre_kws.values() for kw in s}
    if not global_kws:
        return candidates

    def passes(row) -> bool:
        gs = row["genres"] if isinstance(row["genres"], list) else []
        pg = gs[0] if gs else None
        film_kws = set(row["keywords"] if isinstance(row["keywords"], list) else [])
        taste = genre_kws.get(pg, global_kws)
        return bool(film_kws & taste)

    return candidates[candidates.apply(passes, axis=1)]


def _filter_by_quality(candidates: pd.DataFrame) -> pd.DataFrame:
    """Drop films that are obscure or widely panned based on TMDB crowd ratings."""
    mask = (
        candidates["vote_count"].fillna(0) >= _MIN_VOTE_COUNT
    ) & (
        candidates["vote_average"].fillna(0) >= _MIN_VOTE_AVERAGE
    )
    return candidates[mask]


def _run_pipeline_fast(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Instant inference using the pre-built film DB — no API calls."""
    rated = _match_ratings_to_db(df)
    if rated.empty:
        raise HTTPException(500, "None of your rated films were found in the local film DB.")
    rated = rated.dropna(subset=["rating"]).reset_index(drop=True)

    seen_ids = set(rated["tmdb_id"].dropna().astype(int))
    candidates = _film_db[~_film_db["tmdb_id"].isin(seen_ids)].copy()
    candidates = _filter_by_language(rated, candidates)
    candidates = _filter_by_genre(rated, candidates)
    candidates = _filter_by_keyword_overlap(rated, candidates)
    candidates = _filter_by_quality(candidates)

    if candidates.empty:
        raise HTTPException(500, "No candidates after filtering.")

    model = ContentBasedModel()
    model.fit(rated)
    return model.recommend(candidates, n=n)


def _run_pipeline_slow(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Enrich ratings and fetch candidates live from TMDB (~30s first run)."""
    api_key = _require_api_key()
    client = TMDBClient(api_key=api_key, cache_path=_DEFAULT_CACHE)

    df = client.enrich_dataframe(df, verbose=False)
    df = df.dropna(subset=["rating"]).reset_index(drop=True)

    seen_ids = set(df["tmdb_id"].dropna().astype(int).tolist())
    top_films = df.nlargest(10, "rating")["tmdb_id"].dropna().astype(int).tolist()

    candidates = client.fetch_candidates(
        seen_ids=seen_ids,
        n_popular_pages=_POPULAR_PAGES,
        top_films=top_films,
    )
    if candidates.empty:
        raise HTTPException(500, "No candidates fetched — check TMDB_API_KEY")

    candidates = _filter_by_genre(df, candidates)
    candidates = _filter_by_keyword_overlap(df, candidates)
    candidates = _filter_by_quality(candidates)

    model = ContentBasedModel()
    model.fit(df)
    return model.recommend(candidates, n=n)


def _lbd_url(name: str, year: int) -> str:
    """Best-effort Letterboxd film URL from title + year."""
    slug = re.sub(r"[^a-z0-9\s-]", "", name.lower())
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = re.sub(r"-+", "-", slug)
    return f"https://letterboxd.com/film/{slug}/"


def _run_pipeline(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if _film_db is not None:
        return _run_pipeline_fast(df, n)
    return _run_pipeline_slow(df, n)


_UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Letterboxd Recommender</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 560px;
           margin: 80px auto; padding: 0 24px; color: #99aabb; background: #14181c; }
    .logo { display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }
    .logo-mark { width: 36px; height: 36px; border-radius: 50%;
                 background: conic-gradient(#00b020 0deg 120deg, #ff8000 120deg 240deg, #40bcf4 240deg 360deg); }
    .logo-text { font-size: 1.25rem; font-weight: 700; color: #e9e9e9; letter-spacing: .5px; }
    h1 { font-size: 1.6rem; font-weight: 700; color: #e9e9e9; margin: 0 0 8px; }
    .sub { color: #678; margin: 0 0 32px; line-height: 1.6; font-size: 0.95rem; }
    label { display: block; font-weight: 600; color: #c8d8e8; margin-bottom: 6px; font-size: 0.9rem; text-transform: uppercase; letter-spacing: .5px; }
    .field { margin-bottom: 20px; }
    input[type=file] { width: 100%; padding: 12px; border: 1px dashed #2c3440;
                       border-radius: 6px; background: #1e2a38; color: #99aabb; cursor: pointer; font-size: 0.9rem; }
    input[type=file]:hover { border-color: #00e054; }
    input[type=number] { width: 90px; padding: 9px 12px; border: 1px solid #2c3440;
                         border-radius: 6px; font-size: 1rem; background: #1e2a38; color: #e9e9e9; }
    button { background: #00e054; color: #14181c; border: none; padding: 11px 28px;
             border-radius: 6px; font-size: 0.95rem; cursor: pointer; font-weight: 700; letter-spacing: .3px; }
    button:hover { background: #00c040; }
    .hint { font-size: 0.8rem; color: #445566; margin-top: 12px; }
    code { background: #1e2a38; color: #00e054; padding: 1px 6px; border-radius: 3px; font-size: 0.88em; }
  </style>
</head>
<body>
  <div class="logo">
    <div class="logo-mark"></div>
    <span class="logo-text">Letterboxd Recommender</span>
  </div>
  <h1>What to watch next?</h1>
  <p class="sub">
    Upload your Letterboxd <code>ratings.csv</code> and get personalised film
    recommendations trained on your taste profile.
  </p>
  <form method="post" action="/recommend" enctype="multipart/form-data">
    <div class="field">
      <label for="file">ratings.csv</label>
      <input type="file" id="file" name="file" accept=".csv" required>
    </div>
    <div class="field">
      <label for="n">Number of recommendations</label>
      <input type="number" id="n" name="n" value="20" min="1" max="100">
    </div>
    <button type="submit">Get Recommendations</button>
    <p class="hint">Results are usually ready in under 2 seconds.</p>
  </form>
</body>
</html>
"""


def _results_html(recs: pd.DataFrame, n: int, total_rated: int) -> str:
    rows = "".join(
        f"<tr>"
        f"<td class='rank'>{i + 1}</td>"
        f"<td class='title'><a href='{_lbd_url(row['name'], int(row['year']))}' target='_blank' rel='noopener'>{row['name']}</a></td>"
        f"<td class='year'>{int(row['year'])}</td>"
        f"<td class='score'>{row['predicted_rating']:.2f}<span class='denom'> / 5</span></td>"
        f"</tr>"
        for i, (_, row) in enumerate(recs.iterrows())
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Your Recommendations</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 700px;
            margin: 72px auto; padding: 0 24px; color: #99aabb; background: #14181c; }}
    .logo {{ display: flex; align-items: center; gap: 10px; margin-bottom: 28px; }}
    .logo-mark {{ width: 28px; height: 28px; border-radius: 50%;
                  background: conic-gradient(#00b020 0deg 120deg, #ff8000 120deg 240deg, #40bcf4 240deg 360deg); }}
    .logo-text {{ font-size: 1rem; font-weight: 700; color: #e9e9e9; letter-spacing: .5px; }}
    h1 {{ font-size: 1.6rem; font-weight: 700; color: #e9e9e9; margin: 0 0 6px; }}
    .meta {{ color: #556677; margin: 0 0 24px; font-size: 0.88rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ text-align: left; padding: 10px 14px; color: #556677; font-size: 0.78rem;
          text-transform: uppercase; letter-spacing: .6px; border-bottom: 1px solid #2c3440; }}
    td {{ padding: 12px 14px; border-bottom: 1px solid #1e2a38; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #1a2330; }}
    .rank {{ color: #445566; font-size: 0.85rem; width: 32px; }}
    .title a {{ color: #e9e9e9; text-decoration: none; font-weight: 500; }}
    .title a:hover {{ color: #00e054; }}
    .year {{ color: #556677; width: 56px; font-size: 0.9rem; }}
    .score {{ font-weight: 700; color: #00e054; width: 80px; }}
    .denom {{ color: #445566; font-weight: 400; font-size: 0.85em; }}
    .back {{ display: inline-block; margin-top: 28px; color: #678;
             text-decoration: none; font-size: 0.9rem; }}
    .back:hover {{ color: #00e054; }}
  </style>
</head>
<body>
  <div class="logo">
    <div class="logo-mark"></div>
    <span class="logo-text">Letterboxd Recommender</span>
  </div>
  <h1>Your Top {n} Films to Watch</h1>
  <p class="meta">Trained on {total_rated} rated films &middot; content-based ML model</p>
  <table>
    <thead>
      <tr><th>#</th><th>Film</th><th>Year</th><th>Predicted</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <a class="back" href="/">&#8592; Try again</a>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_UPLOAD_HTML)


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(file: UploadFile = File(...), n: int = Form(20)):
    contents = await file.read()
    try:
        df = load_ratings(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse ratings CSV: {exc}")

    total_rated = len(df)
    recs = _run_pipeline(df, n=n)
    return HTMLResponse(_results_html(recs, n, total_rated))
