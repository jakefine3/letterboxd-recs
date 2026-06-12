"""FastAPI web interface for the Letterboxd recommendation system.

Run with:
    uvicorn main:app --reload

Uses the content-based ML model for fast in-browser inference (< 2s on cached data).
For the full hybrid ML+DL model with cross-validated ensemble weights, use the CLI:
    python scripts/recommend.py
"""
import io
import os
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
_POPULAR_PAGES = 5


def _require_api_key() -> str:
    key = os.getenv("TMDB_API_KEY", "")
    if not key:
        raise HTTPException(500, "TMDB_API_KEY not set in .env")
    return key


def _run_pipeline(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Enrich ratings, fetch TMDB candidates, train ML model, return top-n."""
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

    model = ContentBasedModel()
    model.fit(df)
    return model.recommend(candidates, n=n)


_UPLOAD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Letterboxd Recommender</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 640px;
           margin: 72px auto; padding: 0 24px; color: #1a1a1a; background: #f8f8f8; }
    h1 { font-size: 1.75rem; font-weight: 700; margin-bottom: 6px; }
    .sub { color: #555; margin: 0 0 32px; line-height: 1.5; }
    label { display: block; font-weight: 600; margin-bottom: 6px; }
    .field { margin-bottom: 20px; }
    input[type=file] { width: 100%; padding: 10px; border: 1px dashed #bbb;
                       border-radius: 6px; background: #fff; cursor: pointer; }
    input[type=number] { width: 90px; padding: 8px 10px; border: 1px solid #ccc;
                         border-radius: 6px; font-size: 1rem; }
    button { background: #111; color: #fff; border: none; padding: 11px 28px;
             border-radius: 6px; font-size: 1rem; cursor: pointer; font-weight: 600; }
    button:hover { background: #333; }
    .hint { font-size: 0.82rem; color: #888; margin-top: 10px; }
    code { background: #eee; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>Letterboxd Recommender</h1>
  <p class="sub">
    Upload your Letterboxd <code>ratings.csv</code> and get personalised film recommendations
    based on your taste profile.
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
    <p class="hint">
      First run fetches metadata from TMDB (~30s for ~300 films). Subsequent runs use
      the local cache and complete in seconds.
    </p>
  </form>
</body>
</html>
"""


def _results_html(recs: pd.DataFrame, n: int, total_rated: int) -> str:
    rows = "".join(
        f"<tr><td class='rank'>{i + 1}</td>"
        f"<td class='title'>{row['name']}</td>"
        f"<td class='year'>{int(row['year'])}</td>"
        f"<td class='score'>{row['predicted_rating']:.2f} / 5</td></tr>"
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
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 680px;
            margin: 72px auto; padding: 0 24px; color: #1a1a1a; background: #f8f8f8; }}
    h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 4px; }}
    .meta {{ color: #555; margin: 0 0 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff;
             border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    th {{ text-align: left; padding: 12px 16px; background: #111; color: #fff; font-size: 0.85rem; }}
    td {{ padding: 11px 16px; border-bottom: 1px solid #f0f0f0; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafafa; }}
    .rank {{ color: #aaa; font-size: 0.9rem; width: 36px; }}
    .year {{ color: #777; width: 60px; }}
    .score {{ font-weight: 600; width: 90px; }}
    .back {{ display: inline-block; margin-top: 24px; color: #111;
             text-decoration: none; font-weight: 600; }}
    .back:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>Your Top {n} Recommendations</h1>
  <p class="meta">Trained on {total_rated} rated films · content-based ML model</p>
  <table>
    <thead>
      <tr><th>#</th><th>Film</th><th>Year</th><th>Predicted Rating</th></tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <a class="back" href="/">← Try again</a>
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
