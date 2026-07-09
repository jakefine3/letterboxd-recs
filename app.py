"""Letterboxd Recommendations — Streamlit app.

Run with:  uv run streamlit run app.py

Input: a public Letterboxd username (scraped live). The CSV path still exists
in the backend (src/ingestion/csv_loader.py) for offline experimentation.
"""
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion.scraper import LetterboxdScraper, ScrapeError
from src.ingestion.tmdb_client import TMDBClient
from src.models.content_based import ContentBasedModel
from src.models import taste

load_dotenv()

_POSTER_BASE = "https://image.tmdb.org/t/p/w342"
_EMPTY_POSTER = "https://s.ltrbxd.com/static/img/empty-poster-70-BSf-Pjrh.png"

st.set_page_config(page_title="Letterboxd Recs", page_icon="🎬", layout="wide")

# ----------------------------------------------------------------------
# Letterboxd-style theming
# ----------------------------------------------------------------------

_LB_CSS = """
<style>
/* Letterboxd palette: #14181c bg, #2c3440 cards, #9ab text,
   #00e054 green, #40bcf4 blue, #ff8000 orange */

h1, h2, h3 { color: #fff !important; letter-spacing: -0.5px; }

/* tri-dot logo accent */
.lb-dots { font-size: 1.6rem; letter-spacing: 2px; }
.lb-dot-orange { color: #ff8000; }
.lb-dot-green  { color: #00e054; }
.lb-dot-blue   { color: #40bcf4; }

/* movie cards */
.lb-card {
    background: #2c3440;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
    transition: transform .15s ease, box-shadow .15s ease;
    height: 100%;
}
.lb-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 20px rgba(0, 224, 84, .25);
}
.lb-card img {
    border-radius: 4px;
    width: 100%;
    border: 1px solid #456;
}
.lb-card .title {
    color: #fff;
    font-weight: 700;
    font-size: 0.85rem;
    margin: 8px 0 2px 0;
    line-height: 1.2;
}
.lb-card .year { color: #678; font-size: 0.75rem; }
.lb-card .stars { color: #00e054; font-size: 0.9rem; letter-spacing: 1px; }
.lb-card .wl-badge {
    display: inline-block;
    background: rgba(255, 128, 0, .15);
    color: #ff8000;
    border: 1px solid #ff8000;
    border-radius: 3px;
    font-size: .6rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 1px 6px;
    margin-top: 4px;
}

/* taste persona banner */
.lb-persona {
    background: linear-gradient(135deg, #2c3440 0%, #14181c 100%);
    border: 1px solid #456;
    border-left: 4px solid #00e054;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 20px;
}
.lb-persona .label { color: #678; font-size: .8rem; text-transform: uppercase; letter-spacing: 2px; }
.lb-persona .value { color: #00e054; font-size: 1.6rem; font-weight: 800; }

/* stat chips */
.lb-chip {
    display: inline-block;
    background: #2c3440;
    border: 1px solid #456;
    color: #9ab;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px;
    font-size: .85rem;
}
.lb-chip b { color: #40bcf4; }
</style>
"""
st.markdown(_LB_CSS, unsafe_allow_html=True)


def stars(rating: float) -> str:
    """Render a rating as Letterboxd-style stars, e.g. 3.5 -> ★★★½."""
    full = int(rating)
    half = rating - full >= 0.5
    return "★" * full + ("½" if half else "")


# ----------------------------------------------------------------------
# Cached pipeline stages
# ----------------------------------------------------------------------

@st.cache_resource
def get_client() -> TMDBClient:
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        st.error("TMDB_API_KEY not set — copy .env.example to .env and add your key.")
        st.stop()
    return TMDBClient(api_key=api_key)


@st.cache_data(show_spinner=False)
def scrape_profile(username: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scraper = LetterboxdScraper()
    return scraper.fetch_ratings(username), scraper.fetch_watchlist(username)


@st.cache_data(show_spinner=False)
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    enriched = get_client().enrich_dataframe(df.copy(), verbose=False)
    return enriched.dropna(subset=["tmdb_id"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def fetch_candidates(seen_ids: frozenset, top_films: tuple, taste_genres: tuple) -> pd.DataFrame:
    return get_client().fetch_candidates(
        set(seen_ids),
        top_films=list(top_films),
        taste_genres=list(taste_genres),
    )


def top_genres(enriched_df: pd.DataFrame, n: int = 3) -> list[str]:
    import ast
    liked = enriched_df[enriched_df["rating"] >= 4.0]["genres"]
    liked = liked.apply(lambda v: v if isinstance(v, list) else ast.literal_eval(v))
    return liked.explode().value_counts().head(n).index.tolist()


def poster_url(path) -> str:
    return f"{_POSTER_BASE}{path}" if isinstance(path, str) and path else _EMPTY_POSTER


def poster_grid(films: pd.DataFrame, n_cols: int = 5):
    rows = [films.iloc[i:i + n_cols] for i in range(0, len(films), n_cols)]
    for chunk in rows:
        cols = st.columns(n_cols)
        for col, (_, movie) in zip(cols, chunk.iterrows()):
            with col:
                badge = '<div class="wl-badge">ON YOUR WATCHLIST</div>' if movie.get("on_watchlist") else ""
                st.markdown(
                    f"""
                    <div class="lb-card">
                        <img src="{poster_url(movie.get('poster_path'))}" alt="{movie['name']}">
                        <div class="title">{movie['name']}</div>
                        <div class="year">{movie['year']}</div>
                        <div class="stars">{stars(movie['predicted_rating'])}
                            <span style="color:#678;font-size:.75rem;">{movie['predicted_rating']:.1f}</span>
                        </div>
                        {badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ----------------------------------------------------------------------
# Header + input
# ----------------------------------------------------------------------

st.markdown(
    '<div class="lb-dots"><span class="lb-dot-orange">●</span>'
    '<span class="lb-dot-green">●</span>'
    '<span class="lb-dot-blue">●</span></div>',
    unsafe_allow_html=True,
)
st.title("So many movies, so little time.")
st.write("We'll learn your taste from your Letterboxd ratings and find films you'll love.")

col_input, col_slider = st.columns([2, 1])
with col_input:
    username = st.text_input(
        "Letterboxd username",
        placeholder="e.g. jakefine",
        help="Your profile must be public.",
    )
with col_slider:
    n_recs = st.slider("How many picks?", 5, 50, 20, step=5)

hide_mainstream = st.toggle(
    "Prefer hidden gems",
    help="Skews recommendations away from films everyone's already seen.",
)

if not username:
    st.stop()

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

try:
    with st.spinner(f"Reading {username}'s ratings and watchlist from Letterboxd..."):
        ratings_df, watchlist_df = scrape_profile(username.strip().lower())
except ScrapeError as e:
    st.error(str(e))
    st.stop()

n_rated = len(ratings_df)
st.markdown(
    f'<span class="lb-chip"><b>{n_rated}</b> films rated</span>'
    f'<span class="lb-chip"><b>{ratings_df["rating"].mean():.2f}★</b> average</span>'
    f'<span class="lb-chip"><b>{(ratings_df["rating"] >= 4.5).sum()}</b> favorites (4.5★+)</span>',
    unsafe_allow_html=True,
)
if n_rated < 30:
    st.warning("Fewer than 30 ratings — recommendations improve as you rate more films.")

with st.spinner("Enriching your films with TMDB metadata..."):
    enriched = enrich(ratings_df)
    wl_enriched = enrich(watchlist_df) if len(watchlist_df) else watchlist_df

with st.spinner("Training your taste model..."):
    model = ContentBasedModel()
    # watchlist = weak positive signal: films you chose but haven't seen yet
    model.fit(enriched, watchlist=wl_enriched if len(wl_enriched) else None)

tab_recs, tab_taste = st.tabs(["🎬  Your picks", "🧠  Your taste"])

# ----------------------------------------------------------------------
# Recommendations
# ----------------------------------------------------------------------

with tab_recs:
    genres = top_genres(enriched)
    seen = frozenset(enriched["tmdb_id"].dropna().astype(int))
    favorites = tuple(enriched[enriched["rating"] >= 4.5]["tmdb_id"].dropna().astype(int).tolist())

    with st.spinner("Scouting films you haven't seen..."):
        candidates = fetch_candidates(seen, favorites, tuple(genres))

    if hide_mainstream:
        candidates = candidates[candidates["vote_count"] < 5000]
    candidates = candidates[candidates["vote_count"] >= 50]

    # watchlist films are prime candidates — the user already wants to see them
    candidates["on_watchlist"] = False
    if len(wl_enriched):
        wl_pool = wl_enriched.copy()
        wl_pool["on_watchlist"] = True
        candidates = pd.concat([wl_pool, candidates], ignore_index=True)
        candidates = candidates.drop_duplicates(subset="tmdb_id", keep="first")
    candidates = candidates.reset_index(drop=True)

    recs = candidates.copy()
    recs["predicted_rating"] = model.predict(recs)
    recs = recs.sort_values("predicted_rating", ascending=False).head(n_recs)

    st.caption(
        f"Scored {len(candidates)} unseen films against your taste. "
        f"Stars show the rating we think *you'd* give."
    )
    poster_grid(recs)

# ----------------------------------------------------------------------
# Taste profile
# ----------------------------------------------------------------------

with tab_taste:
    st.markdown(
        f"""
        <div class="lb-persona">
            <div class="label">Your film persona</div>
            <div class="value">{taste.persona(enriched)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("What's your taste?")
    for line in taste.taste_summary(enriched):
        st.markdown(f"- {line}")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Directors you trust")
        directors = taste.top_directors(enriched, n=5)
        for _, r in directors.iterrows():
            st.markdown(
                f'<span class="lb-chip"><b>{r["director"]}</b> · '
                f'{stars(r["mean"])} over {int(r["count"])} films</span>',
                unsafe_allow_html=True,
            )

        st.subheader("Recurring themes")
        themes = taste.recurring_themes(enriched, n=10)
        chips = "".join(
            f'<span class="lb-chip">{t} <b>×{c}</b></span>'
            for t, c in themes.itertuples(index=False)
        )
        st.markdown(chips, unsafe_allow_html=True)

    with col2:
        st.subheader("Ratings by decade")
        era = taste.era_profile(enriched).set_index("decade")["mean"]
        st.bar_chart(era, color="#00e054")

        st.subheader("Your rating distribution")
        dist = enriched["rating"].value_counts().sort_index()
        st.bar_chart(dist, color="#40bcf4")
