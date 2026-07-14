"""Letterboxd Recommendations — Streamlit app.

Run with:  uv run streamlit run app.py

Input: a public Letterboxd username (scraped live). The CSV path still exists
in the backend (src/ingestion/csv_loader.py) for offline experimentation.
"""
import os
import random

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.ingestion.scraper import LetterboxdScraper, ScrapeError
from src.ingestion.tmdb_client import TMDBClient
from src.models.content_based import ContentBasedModel
from src.models import taste

load_dotenv()

_POSTER_BASE = "https://image.tmdb.org/t/p/w342"
_MARQUEE_POSTER_BASE = "https://image.tmdb.org/t/p/w185"
_EMPTY_POSTER = "https://s.ltrbxd.com/static/img/empty-poster-70-BSf-Pjrh.png"

_QUOTES = [
    ("Frankly, my dear, I don't give a damn.", "Gone with the Wind", 1939),
    ("Here's looking at you, kid.", "Casablanca", 1942),
    ("I'm gonna make him an offer he can't refuse.", "The Godfather", 1972),
    ("You talking to me?", "Taxi Driver", 1976),
    ("May the Force be with you.", "Star Wars", 1977),
    ("Here's Johnny!", "The Shining", 1980),
    ("Say hello to my little friend!", "Scarface", 1983),
    ("Roads? Where we're going, we don't need roads.", "Back to the Future", 1985),
    ("I'll be back.", "The Terminator", 1984),
    ("Nobody puts Baby in a corner.", "Dirty Dancing", 1987),
    ("You can't handle the truth!", "A Few Good Men", 1992),
    ("Life is like a box of chocolates.", "Forrest Gump", 1994),
    ("Houston, we have a problem.", "Apollo 13", 1995),
    ("I see dead people.", "The Sixth Sense", 1999),
    ("I am serious. And don't call me Shirley.", "Airplane!", 1980),
    ("Why so serious?", "The Dark Knight", 2008),
    ("Keep your friends close, but your enemies closer.", "The Godfather Part II", 1974),
    ("It's alive! It's alive!", "Frankenstein", 1931),
    ("E.T. phone home.", "E.T. the Extra-Terrestrial", 1982),
    ("They may take our lives, but they'll never take our freedom!", "Braveheart", 1995),
    ("To infinity and beyond!", "Toy Story", 1995),
    ("Just keep swimming.", "Finding Nemo", 2003),
    ("Wax on, wax off.", "The Karate Kid", 1984),
    ("I drink your milkshake!", "There Will Be Blood", 2007),
]

st.set_page_config(page_title="Letterboxd Insights", page_icon="🎬", layout="wide")

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

/* site title */
.lb-title {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -1px;
    color: #fff;
    margin-bottom: 0;
}
.lb-title .accent { color: #00e054; }

/* rotating quote */
.lb-quote {
    font-style: italic;
    color: #9ab;
    font-size: 1.15rem;
    margin: 4px 0 18px 0;
}
.lb-quote .src { font-style: normal; color: #678; font-size: .85rem; }
.lb-quote .src b { color: #40bcf4; }

/* insight stat cards */
.lb-stat {
    background: #2c3440;
    border: 1px solid #456;
    border-radius: 8px;
    padding: 16px 12px;
    text-align: center;
    height: 100%;
}
.lb-stat .stat-label { color: #678; font-size: .7rem; text-transform: uppercase; letter-spacing: 1.5px; }
.lb-stat .stat-value { color: #fff; font-size: 1.3rem; font-weight: 800; margin: 4px 0 2px; }
.lb-stat .stat-sub { color: #9ab; font-size: .75rem; }

/* contrarian film rows */
.lb-film-row { padding: 8px 0; border-bottom: 1px solid #2c3440; }
.lb-film-row .fn { color: #fff; font-weight: 600; font-size: .9rem; }
.lb-film-row .fy { color: #678; font-size: .75rem; margin-left: 4px; }
.lb-film-row .you { color: #00e054; font-size: .8rem; }
.lb-film-row .crowd { color: #9ab; font-size: .8rem; }
.lb-film-row .dp { color: #00e054; font-size: .8rem; font-weight: 700; }
.lb-film-row .dn { color: #ff8000; font-size: .8rem; font-weight: 700; }

/* trending poster marquee */
.lb-marquee {
    overflow: hidden;
    margin-top: 40px;
    padding: 12px 0;
    border-top: 1px solid #2c3440;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent);
}
.lb-marquee-track {
    display: flex;
    gap: 14px;
    width: max-content;
    animation: lb-scroll var(--lb-scroll-duration, 300s) linear infinite;
}
.lb-marquee-track:hover { animation-play-state: paused; }
.lb-marquee-track img {
    height: 320px;
    border-radius: 6px;
    border: 1px solid #456;
    transition: transform .15s ease;
}
.lb-marquee-track img:hover { transform: scale(1.06); }
@keyframes lb-scroll {
    from { transform: translateX(0); }
    to   { transform: translateX(-50%); }
}
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

def _tmdb_api_key() -> str | None:
    """Local dev: .env via os.getenv. Streamlit Cloud: st.secrets."""
    key = os.getenv("TMDB_API_KEY")
    if key:
        return key
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return None


def _scraperapi_key() -> str | None:
    key = os.getenv("SCRAPERAPI_KEY")
    if key:
        return key
    try:
        return st.secrets["SCRAPERAPI_KEY"]
    except (KeyError, FileNotFoundError):
        return None


@st.cache_resource
def get_client() -> TMDBClient:
    api_key = _tmdb_api_key()
    if not api_key:
        st.error(
            "TMDB_API_KEY not set. Locally: copy .env.example to .env and add your key. "
            "On Streamlit Cloud: add it under App settings → Secrets."
        )
        st.stop()
    return TMDBClient(api_key=api_key)


@st.cache_data(show_spinner=False)
def scrape_profile(username: str, scraperapi_key: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scraper = LetterboxdScraper(scraperapi_key=scraperapi_key)
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def top_rated_posters() -> list[dict]:
    return get_client().top_rated_posters()


def render_greats_marquee():
    films = top_rated_posters()
    if not films:
        return
    imgs = "".join(
        f'<img src="{_MARQUEE_POSTER_BASE}{f["poster_path"]}" alt="{f["name"]}" title="{f["name"]}">'
        for f in films
    )
    # scale duration with film count so scroll speed stays constant (~35px/s)
    duration = len(films) * 7
    st.markdown(
        f"""
        <div class="lb-marquee" style="--lb-scroll-duration: {duration}s;">
            <div class="lb-marquee-track">{imgs}{imgs}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Header + input
# ----------------------------------------------------------------------

# one quote per visit — survives widget interactions, changes on refresh
if "quote" not in st.session_state:
    st.session_state.quote = random.choice(_QUOTES)
quote, film, year = st.session_state.quote

st.markdown(
    f"""
    <div class=”lb-title”>Letterboxd <span class=”accent”>Insights</span></div>
    <div class=”lb-dots”><span class=”lb-dot-orange”>●</span><span class=”lb-dot-green”>●</span><span class=”lb-dot-blue”>●</span></div>
    <div class=”lb-quote”>”{quote}” <span class=”src”>— <b>{film}</b> ({year})</span></div>
    “””,
    unsafe_allow_html=True,
)
st.write(“Understand your taste. Discover what's next. Enter your public Letterboxd username to get started.”)

col_input, col_slider = st.columns([2, 1])
with col_input:
    username = st.text_input(
        "Letterboxd username",
        placeholder="e.g. martinscorsese",
        help="Your profile must be public.",
    )
with col_slider:
    n_recs = st.slider("How many picks?", 5, 50, 20, step=5)

tog1, tog2 = st.columns(2)
with tog1:
    hide_mainstream = st.toggle(
        "Prefer hidden gems",
        help="Skews recommendations away from films everyone's already seen.",
    )
with tog2:
    hide_watchlist = st.toggle(
        "Exclude watchlist",
        help="Hides films already on your Letterboxd watchlist so you discover something new.",
    )

if not username:
    render_greats_marquee()
    st.stop()

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

ratings_df = None
watchlist_df = None

try:
    with st.spinner(f"Scraping {username}'s ratings from Letterboxd..."):
        ratings_df, watchlist_df = scrape_profile(username.strip().lower(), _scraperapi_key())
except ScrapeError as e:
    if "403" in str(e):
        st.error(
            "Letterboxd is blocking requests from our servers — this is a Cloudflare "
            "restriction on cloud-hosted apps, not an issue with your account."
        )
        st.info(
            "**Upload your ratings.csv as a workaround:**\n\n"
            "1. Go to **letterboxd.com → Profile → Settings → Import & Export**\n"
            "2. Click **Export Your Data** and unzip the downloaded file\n"
            "3. Upload `ratings.csv` below"
        )
    else:
        st.error(str(e))

    uploaded = st.file_uploader("ratings.csv", type="csv", label_visibility="collapsed")
    if uploaded is not None:
        from src.ingestion.csv_loader import load_ratings
        ratings_df = load_ratings(uploaded)
        watchlist_df = pd.DataFrame(columns=["name", "year"])

if ratings_df is None:
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

_enrich_msg = (
    f"Enriching {n_rated} films with TMDB metadata — this takes ~{"30s" if n_rated > 500 else "10s"} "
    f"on first load, then it's cached."
)
with st.spinner(_enrich_msg):
    enriched = enrich(ratings_df)
    wl_enriched = enrich(watchlist_df) if len(watchlist_df) else watchlist_df

with st.spinner("Training your taste model..."):
    model = ContentBasedModel()
    # watchlist = weak positive signal: films you chose but haven't seen yet
    model.fit(enriched, watchlist=wl_enriched if len(wl_enriched) else None)

if model.is_flat_rater:
    st.warning(
        "Your ratings are nearly all the same score, so the model can't learn your preferences "
        "from scores alone. Recommendations are based on the genres you watch most instead."
    )

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

    candidates["on_watchlist"] = False
    if len(wl_enriched) and not hide_watchlist:
        wl_pool = wl_enriched.copy()
        wl_pool["on_watchlist"] = True
        candidates = pd.concat([wl_pool, candidates], ignore_index=True)
        candidates = candidates.drop_duplicates(subset="tmdb_id", keep="first")
    elif len(wl_enriched) and hide_watchlist:
        wl_ids = set(wl_enriched["tmdb_id"].dropna().astype(int))
        candidates = candidates[~candidates["tmdb_id"].isin(wl_ids)]
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
    # --- Persona ---
    st.markdown(
        f"""
        <div class="lb-persona">
            <div class="label">Your film persona</div>
            <div class="value">{taste.persona(enriched)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for line in taste.taste_summary(enriched):
        st.markdown(f"- {line}")

    st.divider()

    # --- Stat row ---
    tendency = taste.critic_tendency(enriched)
    obs = taste.obscurity_profile(enriched)
    lang = taste.language_split(enriched)
    n_enriched = len(enriched)
    pct_foreign = lang["foreign_count"] / n_enriched * 100 if n_enriched else 0

    tend_sign = "+" if tendency["delta"] > 0 else ""
    tend_label = "more generous than avg" if tendency["delta"] > 0 else "harsher than avg"

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, value, sub in [
        (sc1, "Films rated", str(n_enriched), "in your library"),
        (sc2, "vs. TMDB crowd", f"{tend_sign}{tendency['delta']:.2f}★", tend_label),
        (sc3, "Taste profile", obs["label"], f"median {obs['median_vote_count']:,} votes on loved films"),
        (sc4, "Non-English", f"{pct_foreign:.0f}%", f"{lang['foreign_count']} films"),
    ]:
        with col:
            st.markdown(
                f'<div class="lb-stat">'
                f'<div class="stat-label">{label}</div>'
                f'<div class="stat-value" style="font-size:{"1rem" if len(value) > 10 else "1.3rem"}">{value}</div>'
                f'<div class="stat-sub">{sub}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # --- You vs. the crowd ---
    st.subheader("You vs. the crowd")
    loved_more, loved_less = taste.contrarian_picks(enriched, n=5)

    def _film_rows(frame: pd.DataFrame, positive: bool) -> str:
        html = ""
        for _, r in frame.iterrows():
            yr = int(r["year"]) if pd.notna(r["year"]) else "?"
            delta_cls = "dp" if positive else "dn"
            delta_str = f"+{r['delta']:.1f}★" if positive else f"{r['delta']:.1f}★"
            html += (
                f'<div class="lb-film-row">'
                f'<span class="fn">{r["name"]}</span><span class="fy">({yr})</span><br>'
                f'<span class="you">You {r["rating"]:.1f}★</span>'
                f' · <span class="crowd">TMDB {r["tmdb_scaled"]:.1f}★</span>'
                f' · <span class="{delta_cls}">{delta_str}</span>'
                f'</div>'
            )
        return html or '<p style="color:#678;font-size:.85rem">Not enough data yet.</p>'

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Hidden gems — you loved, the crowd slept on**")
        st.markdown(_film_rows(loved_more, positive=True), unsafe_allow_html=True)
    with cc2:
        st.markdown("**Overhyped — TMDB loves, you don't**")
        st.markdown(_film_rows(loved_less, positive=False), unsafe_allow_html=True)

    st.divider()

    # --- Directors / themes / charts ---
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

    # --- Runtime sweet spot ---
    rt = taste.runtime_sweet_spot(enriched)
    if not rt.empty:
        st.subheader("Runtime sweet spot")
        st.caption("Average rating you give by film length.")
        st.bar_chart(rt.set_index("length")["avg_rating"], color="#ff8000")
