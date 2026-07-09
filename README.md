# letterboxd movie recommendation system

a recommendation system built for letterboxd users. learns your taste from your
ratings + TMDB metadata (genres, directors, keywords), then scores unseen films
against it.

## features

- **two input modes**: upload your letterboxd `ratings.csv` export, or just type
  a public letterboxd username (scraped live)
- **TMDB enrichment**: every film gets genres, director, keywords, runtime,
  language — cached locally so repeat runs are free
- **content-based model**: gradient-boosted trees (sklearn) trained on your
  ratings, cross-validated MAE ~0.65 stars
- **candidate pool**: popular + trending + similar-to-your-favorites + a niche
  discovery pass (acclaimed, low-vote-count films in your top genres)
- **watchlist validation**: sanity-checks the model by confirming it scores your
  hand-picked watchlist above the average unseen film
- **taste profile**: see which features (directors, keywords, genres) drive your
  ratings, via permutation importance

## structure

```
app.py                        # streamlit app (main entrypoint)
scripts/run_recommendations.py # CLI: enrich ratings csv -> processed dataset
src/
  ingestion/
    csv_loader.py             # letterboxd ratings.csv -> clean dataframe
    scraper.py                # public profile scraper (curl_cffi + bs4)
    tmdb_client.py            # TMDB API client with local JSON cache
  models/
    features.py               # feature engineering (sklearn transformer)
    content_based.py          # HistGradientBoostingRegressor pipeline
    evaluation.py             # watchlist validation
    collaborative.py          # stub — needs multi-user data
    hybrid.py                 # stub — feature-level hybrid (planned)
notebooks/exploration.ipynb   # data exploration / model experiments
```

## setup

```bash
# install deps (uses uv)
uv sync

# TMDB api key (free: themoviedb.org/settings/api)
cp .env.example .env   # then add your key

# run the app
uv run streamlit run app.py
```

## development roadmap

- [x] csv ingestion + TMDB enrichment
- [x] content-based model + streamlit app
- [x] username scraper
- [ ] collaborative filtering (needs multi-user scrape)
- [ ] feature-level hybrid: decision trees + logistic regression + neural nets
- [ ] public web deployment
