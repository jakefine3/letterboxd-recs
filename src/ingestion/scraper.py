"""BeautifulSoup scraper for Letterboxd public profiles — stub for future use.

Letterboxd doesn't have a public API for user ratings, but film ratings and
watchlists are publicly visible on profile pages. This module is a placeholder
for scraping that data to build the multi-user dataset required by the
collaborative filtering model (see src/models/collaborative.py).

Planned interface:
    scraper = LetterboxdScraper()
    df = scraper.fetch_ratings(username="someuser")  # → same schema as ratings.csv
"""
