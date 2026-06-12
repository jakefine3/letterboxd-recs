"""Collaborative filtering model — stub for future multi-user support.

Collaborative filtering identifies users with similar taste and recommends films
they liked that you haven't seen. It requires ratings from many users to find
meaningful neighbours.

With a single Letterboxd export (~300 ratings), there's no basis for CF.
This module is a placeholder for when multi-user data becomes available —
e.g., via BeautifulSoup scraping of other users' public Letterboxd profiles
(see src/ingestion/scraper.py).

Likely approach when implemented:
  - Build a user-item rating matrix from multiple exports
  - Use SVD / ALS matrix factorisation (surprise or implicit library)
  - Blend CF scores with the hybrid content-based model
"""
