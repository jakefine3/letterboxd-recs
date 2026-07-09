"""Model sanity checks against data the model never trained on.

The watchlist is a natural validation set: films the user hand-picked as
"want to watch". A model that has learned the user's taste should, on
average, score watchlist films higher than a generic candidate pool.
"""
import numpy as np
import pandas as pd


def watchlist_validation(
    model,
    watchlist: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict:
    """Compare predicted ratings for the watchlist vs the candidate pool.

    Parameters
    ----------
    model : fitted ContentBasedModel
    watchlist : enriched DataFrame of the user's watchlist films
    candidates : enriched DataFrame of the generic candidate pool

    Returns a dict with mean scores, the lift between them, and the
    percentile of the watchlist mean within the candidate score
    distribution. Lift > 0 means the model ranks the user's own picks
    above an average unseen film — evidence it captures their taste.
    """
    watchlist_scores = model.predict(watchlist)
    candidate_scores = model.predict(candidates)

    wl_mean = float(np.mean(watchlist_scores))
    cand_mean = float(np.mean(candidate_scores))

    return {
        "watchlist_mean": wl_mean,
        "candidate_mean": cand_mean,
        "lift": wl_mean - cand_mean,
        "watchlist_percentile": float((candidate_scores < wl_mean).mean() * 100),
        "n_watchlist": len(watchlist),
        "n_candidates": len(candidates),
    }
