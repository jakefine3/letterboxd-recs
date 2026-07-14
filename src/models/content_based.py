"""Content-based ML recommendation model.

Architecture
------------
sklearn Pipeline:
  1. FeatureBuilder  — engineers a sparse binary + numeric matrix from TMDB metadata:
                       one-hot genres, frequent directors/keywords, normalised runtime,
                       is_english flag.
  2. HistGradientBoostingRegressor — gradient-boosted trees that predict the user's rating
                                      for a film given its feature vector.

Why HistGradientBoostingRegressor over Ridge
---------------------------------------------
Our features are mostly sparse binary flags (genre_X, director_Y, keyword_Z). A linear model
like Ridge assumes additive effects — it can't capture that the user likes "Horror + A24" more
than either alone. HGBR learns these interaction effects and handles the sparse feature matrix
without needing feature scaling, which removes one potential source of leakage.

Why a Pipeline
--------------
Wrapping FeatureBuilder inside a Pipeline ensures that during cross-validation the scaler
and vocabulary are fit only on the training fold — no leakage from the validation fold.
"""
import ast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.models.features import FeatureBuilder

_FLAT_RATER_STD = 0.5   # below this threshold ratings carry no preference signal
_BAYES_M = 1000.0
_BAYES_C = 6.5


def _parse_genres(val) -> list[str]:
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    return []


class ContentBasedModel:
    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.03,
        max_leaf_nodes: int = 15,
    ):
        # defaults from grid search on real data (2026-07-09):
        # CV MAE 0.554, watchlist lift +0.37 with weight-0.2 weak supervision
        self._pipeline = Pipeline([
            ("features", FeatureBuilder()),
            ("model", HistGradientBoostingRegressor(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_leaf_nodes=max_leaf_nodes,
                random_state=42,
            )),
        ])
        self._fitted = False
        self._flat_rater = False
        self._genre_weights: dict[str, float] = {}

    @property
    def is_flat_rater(self) -> bool:
        return self._flat_rater

    def fit(
        self,
        df: pd.DataFrame,
        watchlist: pd.DataFrame | None = None,
        watchlist_weight: float = 0.2,
    ) -> "ContentBasedModel":
        """Fit on rated films, optionally using the watchlist as weak supervision.

        Watchlist films are probable-positives: the user chose them but hasn't
        seen them. They get a pseudo-rating at the user's 75th percentile and
        a reduced sample weight, so real ratings stay the dominant signal while
        watchlist-like features (genres, directors, themes) get a nudge.

        If the user's rating distribution has very low variance (flat-rater), the
        regression model can't learn preferences — we fall back to genre-frequency
        ranking instead and skip HGBR fitting entirely.
        """
        self._flat_rater = df["rating"].std() < _FLAT_RATER_STD

        if self._flat_rater:
            counts: dict[str, int] = {}
            for genres in df["genres"].apply(_parse_genres):
                for g in genres:
                    counts[g] = counts.get(g, 0) + 1
            total = sum(counts.values()) or 1
            self._genre_weights = {g: c / total for g, c in counts.items()}
            self._fitted = True
            return self

        train = df
        weights = np.ones(len(df), dtype=np.float32)

        if watchlist is not None and len(watchlist) > 0:
            pseudo = watchlist.copy()
            pseudo["rating"] = float(df["rating"].quantile(0.75))
            train = pd.concat([df, pseudo], ignore_index=True)
            weights = np.concatenate([
                weights,
                np.full(len(pseudo), watchlist_weight, dtype=np.float32),
            ])

        y = train["rating"].values.astype(np.float32)
        self._pipeline.fit(train, y, model__sample_weight=weights)
        self._fitted = True
        return self

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        """Return cross-validated MAE and RMSE — useful for comparing against the hybrid."""
        y = df["rating"].values.astype(np.float32)
        mae = -cross_val_score(self._pipeline, df, y, cv=cv, scoring="neg_mean_absolute_error")
        rmse = np.sqrt(-cross_val_score(self._pipeline, df, y, cv=cv, scoring="neg_mean_squared_error"))
        return {
            "mae_mean": float(mae.mean()),
            "mae_std": float(mae.std()),
            "rmse_mean": float(rmse.mean()),
            "rmse_std": float(rmse.std()),
        }

    def _genre_affinity_predict(self, candidates: pd.DataFrame) -> np.ndarray:
        scores = []
        for _, row in candidates.iterrows():
            genres = _parse_genres(row.get("genres", []))
            genre_score = (
                sum(self._genre_weights.get(g, 0.0) for g in genres) / max(len(genres), 1)
            )
            vote_cnt = float(row.get("vote_count") or 0)
            raw_avg = float(row.get("vote_average") or _BAYES_C)
            bayesian = (vote_cnt * raw_avg + _BAYES_M * _BAYES_C) / (vote_cnt + _BAYES_M)
            scores.append(0.7 * genre_score + 0.3 * (bayesian / 10.0))

        raw = np.array(scores, dtype=np.float32)
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            return (1.0 + 4.0 * (raw - lo) / (hi - lo)).astype(np.float32)
        return np.full_like(raw, 3.0)

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        if self._flat_rater:
            return self._genre_affinity_predict(candidates)
        return np.clip(self._pipeline.predict(candidates), 0.5, 5.0).astype(np.float32)

    def recommend(self, candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        scores = self.predict(candidates)
        results = candidates[["name", "year"]].copy()
        results["predicted_rating"] = scores
        return results.sort_values("predicted_rating", ascending=False).head(n)

    def top_features(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return the n most influential features via permutation importance.

        Permutation importance measures how much the MAE increases when each feature
        is randomly shuffled — a model-agnostic way to rank features that works for
        any sklearn estimator, including HGBR which doesn't expose tree-split importances.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before top_features().")
        from sklearn.inspection import permutation_importance
        feature_builder = self._pipeline.named_steps["features"]
        X = feature_builder.transform(df)
        y = df["rating"].values.astype(np.float32)
        model = self._pipeline.named_steps["model"]
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="neg_mean_absolute_error")
        return pd.DataFrame({
            "feature": feature_builder.feature_names_,
            "importance": result.importances_mean,
        }).sort_values("importance", ascending=False).head(n).reset_index(drop=True)
