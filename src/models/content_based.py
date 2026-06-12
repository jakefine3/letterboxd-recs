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
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.models.features import FeatureBuilder


class ContentBasedModel:
    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_leaf_nodes: int = 31,
    ):
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

    def fit(self, df: pd.DataFrame) -> "ContentBasedModel":
        y = df["rating"].values.astype(np.float32)
        self._pipeline.fit(df, y)
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

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
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
