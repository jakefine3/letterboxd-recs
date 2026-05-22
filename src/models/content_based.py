"""Content-based recommendation model.

Trains a regression model on a user's rated films to learn which features
(genres, director, cast, keywords, runtime, language) correlate with high
ratings. Scores unseen candidate films using the same feature space.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from src.models.features import FeatureBuilder


class ContentBasedModel:
    """Predicts a user's rating for any film given its TMDB features.

    Parameters
    ----------
    estimator : sklearn regressor, optional
        Defaults to Ridge regression, which works well with sparse binary
        features. Swap in any sklearn-compatible regressor later.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator or Ridge(alpha=1.0)
        self.feature_builder = FeatureBuilder()
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "ContentBasedModel":
        """Fit the model on a user's rated film DataFrame.

        Expects the output of TMDBClient.enrich_dataframe() —
        columns: name, year, rating, genres, director, cast, keywords,
        runtime, language.
        """
        X = self.feature_builder.fit_transform(df)
        y = self.feature_builder.get_target(df)

        self.estimator.fit(X, y)
        self._fitted = True
        return self

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        """Return cross-validated MAE and RMSE on the training data.

        Useful for understanding how well the model generalises before
        you have a separate test set.
        """
        X = self.feature_builder.fit_transform(df)
        y = self.feature_builder.get_target(df)

        mae_scores = -cross_val_score(self.estimator, X, y, cv=cv, scoring="neg_mean_absolute_error")
        rmse_scores = np.sqrt(-cross_val_score(self.estimator, X, y, cv=cv, scoring="neg_mean_squared_error"))

        return {
            "mae_mean": mae_scores.mean(),
            "mae_std": mae_scores.std(),
            "rmse_mean": rmse_scores.mean(),
            "rmse_std": rmse_scores.std(),
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        """Predict ratings for a DataFrame of candidate films.

        Candidates should have the same TMDB columns as the training data.
        Films with missing features will still get a prediction — they just
        won't activate those feature columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        X = self.feature_builder.transform(candidates)
        predictions = self.estimator.predict(X)

        # clamp to valid rating range
        return np.clip(predictions, 0.5, 5.0)

    def recommend(self, candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return the top-n candidates sorted by predicted rating.

        Parameters
        ----------
        candidates : DataFrame of unseen films with TMDB features
        n : number of recommendations to return
        """
        scores = self.predict(candidates)
        results = candidates[["name", "year"]].copy()
        results["predicted_rating"] = scores
        return results.sort_values("predicted_rating", ascending=False).head(n)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """Return the features with the largest positive/negative coefficients.

        Only works with linear models (Ridge, Lasso, LogisticRegression).
        Shows you what the model thinks you like and dislike.
        """
        if not hasattr(self.estimator, "coef_"):
            raise TypeError(f"{type(self.estimator).__name__} doesn't expose coefficients.")

        coefs = self.estimator.coef_
        names = self.feature_builder.feature_names_

        df = pd.DataFrame({"feature": names, "coefficient": coefs})
        df = df.reindex(df["coefficient"].abs().sort_values(ascending=False).index)
        return df.head(n).reset_index(drop=True)
