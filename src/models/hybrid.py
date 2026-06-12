"""Hybrid recommendation model: weighted ensemble of ML (content-based) and DL (neural).

Architecture
------------
Final prediction:
    score = alpha * ml_prediction + (1 - alpha) * dl_prediction

The two models capture different signals:
  - ContentBasedModel sees structured metadata (genres, director, keywords as discrete flags)
    and learns which specific attributes correlate with high/low ratings.
  - NeuralModel sees free-text semantic meaning of overviews + keywords and captures
    thematic and tonal preferences that don't map cleanly to discrete TMDB fields.

Neither model alone is ideal for a small personal dataset, but their errors tend to be
uncorrelated — ML struggles with unseen directors/keywords, DL struggles when similar-
sounding overviews have very different personal appeal. The ensemble smooths both.

Choosing alpha
--------------
alpha is selected by k-fold cross-validation: for each fold we collect out-of-fold
predictions from both models, then grid-search over alpha ∈ {0.0, 0.1, …, 1.0}
to minimise mean absolute error on the out-of-fold predictions.

To avoid running sentence-transformer encoding once per fold (expensive), embeddings
for the full dataset are pre-computed once and then sliced per fold.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.models.content_based import ContentBasedModel
from src.models.neural import NeuralModel


class HybridModel:
    def __init__(self, cv: int = 5):
        self.cv = cv
        self.ml_model = ContentBasedModel()
        self.dl_model = NeuralModel()
        self.alpha = 0.5    # ML weight — updated by fit()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "HybridModel":
        # encode all texts once; sliced into folds to avoid redundant API calls
        print("  Encoding texts for DL model...")
        all_embeddings = self.dl_model._encode_texts(df)

        print(f"  Cross-validating ensemble weight ({self.cv} folds)...")
        self.alpha = self._find_best_alpha(df, all_embeddings)
        print(f"  alpha = {self.alpha:.2f}  (ML={self.alpha:.0%}, DL={1 - self.alpha:.0%})")

        print("  Training final ML model...")
        self.ml_model.fit(df)

        print("  Training final DL model...")
        self.dl_model.fit(df, embeddings=all_embeddings)

        self._fitted = True
        return self

    def _find_best_alpha(self, df: pd.DataFrame, all_embeddings: np.ndarray) -> float:
        y = df["rating"].values.astype(np.float32)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)

        oof_ml = np.zeros(len(df), dtype=np.float32)
        oof_dl = np.zeros(len(df), dtype=np.float32)

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            print(f"    fold {fold + 1}/{self.cv}", end="\r")
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            ml = ContentBasedModel()
            ml.fit(train_df)
            oof_ml[val_idx] = ml.predict(val_df)

            dl = NeuralModel(epochs=self.dl_model.epochs, patience=self.dl_model.patience)
            dl.fit(train_df, embeddings=all_embeddings[train_idx])
            oof_dl[val_idx] = dl.predict(val_df, embeddings=all_embeddings[val_idx])

        print()

        alphas = np.linspace(0, 1, 11)
        maes = [float(np.abs(a * oof_ml + (1 - a) * oof_dl - y).mean()) for a in alphas]
        return float(alphas[int(np.argmin(maes))])

    def predict(self, candidates: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        ml_preds = self.ml_model.predict(candidates)
        dl_preds = self.dl_model.predict(candidates)
        return np.clip(self.alpha * ml_preds + (1 - self.alpha) * dl_preds, 0.5, 5.0).astype(np.float32)

    def recommend(self, candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        scores = self.predict(candidates)
        results = candidates[["name", "year"]].copy()
        results["predicted_rating"] = scores
        return results.sort_values("predicted_rating", ascending=False).head(n)
