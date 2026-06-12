"""Neural content-based recommendation model.

Architecture
------------
1. Text encoding  — concatenates a film's overview and keywords, then encodes the
                    combined text with the all-MiniLM-L6-v2 sentence-transformer
                    (384-dimensional embedding, ~400 MB, downloaded once to ~/.cache).
2. MLP regressor  — a small PyTorch network that maps the 384-dim embedding to a
                    predicted rating:
                    384 → Linear(128) → ReLU → Dropout(0.3)
                        → Linear(64)  → ReLU → Dropout(0.3)
                        → Linear(1)

Why sentence-transformers over TF-IDF
--------------------------------------
TF-IDF treats overviews as bags of words and misses semantic relationships —
"crime thriller" and "neo-noir detective story" share no tokens but have very
similar meaning. The sentence-transformer produces dense vectors where semantically
similar texts are geometrically close, which gives the MLP a much better signal
to learn from with a small dataset.

Why a small MLP (not a large one)
-----------------------------------
With ~319 training examples, a large network memorises the training set immediately.
The 384→128→64→1 architecture has ~50k parameters, which is still overparameterised,
but Dropout(0.3) + weight decay + early stopping prevent the worst of it. The network
is small enough that the sentence-transformer embedding carries most of the signal,
and the MLP learns a simple non-linear re-weighting on top.

Training details
-----------------
- Ratings are z-score normalised before training so the loss is on a consistent scale.
- A 15% validation split is held out for early stopping (patience = 15 epochs).
- Best checkpoint (lowest validation MSE) is restored after training.
- Adam with weight_decay=1e-4 provides mild L2 regularisation.
"""
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_keywords(val) -> str:
    """Convert a keywords value (list or stringified list) to a space-joined string."""
    if isinstance(val, list):
        return " ".join(val)
    if isinstance(val, str) and val.startswith("["):
        try:
            return " ".join(ast.literal_eval(val))
        except (ValueError, SyntaxError):
            return ""
    return str(val) if val else ""


class _MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NeuralModel:
    """Predicts user ratings from sentence-transformer embeddings of overview + keywords."""

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        val_frac: float = 0.15,
        patience: int = 15,
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.patience = patience

        self._encoder = None
        self._model: _MLP | None = None
        self._fitted = False
        self._rating_mean = 0.0
        self._rating_std = 1.0
        self._device = _get_device()

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.MODEL_NAME, device=str(self._device))
        return self._encoder

    def _encode_texts(self, df: pd.DataFrame) -> np.ndarray:
        """Encode each film as 'overview keywords...' → 384-dim float32 vector."""
        overviews = df["overview"].fillna("").astype(str)
        keywords = df["keywords"].apply(_parse_keywords)
        texts = (overviews + " " + keywords).str.strip().tolist()
        return self._get_encoder().encode(texts, batch_size=64, show_progress_bar=False)

    def fit(self, df: pd.DataFrame, embeddings: np.ndarray | None = None) -> "NeuralModel":
        """Train the MLP on rated films.

        Parameters
        ----------
        embeddings : pre-computed sentence-transformer embeddings (shape [N, 384]).
                     Pass these when calling from HybridModel to avoid re-encoding
                     the same texts on every CV fold.
        """
        X = embeddings if embeddings is not None else self._encode_texts(df)
        y = df["rating"].values.astype(np.float32)

        # z-score normalise targets so MSE loss is scale-independent
        self._rating_mean = float(y.mean())
        self._rating_std = float(y.std()) + 1e-8
        y_norm = (y - self._rating_mean) / self._rating_std

        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_frac))
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        X_tr = torch.tensor(X[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(y_norm[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(self._device)
        y_val = torch.tensor(y_norm[val_idx], dtype=torch.float32).to(self._device)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=self.batch_size, shuffle=True)

        self._model = _MLP(X.shape[1]).to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(self.epochs):
            self._model.train()
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                criterion(self._model(xb), yb).backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = criterion(self._model(X_val), y_val).item()

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._fitted = True
        return self

    def predict(self, candidates: pd.DataFrame, embeddings: np.ndarray | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X = embeddings if embeddings is not None else self._encode_texts(candidates)
        xt = torch.tensor(X, dtype=torch.float32).to(self._device)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(xt).cpu().numpy()
        preds = preds * self._rating_std + self._rating_mean
        return np.clip(preds, 0.5, 5.0).astype(np.float32)

    def recommend(self, candidates: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        scores = self.predict(candidates)
        results = candidates[["name", "year"]].copy()
        results["predicted_rating"] = scores
        return results.sort_values("predicted_rating", ascending=False).head(n)
