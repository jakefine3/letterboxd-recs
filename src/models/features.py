"""Feature engineering: transforms enriched_ratings.csv into a numeric matrix."""
import ast
import math
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

_LOG_VOTE_COUNT_CAP = math.log1p(500_000)  # normalise vote_count on a log scale
_BAYES_M = 1000.0  # confidence weight for Bayesian vote average smoothing
_BAYES_C = 6.5     # global TMDB mean vote_average (approximate)
_YEAR_MIN = 1930.0
_YEAR_MAX = 2025.0


# minimum appearances in the library before a director/actor/keyword
# gets its own column — keeps the matrix dense enough to be useful
_MIN_DIRECTOR_APPEARANCES = 2
_MIN_CAST_APPEARANCES = 3
_MIN_KEYWORD_APPEARANCES = 3

# keywords that are too generic, location-based, or clearly TMDB noise
_KEYWORD_BLOCKLIST = {
    # TMDB housekeeping tags
    "aftercreditsstinger", "duringcreditsstinger", "tuwaderalit",
    # pure metadata — say nothing about content or taste
    "sequel", "based on novel or book", "based on short story",
    # TMDB mood/audience-feeling annotations — appear on almost everything, zero discrimination
    "excited", "amused", "bold", "antagonistic", "dramatic",
}


def _parse_list_col(series: pd.Series) -> pd.Series:
    """Convert a column of stringified Python lists / numpy arrays to actual lists."""
    def parse(val):
        if isinstance(val, list):
            return val
        if hasattr(val, "tolist"):  # numpy ndarray from parquet
            return val.tolist()
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return []
    return series.apply(parse)


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Fits on a rated-film DataFrame and transforms it into a numeric matrix.

    Fit once on your rated library, then use transform() on any set of films
    (including unseen candidates) to score them against the same feature space.
    """

    def __init__(
        self,
        min_director_appearances: int = _MIN_DIRECTOR_APPEARANCES,
        min_cast_appearances: int = _MIN_CAST_APPEARANCES,
        min_keyword_appearances: int = _MIN_KEYWORD_APPEARANCES,
        use_cast: bool = False,
    ):
        self.min_director_appearances = min_director_appearances
        self.min_cast_appearances = min_cast_appearances
        self.min_keyword_appearances = min_keyword_appearances
        self.use_cast = use_cast

        # learned during fit
        self.genres_: list[str] = []
        self.directors_: list[str] = []
        self.cast_: list[str] = []
        self.keywords_: list[str] = []
        self._scaler = MinMaxScaler()
        self.feature_names_: list[str] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y=None) -> "FeatureBuilder":
        genres = _parse_list_col(df["genres"])
        cast = _parse_list_col(df["cast"])
        keywords = _parse_list_col(df["keywords"])

        self.genres_ = sorted(genres.explode().dropna().unique().tolist())

        self.directors_ = _frequent_values(
            df["director"].dropna(), self.min_director_appearances
        )
        self.cast_ = _frequent_values(
            cast.explode().dropna(), self.min_cast_appearances
        ) if self.use_cast else []
        self.keywords_ = _frequent_values(
            keywords.explode().dropna(), self.min_keyword_appearances,
            blocklist=_KEYWORD_BLOCKLIST,
        )

        # keyword affinity map: frequency of each keyword in highly-rated films
        if "rating" in df.columns:
            high = df[df["rating"] >= 4.0]
        else:
            high = df
        high_kws = _parse_list_col(high["keywords"]) if not high.empty else pd.Series([], dtype=object)
        kw_counts = pd.Series([k for lst in high_kws for k in lst]).value_counts()
        self._kw_affinity: dict[str, int] = kw_counts.to_dict()
        self._kw_affinity_max: float = float(kw_counts.iloc[0]) if not kw_counts.empty else 1.0

        # fit scaler on runtime using training data
        runtime = df["runtime"].fillna(df["runtime"].median()).values.reshape(-1, 1)
        self._scaler.fit(runtime)

        self.feature_names_ = (
            [f"genre_{g}" for g in self.genres_]
            + [f"director_{d}" for d in self.directors_]
            + ([f"cast_{c}" for c in self.cast_] if self.use_cast else [])
            + [f"keyword_{k}" for k in self.keywords_]
            + ["runtime_norm", "is_english", "vote_avg_norm", "vote_count_log_norm", "keyword_affinity", "year_norm"]
        )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        genres = _parse_list_col(df["genres"])
        cast = _parse_list_col(df["cast"])
        keywords = _parse_list_col(df["keywords"])

        rows = []
        for i in range(len(df)):
            row_genres = set(genres.iloc[i])
            row_cast = set(cast.iloc[i])
            row_keywords = set(keywords.iloc[i])
            director = df["director"].iloc[i]
            runtime = df["runtime"].iloc[i]
            language = df["language"].iloc[i]

            genre_vec = [1 if g in row_genres else 0 for g in self.genres_]
            director_vec = [1 if d == director else 0 for d in self.directors_]
            cast_vec = [1 if c in row_cast else 0 for c in self.cast_] if self.use_cast else []
            keyword_vec = [1 if k in row_keywords else 0 for k in self.keywords_]

            rt = runtime if pd.notna(runtime) else self._scaler.data_min_[0]
            runtime_norm = self._scaler.transform([[rt]])[0][0]

            is_english = 1 if language == "en" else 0

            vote_avg = df["vote_average"].iloc[i] if "vote_average" in df.columns else None
            vote_cnt = float(df["vote_count"].iloc[i]) if "vote_count" in df.columns and pd.notna(df["vote_count"].iloc[i]) else 0.0
            raw_avg = float(vote_avg) if pd.notna(vote_avg) else _BAYES_C
            # Bayesian smoothing: shrinks low-vote films towards the global mean
            bayesian_avg = (vote_cnt * raw_avg + _BAYES_M * _BAYES_C) / (vote_cnt + _BAYES_M)
            vote_avg_norm = bayesian_avg / 10.0
            vote_count_log_norm = min(math.log1p(vote_cnt) / _LOG_VOTE_COUNT_CAP, 1.0)

            # continuous keyword alignment with the user's highly-rated films
            kw_set = set(keywords.iloc[i])
            raw_affinity = sum(self._kw_affinity.get(k, 0) for k in kw_set)
            keyword_affinity = min(raw_affinity / (self._kw_affinity_max * max(len(kw_set), 1)), 1.0)

            year_val = df["year"].iloc[i] if "year" in df.columns else None
            year_norm = float(np.clip((float(year_val) - _YEAR_MIN) / (_YEAR_MAX - _YEAR_MIN), 0.0, 1.0)) if pd.notna(year_val) else 0.5

            rows.append(genre_vec + director_vec + cast_vec + keyword_vec + [runtime_norm, is_english, vote_avg_norm, vote_count_log_norm, keyword_affinity, year_norm])

        return np.array(rows, dtype=np.float32)

    def get_target(self, df: pd.DataFrame) -> np.ndarray:
        """Return the rating column as a numpy array (your y vector)."""
        return df["rating"].values.astype(np.float32)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _frequent_values(
    series: pd.Series,
    min_count: int,
    blocklist: set[str] | None = None,
) -> list[str]:
    """Return values that appear at least min_count times, sorted by frequency.

    Filters out blocklisted tags and anything that looks like noise
    (non-ASCII, very short strings).
    """
    counts = series.value_counts()
    values = counts[counts >= min_count].index.tolist()
    return [
        v for v in values
        if len(v) > 2
        and v.isascii()
        and (blocklist is None or v not in blocklist)
    ]
