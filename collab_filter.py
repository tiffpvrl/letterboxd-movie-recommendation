import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from lenskit.data import load_movielens
from pathlib import Path
import pickle
import sys

import kagglehub

BASE_DIR = Path(__file__).resolve().parent
MODEL_CACHE = BASE_DIR / "svd_model.pkl"
KAGGLE_DATASET = "justsahil/movielens-32m"


def get_ml_data_path():
    """Download ML-32M from Kaggle (cached after first download)."""
    print(f"Fetching MovieLens 32M from Kaggle ({KAGGLE_DATASET}) ...")
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

    # Kaggle extracts into a subdirectory (e.g. .../versions/1/ml-32m/)
    ml_sub = path / "ml-32m"
    if ml_sub.is_dir():
        path = ml_sub

    print(f"  Data path: {path}")
    return path


def load_movielens_data():
    """
    Load MovieLens 32M via Kaggle + LensKit.

    Downloads from Kaggle on first run (cached afterwards).
    Returns (ratings_df, movies_df).
    """
    data_path = get_ml_data_path()

    print(f"Loading MovieLens from {data_path} via LensKit ...")
    dataset = load_movielens(data_path)

    ratings_df = dataset.interaction_table(format="pandas", original_ids=True)
    ratings_df = ratings_df.rename(columns={"user_id": "userId", "item_id": "movieId"})
    ratings_df = ratings_df[["userId", "movieId", "rating"]]

    movies_path = data_path / "movies.csv"
    movies_df = pd.read_csv(movies_path) if movies_path.exists() else None

    print(f"  {len(ratings_df):,} ratings, {len(movies_df) if movies_df is not None else '?'} movies")
    return ratings_df, movies_df


class SVDRecommender:
    """Truncated-SVD collaborative filtering on a sparse user-item matrix."""

    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.svd = TruncatedSVD(
            n_components=n_factors, algorithm="randomized", random_state=42
        )
        self.movie_ids = None
        self.movie_to_idx = None
        self.global_mean = 0.0
        self.is_fitted = False

    def fit(self, ratings_df):
        """
        Build a sparse user-item matrix and train SVD.

        Args:
            ratings_df: DataFrame with userId, movieId, rating columns.
        """
        print("Building sparse user-item matrix ...")
        user_ids = sorted(ratings_df["userId"].unique())
        self.movie_ids = np.array(sorted(ratings_df["movieId"].unique()))

        user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        self.movie_to_idx = {int(mid): i for i, mid in enumerate(self.movie_ids)}

        rows = ratings_df["userId"].map(user_to_idx).values
        cols = ratings_df["movieId"].map(self.movie_to_idx).values
        vals = ratings_df["rating"].values.astype(np.float32)

        self.global_mean = float(vals.mean())

        sparse_matrix = csr_matrix(
            (vals - self.global_mean, (rows, cols)),
            shape=(len(user_ids), len(self.movie_ids)),
        )
        print(f"  Shape {sparse_matrix.shape}, {sparse_matrix.nnz:,} ratings")

        del rows, cols, vals

        print(f"Training SVD ({self.n_factors} factors) ...")
        self.svd.fit(sparse_matrix)
        self.is_fitted = True

        var_explained = self.svd.explained_variance_ratio_.sum()
        print(f"  Done — explained variance {var_explained:.3f}")

    def recommend(self, user_ratings, n_recommendations=10, movies_df=None):
        """
        Generate recommendations for a new user.

        Args:
            user_ratings: dict {movieId: rating} or DataFrame with movieId, rating.
            n_recommendations: how many results.
            movies_df: DataFrame with movieId, title for display.

        Returns:
            List of (movieId, title, predicted_rating) tuples.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted — call fit() first")

        if isinstance(user_ratings, pd.DataFrame):
            ratings_dict = dict(
                zip(user_ratings["movieId"].astype(int), user_ratings["rating"])
            )
        else:
            ratings_dict = {int(k): float(v) for k, v in user_ratings.items()}

        user_vec = np.zeros((1, len(self.movie_ids)), dtype=np.float32)
        rated_indices = set()
        for mid, rating in ratings_dict.items():
            if mid in self.movie_to_idx:
                idx = self.movie_to_idx[mid]
                user_vec[0, idx] = rating - self.global_mean
                rated_indices.add(idx)

        # Project into latent space, then reconstruct predicted scores
        user_latent = self.svd.transform(user_vec)
        scores = (user_latent @ self.svd.components_).ravel() + self.global_mean

        title_map = {}
        if movies_df is not None:
            title_map = dict(zip(movies_df["movieId"].astype(int), movies_df["title"]))

        ranking = np.argsort(scores)[::-1]
        results = []
        for idx in ranking:
            if idx in rated_indices:
                continue
            mid = int(self.movie_ids[idx])
            title = title_map.get(mid, f"Movie {mid}")
            results.append((mid, title, float(scores[idx])))
            if len(results) >= n_recommendations:
                break

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path=None):
        path = Path(path or MODEL_CACHE)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "n_factors": self.n_factors,
                    "svd": self.svd,
                    "movie_ids": self.movie_ids,
                    "movie_to_idx": self.movie_to_idx,
                    "global_mean": self.global_mean,
                },
                f,
            )
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path=None):
        path = Path(path or MODEL_CACHE)
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(n_factors=data["n_factors"])
        model.svd = data["svd"]
        model.movie_ids = data["movie_ids"]
        model.movie_to_idx = data["movie_to_idx"]
        model.global_mean = data["global_mean"]
        model.is_fitted = True
        print(f"Model loaded ← {path}")
        return model


def get_or_train_model(n_factors=50, force_retrain=False):
    """Load cached SVD model, or train and cache a new one."""
    if MODEL_CACHE.exists() and not force_retrain:
        return SVDRecommender.load()

    ratings_df, _ = load_movielens_data()
    model = SVDRecommender(n_factors=n_factors)
    model.fit(ratings_df)
    model.save()
    return model


if __name__ == "__main__":
    ratings_df, movies_df = load_movielens_data()
    model = SVDRecommender(n_factors=50)
    model.fit(ratings_df)
    model.save()

    first_user = ratings_df["userId"].iloc[0]
    sample = ratings_df[ratings_df["userId"] == first_user]
    sample_dict = dict(zip(sample["movieId"], sample["rating"]))

    recs = model.recommend(sample_dict, n_recommendations=10, movies_df=movies_df)
    print(f"\nTest: Top 10 recs for user {first_user}:")
    for mid, title, score in recs:
        print(f"  {score:.2f}  {title}")
