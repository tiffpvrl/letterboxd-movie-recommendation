import pandas as pd
import numpy as np
import re


# ============================================
# LETTERBOXD DATA LOADING
# ============================================

def load_letterboxd_export(ratings_filepath, liked_filepath=None):
    """
    Load ratings from Letterboxd CSV export and optionally merge with liked films.

    Steps to get your data:
    1. Go to letterboxd.com/settings/data
    2. Request your data export
    3. Download the ZIP file
    4. Extract and use 'ratings.csv' and 'liked/films.csv'

    Args:
        ratings_filepath: Path to ratings.csv
        liked_filepath: Path to liked/films.csv (optional)

    Returns:
        Combined DataFrame with all ratings (liked films added as 5 stars)
    """
    ratings_df = pd.read_csv(ratings_filepath)
    print(f"Loaded {len(ratings_df)} ratings from Letterboxd")

    if liked_filepath:
        ratings_df = merge_liked_films(ratings_df, liked_filepath)

    return ratings_df


def merge_liked_films(ratings_df, liked_filepath):
    """
    Merge liked films into ratings, treating unrated likes as 5-star ratings.
    """
    liked_df = pd.read_csv(liked_filepath)
    print(f"Loaded {len(liked_df)} liked films from Letterboxd")

    rated_films = set(zip(ratings_df["Name"], ratings_df["Year"]))

    new_likes = []
    for _, row in liked_df.iterrows():
        if (row["Name"], row["Year"]) not in rated_films:
            new_likes.append(
                {
                    "Date": row["Date"],
                    "Name": row["Name"],
                    "Year": row["Year"],
                    "Letterboxd URI": row["Letterboxd URI"],
                    "Rating": 5.0,
                }
            )

    if new_likes:
        ratings_df = pd.concat(
            [ratings_df, pd.DataFrame(new_likes)], ignore_index=True
        )
        print(f"Added {len(new_likes)} liked films as 5-star ratings")
    else:
        print("All liked films already have ratings")

    print(f"Total ratings after merge: {len(ratings_df)}")
    return ratings_df


# ============================================
# MOVIE MATCHING (optimised for large datasets)
# ============================================

def clean_title(title):
    """Normalise a movie title for matching."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    # Strip trailing articles — ML format: "Grand Budapest Hotel, The"
    for article in ("the", "a", "an"):
        if title.endswith(" " + article):
            title = title[: -(len(article) + 1)]
    # Strip leading articles — Letterboxd format: "The Grand Budapest Hotel"
    for article in ("the", "a", "an"):
        if title.startswith(article + " "):
            title = title[len(article) + 1 :]
    return title


def _extract_ml_titles(raw_title):
    """
    Extract matchable titles from a MovieLens title string.

    ML titles can look like:
      "Spirited Away (Sen to Chihiro no kamikakushi) (2001)"
      "Grand Budapest Hotel, The (2014)"
      "Toy Story (1995)"

    Returns a list of cleaned title strings to index under.
    """
    # Primary title: everything before the first "("
    primary = raw_title.split("(")[0].strip()

    # Full base: everything before the last "(YYYY)"
    year_match = re.search(r"\((\d{4})\)", raw_title)
    if year_match:
        base = raw_title[: year_match.start()].strip()
    else:
        base = raw_title

    titles = {clean_title(primary)}
    cleaned_base = clean_title(base)
    if cleaned_base:
        titles.add(cleaned_base)

    return list(titles)


def build_movie_index(movies_df):
    """
    Build dictionary indices for O(1) title+year lookups.

    Indexes each ML movie under multiple cleaned title variants
    to handle trailing articles and alternate/foreign titles.

    Returns:
        (title_year_index, title_index)
        title_year_index: dict  (clean_title, year) → movie row
        title_index:      dict  clean_title → movie row
    """
    title_year_index = {}
    title_index = {}

    for _, movie in movies_df.iterrows():
        raw_title = movie["title"]
        year_match = re.search(r"\((\d{4})\)", raw_title)
        variants = _extract_ml_titles(raw_title)

        for clean in variants:
            if not clean:
                continue

            if year_match:
                year = int(year_match.group(1))
                for delta in (-1, 0, 1):
                    title_year_index.setdefault((clean, year + delta), movie)

            title_index.setdefault(clean, movie)

    return title_year_index, title_index


def match_letterboxd_to_movielens(letterboxd_df, movies_df, movie_index=None):
    """
    Match Letterboxd movies to MovieLens movies using pre-built indices.

    Args:
        letterboxd_df: DataFrame from Letterboxd export
        movies_df: MovieLens movies DataFrame
        movie_index: tuple from build_movie_index (optional, built if not provided)

    Returns:
        (matched_df, unmatched_df)
    """
    if movie_index is None:
        movie_index = build_movie_index(movies_df)

    title_year_idx, title_idx = movie_index

    matched, unmatched = [], []

    for _, row in letterboxd_df.iterrows():
        lb_title = row["Name"]
        lb_year = row["Year"]
        lb_rating = row["Rating"]

        if pd.isna(lb_rating):
            continue

        clean = clean_title(lb_title)

        # 1) title + year (±1)
        movie = title_year_idx.get((clean, int(lb_year)))

        # 2) title only
        if movie is None:
            movie = title_idx.get(clean)

        if movie is not None:
            matched.append(
                {
                    "movieId": movie["movieId"],
                    "title": movie["title"],
                    "rating": lb_rating,
                    "letterboxd_title": lb_title,
                    "letterboxd_year": lb_year,
                }
            )
        else:
            unmatched.append(
                {"title": lb_title, "year": lb_year, "rating": lb_rating}
            )

    matched_df = pd.DataFrame(matched)
    unmatched_df = pd.DataFrame(unmatched)

    print(f"Matched: {len(matched_df)} movies")
    print(f"Unmatched: {len(unmatched_df)} movies")

    if len(unmatched_df) > 0:
        print("Sample unmatched:")
        print(unmatched_df.head(10).to_string(index=False))

    return matched_df, unmatched_df


# ============================================
# FULL PIPELINE
# ============================================

def letterboxd_to_recommendations_pipeline(
    letterboxd_ratings_path,
    letterboxd_liked_path=None,
    movies_df=None,
    movie_index=None,
):
    """
    Process a Letterboxd export and match films to MovieLens.

    Args:
        letterboxd_ratings_path: Path to Letterboxd ratings.csv
        letterboxd_liked_path: Path to Letterboxd liked/films.csv (optional)
        movies_df: Pre-loaded MovieLens movies DataFrame
        movie_index: Pre-built index from build_movie_index()

    Returns:
        dict with matched_ratings, unmatched, movies_df  — or None on failure.
    """
    print("=" * 60)
    print("LETTERBOXD → MOVIELENS MATCHING")
    print("=" * 60)

    letterboxd_df = load_letterboxd_export(
        letterboxd_ratings_path, letterboxd_liked_path
    )

    if movies_df is None:
        raise ValueError("movies_df is required")

    if movie_index is None:
        movie_index = build_movie_index(movies_df)

    matched_ratings, unmatched = match_letterboxd_to_movielens(
        letterboxd_df, movies_df, movie_index
    )

    if len(matched_ratings) == 0:
        print("ERROR: No movies matched!")
        return None

    print(f"\nTop rated films:")
    top = matched_ratings.nlargest(5, "rating")[["title", "rating"]]
    print(top.to_string(index=False))

    return {
        "matched_ratings": matched_ratings,
        "unmatched": unmatched,
        "movies_df": movies_df,
    }


# ============================================
# CLI EXAMPLE
# ============================================

if __name__ == "__main__":
    from collab_filter import load_movielens_data, get_or_train_model

    _, movies_df = load_movielens_data()
    movie_index = build_movie_index(movies_df)

    results = letterboxd_to_recommendations_pipeline(
        letterboxd_ratings_path="appendix/letterboxd_export/ratings.csv",
        letterboxd_liked_path="appendix/letterboxd_export/likes/films.csv",
        movies_df=movies_df,
        movie_index=movie_index,
    )

    if results is None:
        raise SystemExit("Pipeline failed — no matches")

    model = get_or_train_model()
    recs = model.recommend(
        results["matched_ratings"], n_recommendations=10, movies_df=movies_df
    )

    print("\nYOUR PERSONALIZED RECOMMENDATIONS:")
    print("=" * 60)
    for mid, title, score in recs:
        print(f"  {score:.2f}  {title}")
