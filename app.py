"""
Flask web app for Letterboxd → Movie recommendations via collaborative filtering.
Upload your Letterboxd ratings.csv and likes/films.csv to get personalized film recommendations.
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify

# Project root
BASE_DIR = Path(__file__).resolve().parent
ML_RATINGS = BASE_DIR / "ml_data" / "ratings.csv"
ML_MOVIES = BASE_DIR / "ml_data" / "movies.csv"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """Process uploaded Letterboxd CSVs and return film recommendations."""
    if "ratings" not in request.files:
        return jsonify({"error": "ratings.csv is required"}), 400

    ratings_file = request.files["ratings"]
    if ratings_file.filename == "":
        return jsonify({"error": "ratings.csv is required"}), 400

    if not ratings_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "ratings must be a CSV file"}), 400

    # Optional likes file
    liked_file = request.files.get("likes")
    if liked_file and liked_file.filename == "":
        liked_file = None

    n_recommendations = request.form.get("n_recommendations", "10", type=int)
    n_recommendations = max(5, min(50, n_recommendations))

    with tempfile.TemporaryDirectory() as tmpdir:
        ratings_path = Path(tmpdir) / "ratings.csv"
        ratings_file.save(ratings_path)

        liked_path = None
        if liked_file and liked_file.filename:
            liked_path = Path(tmpdir) / "films.csv"
            liked_file.save(liked_path)

        try:
            from lb_recs import letterboxd_to_recommendations_pipeline
            from collab_filter import ItemBasedCF, create_user_item_matrix

            results = letterboxd_to_recommendations_pipeline(
                letterboxd_ratings_path=str(ratings_path),
                movielens_ratings_path=str(ML_RATINGS),
                movielens_movies_path=str(ML_MOVIES),
                letterboxd_liked_path=str(liked_path) if liked_path else None,
                n_recommendations=n_recommendations,
            )

            if results is None:
                return jsonify({
                    "error": "No movies could be matched from your Letterboxd data. "
                    "Make sure your ratings.csv has columns: Date, Name, Year, Letterboxd URI, Rating"
                }), 400

            user_item = create_user_item_matrix(results["combined_ratings"])
            model = ItemBasedCF(user_item)
            model.compute_similarity()

            recommendations = model.recommend(
                results["your_user_id"],
                n_recommendations=n_recommendations,
                movies_df=results["movies_df"],
            )

            recs = [
                {"movie_id": int(mid), "title": title, "predicted_rating": round(float(r), 2)}
                for mid, title, r in recommendations
            ]

            matched_count = len(results["matched_ratings"])
            unmatched_count = len(results["unmatched"])

            return jsonify({
                "recommendations": recs,
                "matched_count": matched_count,
                "unmatched_count": unmatched_count,
                "total_input": matched_count + unmatched_count,
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if not ML_RATINGS.exists() or not ML_MOVIES.exists():
        print("ERROR: ml_data/ratings.csv and ml_data/movies.csv must exist.")
        print("Download from https://grouplens.org/datasets/movielens/")
        exit(1)
    app.run(debug=True, port=5000)
