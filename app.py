"""
Flask web app for Letterboxd → Movie recommendations via SVD collaborative filtering.
Upload your Letterboxd ratings.csv and likes/films.csv to get personalised film recs.
"""

import tempfile
from pathlib import Path

from flask import Flask, render_template, request, jsonify

from collab_filter import load_movielens_data, get_or_train_model
from lb_recs import (
    letterboxd_to_recommendations_pipeline,
    build_movie_index,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ── Globals initialised once at startup ──────────────────────────────
_model = None
_movies_df = None
_movie_index = None


def _init():
    """Load MovieLens data, movie index, and SVD model (cached after first train)."""
    global _model, _movies_df, _movie_index
    if _model is not None:
        return

    _, _movies_df = load_movielens_data()
    _movie_index = build_movie_index(_movies_df)
    _model = get_or_train_model()


# ── Routes ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    """Accept uploaded Letterboxd CSVs and return recommendations."""
    if "ratings" not in request.files or request.files["ratings"].filename == "":
        return jsonify({"error": "ratings.csv is required"}), 400

    ratings_file = request.files["ratings"]
    if not ratings_file.filename.lower().endswith(".csv"):
        return jsonify({"error": "ratings must be a CSV file"}), 400

    liked_file = request.files.get("likes")
    if liked_file and liked_file.filename == "":
        liked_file = None

    n_recs = request.form.get("n_recommendations", 10, type=int)
    n_recs = max(5, min(50, n_recs))

    with tempfile.TemporaryDirectory() as tmpdir:
        ratings_path = Path(tmpdir) / "ratings.csv"
        ratings_file.save(ratings_path)

        liked_path = None
        if liked_file and liked_file.filename:
            liked_path = Path(tmpdir) / "films.csv"
            liked_file.save(liked_path)

        try:
            results = letterboxd_to_recommendations_pipeline(
                letterboxd_ratings_path=str(ratings_path),
                letterboxd_liked_path=str(liked_path) if liked_path else None,
                movies_df=_movies_df,
                movie_index=_movie_index,
            )

            if results is None:
                return jsonify({
                    "error": "No movies could be matched from your Letterboxd data. "
                    "Make sure your ratings.csv has columns: Date, Name, Year, Letterboxd URI, Rating"
                }), 400

            recommendations = _model.recommend(
                results["matched_ratings"],
                n_recommendations=n_recs,
                movies_df=_movies_df,
            )

            recs = [
                {"movie_id": int(mid), "title": title, "predicted_rating": round(r, 2)}
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
    print("Initialising model (first run trains SVD — may take a few minutes) …")
    _init()
    print("Ready.\n")
    app.run(debug=False, port=5000)
