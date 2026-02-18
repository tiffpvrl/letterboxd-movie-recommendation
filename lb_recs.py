import pandas as pd
import numpy as np
from datetime import datetime
import re

def load_letterboxd_export(ratings_filepath, liked_filepath=None):
    """
    Load ratings from Letterboxd CSV export and optionally merge with liked films
    
    Steps to get your data:
    1. Go to letterboxd.com/settings/data
    2. Request your data export
    3. Download the ZIP file
    4. Extract and use 'ratings.csv' and 'liked/films.csv'
    
    Letterboxd ratings.csv columns:
    - Date, Name, Year, Letterboxd URI, Rating
    
    Letterboxd liked/films.csv columns:
    - Date, Name, Year, Letterboxd URI
    
    Args:
        ratings_filepath: Path to ratings.csv
        liked_filepath: Path to liked/films.csv (optional)
    
    Returns:
        Combined DataFrame with all ratings (liked films added as 5 stars)
    """
    ratings_df = pd.read_csv(ratings_filepath)
    
    print(f"Loaded {len(ratings_df)} ratings from Letterboxd")
    print(f"Columns: {ratings_df.columns.tolist()}")
    
    # If liked films provided, merge them
    if liked_filepath:
        ratings_df = merge_liked_films(ratings_df, liked_filepath)
    
    return ratings_df


def merge_liked_films(ratings_df, liked_filepath):
    """
    Merge liked films into ratings, treating them as 5-star ratings
    
    Args:
        ratings_df: DataFrame from ratings.csv
        liked_filepath: Path to liked/films.csv
    
    Returns:
        Combined DataFrame with liked films added as 5-star ratings
    """
    liked_df = pd.read_csv(liked_filepath)
    
    print(f"\nLoaded {len(liked_df)} liked films from Letterboxd")
    
    # Create a set of (Name, Year) tuples from ratings for quick lookup
    rated_films = set(zip(ratings_df['Name'], ratings_df['Year']))
    
    # Filter liked films to only those not already rated
    new_likes = []
    for idx, row in liked_df.iterrows():
        film_key = (row['Name'], row['Year'])
        if film_key not in rated_films:
            new_likes.append({
                'Date': row['Date'],
                'Name': row['Name'],
                'Year': row['Year'],
                'Letterboxd URI': row['Letterboxd URI'],
                'Rating': 5.0  # Treat likes as 5-star ratings
            })
    
    # Add new likes to ratings
    if new_likes:
        new_likes_df = pd.DataFrame(new_likes)
        ratings_df = pd.concat([ratings_df, new_likes_df], ignore_index=True)
        print(f"Added {len(new_likes)} liked films as 5-star ratings")
    else:
        print("All liked films already have ratings")
    
    print(f"Total ratings after merge: {len(ratings_df)}")
    
    return ratings_df

def match_letterboxd_to_movielens(letterboxd_df, movies_df):
    """
    Match Letterboxd movies to MovieLens movies
    
    Args:
        letterboxd_df: DataFrame from Letterboxd export
        movies_df: MovieLens movies DataFrame with columns [movieId, title, genres]
    
    Returns:
        DataFrame with matched movies and ratings
    """
    matched_ratings = []
    unmatched = []
    
    for idx, row in letterboxd_df.iterrows():
        letterboxd_title = row['Name']
        letterboxd_year = row['Year']
        letterboxd_rating = row['Rating']
        
        # Skip if no rating
        if pd.isna(letterboxd_rating):
            continue
        
        # Try to match by title and year
        match = find_movie_match(letterboxd_title, letterboxd_year, movies_df)
        
        if match is not None:
            matched_ratings.append({
                'movieId': match['movieId'],
                'title': match['title'],
                'rating': letterboxd_rating,
                'letterboxd_title': letterboxd_title,
                'letterboxd_year': letterboxd_year
            })
        else:
            unmatched.append({
                'title': letterboxd_title,
                'year': letterboxd_year,
                'rating': letterboxd_rating
            })
    
    matched_df = pd.DataFrame(matched_ratings)
    unmatched_df = pd.DataFrame(unmatched)
    
    print(f"\nMatched: {len(matched_df)} movies")
    print(f"Unmatched: {len(unmatched_df)} movies")
    
    if len(unmatched_df) > 0:
        print("\nSample unmatched movies:")
        print(unmatched_df.head(10))
    
    return matched_df, unmatched_df


def find_movie_match(letterboxd_title, letterboxd_year, movies_df):
    """
    Find matching movie in MovieLens dataset
    Uses fuzzy matching on title and year
    """
    # Clean title for matching
    clean_letterboxd = clean_title(letterboxd_title)
    
    # First try: exact title and year match
    for idx, movie in movies_df.iterrows():
        movie_title = movie['title']
        
        # Extract year from MovieLens title (format: "Movie Title (Year)")
        year_match = re.search(r'\((\d{4})\)', movie_title)
        if year_match:
            movie_year = int(year_match.group(1))
            movie_title_clean = clean_title(movie_title.rsplit('(', 1)[0].strip())
            
            # Check if titles match and years are close (within 1 year for errors)
            if (movie_title_clean == clean_letterboxd and 
                abs(movie_year - letterboxd_year) <= 1):
                return movie
    
    # Second try: relaxed matching (title only)
    for idx, movie in movies_df.iterrows():
        movie_title = movie['title']
        movie_title_clean = clean_title(movie_title.rsplit('(', 1)[0].strip())
        
        if movie_title_clean == clean_letterboxd:
            return movie
    
    # Third try: partial matching (contains)
    for idx, movie in movies_df.iterrows():
        movie_title = movie['title']
        movie_title_clean = clean_title(movie_title.rsplit('(', 1)[0].strip())
        
        if clean_letterboxd in movie_title_clean or movie_title_clean in clean_letterboxd:
            year_match = re.search(r'\((\d{4})\)', movie_title)
            if year_match:
                movie_year = int(year_match.group(1))
                if abs(movie_year - letterboxd_year) <= 2:
                    return movie
    
    return None


def clean_title(title):
    """Clean movie title for matching"""
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
    title = title.strip()
    
    # Remove common articles
    for article in ['the', 'a', 'an']:
        if title.startswith(article + ' '):
            title = title[len(article)+1:]
    
    return title

# ============================================
# INTEGRATE INTO COLLABORATIVE FILTERING
# ============================================

def add_user_to_ratings(existing_ratings, user_letterboxd_ratings, new_user_id=None):
    """
    Add your Letterboxd ratings to the MovieLens dataset
    
    Args:
        existing_ratings: MovieLens ratings DataFrame
        user_letterboxd_ratings: Your matched ratings from Letterboxd
        new_user_id: ID to assign to new user (if None, uses max+1)
    
    Returns:
        Combined ratings DataFrame
    """
    if new_user_id is None:
        new_user_id = existing_ratings['userId'].max() + 1
    
    # Create new rows for your ratings
    new_ratings = pd.DataFrame({
        'userId': new_user_id,
        'movieId': user_letterboxd_ratings['movieId'],
        'rating': user_letterboxd_ratings['rating'],
        'timestamp': int(datetime.now().timestamp())
    })
    
    # Combine with existing ratings
    combined_ratings = pd.concat([existing_ratings, new_ratings], ignore_index=True)
    
    print(f"\nAdded user {new_user_id} with {len(new_ratings)} ratings")
    print(f"Total ratings in dataset: {len(combined_ratings)}")
    
    return combined_ratings, new_user_id


def get_personalized_recommendations(ratings_df, movies_df, your_user_id, model_class, n_recommendations=10):
    """
    Get recommendations based on your Letterboxd ratings
    
    Args:
        ratings_df: Combined ratings including your data
        movies_df: MovieLens movies DataFrame
        your_user_id: Your assigned user ID
        model_class: ItemBasedCF, UserBasedCF, or MatrixFactorizationCF
        n_recommendations: Number of recommendations to return
    """
    from collab_filter import create_user_item_matrix
    
    # Create user-item matrix
    user_item = create_user_item_matrix(ratings_df)
    
    # Initialize and fit model
    if model_class.__name__ == 'MatrixFactorizationCF':
        model = model_class(user_item, n_factors=20)
    else:
        model = model_class(user_item)
        model.compute_similarity()
    
    # Get recommendations
    recommendations = model.recommend(your_user_id, n_recommendations, movies_df)
    
    return recommendations


# ============================================
# COMPLETE WORKFLOW EXAMPLE
# ============================================

def letterboxd_to_recommendations_pipeline(
    letterboxd_ratings_path,
    movielens_ratings_path,
    movielens_movies_path,
    letterboxd_liked_path=None,
    n_recommendations=10
):
    """
    Complete pipeline from Letterboxd export to personalized recommendations
    
    Args:
        letterboxd_ratings_path: Path to Letterboxd ratings.csv
        movielens_ratings_path: Path to MovieLens ratings.csv
        movielens_movies_path: Path to MovieLens movies.csv
        letterboxd_liked_path: Path to Letterboxd liked/films.csv (optional)
        n_recommendations: Number of recommendations to generate
    """
    print("="*70)
    print("LETTERBOXD TO MOVIELENS RECOMMENDATION PIPELINE")
    print("="*70)
    
    # Step 1: Load Letterboxd data
    print("\n[1/5] Loading Letterboxd export...")
    letterboxd_df = load_letterboxd_export(
        letterboxd_ratings_path, 
        letterboxd_liked_path
    )
    
    # Step 2: Load MovieLens data
    print("\n[2/5] Loading MovieLens data...")
    movielens_ratings = pd.read_csv(movielens_ratings_path)
    movielens_movies = pd.read_csv(movielens_movies_path)
    print(f"MovieLens: {len(movielens_ratings)} ratings, {len(movielens_movies)} movies")
    
    # Step 3: Match movies
    print("\n[3/5] Matching Letterboxd movies to MovieLens...")
    matched_ratings, unmatched = match_letterboxd_to_movielens(letterboxd_df, movielens_movies)
    
    if len(matched_ratings) == 0:
        print("ERROR: No movies matched! Check your data files.")
        return None
    
    # Step 4: Add to dataset
    print("\n[4/5] Adding your ratings to MovieLens dataset...")
    combined_ratings, your_user_id = add_user_to_ratings(
        movielens_ratings, 
        matched_ratings
    )
    
    # Step 5: Get recommendations
    print("\n[5/5] Generating personalized recommendations...")

    print(f"\nYour User ID: {your_user_id}")
    print(f"Your movies in system: {len(matched_ratings)}")
    print(f"\nTop movies you rated:")
    top_rated = matched_ratings.nlargest(5, 'rating')[['title', 'rating']]
    print(top_rated.to_string(index=False))
    
    return {
        'combined_ratings': combined_ratings,
        'your_user_id': your_user_id,
        'matched_ratings': matched_ratings,
        'unmatched': unmatched,
        'movies_df': movielens_movies
    }


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("""
    HOW TO USE THIS SCRIPT:
    
    1. Export your Letterboxd data:
       - Go to https://letterboxd.com/settings/data
       - Click "Request my data"
       - Wait for email with download link
       - Download and extract the ZIP file
       - Use the 'ratings.csv' file
    
    2. Download MovieLens dataset:
       - Go to https://grouplens.org/datasets/movielens/
       - Download "ml-latest-small" (100k ratings)
       - Extract ratings.csv and movies.csv
    
    3. Run the pipeline:
    """)
    
    results = letterboxd_to_recommendations_pipeline(
        letterboxd_ratings_path='appendix/letterboxd_export/ratings.csv',
        movielens_ratings_path='ml_data/ratings.csv',
        movielens_movies_path='ml_data/movies.csv',
        letterboxd_liked_path='appendix/letterboxd_export/likes/films.csv',
        n_recommendations=10)   
    
    # Then use with CF model:
    from collab_filter import ItemBasedCF, create_user_item_matrix
    
    user_item = create_user_item_matrix(results['combined_ratings'])
    model = ItemBasedCF(user_item)
    model.compute_similarity()
    
    recommendations = model.recommend(
        results['your_user_id'], 
        n_recommendations=10,
        movies_df=results['movies_df']
    )
    
    print("\n🎬 YOUR PERSONALIZED RECOMMENDATIONS:")
    print("="*70)
    for movie_id, title, rating in recommendations:
        print(f"⭐ {rating:.2f} - {title}")