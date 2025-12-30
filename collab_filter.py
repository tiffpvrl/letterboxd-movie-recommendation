import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.sparse import csr_matrix
# from sklearn.neighbors import NearestNeighbors

# 1. Load and prepare data
ratings = pd.read_csv("ml_data/ratings.csv")
movies = pd.read_csv("ml_data/movies.csv")

# 2. User-Item matrix
def create_user_item_matrix(ratings_df):
    """
    Create a user-item matrix where rows=users, columns=movies
    Values are ratings (0 if not rated)
    """
    user_item_matrix = ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    return user_item_matrix

# 3. Item based collaborative filtering
class ItemBasedCF:
    def __init__(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.item_similarity = None
        
    def compute_similarity(self, metric='cosine'):
        """Compute similarity between items (movies)"""
        if metric == 'cosine':
            # Transpose so items are rows
            self.item_similarity = cosine_similarity(self.user_item_matrix.T)
            self.item_similarity = pd.DataFrame(
                self.item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
        return self.item_similarity
    
    def predict_rating(self, user_id, movie_id, k=10):
        """
        Predict rating for a user-movie pair
        k: number of similar items to consider
        """
        if user_id not in self.user_item_matrix.index:
            return None
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Get movies the user has rated
        rated_movies = user_ratings[user_ratings > 0].index
        
        if movie_id not in self.item_similarity.index:
            return None
        
        # Get similarity scores for target movie
        similarities = self.item_similarity.loc[movie_id, rated_movies]
        
        # Get top-k most similar movies that user has rated
        top_k_similar = similarities.nlargest(k)
        
        # Weighted average of ratings
        if top_k_similar.sum() == 0:
            return user_ratings.mean()
        
        weighted_ratings = sum(top_k_similar * user_ratings[top_k_similar.index])
        prediction = weighted_ratings / top_k_similar.sum()
        
        return prediction
    
    def recommend(self, user_id, n_recommendations=5, movies_df=None):
        """Get top N recommendations for a user"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Predict ratings for all unrated movies
        predictions = []
        for movie_id in unrated_movies:
            pred = self.predict_rating(user_id, movie_id)
            if pred is not None:
                # Get movie title if movies_df provided
                if movies_df is not None and movie_id in movies_df['movieId'].values:
                    title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
                    predictions.append((movie_id, title, pred))
                else:
                    predictions.append((movie_id, f'Movie {movie_id}', pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return predictions[:n_recommendations]
    
# 4. Usage
if __name__ == "__main__":
    # Create user-item matrix
    user_item = create_user_item_matrix(ratings)
    
    print("User-Item Matrix Shape:", user_item.shape)
    print("\n" + "="*50)
    
    # Item-Based CF
    print("\n ITEM-BASED COLLABORATIVE FILTERING")
    print("="*50)
    item_cf = ItemBasedCF(user_item)
    item_cf.compute_similarity()
    
    user_id = user_item.index[180]
    recommendations = item_cf.recommend(user_id, n_recommendations=5, movies_df=movies)
    print(f"\nTop 5 recommendations for User {user_id}:")
    for movie_id, title, rating in recommendations:
        print(f"  {title} (ID: {movie_id}): Predicted Rating = {rating:.2f}")
