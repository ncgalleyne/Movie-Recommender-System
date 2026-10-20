from functools import lru_cache
from .recommender_util import sanitize_for_json

@lru_cache(maxsize=1)
def get_sorted_titles(titles_tuple):
    """Cache sorted movie titles for fast autocomplete lookups."""
    return sorted(titles_tuple)

def get_movie_by_id(movie_id, all_movies):
    movie = all_movies[all_movies["movie_id"] == movie_id]

    if movie.empty:
        return {}

    return sanitize_for_json(movie.iloc[0].to_dict())