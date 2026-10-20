from functools import lru_cache
from .recommender_util import sanitize_for_json
import json

@lru_cache(maxsize=1)
def get_sorted_titles(titles_tuple):
    """Cache sorted movie titles for fast autocomplete lookups."""
    return sorted(titles_tuple)

def get_movie_by_id(movie_id, all_movies):
    movie = all_movies[all_movies["movie_id"] == movie_id]

    if movie.empty:
        return {}

    return sanitize_for_json(movie.iloc[0].to_dict())

def get_movie_credits(movie, cast_limit=5):
    def parse_json(value):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []
        return value if isinstance(value, list) else []

    cast = parse_json(movie.get("cast"))
    crew = parse_json(movie.get("crew"))

    return {
        "cast": [member["name"] for member in cast[:cast_limit] if member.get("name")],
        "director": next(
            (
                member["name"]
                for member in crew
                if member.get("job", "").lower() == "director"
            ),
            None,
        ),
    }