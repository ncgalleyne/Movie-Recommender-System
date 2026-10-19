from functools import lru_cache

@lru_cache(maxsize=1)
def get_sorted_titles(titles_tuple):
    """Cache sorted movie titles for fast autocomplete lookups."""
    return sorted(titles_tuple)