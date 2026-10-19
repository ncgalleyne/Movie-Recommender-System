import os
import pandas as pd
import ast
import requests

from functools import lru_cache
from utils.recommender_util import format_overview_html

def safe_movie_id(value):
    """Convert movie_id to int, or None if missing/invalid."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        mid = int(float(value))
        return mid if mid > 0 else None
    except (ValueError, TypeError):
        return None

@lru_cache(maxsize=10000)
def load_title_to_tmdb_id():
    """Fallback map title -> TMDB id from raw CSV files."""
    mapping = {}
    for path, id_col in [
        ("../data/tmdb_5000_movies.csv", "id"),
        ("../data/tmdb_movies.csv", "movie_id"),
    ]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["title", id_col])
        for _, row in df.iterrows():
            title = str(row["title"]).strip()
            mid = safe_movie_id(row[id_col])
            if title and mid and title not in mapping:
                mapping[title] = mid
    return mapping

TITLE_TO_TMDB_ID = {}
TITLE_TO_TMDB_ID.update(load_title_to_tmdb_id())

@lru_cache(maxsize=1)
def load_csv_metadata():
    """
    Fallback loader: if the saved pickle was trained before metadata columns
    were added, pull year/rating/genres/overview from the raw TMDB CSV on disk.
    """
    csv_path = '../data/tmdb_movies.csv'
    if not os.path.exists(csv_path):
        csv_path = '../data/tmdb_5000_movies.csv'
    if not os.path.exists(csv_path):
        return None

    raw = pd.read_csv(
        csv_path,
        usecols=['id', 'release_date', 'vote_average', 'genres', 'overview'],
    )
    raw = raw.rename(columns={'id': 'movie_id'})
    raw['release_year'] = pd.to_datetime(raw['release_date'], errors='coerce').dt.year
    raw['rating'] = raw['vote_average']
    raw['overview_display'] = raw['overview']

    def parse_genres(genres_json):
        try:
            names = [g['name'] for g in ast.literal_eval(genres_json)]
            return ', '.join(names) if names else None
        except (ValueError, SyntaxError, TypeError):
            return None

    raw['genres_display'] = raw['genres'].apply(parse_genres)
    return raw[['movie_id', 'release_year', 'rating', 'genres_display', 'overview_display']]
    
def enrich_movies_with_metadata(movies_df):
    """
    Merge display metadata into the movies dataframe when the pickle
    does not already contain release_year, rating, genres_display, or overview_display.
    """
    metadata_cols = ['release_year', 'rating', 'genres_display', 'overview_display']
    if all(col in movies_df.columns for col in metadata_cols):
        return movies_df

    csv_meta = load_csv_metadata()
    if csv_meta is None:
        return movies_df

    # Only fill in columns that are missing from the pickle
    missing_cols = [col for col in metadata_cols if col not in movies_df.columns]
    merge_cols = ['movie_id'] + missing_cols
    return movies_df.merge(csv_meta[merge_cols], on='movie_id', how='left')

def get_movie_details(row):
    """
    Build display fields for a recommendation card.
    Returns 'N/A' for any missing or empty value.
    """
    details = {
        'movie_id': safe_movie_id(row.get('movie_id')),
        'title': row['title'] if pd.notna(row.get('title')) else 'N/A',
        'year': 'N/A',
        'rating': 'N/A',
        'genres': 'N/A',
        'overview_html': 'N/A',
    }

    if 'release_year' in row.index and pd.notna(row['release_year']):
        details['year'] = str(int(row['release_year']))

    if 'rating' in row.index and pd.notna(row['rating']):
        details['rating'] = f"{float(row['rating']):.1f}/10"

    if 'genres_display' in row.index and pd.notna(row['genres_display']):
        genres_text = str(row['genres_display']).strip()
        if genres_text:
            details['genres'] = genres_text

    overview = row['overview_display'] if 'overview_display' in row.index else None
    details['overview_html'] = format_overview_html(overview)

    return details

def fetch_poster(movie_id, title=None):
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return "https://placehold.co/500x750?text=TMDB+API+Key+Missing"

    mid = resolve_movie_id(movie_id, title)
    if mid is None and title:
        try:
            search = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": api_key, "query": title, "language": "en-US"},
                timeout=15,
            )
            search.raise_for_status()
            results = search.json().get("results", [])
            if results:
                mid = results[0].get("id")
        except requests.RequestException:
            pass

    if mid is None:
        return "https://placehold.co/500x750?text=No+Poster+Found"

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{mid}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
        return "https://placehold.co/500x750?text=No+Poster+Found"
    except requests.RequestException:
        return "https://placehold.co/500x750?text=No+Poster+Found"

def resolve_movie_id(movie_id, title):
    """Get a valid TMDB id from movie_id column or title fallback map."""
    mid = safe_movie_id(movie_id)
    if mid is not None:
        return mid
    return TITLE_TO_TMDB_ID.get(title)