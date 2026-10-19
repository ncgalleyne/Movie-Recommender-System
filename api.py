from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml.recommendation_model import MovieRecommender
from ml.recommend_engine import enrich_movies_with_metadata, get_movie_details, fetch_poster
from ml.search_engine import search_movies, find_exact_title
from utils.recommender_util import format_similarity_score, sanitize_for_json
from utils.search_util import get_sorted_titles
from db.watchlist_db import get_ratings, get_excluded_movie_ids, is_in_watchlist

# Import or initialize your existing recommender,
# movies DataFrame, and helper functions here.

app = FastAPI(title="CineMatch API")

# Allow your React development server to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    movie: str

class MovieSearchResponse(BaseModel):
    query: str
    suggestions: list[str]
    exact_title: str | None

try:
    # Ensure the model files exist
    RECOMMENDER = MovieRecommender.load()
    MOVIES = RECOMMENDER.movies
    MOVIES = enrich_movies_with_metadata(MOVIES)

    ALL_TITLES = get_sorted_titles(tuple(MOVIES['title'].tolist()))
    VALID_TITLES = set(ALL_TITLES)
except Exception as e:
    print(f"Error loading recommender: {e}")
    RECOMMENDER = None
    MOVIES = None
    ALL_TITLES = []
    VALID_TITLES = set()

@app.get("/")
def root():
    return {"message": "CineMatch API is running"}

@app.post("/api/recommend")
def recommend_movies(request: RecommendationRequest):
    movie = request.movie.strip()

    if not movie:
        raise HTTPException(
            status_code=400,
            detail="Movie title cannot be empty",
        )

    recommendations = RECOMMENDER.recommend(movie, top_n=8)
    recommended_movies = []
    recommended_movies_posters = []

    for rec in recommendations:
        movie_row = MOVIES[MOVIES["title"] == rec["title"]].iloc[0]
        details = get_movie_details(movie_row)
        details["similarity"] = format_similarity_score(rec["similarity_score"])
        recommended_movies.append(details)
        movie_id = rec.get("movie_id") or details.get("movie_id")
        recommended_movies_posters.append(
            fetch_poster(movie_id, title=details["title"])
        )

    return {
        "recommendations": recommended_movies,
        "posters": recommended_movies_posters,
    }

@app.get("/api/movies/search", response_model=MovieSearchResponse)
def movie_search(
    q: str = Query(default="", max_length=200),
):
    query = q.strip()

    suggestions = (
        search_movies(query, ALL_TITLES)
        if query
        else []
    )

    exact_title = find_exact_title(query, ALL_TITLES)

    return {
        "query": query,
        "suggestions": suggestions,
        "exact_title": exact_title,
    }

@app.get("/api/for-you")
def get_for_you_recommendations():
    ratings = get_ratings()
    excluded_ids = get_excluded_movie_ids()

    recommendations = RECOMMENDER.recommend_for_profile(ratings, excluded_ids, top_n=8)

    results = []

    for rec in recommendations:
        movie_id = rec.get("movie_id")
        title = rec.get("title", "")

        # Existing movie metadata
        year = rec.get("year")
        rating = rec.get("rating")
        genres = [
            genre.strip()
            for genre in (rec.get("genres_display") or "").split(",")
            if genre.strip()
        ]

        # Convert recommender score into a percentage.
        match = max(0, min(99, round(rec.get("score", 0) * 100)))

        # Resolve poster URL using your existing poster lookup.
        image = fetch_poster(movie_id, title)

        results.append({
            "id": movie_id,
            "title": title,
            "year": int(year) if year is not None else None,
            "runtime": None,  # Populate from movie metadata
            "genres": genres,
            "match": match,
            "image": image,
            "imageAlt": f"Poster for {title}",
            "reason": rec.get("reason", ""),
            "rating": rating if rating is not None else 0,
            "watchlist": is_in_watchlist(movie_id) if movie_id is not None else False,
            "liked": False,      # Populate from saved user feedback
            "disliked": False,   # Populate from saved user feedback
            "hidden": False,     # Populate from saved movie status
            "hiddenGem": False,  # Derive from your chosen criteria
            "score": rec.get("score", 0),
            "releaseDate": int(year) if year is not None else None,
            "similarOnly": False,
        })

    return sanitize_for_json(results)

@app.get("/api/movies/suggestions")
def get_movie_suggestions(q: str = Query(default="", max_length=200)):
    return search_movies(q, ALL_TITLES)
