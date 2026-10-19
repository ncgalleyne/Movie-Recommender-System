from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml.recommendation_model import MovieRecommender
from ml.recommend_engine import enrich_movies_with_metadata, get_movie_details, fetch_poster
from utils.recommender_util import format_similarity_score

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

    try:
        # Ensure the model files exist
        recommender = MovieRecommender.load()
        movies = recommender.movies
        movies = enrich_movies_with_metadata(movies)

        recommendations = recommender.recommend(movie, top_n=8)
        recommended_movies = []
        recommended_movies_posters = []

        for rec in recommendations:
            movie_row = movies[movies["title"] == rec["title"]].iloc[0]
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

    except IndexError:
        raise HTTPException(
            status_code=404,
            detail="Movie details not found",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )