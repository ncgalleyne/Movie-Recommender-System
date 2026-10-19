"""
Training and inference for the content-based recommender.
Uses TF-IDF vectors + genre/keyword/cast-aware hybrid re-ranking.
"""

import os
import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.feature_engineering import build_feature_record

load_dotenv()

MODEL_DIR = "model"
MOVIES_PKL = os.path.join(MODEL_DIR, "movies_dict.pkl")
SIMILARITY_PKL = os.path.join(MODEL_DIR, "similarity.pkl")
METADATA_PKL = os.path.join(MODEL_DIR, "movie_metadata.pkl")
VECTORIZER_PKL = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Hybrid re-ranking weights (cosine base + metadata overlap bonuses)
GENRE_MATCH_BOOST = 0.40
KEYWORD_MATCH_BOOST = 0.25
CAST_MATCH_BOOST = 0.15
DIRECTOR_MATCH_BOOST = 0.12
NO_GENRE_PENALTY = 0.20  # multiply score when zero genre overlap

# Rating weights for profile-based recommendations
RATING_WEIGHTS = {
    5: 2.0,  # Loved it
    4: 1.0,  # Liked it
    3: 0.0,  # Okay
    2: -1.0,  # Disliked it
    1: -2.0,  # Hated it
}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def hybrid_score(
    cosine_sim: float,
    query_meta: dict,
    candidate_meta: dict,
) -> float:
    """Combine cosine similarity with metadata overlap signals."""
    genre_sim = _jaccard(query_meta["genre_set"], candidate_meta["genre_set"])
    keyword_sim = _jaccard(query_meta["keyword_set"], candidate_meta["keyword_set"])
    cast_sim = _jaccard(query_meta["cast_set"], candidate_meta["cast_set"])
    director_sim = _jaccard(query_meta["director_set"], candidate_meta["director_set"])

    score = float(cosine_sim)
    score *= 1.0 + GENRE_MATCH_BOOST * genre_sim
    score *= 1.0 + KEYWORD_MATCH_BOOST * keyword_sim
    score *= 1.0 + CAST_MATCH_BOOST * cast_sim
    score *= 1.0 + DIRECTOR_MATCH_BOOST * director_sim

    if query_meta["genre_set"] and candidate_meta["genre_set"] and genre_sim == 0:
        score *= NO_GENRE_PENALTY

    return score


def format_similarity_score(score: float) -> str:
    """Format hybrid score as percentage (clamped 0–99 for display)."""
    pct = max(0, min(99, round(float(score) * 100)))
    return f"Similarity: {pct}%"


def train_with_audit(movies_df: pd.DataFrame, audit: bool = True) -> dict:
    """
    Train model from preprocessed movie dataframe.
    Returns audit statistics dict.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    audit_log = {"steps": []}

    def log_step(name: str, count: int, detail: str = ""):
        audit_log["steps"].append({"step": name, "count": count, "detail": detail})
        if audit:
            msg = f"[{name}] rows={count}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    df = movies_df.copy()
    log_step("input_movies", len(df))

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(build_feature_record(row))

    features_df = pd.DataFrame(feature_rows)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    log_step("after_feature_engineering", len(df))

    tags = df["tags"].tolist()
    max_features = min(12000, max(2000, len(df) * 4))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        sublinear_tf=True,
    )
    vectors = vectorizer.fit_transform(tags)
    log_step(
        "after_vectorization",
        vectors.shape[0],
        f"shape={vectors.shape}, max_features={max_features}, vocab={len(vectorizer.vocabulary_)}",
    )

    similarity = cosine_similarity(vectors).astype(np.float16)
    log_step(
        "similarity_matrix",
        similarity.shape[0],
        f"shape={similarity.shape}, min={similarity.min():.4f}, max={similarity.max():.4f}",
    )

    assert len(df) == similarity.shape[0] == similarity.shape[1], (
        f"Dimension mismatch: movies={len(df)}, similarity={similarity.shape}"
    )

    output_df = df[
        [
            "movie_id",
            "title",
            "tags",
            "release_year",
            "rating",
            "genres_display",
            "overview_display",
        ]
    ].copy()

    metadata = []
    for _, row in df.iterrows():
        metadata.append(
            {
                "genre_set": list(row["genre_set"]),
                "keyword_set": list(row["keyword_set"]),
                "cast_set": list(row["cast_set"]),
                "director_set": list(row["director_set"]),
            }
        )

    pickle.dump(output_df.to_dict(), open(MOVIES_PKL, "wb"))
    pickle.dump(similarity, open(SIMILARITY_PKL, "wb"))
    pickle.dump(metadata, open(METADATA_PKL, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PKL, "wb"))

    audit_log["final_movie_count"] = len(output_df)
    audit_log["similarity_shape"] = similarity.shape
    return audit_log


class MovieRecommender:
    """Loaded recommender for inference."""

    def __init__(self, movies_df: pd.DataFrame, similarity: np.ndarray, metadata: list):
        self.movies = movies_df
        self.similarity = similarity
        self.metadata = metadata
        self._title_to_pos = {
            title: pos for pos, title in enumerate(movies_df["title"].values)
        }

    @classmethod
    def load(cls):
        movies = pd.DataFrame(pickle.load(open(MOVIES_PKL, "rb")))
        similarity = pickle.load(open(SIMILARITY_PKL, "rb"))
        metadata = pickle.load(open(METADATA_PKL, "rb"))
        if len(movies) != similarity.shape[0] or len(movies) != len(metadata):
            raise ValueError(
                f"Model files mismatch: movies={len(movies)}, "
                f"similarity={similarity.shape}, metadata={len(metadata)}"
            )
        return cls(movies, similarity, metadata)

    def recommend(self, title: str, top_n: int = 8, candidate_pool: int = 50) -> list[dict]:
        if title not in self._title_to_pos:
            raise KeyError(f"Movie not found: {title}")

        query_pos = self._title_to_pos[title]
        query_meta = {
            "genre_set": set(self.metadata[query_pos]["genre_set"]),
            "keyword_set": set(self.metadata[query_pos]["keyword_set"]),
            "cast_set": set(self.metadata[query_pos]["cast_set"]),
            "director_set": set(self.metadata[query_pos]["director_set"]),
        }

        cosine_row = self.similarity[query_pos]
        candidates = np.argsort(cosine_row)[::-1]

        ranked = []
        for pos in candidates:
            if pos == query_pos:
                continue
            cand_meta = {
                "genre_set": set(self.metadata[pos]["genre_set"]),
                "keyword_set": set(self.metadata[pos]["keyword_set"]),
                "cast_set": set(self.metadata[pos]["cast_set"]),
                "director_set": set(self.metadata[pos]["director_set"]),
            }
            hybrid = hybrid_score(float(cosine_row[pos]), query_meta, cand_meta)
            ranked.append((pos, hybrid, float(cosine_row[pos])))
            if len(ranked) >= candidate_pool:
                break

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:top_n]

        if top:
            max_hybrid = top[0][1]
            if max_hybrid > 0:
                top = [(p, h / max_hybrid, c) for p, h, c in top]

        results = []
        for pos, norm_score, raw_cosine in top:
            row = self.movies.iloc[pos]
            results.append(
                {
                    "position": pos,
                    "title": row["title"],
                    "movie_id": (
                        int(float(row["movie_id"]))
                        if pd.notna(row["movie_id"]) and float(row["movie_id"]) > 0
                        else None
                    ),
                    "similarity_score": norm_score,
                    "cosine_score": raw_cosine,
                    "genres_display": row.get("genres_display", "N/A"),
                    "year": row.get("release_year"),
                    "rating": row.get("rating"),
                    "overview_display": row.get("overview_display"),
                }
            )
        return results

    def recommend_for_profile(
        self,
        ratings: list[dict],
        excluded_movie_ids: list[int] = None,
        top_n: int = 20,
    ) -> list[dict]:
        """
        Generate personalized recommendations based on user's movie ratings.

        Args:
            ratings: List of dicts with keys 'movie_id', 'title', 'rating' (1-5)
            excluded_movie_ids: Movie IDs to exclude (watched/dismissed)
            top_n: Number of recommendations to return

        Returns:
            List of ranked movie records with score components and explanations.
        """
        if excluded_movie_ids is None:
            excluded_movie_ids = []

        # Build a movie_id -> position lookup from the model
        movie_id_to_pos = {}
        for pos, row in self.movies.iterrows():
            mid = row.get("movie_id")
            if pd.notna(mid):
                try:
                    mid_int = int(float(mid))
                    if mid_int > 0:
                        movie_id_to_pos[mid_int] = pos
                except (ValueError, TypeError):
                    continue

        # Build list of rated movies with their positions and weights
        rated_movies = []
        rated_movie_ids = set()
        for rating_entry in ratings:
            movie_id = rating_entry.get("movie_id")
            rating = rating_entry.get("rating")

            if movie_id is None or rating is None:
                continue

            try:
                movie_id_int = int(float(movie_id))
            except (ValueError, TypeError):
                continue

            rated_movie_ids.add(movie_id_int)

            if movie_id_int in movie_id_to_pos:
                pos = movie_id_to_pos[movie_id_int]
                weight = RATING_WEIGHTS.get(rating, 0.0)
                if weight != 0.0:  # Skip neutral ratings
                    rated_movies.append((pos, weight))

        if not rated_movies:
            return []

        # Score all candidates
        candidate_scores = {}
        for candidate_pos in range(len(self.movies)):
            candidate_id = self.movies.iloc[candidate_pos].get("movie_id")
            if pd.isna(candidate_id):
                continue
            try:
                candidate_id_int = int(float(candidate_id))
            except (ValueError, TypeError):
                continue

            # Skip excluded movies and already-rated movies
            if candidate_id_int in excluded_movie_ids or candidate_id_int in rated_movie_ids:
                continue

            # Compute weighted similarity score
            score = 0.0
            positive_sim = 0.0
            negative_sim = 0.0

            for rated_pos, weight in rated_movies:
                cosine_sim = float(self.similarity[rated_pos][candidate_pos])
                if weight > 0:
                    positive_sim += weight * cosine_sim
                else:
                    negative_sim += abs(weight) * cosine_sim
                score += weight * cosine_sim

            # Add quality bonus based on TMDB rating
            tmdb_rating = self.movies.iloc[candidate_pos].get("rating")
            if pd.notna(tmdb_rating):
                quality_bonus = (float(tmdb_rating) - 5.0) / 10.0  # Normalize around 5/10
                score += quality_bonus

            candidate_scores[candidate_pos] = {
                "score": score,
                "positive_sim": positive_sim,
                "negative_sim": negative_sim,
            }

        # Sort by score and get top N
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1]["score"], reverse=True
        )[:top_n]

        # Build explanation and return results
        results = []
        for pos, score_data in sorted_candidates:
            row = self.movies.iloc[pos]
            movie_id = row.get("movie_id")
            if pd.isna(movie_id):
                continue
            try:
                movie_id_int = int(float(movie_id))
            except (ValueError, TypeError):
                continue

            # Generate explanation
            parts = []
            if score_data["positive_sim"] > 0:
                parts.append("similar to movies you liked")
            if score_data["negative_sim"] > 0:
                parts.append("different from movies you disliked")
            if pd.notna(row.get("rating")) and float(row.get("rating")) > 7.0:
                parts.append("highly rated by critics")

            explanation = "; ".join(parts) if parts else "based on your viewing preferences"

            results.append(
                {
                    "movie_id": movie_id_int,
                    "title": row["title"],
                    "score": score_data["score"],
                    "reason": explanation,
                    "genres_display": row.get("genres_display", "N/A"),
                    "year": row.get("release_year"),
                    "rating": row.get("rating"),
                    "overview_display": row.get("overview_display"),
                }
            )

        return results


def explain_recommendation(source_title: str, recommended_title: str, recommender: MovieRecommender) -> str:
    """Human-readable explanation for a recommendation pair."""
    q = recommender._title_to_pos[source_title]
    c = recommender._title_to_pos[recommended_title]
    qm = recommender.metadata[q]
    cm = recommender.metadata[c]
    shared_genres = set(qm["genre_set"]) & set(cm["genre_set"])
    shared_kw = set(qm["keyword_set"]) & set(cm["keyword_set"])
    shared_cast = set(qm["cast_set"]) & set(cm["cast_set"])
    shared_dir = set(qm["director_set"]) & set(cm["director_set"])
    parts = []
    if shared_genres:
        parts.append(f"shared genres: {', '.join(sorted(shared_genres))}")
    if shared_kw:
        parts.append(f"shared keywords: {', '.join(sorted(list(shared_kw)[:5]))}")
    if shared_cast:
        parts.append(f"shared cast: {', '.join(sorted(shared_cast))}")
    if shared_dir:
        parts.append(f"shared director(s): {', '.join(sorted(shared_dir))}")
    if not parts:
        parts.append("textual similarity in overview/tags")
    return "; ".join(parts)
