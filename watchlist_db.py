import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "watchlist.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create watchlist, personal_ratings, and movie_status tables if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER NOT NULL UNIQUE,
                title TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS personal_ratings (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movie_status (
                movie_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN ('unseen', 'watched', 'dismissed', 'want_to_watch')
                ),
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_to_watchlist(movie_id: int, title: str) -> bool:
    """
    Add a movie to the watchlist.
    Returns True if added, False if already present.
    """
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
        if existing:
            return False

        conn.execute(
            "INSERT INTO watchlist (movie_id, title, added_at) VALUES (?, ?, ?)",
            (movie_id, title, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return True


def remove_from_watchlist(movie_id: int) -> bool:
    """Remove a movie from the watchlist. Returns True if a row was deleted."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_watchlist():
    """Return all watchlist entries, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, title, added_at
            FROM watchlist
            ORDER BY added_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def is_in_watchlist(movie_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
    return row is not None


def get_watchlist_stats():
    """Return watchlist summary statistics."""
    items = get_watchlist()
    return {
        "total_movies": len(items),
        "titles": [item["title"] for item in items],
    }


def save_rating(movie_id: int, title: str, rating: int) -> bool:
    """
    Save or update a movie rating (1-5 stars).
    Returns True if successful, False if validation fails.
    """
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return False

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO personal_ratings (movie_id, title, rating, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(movie_id) DO UPDATE SET
                title = excluded.title,
                rating = excluded.rating,
                updated_at = excluded.updated_at
            """,
            (movie_id, title, rating, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return True


def get_ratings():
    """Return all personal ratings as a list of dicts."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, title, rating, updated_at
            FROM personal_ratings
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_rating(movie_id: int):
    """Return a single movie's rating dict or None if not found."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT movie_id, title, rating, updated_at FROM personal_ratings WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
    return dict(row) if row else None


def set_movie_status(movie_id: int, title: str, status: str) -> bool:
    """
    Set a movie's status (unseen, watched, dismissed, want_to_watch).
    Returns True if successful, False if invalid status.
    """
    valid_statuses = {"unseen", "watched", "dismissed", "want_to_watch"}
    if status not in valid_statuses:
        return False

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO movie_status (movie_id, title, status, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(movie_id) DO UPDATE SET
                title = excluded.title,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (movie_id, title, status, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return True


def get_excluded_movie_ids():
    """Return list of movie_ids that should be excluded from recommendations (watched or dismissed)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id
            FROM movie_status
            WHERE status IN ('watched', 'dismissed')
            """
        ).fetchall()
    return [row["movie_id"] for row in rows]
