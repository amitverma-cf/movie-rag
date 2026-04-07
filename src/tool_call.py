import re
import sqlite3

import pandas as pd

from src.config import DB_PATH


class SQLiteMovieTool:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path

    def run_sql(self, sql, params=None):
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn, params=params or ())
        finally:
            conn.close()
        return df

    def search_movies(self, query, limit=8):
        ql = query.lower()
        where = ["1=1"]
        params = []

        genre_keywords = [
            "action", "drama", "comedy", "romance", "thriller", "horror", "sci-fi", "science fiction",
            "animation", "adventure", "fantasy", "crime", "mystery", "war", "family",
        ]
        for g in genre_keywords:
            if g in ql:
                genre = "Sci-Fi" if g == "science fiction" else g.title()
                where.append("genre LIKE ?")
                params.append(f"%{genre}%")
                break

        after = re.search(r"after\s+(19\d{2}|20\d{2})", ql)
        if after:
            where.append("release_year > ?")
            params.append(int(after.group(1)))

        before = re.search(r"before\s+(19\d{2}|20\d{2})", ql)
        if before:
            where.append("release_year < ?")
            params.append(int(before.group(1)))

        rating = re.search(r"(?:rating|imdb).{0,10}(\d(?:\.\d)?)", ql)
        if rating:
            where.append("rating >= ?")
            params.append(float(rating.group(1)))

        under_mins = re.search(r"(?:under|below)\s+(\d{2,3})\s*(?:mins|minutes|min)", ql)
        if under_mins:
            where.append("duration_mins <= ?")
            params.append(int(under_mins.group(1)))

        director_match = re.search(
            r"directed by\s+([a-zA-Z\s\.\-]+?)(?=\s+(after|before|with|under|below|rating|imdb|$)|$)",
            query,
            flags=re.IGNORECASE,
        )
        if director_match:
            where.append("director LIKE ?")
            params.append(f"%{director_match.group(1).strip()}%")

        title_hint = None
        similar_match = re.search(r"similar to\s+([a-zA-Z0-9\s:\-']+)", query, flags=re.IGNORECASE)
        if similar_match:
            title_hint = similar_match.group(1).strip()
        if title_hint:
            where.append("(title LIKE ? OR description LIKE ?)")
            params.extend([f"%{title_hint}%", f"%{title_hint}%"])

        sql = f"""
SELECT
  faiss_id,
  imdb_id,
  title,
  rating,
  release_year AS year,
  genre,
  director,
  "cast" AS cast,
  duration_mins AS duration,
  description,
  keywords,
  poster_url,
  trailer_url
FROM movies
WHERE {' AND '.join(where)}
ORDER BY COALESCE(rating, 0) DESC, COALESCE(release_year, 0) DESC
LIMIT ?
"""
        params.append(limit)
        return self.run_sql(sql, params=params)
