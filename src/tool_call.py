import json
import sqlite3

import pandas as pd
from openai import OpenAI

from src.config import DB_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME


PARSER_SYSTEM_PROMPT = """You convert movie queries into strict JSON filters.
Return JSON only, no markdown.

Schema:
{
    "genres": ["Action", "Drama", "Sci-Fi"],
    "imdb_id": null,
    "title_contains": null,
    "cast_contains": null,
    "keyword_contains": null,
    "description_contains": null,
    "year": null,
    "min_year": null,
    "max_year": null,
    "min_rating": null,
    "max_rating": null,
    "min_duration": null,
    "max_duration": null,
    "director": null,
    "title_hint": null
}

Rules:
- Use null for missing fields.
- genres must be a list.
- Keep values concise and SQL-friendly.
"""


class SQLiteMovieTool:
    def __init__(self, db_path=DB_PATH, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model_name=LLM_MODEL_NAME):
        self.db_path = db_path
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name

    def run_sql(self, sql, params=None):
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(sql, conn, params=params or ())
        finally:
            conn.close()
        return df

    @staticmethod
    def _normalize_intent(raw):
        allowed_genres = {
            "action", "drama", "comedy", "romance", "thriller", "horror", "sci-fi", "science fiction",
            "animation", "adventure", "fantasy", "crime", "mystery", "war", "family", "biography", "documentary",
        }
        mapped = {
            "science fiction": "Sci-Fi",
            "sci-fi": "Sci-Fi",
            "action": "Action",
            "drama": "Drama",
            "comedy": "Comedy",
            "romance": "Romance",
            "thriller": "Thriller",
            "horror": "Horror",
            "animation": "Animation",
            "adventure": "Adventure",
            "fantasy": "Fantasy",
            "crime": "Crime",
            "mystery": "Mystery",
            "war": "War",
            "family": "Family",
            "biography": "Biography",
            "documentary": "Documentary",
        }

        intent = {
            "genres": [],
            "imdb_id": None,
            "title_contains": None,
            "cast_contains": None,
            "keyword_contains": None,
            "description_contains": None,
            "year": None,
            "min_year": None,
            "max_year": None,
            "min_rating": None,
            "max_rating": None,
            "min_duration": None,
            "max_duration": None,
            "director": None,
            "title_hint": None,
        }
        if not isinstance(raw, dict):
            return intent

        genres = raw.get("genres")
        if isinstance(genres, str):
            genres = [genres]
        if isinstance(genres, list):
            normalized = []
            for g in genres:
                gl = str(g or "").strip().lower()
                if gl in allowed_genres and mapped.get(gl):
                    normalized.append(mapped[gl])
            intent["genres"] = list(dict.fromkeys(normalized))

        for key in ("year", "min_year", "max_year", "min_duration", "max_duration"):
            val = raw.get(key)
            try:
                intent[key] = int(val) if val is not None else None
            except (TypeError, ValueError):
                intent[key] = None

        for key in ("min_rating", "max_rating"):
            val = raw.get(key)
            try:
                intent[key] = float(val) if val is not None else None
            except (TypeError, ValueError):
                intent[key] = None

        director = raw.get("director")
        intent["director"] = str(director).strip() if director else None

        imdb_id = raw.get("imdb_id")
        intent["imdb_id"] = str(imdb_id).strip() if imdb_id else None

        title_contains = raw.get("title_contains")
        intent["title_contains"] = str(title_contains).strip() if title_contains else None

        cast_contains = raw.get("cast_contains")
        intent["cast_contains"] = str(cast_contains).strip() if cast_contains else None

        keyword_contains = raw.get("keyword_contains")
        intent["keyword_contains"] = str(keyword_contains).strip() if keyword_contains else None

        description_contains = raw.get("description_contains")
        intent["description_contains"] = str(description_contains).strip() if description_contains else None

        title_hint = raw.get("title_hint")
        intent["title_hint"] = str(title_hint).strip() if title_hint else None

        return intent

    def _parse_query_with_llm(self, query):
        if not self.api_key:
            raise ValueError("Missing API key. Set LLM_API_KEY environment variable.")

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        res = client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        content = (res.choices[0].message.content or "{}").strip()
        try:
            raw = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM parser did not return valid JSON.") from exc
        return self._normalize_intent(raw)

    def search_movies(self, query, limit=8):
        intent = self._normalize_intent(self._parse_query_with_llm(query))
        where = ["1=1"]
        params = []

        if intent["genres"]:
            where.append("(" + " OR ".join(["genre LIKE ?"] * len(intent["genres"])) + ")")
            params.extend([f"%{g}%" for g in intent["genres"]])

        if intent["imdb_id"]:
            where.append("imdb_id = ?")
            params.append(intent["imdb_id"])

        if intent["title_contains"]:
            where.append("title LIKE ?")
            params.append(f"%{intent['title_contains']}%")

        if intent["cast_contains"]:
            where.append('"cast" LIKE ?')
            params.append(f"%{intent['cast_contains']}%")

        if intent["keyword_contains"]:
            where.append("keywords LIKE ?")
            params.append(f"%{intent['keyword_contains']}%")

        if intent["description_contains"]:
            where.append("description LIKE ?")
            params.append(f"%{intent['description_contains']}%")

        if intent["year"] is not None:
            where.append("release_year = ?")
            params.append(int(intent["year"]))

        if intent["min_year"] is not None:
            where.append("release_year >= ?")
            params.append(int(intent["min_year"]))

        if intent["max_year"] is not None:
            where.append("release_year <= ?")
            params.append(int(intent["max_year"]))

        if intent["min_rating"] is not None:
            where.append("rating >= ?")
            params.append(float(intent["min_rating"]))

        if intent["max_rating"] is not None:
            where.append("rating <= ?")
            params.append(float(intent["max_rating"]))

        if intent["max_duration"] is not None:
            where.append("duration_mins <= ?")
            params.append(int(intent["max_duration"]))

        if intent["min_duration"] is not None:
            where.append("duration_mins >= ?")
            params.append(int(intent["min_duration"]))

        if intent["director"]:
            where.append("director LIKE ?")
            params.append(f"%{intent['director']}%")

        if intent["title_hint"]:
            where.append("(title LIKE ? OR description LIKE ? OR keywords LIKE ?)")
            params.extend([f"%{intent['title_hint']}%", f"%{intent['title_hint']}%", f"%{intent['title_hint']}%"])

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
