import json
import sqlite3
import time

import pandas as pd
from tqdm.auto import tqdm

from src.config import DB_PATH


def patch_missing_trailers(work_dir="data", sleep_seconds=1.5, commit_every=50):
    try:
        from youtube_search import YoutubeSearch
    except ImportError as e:
        raise ImportError("Missing dependency 'youtube-search'. Install it with: pip install youtube-search") from e

    db_path = DB_PATH if work_dir == "data" else f"{work_dir}/moviemate.db"
    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()

    cur.execute(
        """
SELECT imdb_id, title, release_year
FROM movies
WHERE trailer_url IS NULL OR TRIM(COALESCE(trailer_url, '')) = ''
"""
    )
    missing_trailers = cur.fetchall()

    print(f"Found {len(missing_trailers)} movies missing trailers. Searching YouTube...")

    updated = 0
    failed = 0
    for i, (imdb_id, title, year) in enumerate(tqdm(missing_trailers, desc="Fixing trailers"), start=1):
        try:
            movie_title = (title or "").strip()
            if not movie_title:
                failed += 1
                continue

            search_query = f"{movie_title} trailer"
            raw = YoutubeSearch(search_query, max_results=1).to_json()
            data = json.loads(raw) if raw else {}
            videos = data.get("videos") or []

            yt_link = None
            if videos:
                first = videos[0] or {}
                suffix = (first.get("url_suffix") or "").strip()
                if suffix:
                    yt_link = suffix if suffix.startswith("http") else f"https://www.youtube.com{suffix}"
                elif first.get("id"):
                    yt_link = f"https://www.youtube.com/watch?v={first['id']}"

            if yt_link:
                cur.execute(
                    "UPDATE movies SET trailer_url = ? WHERE imdb_id = ?",
                    (yt_link, imdb_id),
                )
                updated += 1

            if i % commit_every == 0:
                conn.commit()
            time.sleep(sleep_seconds)
        except Exception as e:
            print(f"Failed to find trailer for {title} ({year}): {e}")
            failed += 1

    conn.commit()
    conn.close()
    print(f"Trailer patching complete. Updated: {updated}, Failed: {failed}")


def _missing(value):
    return value is None or str(value).strip() == "" or value == "FETCH_FAILED"


def generate_missing_keywords_with_keybert(work_dir="data", keybert_model="all-MiniLM-L6-v2"):
    from keybert import KeyBERT

    db_path = DB_PATH if work_dir == "data" else f"{work_dir}/moviemate.db"
    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()

    cur.execute(
        """
SELECT faiss_id, imdb_id, title, description, keywords, poster_url, "cast", director, trailer_url
FROM movies ORDER BY faiss_id
"""
    )
    rows = cur.fetchall()

    report = []
    for r in rows:
        missing_columns = []
        if _missing(r[3]):
            missing_columns.append("description")
        if _missing(r[4]):
            missing_columns.append("keywords")
        if _missing(r[5]):
            missing_columns.append("poster_url")
        if _missing(r[6]):
            missing_columns.append("cast")
        if _missing(r[7]):
            missing_columns.append("director")
        if _missing(r[8]):
            missing_columns.append("trailer_url")
        if missing_columns:
            report.append((r[0], r[1], r[2], ", ".join(missing_columns)))

    print(f"Rows still incomplete before KeyBERT: {len(report)}")
    if report:
        print(pd.DataFrame(report, columns=["faiss_id", "imdb_id", "title", "missing_columns"]).head(200))

    cur.execute(
        """
SELECT faiss_id, description FROM movies
WHERE (keywords IS NULL OR TRIM(keywords)='')
  AND description IS NOT NULL
  AND TRIM(description)!=''
  AND description!='FETCH_FAILED'
"""
    )
    kw_rows = cur.fetchall()
    print(f"Rows where keywords will be generated from description: {len(kw_rows)}")

    if kw_rows:
        kb = KeyBERT(model=keybert_model)
        for faiss_id, desc in kw_rows:
            kws = kb.extract_keywords(desc, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=8)
            text = ", ".join([k for k, _ in kws if k])
            if text:
                cur.execute("UPDATE movies SET keywords=? WHERE faiss_id=?", (text, faiss_id))
        conn.commit()

    cur.execute(
        """
SELECT faiss_id, imdb_id, title, description, keywords, poster_url, "cast", director, trailer_url
FROM movies ORDER BY faiss_id
"""
    )
    rows2 = cur.fetchall()
    report2 = []

    for r in rows2:
        missing_columns = []
        if _missing(r[3]):
            missing_columns.append("description")
        if _missing(r[4]):
            missing_columns.append("keywords")
        if _missing(r[5]):
            missing_columns.append("poster_url")
        if _missing(r[6]):
            missing_columns.append("cast")
        if _missing(r[7]):
            missing_columns.append("director")
        if _missing(r[8]):
            missing_columns.append("trailer_url")
        if missing_columns:
            report2.append((r[0], r[1], r[2], ", ".join(missing_columns)))

    print(f"Rows still incomplete after KeyBERT: {len(report2)}")
    if report2:
        print(pd.DataFrame(report2, columns=["faiss_id", "imdb_id", "title", "missing_columns"]).head(200))

    conn.close()


if __name__ == "__main__":
    patch_missing_trailers()
    generate_missing_keywords_with_keybert()
