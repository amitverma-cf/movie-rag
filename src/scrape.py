import os
import sqlite3
import time

import pandas as pd
import requests

from src.config import BATCH_SIZE, DB_PATH, FAISS_PATH, MIN_VOTES, REQ_DELAY, TMDB_API_KEY, TMDB_BASE, TMDB_IMG


def db_connect(path):
    conn = sqlite3.connect(path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=60000;")
    return conn


def q(cur, sql, params=(), retry=8):
    for i in range(retry):
        try:
            cur.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < retry - 1:
                time.sleep(0.5 * (i + 1))
            else:
                raise


def cmt(conn, retry=8):
    for i in range(retry):
        try:
            conn.commit()
            return
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < retry - 1:
                time.sleep(0.5 * (i + 1))
            else:
                raise


def txt(values, n=None):
    if not values:
        return None
    out = [str(x) for x in values if x]
    return ", ".join(out[:n] if n else out) or None


def tmdb(path, api_key, params=None):
    p = {"api_key": api_key}
    p.update(params or {})
    r = requests.get(f"{TMDB_BASE}{path}", params=p, timeout=20)
    return r.json() if r.status_code == 200 else None


def tmdb_meta(imdb_id, api_key):
    f = tmdb(f"/find/{imdb_id}", api_key, {"external_source": "imdb_id"})
    movie_results = (f or {}).get("movie_results") or []
    if not movie_results:
        return None

    movie_id = movie_results[0].get("id")
    d = tmdb(f"/movie/{movie_id}", api_key, {"append_to_response": "credits,keywords,videos"}) if movie_id else None
    if not d:
        return None

    kw = txt([x.get("name") for x in ((d.get("keywords") or {}).get("keywords") or [])], 20)
    if not kw:
        kw = txt(d.get("genres") and [g.get("name") for g in d.get("genres")], 20)

    vids = (d.get("videos") or {}).get("results") or []
    yt = next((v for v in vids if v.get("site") == "YouTube" and v.get("type") == "Trailer"), None)
    trailer = f"https://www.youtube.com/watch?v={yt['key']}" if yt and yt.get("key") else None

    return {
        "description": d.get("overview"),
        "keywords": kw,
        "poster_url": f"{TMDB_IMG}{d['poster_path']}" if d.get("poster_path") else None,
        "cast": txt([x.get("name") for x in ((d.get("credits") or {}).get("cast") or [])], 4),
        "director": txt([x.get("name") for x in ((d.get("credits") or {}).get("crew") or []) if x.get("job") == "Director"]),
        "trailer_url": trailer,
    }


def scrape_movies(tmdb_api_key=TMDB_API_KEY, work_dir="data", min_votes=MIN_VOTES, batch_size=BATCH_SIZE, req_delay=REQ_DELAY):
    if not tmdb_api_key:
        raise ValueError("TMDB API key is required")

    os.makedirs(work_dir, exist_ok=True)
    db_path = DB_PATH if work_dir == "data" else f"{work_dir}/moviemate.db"
    faiss_path = FAISS_PATH if work_dir == "data" else f"{work_dir}/movies.faiss"

    print(f"Working folder: {os.path.abspath(work_dir)}")
    print(f"DB path: {os.path.abspath(db_path)}")
    print(f"FAISS path: {os.path.abspath(faiss_path)}")

    conn = db_connect(db_path)
    cur = conn.cursor()

    q(
        cur,
        """
CREATE TABLE IF NOT EXISTS movies (
  faiss_id INTEGER PRIMARY KEY AUTOINCREMENT, imdb_id TEXT UNIQUE, title TEXT, release_year INTEGER,
  rating REAL, duration_mins INTEGER, genre TEXT, director TEXT, cast TEXT, description TEXT,
  keywords TEXT, poster_url TEXT, trailer_url TEXT
)""",
    )
    cmt(conn)

    q(cur, "PRAGMA table_info(movies)")
    cols = {r[1] for r in cur.fetchall()}
    if "trailer_url" not in cols:
        q(cur, "ALTER TABLE movies ADD COLUMN trailer_url TEXT")
        cmt(conn)

    q(cur, "SELECT COUNT(*) FROM movies")
    n = cur.fetchone()[0]

    if n == 0:
        print("Downloading IMDb TSV files...")
        basics = pd.read_csv(
            "https://datasets.imdbws.com/title.basics.tsv.gz",
            sep="\t",
            low_memory=False,
            na_values="\\N",
        )
        ratings = pd.read_csv(
            "https://datasets.imdbws.com/title.ratings.tsv.gz",
            sep="\t",
            low_memory=False,
            na_values="\\N",
        )

        df = basics[basics["titleType"] == "movie"].merge(ratings, on="tconst", how="inner")
        df = df[df["numVotes"] > min_votes].sort_values("numVotes", ascending=False)
        print(f"Inserting {len(df)} movies...")

        for _, x in df.iterrows():
            q(
                cur,
                "INSERT OR IGNORE INTO movies (imdb_id,title,release_year,rating,duration_mins,genre) VALUES (?,?,?,?,?,?)",
                (x["tconst"], x["primaryTitle"], x["startYear"], x["averageRating"], x["runtimeMinutes"], x["genres"]),
            )

        cmt(conn)
        q(cur, "SELECT COUNT(*) FROM movies")
        print(f"Database initialized with {cur.fetchone()[0]} movies")
    else:
        print(f"Reusing existing DB with {n} movies")

    for pass_no in range(1):
        q(
            cur,
            """
SELECT imdb_id, description, keywords, poster_url, "cast", director, trailer_url
FROM movies
WHERE description IS NULL OR TRIM(COALESCE(description,''))='' OR description='FETCH_FAILED'
   OR keywords IS NULL OR TRIM(COALESCE(keywords,''))=''
   OR poster_url IS NULL OR TRIM(COALESCE(poster_url,''))=''
   OR "cast" IS NULL OR TRIM(COALESCE("cast",''))=''
   OR director IS NULL OR TRIM(COALESCE(director,''))=''
""",
        )

        pending = cur.fetchall()
        print(f"Pass {pass_no + 1}: rows needing fill = {len(pending)}")
        if not pending:
            break

        updated = 0
        failed = 0
        for i, (imdb_id, od, ok, op, oc, odr, ot) in enumerate(pending, start=1):
            try:
                m = tmdb_meta(imdb_id, tmdb_api_key)
                if not m:
                    failed += 1
                    continue

                nd = m["description"] if (not od or str(od).strip() == "" or od == "FETCH_FAILED") and m["description"] else od
                nk = m["keywords"] if (not ok or str(ok).strip() == "") and m["keywords"] else ok
                np = m["poster_url"] if (not op or str(op).strip() == "") and m["poster_url"] else op
                nc = m["cast"] if (not oc or str(oc).strip() == "") and m["cast"] else oc
                ndr = m["director"] if (not odr or str(odr).strip() == "") and m["director"] else odr
                nt = m["trailer_url"] if (not ot or str(ot).strip() == "") and m["trailer_url"] else ot

                q(
                    cur,
                    "UPDATE movies SET description=?, keywords=?, poster_url=?, \"cast\"=?, director=?, trailer_url=? WHERE imdb_id=?",
                    (nd, nk, np, nc, ndr, nt, imdb_id),
                )
                updated += 1
            except Exception:
                failed += 1

            if i % batch_size == 0:
                cmt(conn)
            time.sleep(req_delay)

        cmt(conn)
        print(f"Pass {pass_no + 1} done. Updated: {updated}, Failed: {failed}")

    q(
        cur,
        """
SELECT COUNT(*) FROM movies
WHERE description IS NULL OR TRIM(COALESCE(description,''))='' OR description='FETCH_FAILED'
   OR keywords IS NULL OR TRIM(COALESCE(keywords,''))=''
   OR poster_url IS NULL OR TRIM(COALESCE(poster_url,''))=''
   OR "cast" IS NULL OR TRIM(COALESCE("cast",''))=''
   OR director IS NULL OR TRIM(COALESCE(director,''))=''
   OR trailer_url IS NULL OR TRIM(COALESCE(trailer_url,''))=''
""",
    )
    print(f"Remaining rows with null/empty metadata fields: {cur.fetchone()[0]}")

    conn.close()


if __name__ == "__main__":
    key = TMDB_API_KEY or input("Enter TMDB API key: ").strip()
    scrape_movies(key)
