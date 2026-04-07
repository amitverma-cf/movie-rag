import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import DB_PATH


def load_movies(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    movies = pd.read_sql_query(
        """
SELECT
  faiss_id, imdb_id, title, rating, release_year AS year, genre, director,
    "cast" AS cast, duration_mins AS duration, description, keywords, poster_url, trailer_url
FROM movies
ORDER BY faiss_id
""",
        conn,
    )
    conn.close()

    for c in ["title", "genre", "director", "cast", "description"]:
        movies[c] = movies[c].fillna("").astype(str).str.strip()

    movies["rating"] = pd.to_numeric(movies["rating"], errors="coerce")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["duration"] = pd.to_numeric(movies["duration"], errors="coerce")
    movies = movies[movies["title"].ne("")].drop_duplicates("imdb_id").reset_index(drop=True)

    return movies


def build_eda_artifacts(movies):
    summary = movies[["rating", "year", "duration"]].describe().T
    summary.insert(0, "rows", len(movies))

    fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    sns.histplot(movies["rating"].dropna(), bins=20, kde=True, ax=ax[0])
    ax[0].set_title("Rating Distribution")

    g = movies["genre"].str.split(",").explode().str.strip()
    top_g = g[g.ne("")].value_counts().head(12)
    sns.barplot(x=top_g.values, y=top_g.index, ax=ax[1])
    ax[1].set_title("Top Genres")

    trend = movies.dropna(subset=["year", "rating"]).groupby("year", as_index=False)["rating"].mean()
    sns.lineplot(data=trend, x="year", y="rating", ax=ax[2])
    ax[2].set_title("Avg Rating by Year")

    plt.tight_layout()
    return summary, fig


def run_eda(movies):
    summary, fig = build_eda_artifacts(movies)
    print("Rows:", len(movies))
    print(summary)
    plt.show()


if __name__ == "__main__":
    movies_df = load_movies()
    run_eda(movies_df)
