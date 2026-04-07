import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "test+prompts.json"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "moviemate.db"


def find_table(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        return None
    for t in tables:
        if 'movie' in t.lower():
            return t
    return tables[0]


def sample_rows_from_db(limit: int = 100) -> List[Dict[str, Any]]:
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    try:
        table = find_table(conn)
        if not table:
            return []
        cur = conn.execute(f"SELECT * FROM {table} ORDER BY RANDOM() LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return rows
    finally:
        conn.close()


def synthesize_rows(n: int) -> List[Dict[str, Any]]:
    genres = [
        'Drama', 'Comedy', 'Action', 'Thriller', 'Sci-Fi', 'Animation', 'Family', 'Mystery', 'War', 'Romance'
    ]
    directors = [
        'Christopher Nolan', 'Steven Spielberg', 'Quentin Tarantino', 'Greta Gerwig', 'Martin Scorsese',
        'Hayao Miyazaki', 'Ridley Scott', 'Patty Jenkins', 'James Cameron', 'Denis Villeneuve'
    ]
    rows = []
    for i in range(n):
        g = random.choice(genres)
        row = {
            'title': f'Sample Movie {i+1}',
            'genre': g,
            'year': random.randint(1970, 2024),
            'rating': round(random.uniform(4.0, 9.5), 1),
            'duration': random.randint(80, 180),
            'director': random.choice(directors),
        }
        rows.append(row)
    return rows


def make_prompt_from_row(row: Dict[str, Any], difficulty: str, idx: int) -> Dict[str, Any]:
    # Pull values with safe defaults
    title = row.get('title')
    genre = str(row.get('genre', '')).split(',')[0].strip()
    try:
        year = int(row.get('year'))
    except Exception:
        year = None
    try:
        rating = float(row.get('rating'))
    except Exception:
        rating = None
    try:
        duration = int(row.get('duration'))
    except Exception:
        duration = None
    director = str(row.get('director', ''))

    constraints: Dict[str, Any] = {}
    question = ''
    expected_answer = ''

    if difficulty == 'easy':
        # Simple single-constraint prompts
        if genre:
            question = f"Suggest {genre} movies."
            expected_answer = f"Should return {genre} movies."
            constraints['genre_contains'] = genre
        elif director:
            question = f"Movies directed by {director}."
            expected_answer = f"Should return movies where director contains {director}."
            constraints['director_contains'] = director
        else:
            # fallback
            question = "Recommend popular movies."
            expected_answer = "Should return general movie recommendations."

    elif difficulty == 'medium':
        # Combine two constraints
        if genre and rating is not None:
            min_rating = round(max(0.0, min(10.0, rating - random.uniform(0.0, 1.5))), 1)
            question = f"Good {genre} movies with rating above {min_rating}."
            expected_answer = f"Should prioritize {genre} movies with rating >= {min_rating}."
            constraints['genre_contains'] = genre
            constraints['min_rating'] = min_rating
        elif director and year is not None:
            qyear = max(1950, year - random.randint(0, 10))
            question = f"Movies directed by {director} after {qyear}."
            expected_answer = f"Should return movies where director contains {director} and year >= {qyear}."
            constraints['director_contains'] = director
            constraints['min_year'] = qyear
        else:
            # fallback to genre+year
            if genre and year is not None:
                q = max(1900, year - random.randint(0, 5))
                question = f"{genre} movies released after {q}."
                expected_answer = f"Should return {genre} movies with release year >= {q}."
                constraints['genre_contains'] = genre
                constraints['min_year'] = q

    else:  # hard
        # Combine 3 constraints where possible
        parts = []
        if genre:
            constraints['genre_contains'] = genre
            parts.append(genre)
        if year is not None:
            miny = year - random.randint(0, 5)
            constraints['min_year'] = miny
            parts.append(f"after {miny}")
        if rating is not None:
            minr = round(max(0.0, min(10.0, rating - random.uniform(0.0, 1.0))), 1)
            constraints['min_rating'] = minr
            parts.append(f"rating >= {minr}")
        if duration is not None and random.random() < 0.5:
            maxd = min(300, duration + random.randint(0, 20))
            constraints['max_duration'] = maxd
            parts.append(f"duration <= {maxd} mins")
        if director and random.random() < 0.4:
            constraints['director_contains'] = director
            parts.append(f"director {director}")

        if parts:
            question = f"Show {' '.join(parts)} {genre if genre else ''} movies.".strip()
            expected_answer = "Should return movies matching multiple constraints."
        else:
            question = "Find movies with several constraints."
            expected_answer = "Should return movies matching the requested constraints."

    return {
        'id': idx,
        'question': question,
        'expected_answer': expected_answer,
        'expected_constraints': constraints,
    }


def main(total: int = 100):
    # Try DB first
    rows = sample_rows_from_db(limit=total)
    if not rows:
        rows = synthesize_rows(total)

    prompts: List[Dict[str, Any]] = []
    difficulties = ['easy', 'medium', 'hard']
    # Shuffle rows and assign difficulties evenly
    random.shuffle(rows)
    for i in range(total):
        row = rows[i % len(rows)]
        # rotate difficulty to get distribution
        difficulty = difficulties[i % len(difficulties)]
        p = make_prompt_from_row(row, difficulty, i + 1)
        prompts.append(p)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open('w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    print(f'Wrote {len(prompts)} prompts to {OUT_PATH}')


if __name__ == '__main__':
    main(100)
