import math
import os
import sqlite3
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.config import DB_PATH, EMBEDDING_MODEL_NAME, FAISS_PATH

WORK_DIR = os.path.dirname(DB_PATH) or "data"
FAISS_TMP_PATH = f"{WORK_DIR}/movies.tmp.faiss"
EMB_MODEL = EMBEDDING_MODEL_NAME


def bucket_year(y):
    if y is None:
        return ""
    try:
        y = int(y)
    except (TypeError, ValueError):
        return ""
    return f"Released in {y // 10 * 10}s"


def bucket_rating(r):
    if r is None:
        return ""
    try:
        r = float(r)
    except (TypeError, ValueError):
        return ""
    if r >= 8:
        return "Highly rated"
    if r >= 6:
        return "Good rating"
    return "Average rating"


def bucket_duration(d):
    if d is None:
        return ""
    try:
        d = int(d)
    except (TypeError, ValueError):
        return ""
    if d < 90:
        return "Short movie"
    if d <= 120:
        return "Standard length movie"
    return "Long movie"


def mean_pool(last_hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_states)
    masked = last_hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def get_existing_ids(index_obj):
    if hasattr(index_obj, "id_map"):
        return set(int(x) for x in faiss.vector_to_array(index_obj.id_map))
    return set()


def build_or_update_faiss(work_dir=WORK_DIR, emb_model=EMB_MODEL):
    os.makedirs(work_dir, exist_ok=True)
    db_path = f"{work_dir}/moviemate.db"
    faiss_path = f"{work_dir}/movies.faiss"
    faiss_tmp_path = f"{work_dir}/movies.tmp.faiss"

    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()

    cur.execute(
        """
DELETE FROM movies
WHERE title IS NULL OR TRIM(title)=''
   OR genre IS NULL OR TRIM(genre)=''
   OR director IS NULL OR TRIM(director)=''
   OR "cast" IS NULL OR TRIM("cast")=''
   OR description IS NULL OR TRIM(description)='' OR description='FETCH_FAILED'
   OR keywords IS NULL OR TRIM(keywords)=''
"""
    )
    conn.commit()

    cur.execute(
        'SELECT faiss_id, title, genre, director, "cast", description, keywords, release_year, rating, duration_mins '
        'FROM movies ORDER BY faiss_id'
    )
    rows = cur.fetchall()
    print(f"Rows available after cleanup: {len(rows)}")

    if not rows:
        print("No rows left for embedding.")
        conn.close()
        return

    idx = None
    existing_ids = set()
    rebuild_all = False

    if os.path.exists(faiss_path):
        try:
            idx = faiss.read_index(faiss_path)
            existing_ids = get_existing_ids(idx)
            if not hasattr(idx, "add_with_ids") or not hasattr(idx, "id_map"):
                rebuild_all = True
                print("Existing FAISS index does not expose ID map/add_with_ids; full rebuild required.")
            else:
                print(f"Existing FAISS found: vectors={idx.ntotal}, known_ids={len(existing_ids)}")
        except Exception as e:
            print(f"Could not read existing FAISS ({e}); full rebuild required.")
            idx = None
            existing_ids = set()
            rebuild_all = True
    else:
        print("No existing FAISS found; creating a new index.")

    missing_rows = [r for r in rows if int(r[0]) not in existing_ids]
    print(f"Missing embeddings detected: {len(missing_rows)}")

    if not rebuild_all and len(missing_rows) == 0:
        print("All rows already have embeddings. Skipping embedding step.")
        conn.close()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_mode = device == "cpu"
    print(f"Loading {emb_model} on {device.upper()}...")

    max_len = 384 if cpu_mode else 2048
    batch_size = 8 if cpu_mode else 16
    save_every = 25

    tok = AutoTokenizer.from_pretrained(emb_model)
    mdl = AutoModel.from_pretrained(emb_model, dtype="auto").to(device).eval()
    dim = int(getattr(mdl.config, "hidden_size"))

    if idx is not None and not rebuild_all and idx.d != dim:
        print(f"Existing index dim={idx.d} differs from model dim={dim}; full rebuild required.")
        rebuild_all = True

    if rebuild_all or idx is None:
        idx = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        target_rows = rows
        print(f"Rebuilding full index from {len(target_rows)} rows...")
    else:
        target_rows = missing_rows
        print(f"Appending {len(target_rows)} missing rows to existing index...")

    ids = [int(r[0]) for r in target_rows]
    texts = [
        (
            "Represent this movie for retrieval:\n"
            f"Title: {r[1]}\n"
            f"Genre: {r[2]}\n"
            f"Director: {r[3]}\n"
            f"Cast: {r[4]}\n"
            f"Keywords: {r[6]}\n"
            f"Summary: {r[5]}\n\n"
            f"{bucket_year(r[7])}\n"
            f"{bucket_rating(r[8])}\n"
            f"{bucket_duration(r[9])}"
        ).strip()
        for r in target_rows
    ]

    total_batches = math.ceil(len(texts) / batch_size) if texts else 0
    started = time.time()

    if texts:
        with torch.no_grad():
            for bno, i in enumerate(tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding batches"), start=1):
                batch_texts = texts[i : i + batch_size]
                batch_ids = np.asarray(ids[i : i + batch_size], dtype=np.int64)

                inp = tok(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
                inp = {k: v.to(device) for k, v in inp.items()}

                out = mdl(**inp)
                vec = mean_pool(out.last_hidden_state, inp["attention_mask"])
                vec = F.normalize(vec, p=2, dim=1)
                arr = vec.float().cpu().numpy().astype(np.float32)

                idx.add_with_ids(arr, batch_ids)

                if (bno % save_every == 0) or (bno == total_batches):
                    faiss.write_index(idx, faiss_tmp_path)
                    elapsed = time.time() - started
                    done = min(i + len(batch_texts), len(texts))
                    rate = done / elapsed if elapsed > 0 else 0.0
                    eta = (len(texts) - done) / rate if rate > 0 else float("inf")
                    print(
                        f"Progress: {done}/{len(texts)} rows | "
                        f"rate={rate:.2f} rows/s | ETA={eta/60:.1f} min | vectors={idx.ntotal}"
                    )

    faiss.write_index(idx, faiss_tmp_path)
    os.replace(faiss_tmp_path, faiss_path)
    print(
        f"FAISS saved: {faiss_path} | vectors={idx.ntotal} | dim={idx.d} | max_len={max_len} | batch_size={batch_size}"
    )

    conn.close()


if __name__ == "__main__":
    build_or_update_faiss()
