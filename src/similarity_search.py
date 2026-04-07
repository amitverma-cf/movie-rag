import faiss
import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer

from src.config import EMBEDDING_MODEL_NAME, FAISS_PATH
from src.eda import load_movies

EMB_MODEL = EMBEDDING_MODEL_NAME


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bsz = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(bsz, device=last_hidden_states.device), seq_lens]


class Retriever:
    def __init__(self, faiss_path=FAISS_PATH, emb_model=EMB_MODEL):
        self.index = faiss.read_index(faiss_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(emb_model)
        self.encoder = AutoModel.from_pretrained(emb_model, dtype="auto").to(self.device).eval()

    def embed_query(self, query):
        with torch.no_grad():
            inputs = self.tokenizer([query], padding=True, truncation=True, max_length=384, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.encoder(**inputs)
            vec = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.float().cpu().numpy().astype(np.float32)

    def retrieve(self, query, movies, k=8):
        q = self.embed_query(query)
        scores, ids = self.index.search(q, k)
        ids_list = [int(x) for x in ids[0] if int(x) != -1]

        out = movies[movies["faiss_id"].isin(ids_list)].copy()
        out["faiss_id"] = out["faiss_id"].astype(int)
        score_map = {int(i): float(s) for i, s in zip(ids[0], scores[0]) if int(i) != -1}
        out["score"] = out["faiss_id"].map(score_map)
        out = out.sort_values("score", ascending=False).reset_index(drop=True)
        return out


if __name__ == "__main__":
    movies_df = load_movies()
    retriever = Retriever()
    print(retriever.retrieve("highly rated sci-fi movies after 2010", movies_df, 8)[["title", "year", "rating", "genre", "score"]])
