import re
import time

import pandas as pd

from src.eda import load_movies
from src.llm import MovieLLM
from src.similarity_search import Retriever
from src.tool_call import SQLiteMovieTool


def apply_simple_filters(df, q):
    if df is None or df.empty:
        return df

    ql = q.lower()

    m = re.search(r"after\s+(19\d{2}|20\d{2})", ql)
    if m:
        df = df[df["year"] > int(m.group(1))]

    m = re.search(r"before\s+(19\d{2}|20\d{2})", ql)
    if m:
        df = df[df["year"] < int(m.group(1))]

    m = re.search(r"(?:rating|imdb).{0,10}(\d(?:\.\d)?)", ql)
    if m:
        df = df[df["rating"] >= float(m.group(1))]

    m = re.search(r"(?:under|below)\s+(\d{2,3})\s*(?:mins|minutes|min)", ql)
    if m and "duration" in df.columns:
        df = df[df["duration"] <= int(m.group(1))]

    return df.reset_index(drop=True)


def _history_to_text(history):
    if not history:
        return ""

    lines = []
    for item in history[-12:]:
        if isinstance(item, dict):
            role = str(item.get("role", "")).strip().lower()
            content = item.get("content", "")
            if role in ("user", "assistant"):
                lines.append(f"{role.capitalize()}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            lines.append(f"User: {item[0]}")
            lines.append(f"Assistant: {item[1]}")
    return "\n".join(lines[-12:])


class MovieAgent:
    def __init__(self):
        self.movies = load_movies()
        self.retriever = Retriever()
        self.tool = SQLiteMovieTool()
        self.llm = MovieLLM()

    def _retrieve(self, query, mode, k=8):
        use_rag = mode in ("rag", "rag+tool")
        use_tool = mode in ("tool", "rag+tool")

        rag_hits = pd.DataFrame()
        if use_rag:
            rag_hits = apply_simple_filters(self.retriever.retrieve(query, self.movies, k=max(12, k)), query)
            rag_hits["source"] = "RAG"

        tool_hits = pd.DataFrame()
        if use_tool:
            tool_hits = apply_simple_filters(self.tool.search_movies(query, limit=max(12, k)), query)
            tool_hits["source"] = "SQL_TOOL"

        merged = pd.concat([rag_hits, tool_hits], ignore_index=True) if (not rag_hits.empty or not tool_hits.empty) else pd.DataFrame()
        if not merged.empty:
            dedupe_key = "imdb_id" if "imdb_id" in merged.columns else "title"
            merged = merged.drop_duplicates(subset=[dedupe_key]).reset_index(drop=True)
        return rag_hits, tool_hits, merged

    @staticmethod
    def _fallback_answer(query, merged):
        if merged is None or merged.empty:
            return f"No strong matches found for: {query}. Try adding genre, year, or rating constraints."
        lines = [
            f"- {r.title} ({int(r.year) if pd.notna(r.year) else 'NA'}) | rating={r.rating} | director={r.director}"
            for _, r in merged.head(5).iterrows()
        ]
        return "Top matches:\n" + "\n".join(lines)

    def run(self, query, mode="rag+tool", history=None, k=8):
        started = time.perf_counter()
        rag_hits, tool_hits, merged = self._retrieve(query, mode=mode, k=k)
        retrieval_latency = time.perf_counter() - started

        history_text = _history_to_text(history or [])
        try:
            llm_result = self.llm.llm_answer(query=query, rag_results=rag_hits, tool_results=tool_hits, merged_results=merged, history_text=history_text)
            answer_text = llm_result["text"]
            analytics = llm_result["analytics"]
        except ValueError:
            answer_text = self._fallback_answer(query, merged)
            analytics = {
                "model": "fallback-no-llm-key",
                "latency_sec": round(retrieval_latency, 3),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tokens_per_sec": 0.0,
            }

        return {
            "answer": answer_text,
            "analytics": analytics,
            "movies": merged.head(8),
            "rag_count": int(len(rag_hits)),
            "tool_count": int(len(tool_hits)),
        }