import time

import pandas as pd
from openai import OpenAI

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME, SYSTEM_PROMPT

class MovieLLM:
    def __init__(self, base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model_name=LLM_MODEL_NAME, prompt_template=SYSTEM_PROMPT):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_template = prompt_template

    def _context_block(self, title, results):
        if results is None or results.empty:
            return f"{title}: none"
        lines = [
            f"{i+1}. {r.title} ({int(r.year) if pd.notna(r.year) else 'NA'}), "
            f"rating={r.rating}, genre={r.genre}, director={r.director}, cast={getattr(r, 'cast', '')}"
            for i, r in results.head(8).iterrows()
        ]
        return f"{title}:\n" + "\n".join(lines)

    def llm_answer(self, query, rag_results=None, tool_results=None, merged_results=None, history_text=""):
        if not self.api_key:
            raise ValueError("Missing API key. Set LLM_API_KEY environment variable.")

        context = "\n\n".join([
            self._context_block("RAG Results", rag_results),
            self._context_block("SQL Tool Results", tool_results),
            self._context_block("Merged Results", merged_results),
        ])

        prompt = (
            f"History:\n{history_text}\n\n"
            f"Query:\n{query}\n\nRetrieved Movies:\n{context}"
        )

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        started = time.perf_counter()
        res = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.prompt_template},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )

        elapsed = max(time.perf_counter() - started, 1e-9)
        usage = getattr(res, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
        tokens_per_sec = completion_tokens / elapsed if completion_tokens > 0 else 0.0

        return {
            "text": res.choices[0].message.content,
            "analytics": {
                "model": self.model_name,
                "latency_sec": round(elapsed, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_sec": round(tokens_per_sec, 2),
            },
        }
