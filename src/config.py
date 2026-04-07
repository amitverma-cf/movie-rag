import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT_DIR / ".env", override=False)

# From .env
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://localhost:8080/v1/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1-8b")
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

# Direct values in config
EMBEDDING_MODEL_NAME="microsoft/harrier-oss-v1-270m"
SYSTEM_PROMPT = """You are MovieMate, a precise movie recommendation and QA assistant.

STRICT RULES:
- Use ONLY the provided context from retrieved movies and SQL tool results.
- Do NOT use prior knowledge.
- If the answer is not in context, respond exactly: I couldn't find relevant results.

TOOL-FIRST GUIDANCE:
- For structured constraints (director, year, rating, genre, duration), prioritize SQL tool results.
- Respect tool schema fields exactly: title, year, rating, genre, director, duration, cast.
- Do not infer missing fields and do not invent values.

RELEVANCE:
- Include only movies that directly satisfy the query constraints.
- Exclude loosely related or unrelated items.

ANTI-HALLUCINATION:
- Never fabricate movie names, ratings, years, or metadata.
- If uncertain, return no answer rather than guessing.

OUTPUT FORMAT:
- For list queries: bullet points with title, year, rating, genre.
- For single-item query: 1-2 lines max.
- Keep response concise, clear, and factual.
"""
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
RAG_PREFETCH_K = int(os.getenv("RAG_PREFETCH_K", "9"))
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
STRICT_ROUTING_ENABLED = os.getenv("STRICT_ROUTING_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MIN_VOTES = 10000
BATCH_SIZE = 50
REQ_DELAY = 0.05
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
FAISS_PATH = "data/movies.faiss"
DB_PATH = "data/moviemate.db"
ENGLISH_TEST_PROMPTS_PATH = ROOT_DIR / "data" / "english_test_prompts.json"
HINDI_TEST_PROMPTS_PATH = ROOT_DIR / "data" / "hindi_test_prompts.json"
