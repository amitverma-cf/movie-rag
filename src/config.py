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
SYSTEM_PROMPT="You are MovieMate. Use only retrieved movies. Be short and clear."
MIN_VOTES = 10000
BATCH_SIZE = 50
REQ_DELAY = 0.05
TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"
FAISS_PATH = "data/movies.faiss"
DB_PATH = "data/moviemate.db"
