"""Central configuration: paths, model and retrieval settings (override via environment / .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / "backend" / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
STUDENT_PROFILES = SYNTHETIC_DIR / "student_profiles.json"
COURSES_DB = PROCESSED_DIR / "courses.db"   # structured store built by extract_data.py

CHROMA_DIR = ROOT / "backend" / "chroma_db"
COLLECTION = "vu_academic_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = int(os.getenv("TOP_K", "5"))
# Below this cosine similarity the best document chunk is treated as "nothing relevant retrieved"
# (calibrated: in-scope eval questions score >= 0.45, off-topic ones 0.13-0.38; see eval/README note).
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.40"))

# LLM providers: Groq models first (lowest latency), Gemini models as automatic fallback (see llm.py).
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODELS = os.getenv("GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b").split(",")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")  # low = fastest
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODELS = os.getenv("GEMINI_MODELS", "gemini-flash-lite-latest,gemini-3.5-flash-lite").split(",")
LLM_PIN = os.getenv("LLM_PIN", "")      # e.g. "groq:openai/gpt-oss-120b" — used by the eval for a fair comparison
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "20"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))  # room for full course lists
