"""Central configuration: paths, model and retrieval settings (override via environment / .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")

# Vector index built by ingest.py: embeddings.npy (one L2-normalised row per chunk) + chunks.json (text + metadata).
# ingest.py also copies the few data files the API reads at runtime into index/data/, because a deployment of
# backend/ alone (Vercel, Root Directory = backend) does not contain the repo's data/ folder.
INDEX_DIR = BACKEND_DIR / "index"
RUNTIME_DATA_FILES = ["processed/term_context.md", "processed/course_catalogue.json", "processed/courses.db",
                      "synthetic/student_profiles.json"]
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_MODEL_DIR = BACKEND_DIR / "models" / EMBED_MODEL  # official ONNX export + tokenizer (see embedder.py)

DATA_DIR = ROOT / "data" if (ROOT / "data").is_dir() else INDEX_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
STUDENT_PROFILES = SYNTHETIC_DIR / "student_profiles.json"
COURSES_DB = PROCESSED_DIR / "courses.db"   # structured store built by extract_data.py
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
