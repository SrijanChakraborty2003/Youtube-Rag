import os
from dotenv import load_dotenv

# Base Directory of Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Initial environment load from .env
load_dotenv(ENV_PATH, override=True)

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ChromaDB Storage Configuration
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "youtube_rag_chunks"

# FlashRank Re-ranker Model Configuration
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

# LLM Configuration (Ollama)
OLLAMA_MODEL_NAME = "gpt-oss:120b-cloud"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval & Re-ranking Parameters
NUM_EXPANDED_QUERIES = 5
TOP_DENSE_PER_QUERY = 10
TOP_BM25_PER_QUERY = 10
RRF_TOP_CANDIDATES = 10
FINAL_TOP_K = 5

# Chat Memory & Recursive Summary Parameters
CHAT_BUFFER_SIZE = 8  # Keep last 8 messages (4 turns) verbatim in context

# SQLite Database
DB_PATH = os.path.join(BASE_DIR, "rag_app.db")

# SMTP Messaging Service Configuration Helper
def get_smtp_config():
    """
    Dynamically re-reads .env so credential changes take effect immediately
    without requiring a server restart.
    """
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH, override=True)
    return {
        "host": os.environ.get("SMTP_HOST", "").strip(),
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "pass": os.environ.get("SMTP_PASS", "").strip(),
        "from": os.environ.get("SMTP_FROM", "").strip() or os.environ.get("SMTP_USER", "").strip()
    }

# Backward compatibility module-level variables
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "").strip()
SMTP_FROM = os.environ.get("SMTP_FROM", "").strip()
