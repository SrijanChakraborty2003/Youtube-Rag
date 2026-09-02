import os

# Base Directory of Project Root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Embedding Model Configuration
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ChromaDB Storage Configuration
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "youtube_rag_chunks"

# FlashRank Re-ranker Model Configuration
FLASHRANK_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"
