import typing
from typing import List, Union
from sentence_transformers import SentenceTransformer
from rag.config import EMBEDDING_MODEL_NAME

class EmbeddingManager:
    _instance = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            print(f"[EMBEDDINGS] Loading Bi-Encoder Embedding Model: {model_name}...")
            cls._instance.model = SentenceTransformer(model_name)
            cls._instance.model_name = model_name
            print(f"[EMBEDDINGS] Model loaded successfully.")
        return cls._instance

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string into a dense float vector.
        """
        if not text.strip():
            return []
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embeds a list of strings in batches into dense float vectors.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()
