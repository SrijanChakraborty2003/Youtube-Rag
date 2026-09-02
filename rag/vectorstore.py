import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from rag.config import CHROMA_DB_DIR, COLLECTION_NAME
from rag.embeddings import EmbeddingManager

class VectorStoreManager:
    def __init__(self, db_dir: str = CHROMA_DB_DIR, collection_name: str = COLLECTION_NAME):
        self.db_dir = db_dir
        self.collection_name = collection_name
        
        os.makedirs(self.db_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedding_manager = EmbeddingManager()
        print(f"[VECTORSTORE] Initialized ChromaDB at '{self.db_dir}' (Collection: '{self.collection_name}')")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Generates embeddings for chunk plain_text and stores/upserts them along
        with rich metadata into ChromaDB.
        """
        if not chunks:
            print("[VECTORSTORE] No chunks to upsert.")
            return 0

        ids = []
        documents = []
        metadatas = []
        plain_texts = []

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            plain_text = chunk.get("plain_text", chunk.get("text", ""))
            
            ids.append(chunk_id)
            documents.append(chunk.get("text", plain_text))
            plain_texts.append(plain_text)

            cues_str = json.dumps(chunk.get("cues", []), ensure_ascii=False)
            metadata = {
                "chunk_id": chunk_id,
                "video_title": str(chunk.get("video_title", "")),
                "video_url": str(chunk.get("video_url", "")),
                "start_seconds": float(chunk.get("start_seconds", 0.0)),
                "end_seconds": float(chunk.get("end_seconds", 0.0)),
                "start_timestamp": str(chunk.get("start_timestamp", "")),
                "end_timestamp": str(chunk.get("end_timestamp", "")),
                "timestamp_url": str(chunk.get("timestamp_url", "")),
                "plain_text": plain_text,
                "cues_json": cues_str
            }
            metadatas.append(metadata)

        print(f"[VECTORSTORE] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_manager.embed_texts(plain_texts)

        print(f"[VECTORSTORE] Upserting {len(chunks)} vectors to ChromaDB...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"[VECTORSTORE] Upsert complete. Collection count: {self.collection.count()}")
        return len(chunks)

    def query_similar(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Queries ChromaDB for top_k most similar chunks using dense vector search.
        """
        if self.collection.count() == 0:
            print("[VECTORSTORE] Vector store is empty.")
            return []

        query_embedding = self.embedding_manager.embed_text(query_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            res_ids = results["ids"][0]
            res_docs = results.get("documents", [[]])[0]
            res_metas = results.get("metadatas", [[]])[0]
            res_dists = results.get("distances", [[]])[0]

            for idx in range(len(res_ids)):
                meta = dict(res_metas[idx]) if idx < len(res_metas) else {}
                cues_json = meta.get("cues_json", "[]")
                try:
                    cues = json.loads(cues_json)
                except Exception:
                    cues = []
                meta["cues"] = cues

                formatted_results.append({
                    "chunk_id": res_ids[idx],
                    "document": res_docs[idx] if idx < len(res_docs) else "",
                    "metadata": meta,
                    "distance": float(res_dists[idx]) if idx < len(res_dists) else 0.0
                })

        return formatted_results

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieves all indexed chunks from ChromaDB for sparse index construction (BM25).
        """
        count = self.collection.count()
        if count == 0:
            return []

        all_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        chunks = []
        ids = all_data.get("ids", [])
        docs = all_data.get("documents", [])
        metas = all_data.get("metadatas", [])

        for idx in range(len(ids)):
            meta = dict(metas[idx]) if idx < len(metas) else {}
            cues_json = meta.get("cues_json", "[]")
            try:
                cues = json.loads(cues_json)
            except Exception:
                cues = []
            
            chunk_obj = {
                "chunk_id": ids[idx],
                "video_title": meta.get("video_title", ""),
                "video_url": meta.get("video_url", ""),
                "start_seconds": meta.get("start_seconds", 0.0),
                "end_seconds": meta.get("end_seconds", 0.0),
                "start_timestamp": meta.get("start_timestamp", ""),
                "end_timestamp": meta.get("end_timestamp", ""),
                "timestamp_url": meta.get("timestamp_url", ""),
                "text": docs[idx],
                "plain_text": meta.get("plain_text", docs[idx]),
                "cues": cues
            }
            chunks.append(chunk_obj)

        return chunks
