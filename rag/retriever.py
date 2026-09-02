import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from rag.config import FLASHRANK_MODEL_NAME
from rag.vectorstore import VectorStoreManager

def SimpleTokenizer(text: str) -> List[str]:
    """
    Simple lowercase word tokenizer for BM25 keyword matching.
    """
    return re.findall(r'\w+', text.lower())

class HybridRetriever:
    def __init__(self, vectorstore: VectorStoreManager = None, reranker_model: str = FLASHRANK_MODEL_NAME):
        self.vectorstore = vectorstore or VectorStoreManager()
        self.all_chunks: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self.chunk_id_map: Dict[str, Dict[str, Any]] = {}

        # Initialize FlashRank Cross-Encoder Re-ranker
        print(f"[RETRIEVER] Initializing FlashRank Cross-Encoder ({reranker_model})...")
        self.ranker = Ranker(model_name=reranker_model)
        print("[RETRIEVER] FlashRank initialized successfully.")

        self.refresh_index()

    def refresh_index(self):
        """
        Loads all chunks from ChromaDB and builds/updates the BM25 sparse index.
        """
        print("[RETRIEVER] Loading chunks for BM25 index...")
        self.all_chunks = self.vectorstore.get_all_chunks()
        self.chunk_id_map = {c["chunk_id"]: c for c in self.all_chunks}

        if self.all_chunks:
            corpus = [SimpleTokenizer(c["plain_text"]) for c in self.all_chunks]
            self.bm25 = BM25Okapi(corpus)
            print(f"[RETRIEVER] BM25 Index built over {len(self.all_chunks)} chunks.")
        else:
            self.bm25 = None
            print("[RETRIEVER] Warning: BM25 corpus is empty.")

    def dense_search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs dense vector retrieval via ChromaDB.
        """
        return self.vectorstore.query_similar(query_text, top_k=top_k)

    def sparse_search(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs sparse keyword retrieval via BM25.
        """
        if not self.bm25 or not self.all_chunks:
            return []

        query_tokens = SimpleTokenizer(query_text)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top_indices = scored_indices[:min(top_k, len(scored_indices))]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] <= 0:
                continue
            chunk = self.all_chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document": chunk["text"],
                "metadata": chunk,
                "score": float(scores[idx]),
                "rank": rank
            })
        return results

    def hybrid_search(
        self,
        query_text: str,
        top_dense: int = 10,
        top_bm25: int = 10,
        rrf_k: float = 60.0,
        top_candidates_count: int = 10,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF) and FlashRank Re-ranking.
        Steps:
        1. Query ChromaDB for top_dense vectors.
        2. Query BM25 for top_bm25 keyword matches.
        3. Combine using RRF score: 1/(k + rank_dense) + 1/(k + rank_bm25).
        4. Pass candidate chunks to FlashRank Cross-Encoder for precision re-scoring.
        5. Return top final_top_k results.
        """
        # Always refresh map/index if new items were added
        if self.vectorstore.collection.count() != len(self.all_chunks):
            self.refresh_index()

        if not self.all_chunks:
            print("[RETRIEVER] Knowledge base is empty.")
            return []

        # 1. Dense Search
        dense_results = self.dense_search(query_text, top_k=top_dense)
        
        # 2. Sparse BM25 Search
        sparse_results = self.sparse_search(query_text, top_k=top_bm25)

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}

        # Dense ranks
        for rank, item in enumerate(dense_results, 1):
            cid = item["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        # Sparse ranks
        for item in sparse_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        if not rrf_scores:
            return []

        # Sort by RRF score descending
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:min(top_candidates_count, len(sorted_candidates))]

        # 4. Prepare for FlashRank Re-ranking
        passages = []
        candidate_chunk_objs = []

        for cid, rrf_score in top_candidates:
            chunk = self.chunk_id_map.get(cid)
            if not chunk:
                continue
            
            # Plain text context for cross-encoder reranking
            text_content = chunk.get("plain_text", chunk.get("text", ""))
            passages.append({
                "id": cid,
                "text": text_content,
                "meta": chunk
            })
            candidate_chunk_objs.append(chunk)

        if not passages:
            return []

        # Rerank with FlashRank
        rerank_req = RerankRequest(query=query_text, passages=passages)
        reranked_results = self.ranker.rerank(rerank_req)

        # 5. Format Top Results
        final_results = []
        for item in reranked_results[:final_top_k]:
            cid = item["id"]
            chunk = self.chunk_id_map.get(cid, {})
            score = float(item["score"])
            
            final_results.append({
                "chunk_id": cid,
                "rerank_score": round(score, 4),
                "video_title": chunk.get("video_title", ""),
                "video_url": chunk.get("video_url", ""),
                "start_seconds": chunk.get("start_seconds", 0.0),
                "end_seconds": chunk.get("end_seconds", 0.0),
                "start_timestamp": chunk.get("start_timestamp", ""),
                "end_timestamp": chunk.get("end_timestamp", ""),
                "timestamp_url": chunk.get("timestamp_url", ""),
                "text": chunk.get("text", ""),
                "plain_text": chunk.get("plain_text", ""),
                "cues": chunk.get("cues", [])
            })

        return final_results
