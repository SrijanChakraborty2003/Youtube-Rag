import re
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest

from rag.config import (
    FLASHRANK_MODEL_NAME,
    TOP_DENSE_PER_QUERY,
    TOP_BM25_PER_QUERY,
    RRF_TOP_CANDIDATES,
    FINAL_TOP_K,
)
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

    def dense_search(
        self,
        query_text: str,
        top_k: int = 10,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs dense vector retrieval via ChromaDB, optionally scoped to chat_id and user_id.
        """
        return self.vectorstore.query_similar(query_text, top_k=top_k, chat_id=chat_id, user_id=user_id)

    def sparse_search(
        self,
        query_text: str,
        top_k: int = 10,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs sparse keyword retrieval via BM25, optionally scoped to chat_id and user_id.
        """
        query_tokens = SimpleTokenizer(query_text)
        if not query_tokens:
            return []

        # If scoped to a specific chat, build a scoped BM25 index on the fly
        if chat_id:
            chat_chunks = self.vectorstore.get_all_chunks(chat_id=chat_id, user_id=user_id)
            if not chat_chunks:
                return []
            corpus = [SimpleTokenizer(c["plain_text"]) for c in chat_chunks]
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            top_indices = scored_indices[:min(top_k, len(scored_indices))]
            results = []
            for rank, idx in enumerate(top_indices, 1):
                if scores[idx] <= 0:
                    continue
                chunk = chat_chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "document": chunk["text"],
                    "metadata": chunk,
                    "score": float(scores[idx]),
                    "rank": rank
                })
            return results

        if not self.bm25 or not self.all_chunks:
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
        top_dense: int = TOP_DENSE_PER_QUERY,
        top_bm25: int = TOP_BM25_PER_QUERY,
        rrf_k: float = 60.0,
        top_candidates_count: int = RRF_TOP_CANDIDATES,
        final_top_k: int = FINAL_TOP_K,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs Single-Query Hybrid Search using Reciprocal Rank Fusion (RRF) and FlashRank Re-ranking.
        """
        return self.multi_query_hybrid_search(
            original_query=query_text,
            queries=[query_text],
            top_dense=top_dense,
            top_bm25=top_bm25,
            rrf_k=rrf_k,
            rrf_top_candidates=top_candidates_count,
            final_top_k=final_top_k,
            chat_id=chat_id,
            user_id=user_id
        )

    def multi_query_hybrid_search(
        self,
        original_query: str,
        queries: List[str],
        top_dense: int = TOP_DENSE_PER_QUERY,
        top_bm25: int = TOP_BM25_PER_QUERY,
        rrf_k: float = 60.0,
        rrf_top_candidates: int = RRF_TOP_CANDIDATES,
        final_top_k: int = FINAL_TOP_K,
        chat_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes multiple generated queries in parallel, scoped to a specific chat_id/user_id.
        """
        # Map of chunks for lookup
        if chat_id:
            chat_chunks = self.vectorstore.get_all_chunks(chat_id=chat_id, user_id=user_id)
            if not chat_chunks:
                print(f"[RETRIEVER] No chunks indexed for chat_id={chat_id}.")
                return []
            chunk_lookup = {c["chunk_id"]: c for c in chat_chunks}
        else:
            if self.vectorstore.collection.count() != len(self.all_chunks):
                self.refresh_index()
            if not self.all_chunks:
                return []
            chunk_lookup = self.chunk_id_map

        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if not clean_queries:
            clean_queries = [original_query]

        # 1. Parallel execution of dense and sparse search for each query
        def _search_query(q: str) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
            d_res = self.dense_search(q, top_k=top_dense, chat_id=chat_id, user_id=user_id)
            s_res = self.sparse_search(q, top_k=top_bm25, chat_id=chat_id, user_id=user_id)
            return q, d_res, s_res

        max_workers = min(len(clean_queries), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            search_outputs = list(executor.map(_search_query, clean_queries))

        # 2. Reciprocal Rank Fusion across all query executions
        rrf_scores: Dict[str, float] = {}

        for q_text, dense_results, sparse_results in search_outputs:
            for rank, item in enumerate(dense_results, 1):
                cid = item["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

            for item in sparse_results:
                cid = item["chunk_id"]
                rank = item["rank"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank))

        if not rrf_scores:
            return []

        # Sort candidate chunks by RRF score descending -> Select Top candidates
        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:min(rrf_top_candidates, len(sorted_candidates))]

        # 3. FlashRank Cross-Encoder Re-ranking on the top candidates
        passages = []
        for cid, rrf_score in top_candidates:
            chunk = chunk_lookup.get(cid)
            if not chunk:
                continue

            text_content = chunk.get("plain_text", chunk.get("text", ""))
            passages.append({
                "id": cid,
                "text": text_content,
                "meta": chunk
            })

        if not passages:
            return []

        # Re-rank candidate passages against the original user query
        rerank_req = RerankRequest(query=original_query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_req)

        # 4. Extract and Format Top N chunks
        final_results = []
        for item in reranked_results[:final_top_k]:
            cid = item["id"]
            chunk = chunk_lookup.get(cid, {})
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

