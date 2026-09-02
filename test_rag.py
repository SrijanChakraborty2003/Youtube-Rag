import os
import json
from rag.ingest import ingest_video
from rag.retriever import HybridRetriever
from rag.vectorstore import VectorStoreManager

def main():
    print("=" * 70)
    print("EMPIRICAL VERIFICATION: Phase 2 (Embeddings & ChromaDB) & Phase 3 (Hybrid Search + FlashRank)")
    print("=" * 70)

    # 1. Ingest existing video folder
    video_folder = os.path.join(os.path.dirname(__file__), "op", "Needle 2 The 45M Parameter Model That Runs Everywhere")
    print(f"\nStep 1: Running video ingestion on: {video_folder}")
    chunks = ingest_video(video_folder)
    print(f"Ingested {len(chunks)} chunks into ChromaDB.")

    # 2. Check VectorStore directly
    vstore = VectorStoreManager()
    print(f"\nStep 2: ChromaDB Total Count: {vstore.collection.count()}")

    # 3. Test Hybrid Retrieval & FlashRank Reranking
    print("\nStep 3: Initializing HybridRetriever (BM25 + ChromaDB + FlashRank)...")
    retriever = HybridRetriever(vectorstore=vstore)

    test_queries = [
        "how to set the volume to 50 on Android",
        "how many parameters does Needle 2 have and RAM usage",
        "deploy on Android phone using Termux"
    ]

    for q_idx, query in enumerate(test_queries, 1):
        print("\n" + "-" * 70)
        print(f"QUERY #{q_idx}: '{query}'")
        print("-" * 70)

        results = retriever.hybrid_search(
            query_text=query,
            top_dense=10,
            top_bm25=10,
            final_top_k=3
        )

        for rank, res in enumerate(results, 1):
            print(f"\n  Match #{rank} [Re-rank Score: {res['rerank_score']:.4f}]")
            print(f"  Chunk ID   : {res['chunk_id']}")
            print(f"  Time Range : {res['start_timestamp']} --> {res['end_timestamp']}")
            print(f"  Deep Link  : {res['timestamp_url']}")
            snippet = res['plain_text'][:160] + "..." if len(res['plain_text']) > 160 else res['plain_text']
            print(f"  Snippet    : {snippet}")

    print("\n" + "=" * 70)
    print("Empirical Verification Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
