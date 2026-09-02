import os
import json
from typing import List, Dict, Any
from rag.chunker import parse_srt_file, create_sliding_window_chunks
from rag.vectorstore import VectorStoreManager

def ingest_video(
    video_folder: str,
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 5.0,
    auto_vectorize: bool = True
) -> List[Dict[str, Any]]:
    """
    Reads audio.srt and metadata.json from video_folder, performs 60s/5s sliding
    window chunking, saves the chunks to chunks.json, embeds and indexes chunks into
    ChromaDB, and returns the chunk list.
    """
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder does not exist: {video_folder}")

    srt_path = os.path.join(video_folder, "audio.srt")
    metadata_path = os.path.join(video_folder, "metadata.json")

    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")

    # Load metadata if available
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[INGEST] Warning: Could not parse metadata.json in {video_folder}: {e}")

    if "title" not in metadata:
        metadata["title"] = os.path.basename(os.path.abspath(video_folder))
    if "url" not in metadata:
        metadata["url"] = ""

    print(f"[INGEST] Parsing SRT file for chunking: {srt_path}")
    cues = parse_srt_file(srt_path)
    
    print(f"[INGEST] Creating sliding window chunks (window={chunk_seconds}s, overlap={overlap_seconds}s)...")
    chunks = create_sliding_window_chunks(
        cues=cues,
        video_metadata=metadata,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds
    )

    # Save chunks to chunks.json in the video folder
    chunks_path = os.path.join(video_folder, "chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"[INGEST] Successfully generated {len(chunks)} chunks and saved to: {chunks_path}")

    # Automatically vectorize and index into ChromaDB
    if auto_vectorize and chunks:
        print(f"[INGEST] Automatically indexing {len(chunks)} chunks into ChromaDB Vector Store...")
        try:
            vstore = VectorStoreManager()
            vstore.upsert_chunks(chunks)
            print(f"[INGEST] Vector indexing complete!")
        except Exception as vec_err:
            print(f"[INGEST] Warning: Failed to vectorize chunks into ChromaDB: {vec_err}")

    return chunks

