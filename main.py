import os
import sys
import time
import argparse
from werkzeug.serving import make_server

# Reconfigure stdout/stderr to support printing unicode characters on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web.chat_app import app as chat_app

def print_welcome_banner():
    print("\n" + "=" * 76)
    print(" 🎬 VIDEO KNOWLEDGE RAG SYSTEM (MULTI-USER ISOLATED CHATS)")
    print("=" * 76)
    print(" 💬 Web Application URL : http://localhost:5000")
    print("=" * 76)
    print(" Production Architecture:")
    print("   • Multi-User Privacy  : Email & 6-Digit OTP Authentication (Zero Data Leakage)")
    print("   • Video Isolation     : Each Chat is an Isolated Knowledge Base")
    print("   • Long-Term Memory    : Immediate 8-Message Buffer + Recursive Rolling Summarization")
    print("   • Ingestion Pipeline  : Direct in-app background worker with real-time progress")
    print("   • LLM Synthesis       : gpt-oss:120b-cloud (via Ollama)")
    print("   • ASR Transcription   : NVIDIA NeMo Parakeet 0.6B CTC (Local GPU)")
    print("   • Chunker             : 60s Sliding Window (5s overlap) with Subtitle Cues")
    print("   • Dense Vector Store  : BAAI/bge-small-en-v1.5 + ChromaDB (Cosine)")
    print("   • Sparse Search       : Scoped BM25Okapi Keyword Matching")
    print("   • Query Expansion     : 5-Query Parallel Expansion via gpt-oss:120b-cloud")
    print("   • Fusion & Re-ranker  : Reciprocal Rank Fusion (Top 10) + FlashRank (Top 5)")
    print("   • Citations           : Exact Second Deep-Links [Video Title @ MM:SS](&t=XXs)")
    print("=" * 76)
    print(" Press Ctrl+C at any time to shut down the server.\n")

def run_server(port: int = 5000, host: str = "0.0.0.0"):
    print_welcome_banner()
    server = make_server(host, port, chat_app, threaded=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping web server...")
        server.server_close()
        print("[SHUTDOWN] Server stopped gracefully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Knowledge RAG System")
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port to run the web server on (default: 5000)")
    parser.add_argument("-H", "--host", default="0.0.0.0", help="Host to bind the web server to (default: 0.0.0.0)")

    args = parser.parse_args()
    run_server(port=args.port, host=args.host)
