import sys
import argparse
from rag.agent import VideoRAGAgent
from rag.config import OLLAMA_MODEL_NAME

# Ensure stdout supports UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def print_banner():
    print("=" * 72)
    print(" 🎬 VIDEO KNOWLEDGE RAG - CONVERSATIONAL AGENT")
    print(f" Model       : {OLLAMA_MODEL_NAME} (via Ollama)")
    print(" Architecture: 5-Query Expansion -> Parallel BM25 + Dense (10/10)")
    print("               -> RRF (Top 10) -> FlashRank Cross-Encoder (Top 5)")
    print("               -> Grounded Answer with Clickable Deep-Link Citations")
    print("=" * 72)
    print("Commands:")
    print("  /clear   - Reset conversation memory")
    print("  /chunks  - Toggle printing retrieved chunks & links")
    print("  exit     - Quit chat")
    print("=" * 72 + "\n")

def run_chat_loop(show_chunks_initially: bool = True):
    print_banner()
    print("[CHAT] Initializing agent and index...")
    agent = VideoRAGAgent()
    show_chunks = show_chunks_initially

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "/exit", "/quit"]:
            print("Exiting chat. Goodbye!")
            break

        if user_input.lower() == "/clear":
            agent.reset_memory()
            print("🧹 [CHAT] Conversation memory reset.")
            continue

        if user_input.lower() == "/chunks":
            show_chunks = not show_chunks
            status = "ENABLED" if show_chunks else "DISABLED"
            print(f"⚙️ [CHAT] Chunk display is now {status}.")
            continue

        print("\n" + "-" * 72)
        top_chunks = []
        full_answer = ""

        for event in agent.stream_chat(user_input):
            event_type = event.get("type")
            if event_type == "status":
                print(f"⏳ {event.get('message')}")
            elif event_type == "expanded_queries":
                queries = event.get("queries", [])
                print(f"🔍 Generated {len(queries)} Sub-Queries:")
                for q_idx, q in enumerate(queries, 1):
                    print(f"   {q_idx}. {q}")
            elif event_type == "chunks":
                top_chunks = event.get("chunks", [])
                if show_chunks:
                    print(f"\n📑 Top {len(top_chunks)} Precision Chunks Selected:")
                    for idx, chunk in enumerate(top_chunks, 1):
                        title = chunk.get("video_title", "Unknown")
                        ts = f"{chunk.get('start_timestamp')} --> {chunk.get('end_timestamp')}"
                        url = chunk.get("timestamp_url", "")
                        score = chunk.get("rerank_score", 0.0)
                        print(f"   [{idx}] Score: {score:.4f} | {ts} | {title}")
                        print(f"       🔗 Link: {url}")
                print("\n🤖 Assistant: ", end="", flush=True)
            elif event_type == "delta":
                print(event.get("content", ""), end="", flush=True)
            elif event_type == "done":
                full_answer = event.get("answer", "")
                print("\n" + "-" * 72)

def run_single_query(query: str):
    agent = VideoRAGAgent()
    print(f"\nProcessing query: {query}\n")
    result = agent.chat(query)
    print("\nExpanded Queries:")
    for idx, q in enumerate(result["expanded_queries"], 1):
        print(f"  {idx}. {q}")
    print(f"\nTop {len(result['chunks'])} Chunks:")
    for idx, c in enumerate(result["chunks"], 1):
        print(f"  [{idx}] {c.get('start_timestamp')} --> {c.get('end_timestamp')} | {c.get('video_title')}")
        print(f"      {c.get('timestamp_url')}")
    print("\nAnswer:\n")
    print(result["answer"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conversational Video RAG CLI")
    parser.add_argument("-q", "--query", help="Run a single query non-interactively")
    parser.add_argument("--no-chunks", action="store_true", help="Hide retrieved chunk details in CLI")
    args = parser.parse_args()

    if args.query:
        run_single_query(args.query)
    else:
        run_chat_loop(show_chunks_initially=not args.no_chunks)
