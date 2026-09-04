import os
import sys
import time
import json
import uuid
import threading
from typing import Dict, Any, Optional
from flask import Flask, render_template, request, jsonify, make_response, Response, stream_with_context
from flask_cors import CORS

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from rag.database import db
from rag.auth import request_otp, verify_otp_and_login, authenticate_token
from rag.agent import VideoRAGAgent
from rag.vectorstore import VectorStoreManager
from audio_downloader import download_audio
from audio_transcriber import transcribe_audio, save_srt
from rag.ingest import ingest_video

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static")
)
CORS(app, supports_credentials=True)

OP_DIR = os.path.join(BASE_DIR, "op")
os.makedirs(OP_DIR, exist_ok=True)

# Lazy-loaded singleton agent
_agent_instance = None

def get_agent():
    global _agent_instance
    if _agent_instance is None:
        print("[CHAT_SERVER] Initializing VideoRAGAgent...")
        _agent_instance = VideoRAGAgent()
        print("[CHAT_SERVER] VideoRAGAgent ready.")
    return _agent_instance

def get_auth_user() -> Optional[Dict[str, Any]]:
    """
    Extracts and authenticates user from Authorization header or cookie.
    """
    token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if not token:
        token = request.cookies.get("rag_session")
    if not token:
        return None
    return authenticate_token(token)

# --- Background Ingestion Worker ---
def run_ingestion_for_chat(job_id: str, chat_id: str, user_id: str, url: str):
    db.update_ingest_job(job_id, "processing", "Connecting to YouTube and discovering videos...", 5, f"Started extraction for: {url}")
    chat_dir = os.path.join(OP_DIR, chat_id)
    os.makedirs(chat_dir, exist_ok=True)

    tracks_processed = []

    def on_progress(data):
        stage = data.get("stage")
        total = max(1, data.get("total", 1))
        idx = max(1, data.get("idx", 1))
        title = data.get("title", "")

        if stage == "discovered":
            step = f"[0/{total} vids] Discovered {total} video(s)..."
            db.update_ingest_job(job_id, "processing", step, 8, step)
        elif stage == "downloading":
            # Progress calculation: Video slice = 100 / total
            base_pct = int(((idx - 1) / total) * 100)
            dl_pct = base_pct + int((1 / total) * 20)
            step = f"[{idx}/{total} vids] Downloading audio: '{title}'..."
            db.update_ingest_job(job_id, "processing", step, min(99, max(5, dl_pct)), f"[DOWNLOAD {idx}/{total}] {title}")

    def on_complete_track(track):
        title = track["title"]
        video_folder = track["video_folder"]
        audio_path = track["audio_path"]
        idx = track.get("idx", len(tracks_processed) + 1)
        total = max(1, track.get("total", 1))

        base_pct = int(((idx - 1) / total) * 100)

        # 1. Transcribe audio stage
        asr_pct = base_pct + int((1 / total) * 55)
        step_asr = f"[{idx}/{total} vids] Transcribing audio with GPU Parakeet ASR: '{title}'..."
        db.update_ingest_job(job_id, "processing", step_asr, min(99, asr_pct), f"[ASR {idx}/{total}] Transcribing: {title}")

        if os.path.exists(audio_path):
            try:
                segments = transcribe_audio(audio_file=audio_path, chunk_seconds=60.0, overlap_seconds=1.0)
                srt_path = os.path.join(video_folder, "audio.srt")
                save_srt(segments, srt_path)

                # 2. Chunk and embed stage
                embed_pct = base_pct + int((1 / total) * 85)
                step_embed = f"[{idx}/{total} vids] Vectorizing subtitles & embedding: '{title}'..."
                db.update_ingest_job(job_id, "processing", step_embed, min(99, embed_pct), f"[INGEST {idx}/{total}] Embedding {title} into ChromaDB...")

                chunks = ingest_video(
                    video_folder,
                    chunk_seconds=60.0,
                    overlap_seconds=5.0,
                    chat_id=chat_id,
                    user_id=user_id
                )

                # 3. Clean up WAV
                try:
                    os.remove(audio_path)
                except Exception:
                    pass

                # 4. Save video record to database
                db.add_chat_video(
                    chat_id=chat_id,
                    user_id=user_id,
                    video_title=title,
                    video_url=track.get("url", url),
                    folder_name=os.path.basename(video_folder),
                    chunk_count=len(chunks),
                    cues_count=sum(len(c.get("cues", [])) for c in chunks)
                )

                # Update chat title if it's default
                chat = db.get_chat(chat_id, user_id)
                if chat and (chat["title"] in ("New Video Chat", "Untitled Chat") or not chat["title"]):
                    db.update_chat_title(chat_id, title)

                tracks_processed.append(title)

                # 5. Video indexed
                complete_pct = int((idx / total) * 100)
                step_done = f"[{idx}/{total} vids] Indexed '{title}' ({len(chunks)} chunks ready)!"
                db.update_ingest_job(job_id, "processing", step_done, min(99, complete_pct), f"[DONE {idx}/{total}] Indexed: {title}")

            except Exception as e:
                db.update_ingest_job(job_id, "failed", f"Failed transcribing {title}: {e}", 0, f"[ERROR] {e}")
                print(f"[INGESTION] Error processing {title}: {e}", file=sys.stderr)

    try:
        download_audio(
            url=url,
            output_dir=chat_dir,
            on_complete_callback=on_complete_track,
            on_progress_callback=on_progress
        )
        total_indexed = len(tracks_processed)
        db.update_ingest_job(
            job_id,
            "completed",
            f"Successfully indexed all {total_indexed} video(s) into chat knowledge base!",
            100,
            f"[SUCCESS] All {total_indexed} videos indexed. Chat is ready!"
        )
    except Exception as e:
        db.update_ingest_job(job_id, "failed", f"Pipeline failed: {e}", 0, f"[FATAL] {e}")
        print(f"[INGESTION] Job {job_id} failed: {e}", file=sys.stderr)


# =========================================================================
# WEB ROUTES
# =========================================================================

@app.route("/")
def index():
    return render_template("chat.html")

# --- Authentication Endpoints ---
@app.route("/api/auth/request-otp", methods=["POST"])
def api_request_otp():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").strip()
    if not email or "@" not in email:
        return jsonify({"status": "error", "message": "Please provide a valid email address."}), 400
    res = request_otp(email)
    return jsonify(res)

@app.route("/api/auth/verify-otp", methods=["POST"])
def api_verify_otp():
    data = request.get_json(force=True, silent=True) or {}
    email = data.get("email", "").strip()
    code = data.get("code", "").strip()
    if not email or not code:
        return jsonify({"status": "error", "message": "Email and verification code are required."}), 400

    res = verify_otp_and_login(email, code)
    if res["status"] == "success":
        response = make_response(jsonify(res))
        response.set_cookie("rag_session", res["session_token"], max_age=7*86400, httponly=False, samesite="Lax")
        return response
    return jsonify(res), 400

@app.route("/api/auth/me", methods=["GET"])
def api_auth_me():
    user = get_auth_user()
    if not user:
        return jsonify({"status": "unauthenticated"}), 401
    return jsonify({"status": "authenticated", "user": user})

@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    token = request.cookies.get("rag_session")
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
    if token:
        db.delete_session(token)
    response = make_response(jsonify({"status": "success", "message": "Logged out"}))
    response.delete_cookie("rag_session")
    return response

# --- Chat Management Endpoints ---
@app.route("/api/chats", methods=["GET"])
def api_list_chats():
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chats = db.get_chats_for_user(user["id"])
    active_jobs = db.get_active_jobs_for_user(user["id"])
    job_map = {j["chat_id"]: j for j in active_jobs}

    for c in chats:
        cid = c["chat_id"]
        c["videos"] = db.get_chat_videos(cid)
        c["active_job"] = job_map.get(cid)

    return jsonify({"status": "success", "chats": chats})

@app.route("/api/chats", methods=["POST"])
def api_create_chat():
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()
    title = data.get("title", "").strip() or "New Video Chat"

    chat = db.create_chat(user["id"], title)
    chat_id = chat["chat_id"]

    job_id = None
    if url:
        job_id = str(uuid.uuid4())
        db.create_ingest_job(job_id, chat_id, user["id"], url)
        thread = threading.Thread(
            target=run_ingestion_for_chat,
            args=(job_id, chat_id, user["id"], url),
            daemon=True
        )
        thread.start()

    return jsonify({
        "status": "success",
        "chat": chat,
        "job_id": job_id
    })

@app.route("/api/chats/<chat_id>", methods=["GET"])
def api_get_chat(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        return jsonify({"status": "error", "message": "Chat not found"}), 404

    chat["videos"] = db.get_chat_videos(chat_id)
    chat["messages"] = db.get_chat_messages(chat_id)
    return jsonify({"status": "success", "chat": chat})

@app.route("/api/chats/<chat_id>", methods=["DELETE"])
def api_delete_chat(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        return jsonify({"status": "error", "message": "Chat not found"}), 404

    # 1. Delete ChromaDB vector chunks for this chat
    vstore = VectorStoreManager()
    deleted_chunks = vstore.delete_chat_chunks(chat_id)

    # 2. Delete local files for this chat from disk
    chat_folder = os.path.join(OP_DIR, chat_id)
    if os.path.exists(chat_folder):
        import shutil
        shutil.rmtree(chat_folder, ignore_errors=True)

    # 3. Delete chat and its cascaded records from SQLite
    db.delete_chat(chat_id, user["id"])

    return jsonify({
        "status": "success",
        "message": f"Chat, conversation history, and {deleted_chunks} vector chunks deleted successfully."
    })

@app.route("/api/chats/<chat_id>", methods=["PATCH", "PUT"])
def api_rename_chat(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        return jsonify({"status": "error", "message": "Chat not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    new_title = data.get("title", "").strip()
    if not new_title:
        return jsonify({"status": "error", "message": "Chat title cannot be empty"}), 400

    db.update_chat_title(chat_id, new_title)
    return jsonify({
        "status": "success",
        "message": f"Chat renamed to '{new_title}'",
        "title": new_title
    })

@app.route("/api/chats/<chat_id>/messages", methods=["GET"])

def api_get_chat_messages(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    messages = db.get_chat_messages(chat_id)
    return jsonify({"status": "success", "messages": messages})

@app.route("/api/chats/<chat_id>/messages", methods=["POST"])
def api_send_chat_message(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        return jsonify({"status": "error", "message": "Chat not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"status": "error", "message": "Query cannot be empty"}), 400

    agent = get_agent()

    # If non-streaming is explicitly requested via ?stream=false
    if request.args.get("stream", "true").lower() == "false":
        try:
            result = agent.chat_scoped(query, chat_id=chat_id, user_id=user["id"])
            return jsonify({
                "status": "success",
                "query": query,
                "expanded_queries": result["expanded_queries"],
                "chunks": result["chunks"],
                "answer": result["answer"],
                "summary": result.get("summary", "")
            })
        except Exception as e:
            print(f"[CHAT_SERVER] Error during scoped chat: {e}", file=sys.stderr)
            return jsonify({"status": "error", "message": str(e)}), 500

    # Default: Server-Sent Events (SSE) streaming response
    def generate():
        try:
            for event in agent.chat_scoped_stream(query, chat_id=chat_id, user_id=user["id"]):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            print(f"[CHAT_SERVER] Error during streaming: {e}", file=sys.stderr)
            err_payload = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

import urllib.parse

def extract_youtube_identifiers(url: str) -> Dict[str, Optional[str]]:
    """
    Extracts video_id and playlist_id from any YouTube URL variation.
    """
    video_id = None
    playlist_id = None
    if not url:
        return {"video_id": None, "playlist_id": None}
    try:
        parsed = urllib.parse.urlparse(url.strip())
        query = urllib.parse.parse_qs(parsed.query)

        if "list" in query and query["list"]:
            playlist_id = query["list"][0]

        if "v" in query and query["v"]:
            video_id = query["v"][0]
        elif parsed.netloc in ("youtu.be", "www.youtu.be"):
            parts = parsed.path.strip("/").split("/")
            if parts and parts[0]:
                video_id = parts[0]
        elif "/embed/" in parsed.path:
            video_id = parsed.path.split("/embed/")[1].split("/")[0]
        elif "/v/" in parsed.path:
            video_id = parsed.path.split("/v/")[1].split("/")[0]
        elif "/shorts/" in parsed.path:
            video_id = parsed.path.split("/shorts/")[1].split("/")[0]
    except Exception:
        pass
    return {"video_id": video_id, "playlist_id": playlist_id}


def check_duplicate_video_in_chat(chat_id: str, new_url: str) -> tuple[bool, str]:
    """
    Checks if a video or playlist is already added or currently being processed in this chat.
    Returns: (is_duplicate: bool, message: str)
    """
    new_ids = extract_youtube_identifiers(new_url)
    new_vid = new_ids["video_id"]
    new_pid = new_ids["playlist_id"]

    # 1. Check existing videos in SQLite chat_videos
    existing_videos = db.get_chat_videos(chat_id)
    for v in existing_videos:
        v_url = v.get("video_url", "")
        v_title = v.get("video_title", "Untitled Video")
        v_ids = extract_youtube_identifiers(v_url)

        # Check playlist match
        if new_pid and (v_ids["playlist_id"] == new_pid or f"list={new_pid}" in v_url):
            return True, "This playlist is already added in this chat."

        # Check video match
        if new_vid and v_ids["video_id"] == new_vid:
            return True, f"This video ('{v_title}') is already added in this chat."

        # Match exact URL (ignoring trailing slash and whitespace)
        if v_url and new_url.strip().rstrip("/") == v_url.strip().rstrip("/"):
            return True, f"This video ('{v_title}') is already added in this chat."

    # 2. Check metadata.json files in op/<chat_id> for videos ingested from playlists
    chat_op_dir = os.path.join(OP_DIR, chat_id)
    if os.path.isdir(chat_op_dir) and (new_vid or new_pid):
        for root, dirs, files in os.walk(chat_op_dir):
            if "metadata.json" in files:
                meta_path = os.path.join(root, "metadata.json")
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    m_url = meta.get("url") or meta.get("webpage_url") or ""
                    m_title = meta.get("title", "Untitled Video")
                    m_ids = extract_youtube_identifiers(m_url)

                    if new_vid and m_ids["video_id"] == new_vid:
                        return True, f"This video ('{m_title}') is already added in this chat."
                    if new_pid and (m_ids["playlist_id"] == new_pid or f"list={new_pid}" in m_url):
                        return True, "This playlist is already added in this chat."
                except Exception:
                    pass

    # 3. Check active/queued ingestion jobs for this chat
    active_jobs = db.get_active_jobs_for_chat(chat_id)
    for job in active_jobs:
        job_url = job.get("url", "")
        job_ids = extract_youtube_identifiers(job_url)

        if new_pid and job_ids["playlist_id"] == new_pid:
            return True, "This playlist is currently being processed in this chat."

        if new_vid and job_ids["video_id"] == new_vid:
            return True, "This video is currently being processed in this chat."

    return False, ""


@app.route("/api/chats/<chat_id>/videos", methods=["POST"])
def api_add_video_to_chat(chat_id):
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    chat = db.get_chat(chat_id, user["id"])
    if not chat:
        return jsonify({"status": "error", "message": "Chat not found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"status": "error", "message": "YouTube URL is required."}), 400

    # Prevent duplicate video or playlist addition to this chat
    is_dup, dup_msg = check_duplicate_video_in_chat(chat_id, url)
    if is_dup:
        return jsonify({"status": "error", "message": dup_msg}), 400

    job_id = str(uuid.uuid4())
    db.create_ingest_job(job_id, chat_id, user["id"], url)
    thread = threading.Thread(
        target=run_ingestion_for_chat,
        args=(job_id, chat_id, user["id"], url),
        daemon=True
    )
    thread.start()

    return jsonify({"status": "success", "job_id": job_id})

@app.route("/api/jobs/active", methods=["GET"])
def api_get_active_jobs():
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    jobs = db.get_active_jobs_for_user(user["id"])
    return jsonify({"status": "success", "jobs": jobs})

@app.route("/api/jobs/completed", methods=["GET"])
def api_get_completed_jobs():
    user = get_auth_user()
    if not user:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    since = request.args.get("since", 0.0, type=float)
    jobs = db.get_recent_completed_jobs_for_user(user["id"], since)
    return jsonify({"status": "success", "jobs": jobs, "server_time": time.time()})

def create_chat_server():
    return app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
