import os
import sqlite3
import json
import time
import uuid
from typing import List, Dict, Any, Optional

from rag.config import BASE_DIR

DB_PATH = os.path.join(BASE_DIR, "rag_app.db")

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL
                )
            """)

            # OTP codes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL,
                    code TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    verified INTEGER DEFAULT 0
                )
            """)

            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Chat messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Chat videos table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_videos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    video_title TEXT NOT NULL,
                    video_url TEXT NOT NULL,
                    folder_name TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    cues_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    progress INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            # Ingestion jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    job_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    step TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    logs_json TEXT DEFAULT '[]',
                    created_at REAL NOT NULL,
                    completed_at REAL DEFAULT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(chat_id) ON DELETE CASCADE,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)

            conn.commit()

    # --- User Management ---
    def get_or_create_user(self, email: str) -> Dict[str, Any]:
        email = email.lower().strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cursor.fetchone()
            if row:
                return dict(row)

            user_id = str(uuid.uuid4())
            now = time.time()
            cursor.execute(
                "INSERT INTO users (id, email, created_at) VALUES (?, ?, ?)",
                (user_id, email, now)
            )
            conn.commit()
            return {"id": user_id, "email": email, "created_at": now}

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # --- OTP Codes ---
    def store_otp(self, email: str, code: str, expires_at: float):
        email = email.lower().strip()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO otp_codes (email, code, expires_at, verified) VALUES (?, ?, ?, 0)",
                (email, code, expires_at)
            )
            conn.commit()

    def verify_otp_code(self, email: str, code: str) -> bool:
        email = email.lower().strip()
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM otp_codes
                WHERE email = ? AND code = ? AND expires_at > ? AND verified = 0
                ORDER BY id DESC LIMIT 1
            """, (email, code, now))
            row = cursor.fetchone()
            if row:
                cursor.execute("UPDATE otp_codes SET verified = 1 WHERE id = ?", (row["id"],))
                conn.commit()
                return True
            return False

    # --- Sessions ---
    def create_session(self, user_id: str, token: str, expires_at: float):
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now, expires_at)
            )
            conn.commit()

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT s.*, u.email FROM sessions s
                JOIN users u ON s.user_id = u.id
                WHERE s.token = ? AND s.expires_at > ?
            """, (token, now))
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_session(self, token: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()

    # --- Chats ---
    def create_chat(self, user_id: str, title: str) -> Dict[str, Any]:
        chat_id = str(uuid.uuid4())
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chats (chat_id, user_id, title, summary, created_at, updated_at) VALUES (?, ?, ?, '', ?, ?)",
                (chat_id, user_id, title, now, now)
            )
            conn.commit()
            return {
                "chat_id": chat_id,
                "user_id": user_id,
                "title": title,
                "summary": "",
                "created_at": now,
                "updated_at": now
            }

    def get_chats_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, 
                       (SELECT COUNT(*) FROM chat_videos v WHERE v.chat_id = c.chat_id) as video_count,
                       (SELECT SUM(chunk_count) FROM chat_videos v WHERE v.chat_id = c.chat_id) as total_chunks
                FROM chats c
                WHERE c.user_id = ?
                ORDER BY c.updated_at DESC
            """, (user_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["total_chunks"] = d["total_chunks"] or 0
                results.append(d)
            return results

    def get_chat(self, chat_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_chat_title(self, chat_id: str, title: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?",
                (title, time.time(), chat_id)
            )
            conn.commit()

    def update_chat_summary(self, chat_id: str, summary: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chats SET summary = ?, updated_at = ? WHERE chat_id = ?",
                (summary, time.time(), chat_id)
            )
            conn.commit()

    def delete_chat(self, chat_id: str, user_id: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE chat_id = ? AND user_id = ?", (chat_id, user_id))
            conn.commit()
            return cursor.rowcount > 0

    # --- Chat Messages ---
    def add_chat_message(self, chat_id: str, user_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> int:
        now = time.time()
        meta_str = json.dumps(metadata or {}, ensure_ascii=False)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_messages (chat_id, user_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chat_id, user_id, role, content, meta_str, now))
            # Bump chat updated_at
            cursor.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
            conn.commit()
            return cursor.lastrowid

    def get_chat_messages(self, chat_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if limit:
                cursor.execute("""
                    SELECT * FROM (
                        SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?
                    ) ORDER BY id ASC
                """, (chat_id, limit))
            else:
                cursor.execute("SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))

            rows = cursor.fetchall()
            messages = []
            for r in rows:
                d = dict(r)
                try:
                    d["metadata"] = json.loads(d.get("metadata_json", "{}"))
                except Exception:
                    d["metadata"] = {}
                messages.append(d)
            return messages

    # --- Chat Videos ---
    def add_chat_video(
        self,
        chat_id: str,
        user_id: str,
        video_title: str,
        video_url: str,
        folder_name: str,
        chunk_count: int = 0,
        cues_count: int = 0,
        status: str = "completed"
    ) -> int:
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_videos (chat_id, user_id, video_title, video_url, folder_name, chunk_count, cues_count, status, progress, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 100, ?)
            """, (chat_id, user_id, video_title, video_url, folder_name, chunk_count, cues_count, status, now))
            cursor.execute("UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id))
            conn.commit()
            return cursor.lastrowid


    def get_chat_videos(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chat_videos WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    # --- Ingestion Jobs ---
    def create_ingest_job(self, job_id: str, chat_id: str, user_id: str, url: str) -> Dict[str, Any]:
        now = time.time()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ingest_jobs (job_id, chat_id, user_id, url, status, step, progress, logs_json, created_at)
                VALUES (?, ?, ?, ?, 'queued', 'Queued in pipeline...', 5, '[]', ?)
            """, (job_id, chat_id, user_id, url, now))
            conn.commit()
            return {
                "job_id": job_id,
                "chat_id": chat_id,
                "user_id": user_id,
                "url": url,
                "status": "queued",
                "step": "Queued in pipeline...",
                "progress": 5,
                "logs": [],
                "created_at": now
            }

    def update_ingest_job(self, job_id: str, status: str, step: str, progress: int, log_line: Optional[str] = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT logs_json FROM ingest_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            logs = json.loads(row["logs_json"]) if row and row["logs_json"] else []

            if log_line:
                logs.append(log_line)

            completed_at = time.time() if status in ("completed", "failed") else None
            cursor.execute("""
                UPDATE ingest_jobs
                SET status = ?, step = ?, progress = ?, logs_json = ?, completed_at = COALESCE(?, completed_at)
                WHERE job_id = ?
            """, (status, step, progress, json.dumps(logs, ensure_ascii=False), completed_at, job_id))
            conn.commit()

    def get_ingest_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ingest_jobs WHERE job_id = ?", (job_id,))
            row = cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["logs"] = json.loads(d.get("logs_json", "[]"))
            return d

    def get_active_jobs_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ingest_jobs
                WHERE user_id = ? AND status IN ('queued', 'processing')
                ORDER BY created_at ASC
            """, (user_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["logs"] = json.loads(d.get("logs_json", "[]"))
                results.append(d)
            return results

    def get_active_jobs_for_chat(self, chat_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ingest_jobs
                WHERE chat_id = ? AND status IN ('queued', 'processing')
                ORDER BY created_at ASC
            """, (chat_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["logs"] = json.loads(d.get("logs_json", "[]"))
                results.append(d)
            return results

    def get_recent_completed_jobs_for_user(self, user_id: str, since_timestamp: float) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT j.*, c.title as chat_title FROM ingest_jobs j
                JOIN chats c ON j.chat_id = c.chat_id
                WHERE j.user_id = ? AND j.status = 'completed' AND j.completed_at > ?
                ORDER BY j.completed_at ASC
            """, (user_id, since_timestamp))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["logs"] = json.loads(d.get("logs_json", "[]"))
                results.append(d)
            return results

# Global singleton database instance
db = Database()
