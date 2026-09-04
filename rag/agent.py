import os
import re
import json
from typing import List, Dict, Any, Optional, Generator
import ollama

from rag.config import (
    BASE_DIR,
    OLLAMA_MODEL_NAME,
    OLLAMA_BASE_URL,
    NUM_EXPANDED_QUERIES,
    TOP_DENSE_PER_QUERY,
    TOP_BM25_PER_QUERY,
    RRF_TOP_CANDIDATES,
    FINAL_TOP_K,
)
from rag.retriever import HybridRetriever

class ConversationMemory:
    """
    Manages multi-turn conversation history for the agent.
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.history.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.history.append({"role": "assistant", "content": content})
        self._trim()

    def _trim(self):
        # Keep up to max_turns * 2 messages (pairs of user/assistant)
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def get_context_summary(self, max_recent: int = 3) -> str:
        """
        Returns a concise string representation of the recent conversation turns.
        """
        recent = self.history[-(max_recent * 2):]
        if not recent:
            return ""
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()


class QueryTransformer:
    """
    Transforms a user query into 5 similar / varied search queries using gpt-oss:120b-cloud.
    """
    def __init__(self, model_name: str = OLLAMA_MODEL_NAME, client: Optional[ollama.Client] = None):
        self.model_name = model_name
        self.client = client or ollama.Client(host=OLLAMA_BASE_URL)

    def is_correlated_with_history(self, user_query: str, chat_context: str) -> bool:
        """
        Determines whether the user's latest query is directly co-related with or dependent upon
        the prior conversation history (e.g. uses pronouns, asks follow-up questions, or references
        prior tools/concepts).
        """
        if not chat_context or not chat_context.strip():
            return False

        system_prompt = (
            "You are an expert conversational dependency classifier.\n"
            "Your task is to determine whether the user's latest query is DEPENDENT ON or DIRECTLY CO-RELATED with "
            "the prior conversation history, or if it is a fresh, standalone, independent question.\n\n"
            "Rules for YES (Co-related / Follow-up / Dependent):\n"
            "- The query uses ambiguous pronouns or relative references ('it', 'its', 'that', 'this', 'these', 'those', 'they', 'the tool', 'the method', 'the previous step').\n"
            "- The query is an explicit follow-up question (e.g., 'why?', 'how so?', 'what else?', 'can you show an example?').\n"
            "- The query directly continues the specific topic, tool, or entity discussed in the immediately preceding turn.\n"
            "- The query cannot be fully understood or searched accurately without knowing the prior context.\n\n"
            "Rules for NO (Independent / Standalone / New Topic):\n"
            "- The query is self-contained with its own distinct subject, technology, or topic.\n"
            "- The query introduces a new video topic, question, or fresh concept that does not require prior context.\n\n"
            "Return ONLY 'YES' or 'NO' and nothing else."
        )

        user_prompt = (
            f"Prior Conversation History:\n{chat_context}\n\n"
            f"Current User Query: {user_query}\n\n"
            "Is the current query directly co-related with or dependent on the prior conversation history? (YES/NO):"
        )

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.0}
            )
            decision = response["message"]["content"].strip().upper()
            is_corr = decision.startswith("YES") or "YES" in decision
            return is_corr
        except Exception as e:
            print(f"[QUERY_TRANSFORMER] Warning: Correlation check failed ({e}). Falling back to heuristic.")
            pronouns = [
                r"\bit\b", r"\bits\b", r"\bthis\b", r"\bthat\b", r"\bthese\b",
                r"\bthose\b", r"\bthey\b", r"\bthem\b", r"\bthe previous\b",
                r"\bthe above\b", r"\bthat video\b", r"\bthat tool\b", r"\bwhat about\b",
                r"\bwhy\b", r"\bhow so\b", r"\belaborate\b", r"\band then\b", r"\bcontinue\b"
            ]
            q_lower = user_query.lower()
            return any(re.search(p, q_lower) for p in pronouns)

    def classify_intent(self, user_query: str) -> str:
        """
        Classifies the query into one of:
        - "DIRECT_CHAT": Greetings, casual dialogue, math, general coding/concepts not asking about video
        - "VIDEO_SUMMARY": Comprehensive summary, outline, chapters, key takeaways of the whole video
        - "RAG": Specific questions seeking facts, tools, steps, or details in the video
        """
        q = user_query.strip()
        q_lower = q.lower()

        # 1. High-precision regex heuristics (instant zero-latency decision)
        if re.match(r'^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|sup|greetings|who\s+are\s+you|what\s+can\s+you\s+do)[\s!.]*$', q_lower):
            return "DIRECT_CHAT"
        if re.match(r'^(thanks|thank\s+you|thx|cool|awesome|great|ok|okay|got\s+it)[\s!.]*$', q_lower):
            return "DIRECT_CHAT"

        # Workspace & chat video library meta-questions (zero RAG needed)
        workspace_keywords = [
            r"\bhow\s+many\s+(vids|videos|files|recordings|playlists|content)\b",
            r"\bhow\s+many\s+(vids|videos)\s+(are\s+there|in\s+this\s+chat|do\s+we\s+have|here)\b",
            r"\bwhat\s+(vids|videos)\s+(are\s+there|are\s+in\s+this\s+chat|do\s+we\s+have|are\s+here|do\s+i\s+have)\b",
            r"\b(list|show|which)\s+(the\s+)?(vids|videos|files|content)\b",
            r"\bwhat\s+(are\s+the\s+names\s+of\s+)?(the\s+)?videos\s+(in\s+this\s+chat|here)\b",
            r"\bhow\s+many\s+videos\s+this\s+chat\s+has\b",
            r"\bwhat\s+videos\s+do\s+you\s+have\b"
        ]
        if any(re.search(pat, q_lower) for pat in workspace_keywords):
            return "DIRECT_CHAT"

        # Explicit whole-video summary / outline / takeaways
        summary_keywords = [
            r"\bsummarize\s+(the|this|entire|whole)?\s*video\b",
            r"\bsummarise\s+(the|this|entire|whole)?\s*video\b",
            r"\bsummary\s+of\s+(the|this|entire|whole)?\s*video\b",
            r"\bgive\s+me\s+(a\s+)?summary\b",
            r"\bvideo\s+summary\b",
            r"\bvideo\s+overview\b",
            r"\bwhat\s+is\s+this\s+video\s+about\b",
            r"\bwhat\s+is\s+the\s+video\s+about\b",
            r"\bkey\s+takeaways\b",
            r"\bvideo\s+chapters\b",
            r"\bchapter\s+outline\b",
            r"\boutline\s+(the|this)?\s*video\b",
            r"\btldr\s+of\s+(the|this)?\s*video\b",
            r"\bwhat\s+happens\s+in\s+this\s+video\b"
        ]
        if any(re.search(pat, q_lower) for pat in summary_keywords):
            return "VIDEO_SUMMARY"

        # Explicit generic coding / math / conceptual questions not tied to the video
        if re.search(r'\b(write|create|code|script)\b.*\b(python|javascript|java|c\+\+|sql|html|css)\b', q_lower) and not re.search(r'\b(video|tutorial|speaker|author|he|she|shown|mentioned)\b', q_lower):
            return "DIRECT_CHAT"

        # 2. LLM classifier for ambiguous queries via gpt-oss:120b-cloud
        system_prompt = (
            "You are an intent classification router for an AI video assistant.\n"
            "Classify the user's latest query into EXACTLY ONE of these 3 categories:\n"
            "1. DIRECT_CHAT: Greetings, conversational chat, questions asking how many or which videos are in this chat, or general knowledge/coding questions.\n"
            "2. VIDEO_SUMMARY: Requests to summarize the full video, give an overview, list key takeaways, or provide video chapters/timeline from beginning to end.\n"
            "3. RAG: Specific questions asking about facts, steps, commands, tools, quotes, or details within the video.\n\n"
            "Reply with ONLY one category name: DIRECT_CHAT, VIDEO_SUMMARY, or RAG. No explanation."
        )

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User Query: {user_query}"}
                ],
                options={"temperature": 0.0}
            )
            ans = response["message"]["content"].strip().upper()
            if "VIDEO_SUMMARY" in ans:
                return "VIDEO_SUMMARY"
            if "DIRECT_CHAT" in ans:
                return "DIRECT_CHAT"
            return "RAG"
        except Exception as e:
            print(f"[QUERY_TRANSFORMER] Warning: Intent classification failed ({e}). Defaulting to RAG.")
            return "RAG"

    def expand_query(self, user_query: str, chat_context: str = "", num_queries: int = NUM_EXPANDED_QUERIES) -> List[str]:
        """
        Generates num_queries diverse search queries based on the user question and conversation context.
        """
        system_prompt = (
            "You are an expert search query optimizer for a video transcript RAG system. "
            f"Your task is to take the user's latest query and generate exactly {num_queries} diverse, "
            "effective search queries to retrieve relevant video subtitle segments.\n"
            "Rules:\n"
            "1. If recent conversation context is provided, resolve ambiguities, pronouns, or references to prior topics. If no context is provided, generate search queries strictly from the user question itself.\n"
            "2. Create varied perspectives: include keyword variations, technical terminology, conceptual queries, and direct natural language questions.\n"
            f"3. Return ONLY a JSON list of exactly {num_queries} strings, nothing else. No markdown formatting, no explanations."
        )

        user_prompt = f"User Question: {user_query}"
        if chat_context.strip():
            user_prompt = f"Recent Conversation Context:\n{chat_context}\n\n{user_prompt}"

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={"temperature": 0.3}
            )
            raw_text = response["message"]["content"].strip()
            queries = self._parse_queries(raw_text, user_query, num_queries)
            return queries
        except Exception as e:
            print(f"[QUERY_TRANSFORMER] Warning: Query expansion failed ({e}). Falling back to original query.")
            return [user_query]

    def _parse_queries(self, raw_text: str, original_query: str, target_count: int) -> List[str]:
        """
        Parses JSON list or line-delimited queries from LLM output.
        """
        # Try JSON extraction
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    cleaned = [str(q).strip(' "\'') for q in parsed if str(q).strip(' "\'')]
                    if cleaned:
                        if original_query not in cleaned:
                            cleaned.insert(0, original_query)
                        return cleaned[:target_count]
            except Exception:
                pass

        # Fallback line-by-line extraction
        lines = [re.sub(r'^\s*(\d+[\.\)]|-|\*)\s*', '', line).strip(' "\'') for line in raw_text.splitlines()]
        cleaned = [line for line in lines if len(line) > 3 and not line.startswith("```")]
        if original_query not in cleaned:
            cleaned.insert(0, original_query)

        if not cleaned:
            cleaned = [original_query]

        # Fill if short
        while len(cleaned) < target_count:
            cleaned.append(f"{original_query} details")

        return cleaned[:target_count]


from rag.chunker import extract_video_id

def parse_any_timestamp(ts_str: str) -> Optional[int]:
    """
    Parses timestamps like '01:21', '00:01:21', '1:21', '05:54' into total seconds.
    """
    match = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', ts_str)
    if not match:
        return None
    parts = [int(p) for p in match.groups() if p is not None]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]

def resolve_exact_cue_url(chunks: List[Dict[str, Any]], title_hint: str, target_seconds: Optional[int]) -> str:
    """
    Finds the exact cue deep-link URL matching target_seconds across the retrieved chunks.
    """
    matched_chunks = [
        c for c in chunks
        if title_hint and (
            c.get("video_title", "").lower() in title_hint.lower()
            or title_hint.lower() in c.get("video_title", "").lower()
        )
    ]
    if not matched_chunks:
        matched_chunks = chunks

    if target_seconds is not None:
        best_cue = None
        best_diff = float("inf")

        for c in matched_chunks:
            cues = c.get("cues", [])
            for cue in cues:
                cue_start = cue.get("start_seconds", 0.0)
                cue_end = cue.get("end_seconds", cue_start + 5.0)

                # Check if timestamp falls within the cue window
                if cue_start <= target_seconds <= cue_end:
                    return cue.get("timestamp_url", "")

                diff = abs(cue_start - target_seconds)
                if diff < best_diff:
                    best_diff = diff
                    best_cue = cue

        if best_cue and best_diff <= 15.0:
            return best_cue.get("timestamp_url", "")

        # Fallback: dynamically construct timestamp URL using target_seconds
        for c in matched_chunks:
            v_url = c.get("video_url", "")
            if v_url:
                v_id = extract_video_id(v_url)
                if v_id and v_id != "video":
                    return f"https://www.youtube.com/watch?v={v_id}&t={target_seconds}s"
                sep = "&" if "?" in v_url else "?"
                return f"{v_url}{sep}t={target_seconds}s"

    if matched_chunks:
        return matched_chunks[0].get("timestamp_url", "")
    return ""


class CitationPromptEngine:
    """
    Builds strict citation prompts using granular cue timestamps and ensures clickable deep links.
    """
    @staticmethod
    def build_system_prompt(video_library_header: str = "") -> str:
        prompt = (
            "You are an expert Video Knowledge Conversational Agent. Your mission is to provide accurate, "
            "deeply grounded answers with exact timestamp citations using the provided video transcript cues.\n\n"
        )
        if video_library_header:
            prompt += f"{video_library_header}\n\n"
        prompt += (
            "CRITICAL TIMESTAMP & CITATION RULES:\n"
            "1. The transcript is broken down into fine-grained cues. Each sentence is prefixed with its exact timestamp "
            "[MM:SS] (or [HH:MM:SS]) and its direct YouTube URL in parentheses.\n"
            "2. ALWAYS cite the EXACT cue timestamp where the specific fact, tool, step, quote, or claim is mentioned. "
            "NEVER default to the start of the chunk window (e.g. 00:00:00 or 00:05:25) if the fact was stated later in the chunk!\n"
            "3. MANDATORY CITATION FORMAT:\n"
            "   Every factual claim MUST include a clickable Markdown hyperlink in this exact format:\n"
            "   [<Video Title> @ <MM:SS>](<timestamp_url>)\n"
            "   Example: 'The tutorial introduces Unsloth Studio, a local GUI for fine-tuning, at [How to Fine-Tune any AI Model Locally @ 01:21](https://www.youtube.com/watch?v=4JofSJIrjwU&t=80s).'\n"
            "4. ACCURACY & GROUNDING:\n"
            "   Do NOT hallucinate or guess timestamps or facts. Match your timestamp and URL directly to the cue line containing that information.\n"
            "5. STRUCTURE:\n"
            "   Synthesize a comprehensive, well-structured answer with bullet points or numbered steps where appropriate."
        )
        return prompt

    @staticmethod
    def format_context_chunks(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant video chunks found."

        chunk_texts = []
        for idx, chunk in enumerate(chunks, 1):
            title = chunk.get("video_title", "Unknown Video")
            start_ts = chunk.get("start_timestamp", "00:00:00")
            end_ts = chunk.get("end_timestamp", "00:00:00")
            chunk_url = chunk.get("timestamp_url", chunk.get("video_url", ""))
            cues = chunk.get("cues", [])

            lines = [
                f"--- [CHUNK {idx}] ---",
                f"Video Title   : {title}",
                f"Time Window   : {start_ts} --> {end_ts}",
                f"Chunk Base URL: {chunk_url}",
                "Transcript Cues (with exact per-sentence timestamps & deep links):"
            ]

            if cues:
                for cue in cues:
                    cue_ts = cue.get("start_timestamp", "")
                    cue_url = cue.get("timestamp_url", "")
                    cue_txt = cue.get("text", "").strip()
                    if cue_txt:
                        lines.append(f"  [{cue_ts}] ({cue_url}) {cue_txt}")
            else:
                lines.append(f"  {chunk.get('text', chunk.get('plain_text', ''))}")

            chunk_texts.append("\n".join(lines))

        return "\n\n".join(chunk_texts)

    @staticmethod
    def linkify_citations(text: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Post-processor that inspects all citations in the LLM's response, verifies their timestamps
        against the chunk cues, and guarantees every citation is a clickable Markdown hyperlink
        pointing to the exact second in the video.
        """
        if not chunks:
            return text

        # 1. Convert Chinese brackets 【Title @ 00:01:21】 to [Title @ 00:01:21]
        text = re.sub(r'【([^】]+)】', r'[\1]', text)

        # 2. Match all bracket citations: [Anchor](URL) or [Anchor]
        def process_citation(m):
            full_match = m.group(0)
            anchor = m.group(1).strip()
            existing_url = m.group(2) if m.lastindex >= 2 and m.group(2) else None

            # Skip markdown code blocks, numbers, checkboxes or simple lists like [1], [x]
            if re.match(r'^\d+$|^x$|^\s*$', anchor, re.IGNORECASE):
                return full_match

            # Check if this bracket looks like a citation
            ts_sec = parse_any_timestamp(anchor)
            has_at = "@" in anchor
            is_title = any(
                c.get("video_title", "") and c.get("video_title", "").lower() in anchor.lower()
                for c in chunks
            )

            if not (has_at or ts_sec is not None or is_title):
                return full_match

            title_hint = anchor.split("@")[0].strip() if "@" in anchor else anchor

            # Resolve the most precise cue URL for this timestamp
            accurate_url = resolve_exact_cue_url(chunks, title_hint, ts_sec)
            if not accurate_url and existing_url:
                accurate_url = existing_url

            if accurate_url:
                return f"[{anchor}]({accurate_url})"
            return full_match

        pattern = r'\[([^\]]+)\](?:\(([^)]+)\))?'
        text = re.sub(pattern, process_citation, text)
        return text




from rag.database import db
from rag.config import CHAT_BUFFER_SIZE

def update_recursive_summary(
    existing_summary: str,
    older_messages: List[Dict[str, Any]],
    client: ollama.Client,
    model_name: str
) -> str:
    """
    Recursively condenses messages older than the last 8 turns into an updated, dense summary.
    """
    if not older_messages:
        return existing_summary

    convo_lines = []
    for m in older_messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        convo_lines.append(f"{role}: {m.get('content', '')}")
    convo_str = "\n".join(convo_lines)

    system_prompt = (
        "You are an expert conversation context summarizer. "
        "Your task is to produce a consolidated, dense summary of the conversation so far, "
        "incorporating the previous summary and the new conversation history. "
        "Keep key user intents, entities, video topics discussed, technical details, and answers. "
        "Return ONLY the updated summary in 1-3 clear paragraphs. No meta-commentary."
    )

    user_prompt = ""
    if existing_summary.strip():
        user_prompt += f"Previous Conversation Summary:\n{existing_summary}\n\n"
    user_prompt += f"New Conversation Turns to Fold Into Summary:\n{convo_str}\n\n"
    user_prompt += "Please provide the updated consolidated conversation summary."

    try:
        response = client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.2}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"[AGENT] Warning: Recursive summary generation failed: {e}")
        return existing_summary


def format_chat_video_library_header(chat_id: str) -> str:
    """
    Returns a clean, human-readable catalog of all videos registered in this chat workspace.
    Injected into prompts so the LLM is always aware of the active chat's video library.
    """
    videos = db.get_chat_videos(chat_id)
    if not videos:
        return "Active Chat Video Library: No videos are currently indexed in this chat workspace."

    lines = [f"Active Chat Video Library ({len(videos)} video{'s' if len(videos) != 1 else ''} in this workspace):"]
    for idx, v in enumerate(videos, 1):
        v_title = v.get("video_title", "Untitled Video")
        v_url = v.get("video_url", "")
        chunks_count = v.get("chunk_count", 0)
        lines.append(f"  {idx}. \"{v_title}\" ({v_url}) - {chunks_count} segments")
    return "\n".join(lines)


def load_chat_video_summaries(chat_id: str) -> List[Dict[str, Any]]:
    """
    Loads video transcripts / timeline chunks for all videos in this chat.
    """
    videos = db.get_chat_videos(chat_id)
    video_data = []

    for v in videos:
        fname = v.get("folder_name", "")
        possible_paths = [
            os.path.join(BASE_DIR, "op", chat_id, fname),
            os.path.join(BASE_DIR, "op", fname),
        ]
        v_folder = None
        for p in possible_paths:
            if os.path.isdir(p):
                v_folder = p
                break

        # If not found directly, search recursively in op/chat_id (for playlists with nested subfolders)
        if not v_folder and fname:
            chat_op_dir = os.path.join(BASE_DIR, "op", chat_id)
            if os.path.isdir(chat_op_dir):
                for root, dirs, files in os.walk(chat_op_dir):
                    if os.path.basename(root).lower() == fname.lower():
                        v_folder = root
                        break

        chunks = []
        if v_folder:
            chunks_file = os.path.join(v_folder, "chunks.json")
            if os.path.exists(chunks_file):
                try:
                    with open(chunks_file, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
                except Exception as e:
                    print(f"[AGENT] Warning: could not load chunks.json from {v_folder}: {e}")

        video_data.append({
            "title": v.get("video_title", "Untitled Video"),
            "url": v.get("video_url", ""),
            "chunks": chunks,
            "chunk_count": v.get("chunk_count", len(chunks))
        })
    return video_data


def format_whole_video_context(video_data: List[Dict[str, Any]]) -> str:
    """
    Formats chronological video transcripts across all chunks for full-video summarization.
    """
    if not video_data:
        return "No video transcripts found in this chat."

    output = []
    for v_idx, v in enumerate(video_data, 1):
        v_title = v["title"]
        v_url = v["url"]
        chunks = v["chunks"]

        output.append(f"=== VIDEO {v_idx}: {v_title} ({v_url}) ===")
        if not chunks:
            output.append("  (Transcript chunks are being processed or unavailable)")
            continue

        start_time = chunks[0].get("start_timestamp", "00:00:00")
        end_time = chunks[-1].get("end_timestamp", "00:00:00")
        output.append(f"Full Duration: {start_time} to {end_time} ({len(chunks)} total segments)")
        output.append("Chronological Transcript Segments:")

        # If chunks are many (e.g. > 35), sample them evenly to fit within context
        sample_step = max(1, len(chunks) // 35) if len(chunks) > 35 else 1
        for idx in range(0, len(chunks), sample_step):
            c = chunks[idx]
            s_ts = c.get("start_timestamp", "")
            e_ts = c.get("end_timestamp", "")
            ts_url = c.get("timestamp_url", "")
            plain = c.get("plain_text", c.get("text", "")).strip()
            if len(plain) > 300:
                plain = plain[:300] + "..."
            output.append(f"  [{s_ts} - {e_ts}] ({ts_url}) {plain}")

    return "\n".join(output)


class VideoRAGAgent:
    """
    Full Phase 4 Conversational Agent:
    - Query Transformation (gpt-oss:120b-cloud -> 5 queries)
    - Parallel Hybrid Retrieval (Dense 10 + BM25 10)
    - Reciprocal Rank Fusion (RRF -> Top 10 chunks)
    - FlashRank Cross-Encoder Re-ranking (Top 5 chunks)
    - Grounded Synthesis with Clickable Deep-Link Citations
    - Multi-turn Conversation Memory & 8-Message Buffer with Recursive Summarization
    - Per-Chat / Per-User Data Isolation
    """
    def __init__(
        self,
        model_name: str = OLLAMA_MODEL_NAME,
        retriever: Optional[HybridRetriever] = None,
        client: Optional[ollama.Client] = None
    ):
        self.model_name = model_name
        self.client = client or ollama.Client(host=OLLAMA_BASE_URL)
        self.retriever = retriever or HybridRetriever()
        self.query_transformer = QueryTransformer(model_name=self.model_name, client=self.client)
        self.memory = ConversationMemory()

    def chat_scoped(
        self,
        user_query: str,
        chat_id: str,
        user_id: str,
        num_queries: int = NUM_EXPANDED_QUERIES,
        top_dense: int = TOP_DENSE_PER_QUERY,
        top_bm25: int = TOP_BM25_PER_QUERY,
        rrf_top_candidates: int = RRF_TOP_CANDIDATES,
        final_top_k: int = FINAL_TOP_K
    ) -> Dict[str, Any]:
        """
        Executes a conversational turn strictly scoped to chat_id and user_id.
        Maintains an 8-message buffer and folds older turns into a recursive summary.
        """
        chat = db.get_chat(chat_id, user_id)
        if not chat:
            raise ValueError(f"Chat '{chat_id}' not found for user.")

        # Save user message to database
        db.add_chat_message(chat_id, user_id, "user", user_query)

        # Retrieve all messages for this chat
        all_messages = db.get_chat_messages(chat_id)
        # Exclude the very last message (which is the current query) for memory context
        prior_messages = all_messages[:-1]

        # Process 8-message buffer and recursive summary
        existing_summary = chat.get("summary", "")
        if len(prior_messages) > CHAT_BUFFER_SIZE:
            older = prior_messages[:-CHAT_BUFFER_SIZE]
            buffer = prior_messages[-CHAT_BUFFER_SIZE:]
            print(f"[AGENT] Updating recursive summary for {len(older)} older messages in chat {chat_id}...")
            updated_summary = update_recursive_summary(existing_summary, older, self.client, self.model_name)
            db.update_chat_summary(chat_id, updated_summary)
        else:
            buffer = prior_messages
            updated_summary = existing_summary

        # Step 0: Classify query intent
        intent = self.query_transformer.classify_intent(user_query)
        print(f"[AGENT] Detected query intent: {intent} (chat_id={chat_id})")

        # --- ROUTE 1: DIRECT GENERAL KNOWLEDGE CHAT (BYPASS RAG) ---
        if intent == "DIRECT_CHAT":
            print(f"[AGENT] Executing DIRECT_CHAT intent (bypassing video search)...")
            video_lib_header = format_chat_video_library_header(chat_id)
            system_prompt = (
                "You are an expert, helpful conversational assistant for this Video Knowledge workspace.\n\n"
                f"{video_lib_header}\n\n"
                "INSTRUCTIONS:\n"
                "- If the user asks workspace questions (e.g. how many videos are here, what videos exist in this chat, "
                "or asks for a list/names of the videos), answer accurately and directly using the Active Chat Video Library above.\n"
                "- For greetings, social pleasantries, or general questions, reply politely and helpfully.\n"
                "- Format code snippets cleanly in markdown if applicable. Do not invent fake video references."
            )
            messages = [{"role": "system", "content": system_prompt}]
            for m in buffer:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_query})

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.3}
            )
            answer = response["message"]["content"].strip()
            db.add_chat_message(
                chat_id,
                user_id,
                "assistant",
                answer,
                metadata={"intent": "direct_chat"}
            )
            return {
                "query": user_query,
                "intent": "direct_chat",
                "expanded_queries": [],
                "chunks": [],
                "answer": answer,
                "summary": updated_summary
            }

        # --- ROUTE 2: FULL VIDEO SUMMARY & CHAPTERS (WHOLE-VIDEO MODE) ---
        elif intent == "VIDEO_SUMMARY":
            print(f"[AGENT] Executing VIDEO_SUMMARY intent for chat {chat_id}...")
            video_data = load_chat_video_summaries(chat_id)
            context_text = format_whole_video_context(video_data)
            all_chunks = [c for v in video_data for c in v["chunks"]]

            system_prompt = (
                "You are an expert video content analyst. Provide a comprehensive, rich, and highly structured summary "
                "of the entire video based on its chronological transcript.\n\n"
                "YOUR RESPONSE MUST INCLUDE THE FOLLOWING SECTIONS:\n"
                "1. 📌 **Executive Overview**: A clear 2-3 paragraph summary of the video's premise, objectives, and overall outcome.\n"
                "2. ⏱ **Chapter-by-Chapter Timeline**: A chronological breakdown of major segments with exact timestamps in this format: "
                "[Chapter Name @ MM:SS](timestamp_url) - 1-2 sentence description of what happens.\n"
                "3. 💡 **Key Takeaways & Concepts**: Bulleted list of critical techniques, tools, commands, or arguments presented.\n"
                "4. 🎯 **Conclusion & Action Items**: Target audience, next steps, or practical conclusions."
            )

            user_content = (
                f"Full Video Transcript & Timeline Across All Segments:\n{context_text}\n\n"
                f"User Request: {user_query}\n\n"
                "Synthesize a thorough, complete whole-video summary and timeline."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.2}
            )
            raw_answer = response["message"]["content"].strip()
            answer = CitationPromptEngine.linkify_citations(raw_answer, all_chunks)

            db.add_chat_message(
                chat_id,
                user_id,
                "assistant",
                answer,
                metadata={"intent": "video_summary"}
            )
            return {
                "query": user_query,
                "intent": "video_summary",
                "expanded_queries": ["Executive Overview", "Chapter Timeline", "Key Takeaways"],
                "chunks": all_chunks[:5] if all_chunks else [],
                "answer": answer,
                "summary": updated_summary
            }

        # --- ROUTE 3: TARGETED VIDEO RAG (HYBRID SEARCH & FLASHRANK) ---
        # Check if query is directly co-related with previous chat history
        recent_history_str = ""
        if buffer:
            recent_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in buffer[-4:]])

        is_correlated = False
        context_to_inject = ""
        if recent_history_str.strip():
            print(f"[AGENT] Checking if query is co-related with chat history (chat_id={chat_id})...")
            is_correlated = self.query_transformer.is_correlated_with_history(
                user_query=user_query,
                chat_context=recent_history_str
            )

        if is_correlated:
            context_to_inject = ""
            if updated_summary:
                context_to_inject += f"Prior Conversation Summary: {updated_summary}\n"
            context_to_inject += f"Recent Turns:\n{recent_history_str}"
            print(f"[AGENT] Query is CO-RELATED with chat history. Injecting context for query expansion.")
        else:
            context_to_inject = ""
            print(f"[AGENT] Query is INDEPENDENT (not co-related). Generating search queries without history.")

        # Step 1: Query Expansion via gpt-oss:120b-cloud
        print(f"[AGENT] Expanding query into {num_queries} variations (chat_id={chat_id})...")
        expanded_queries = self.query_transformer.expand_query(
            user_query=user_query,
            chat_context=context_to_inject,
            num_queries=num_queries
        )
        print(f"[AGENT] Generated {len(expanded_queries)} search queries: {expanded_queries}")

        # Step 2: Parallel Scoped Hybrid Retrieval + FlashRank re-ranking
        print(f"[AGENT] Executing scoped hybrid retrieval & FlashRank re-ranking for chat {chat_id}...")
        top_chunks = self.retriever.multi_query_hybrid_search(
            original_query=user_query,
            queries=expanded_queries,
            top_dense=top_dense,
            top_bm25=top_bm25,
            rrf_top_candidates=rrf_top_candidates,
            final_top_k=final_top_k,
            chat_id=chat_id,
            user_id=user_id
        )
        print(f"[AGENT] Retrieved {len(top_chunks)} precision chunks for chat {chat_id}.")

        # Step 3: Build Prompt
        context_str_chunks = CitationPromptEngine.format_context_chunks(top_chunks)
        video_lib_header = format_chat_video_library_header(chat_id)
        system_prompt = CitationPromptEngine.build_system_prompt(video_library_header=video_lib_header)

        messages = [{"role": "system", "content": system_prompt}]

        if updated_summary and is_correlated:
            messages.append({
                "role": "system",
                "content": f"Prior Conversation Summary (context before recent messages):\n{updated_summary}"
            })

        # Add recent buffer messages
        for m in buffer:
            messages.append({"role": m["role"], "content": m["content"]})

        # Add current user prompt with chunks
        user_content = (
            f"Context Chunks (Strictly from this chat's video library):\n{context_str_chunks}\n\n"
            f"User Question: {user_query}\n\n"
            f"Provide an accurate, deeply grounded answer citing exact timestamps [Video Title @ MM:SS](URL)."
        )
        messages.append({"role": "user", "content": user_content})

        # Step 4: Synthesize Answer
        print(f"[AGENT] Synthesizing answer with {self.model_name}...")
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": 0.2}
        )
        answer = response["message"]["content"].strip()
        answer = CitationPromptEngine.linkify_citations(answer, top_chunks)

        # Step 5: Save assistant response to database
        db.add_chat_message(
            chat_id,
            user_id,
            "assistant",
            answer,
            metadata={
                "intent": "rag",
                "expanded_queries": expanded_queries,
                "chunks": top_chunks
            }
        )

        return {
            "query": user_query,
            "intent": "rag",
            "expanded_queries": expanded_queries,
            "chunks": top_chunks,
            "answer": answer,
            "summary": updated_summary
        }

    def chat_scoped_stream(
        self,
        user_query: str,
        chat_id: str,
        user_id: str,
        num_queries: int = NUM_EXPANDED_QUERIES,
        top_dense: int = TOP_DENSE_PER_QUERY,
        top_bm25: int = TOP_BM25_PER_QUERY,
        rrf_top_candidates: int = RRF_TOP_CANDIDATES,
        final_top_k: int = FINAL_TOP_K
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming execution of a conversational turn strictly scoped to chat_id and user_id.
        Yields real-time events:
        - {"event": "status", "message": "..."}
        - {"event": "metadata", "expanded_queries": [...], "chunks": [...]}
        - {"event": "token", "delta": "..."}
        - {"event": "done", "answer": "...", "expanded_queries": [...], "chunks": [...], "summary": "..."}
        """
        chat = db.get_chat(chat_id, user_id)
        if not chat:
            yield {"event": "error", "message": f"Chat '{chat_id}' not found for user."}
            return

        # Save user message to database
        db.add_chat_message(chat_id, user_id, "user", user_query)

        # Retrieve all messages for this chat
        all_messages = db.get_chat_messages(chat_id)
        prior_messages = all_messages[:-1]

        # Process 8-message buffer and recursive summary
        existing_summary = chat.get("summary", "")
        if len(prior_messages) > CHAT_BUFFER_SIZE:
            older = prior_messages[:-CHAT_BUFFER_SIZE]
            buffer = prior_messages[-CHAT_BUFFER_SIZE:]
            yield {"event": "status", "message": "Updating conversation summary..."}
            updated_summary = update_recursive_summary(existing_summary, older, self.client, self.model_name)
            db.update_chat_summary(chat_id, updated_summary)
        else:
            buffer = prior_messages
            updated_summary = existing_summary

        # Step 0: Classify query intent
        yield {"event": "status", "message": "Analyzing query intent..."}
        intent = self.query_transformer.classify_intent(user_query)

        # --- ROUTE 1: DIRECT GENERAL KNOWLEDGE CHAT (BYPASS RAG) ---
        if intent == "DIRECT_CHAT":
            yield {"event": "status", "message": "Answering with chat workspace context (bypassing video search)..."}
            video_lib_header = format_chat_video_library_header(chat_id)
            system_prompt = (
                "You are an expert, helpful conversational assistant for this Video Knowledge workspace.\n\n"
                f"{video_lib_header}\n\n"
                "INSTRUCTIONS:\n"
                "- If the user asks workspace questions (e.g. how many videos are here, what videos exist in this chat, "
                "or asks for a list/names of the videos), answer accurately and directly using the Active Chat Video Library above.\n"
                "- For greetings, social pleasantries, or general questions, reply politely and helpfully.\n"
                "- Format code snippets cleanly in markdown if applicable. Do not invent fake video references."
            )
            messages = [{"role": "system", "content": system_prompt}]
            for m in buffer:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": user_query})

            try:
                stream = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.3}
                )
                accumulated_tokens = []
                for chunk in stream:
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        accumulated_tokens.append(delta)
                        yield {"event": "token", "delta": delta}

                answer = "".join(accumulated_tokens).strip()
                db.add_chat_message(
                    chat_id,
                    user_id,
                    "assistant",
                    answer,
                    metadata={"intent": "direct_chat"}
                )
                yield {
                    "event": "done",
                    "intent": "direct_chat",
                    "answer": answer,
                    "expanded_queries": [],
                    "chunks": [],
                    "summary": updated_summary
                }
            except Exception as e:
                print(f"[AGENT] Error during DIRECT_CHAT synthesis: {e}")
                yield {"event": "error", "message": f"Direct chat error: {str(e)}"}
            return

        # --- ROUTE 2: FULL VIDEO SUMMARY & CHAPTERS (WHOLE-VIDEO MODE) ---
        elif intent == "VIDEO_SUMMARY":
            yield {"event": "status", "message": "Compiling full video transcript for whole-video summary & chapters..."}
            video_data = load_chat_video_summaries(chat_id)
            context_text = format_whole_video_context(video_data)
            all_chunks = [c for v in video_data for c in v["chunks"]]

            system_prompt = (
                "You are an expert video content analyst. Provide a comprehensive, rich, and highly structured summary "
                "of the entire video based on its chronological transcript.\n\n"
                "YOUR RESPONSE MUST INCLUDE THE FOLLOWING SECTIONS:\n"
                "1. 📌 **Executive Overview**: A clear 2-3 paragraph summary of the video's premise, objectives, and overall outcome.\n"
                "2. ⏱ **Chapter-by-Chapter Timeline**: A chronological breakdown of major segments with exact timestamps in this format: "
                "[Chapter Name @ MM:SS](timestamp_url) - 1-2 sentence description of what happens.\n"
                "3. 💡 **Key Takeaways & Concepts**: Bulleted list of critical techniques, tools, commands, or arguments presented.\n"
                "4. 🎯 **Conclusion & Action Items**: Target audience, next steps, or practical conclusions."
            )

            user_content = (
                f"Full Video Transcript & Timeline Across All Segments:\n{context_text}\n\n"
                f"User Request: {user_query}\n\n"
                "Synthesize a thorough, complete whole-video summary and timeline."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            yield {"event": "status", "message": "Streaming video summary & chapter outline from gpt-oss:120b-cloud..."}
            try:
                stream = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                    options={"temperature": 0.2}
                )
                accumulated_tokens = []
                for chunk in stream:
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        accumulated_tokens.append(delta)
                        yield {"event": "token", "delta": delta}

                raw_answer = "".join(accumulated_tokens).strip()
                answer = CitationPromptEngine.linkify_citations(raw_answer, all_chunks)

                db.add_chat_message(
                    chat_id,
                    user_id,
                    "assistant",
                    answer,
                    metadata={"intent": "video_summary"}
                )
                yield {
                    "event": "done",
                    "intent": "video_summary",
                    "answer": answer,
                    "expanded_queries": ["Executive Overview", "Chapter Timeline", "Key Takeaways"],
                    "chunks": all_chunks[:5] if all_chunks else [],
                    "summary": updated_summary
                }
            except Exception as e:
                print(f"[AGENT] Error during VIDEO_SUMMARY synthesis: {e}")
                yield {"event": "error", "message": f"Summary generation error: {str(e)}"}
            return

        # --- ROUTE 3: TARGETED VIDEO RAG (HYBRID SEARCH & FLASHRANK) ---
        # Check if query is directly co-related with previous chat history
        recent_history_str = ""
        if buffer:
            recent_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in buffer[-4:]])

        is_correlated = False
        context_to_inject = ""
        if recent_history_str.strip():
            yield {"event": "status", "message": "Checking query correlation with chat history..."}
            is_correlated = self.query_transformer.is_correlated_with_history(
                user_query=user_query,
                chat_context=recent_history_str
            )

        if is_correlated:
            context_to_inject = ""
            if updated_summary:
                context_to_inject += f"Prior Conversation Summary: {updated_summary}\n"
            context_to_inject += f"Recent Turns:\n{recent_history_str}"
            print(f"[AGENT] Streaming: Query is CO-RELATED with chat history. Injecting context.")
        else:
            context_to_inject = ""
            print(f"[AGENT] Streaming: Query is INDEPENDENT. No history injected.")

        # Step 1: Query Expansion via gpt-oss:120b-cloud
        yield {"event": "status", "message": "Generating 5 search variations..."}
        expanded_queries = self.query_transformer.expand_query(
            user_query=user_query,
            chat_context=context_to_inject,
            num_queries=num_queries
        )
        yield {"event": "metadata", "expanded_queries": expanded_queries}

        # Step 2: Parallel Scoped Hybrid Retrieval + FlashRank re-ranking
        yield {"event": "status", "message": "Searching video transcripts with FlashRank re-ranking..."}
        top_chunks = self.retriever.multi_query_hybrid_search(
            original_query=user_query,
            queries=expanded_queries,
            top_dense=top_dense,
            top_bm25=top_bm25,
            rrf_top_candidates=rrf_top_candidates,
            final_top_k=final_top_k,
            chat_id=chat_id,
            user_id=user_id
        )
        yield {"event": "metadata", "chunks": top_chunks}

        # Step 3: Build Prompt
        context_str_chunks = CitationPromptEngine.format_context_chunks(top_chunks)
        video_lib_header = format_chat_video_library_header(chat_id)
        system_prompt = CitationPromptEngine.build_system_prompt(video_library_header=video_lib_header)

        messages = [{"role": "system", "content": system_prompt}]

        if updated_summary and is_correlated:
            messages.append({
                "role": "system",
                "content": f"Prior Conversation Summary (context before recent messages):\n{updated_summary}"
            })

        for m in buffer:
            messages.append({"role": m["role"], "content": m["content"]})

        user_content = (
            f"Context Chunks (Strictly from this chat's video library):\n{context_str_chunks}\n\n"
            f"User Question: {user_query}\n\n"
            f"Provide an accurate, deeply grounded answer citing exact timestamps [Video Title @ MM:SS](URL)."
        )
        messages.append({"role": "user", "content": user_content})

        # Step 4: Stream Synthesize Answer
        yield {"event": "status", "message": "Streaming answer from gpt-oss:120b-cloud..."}
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={"temperature": 0.2}
            )

            accumulated_tokens = []
            for chunk in stream:
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    accumulated_tokens.append(delta)
                    yield {"event": "token", "delta": delta}

            raw_answer = "".join(accumulated_tokens).strip()
            answer = CitationPromptEngine.linkify_citations(raw_answer, top_chunks)

            # Step 5: Save assistant response to database
            db.add_chat_message(
                chat_id,
                user_id,
                "assistant",
                answer,
                metadata={
                    "expanded_queries": expanded_queries,
                    "chunks": top_chunks
                }
            )

            yield {
                "event": "done",
                "answer": answer,
                "expanded_queries": expanded_queries,
                "chunks": top_chunks,
                "summary": updated_summary
            }

        except Exception as e:
            print(f"[AGENT] Error during streaming synthesis: {e}")
            yield {"event": "error", "message": f"Synthesis error: {str(e)}"}

    def chat(
        self,
        user_query: str,
        num_queries: int = NUM_EXPANDED_QUERIES,
        top_dense: int = TOP_DENSE_PER_QUERY,
        top_bm25: int = TOP_BM25_PER_QUERY,
        rrf_top_candidates: int = RRF_TOP_CANDIDATES,
        final_top_k: int = FINAL_TOP_K
    ) -> Dict[str, Any]:
        """
        Legacy global chat turn (for CLI / testing without chat_id).
        """
        context_summary = self.memory.get_context_summary(max_recent=2)
        is_correlated = self.query_transformer.is_correlated_with_history(user_query, context_summary) if context_summary else False
        context_to_inject = context_summary if is_correlated else ""
        expanded_queries = self.query_transformer.expand_query(
            user_query=user_query,
            chat_context=context_to_inject,
            num_queries=num_queries
        )
        top_chunks = self.retriever.multi_query_hybrid_search(
            original_query=user_query,
            queries=expanded_queries,
            top_dense=top_dense,
            top_bm25=top_bm25,
            rrf_top_candidates=rrf_top_candidates,
            final_top_k=final_top_k
        )
        context_str = CitationPromptEngine.format_context_chunks(top_chunks)
        system_prompt = CitationPromptEngine.build_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]
        for turn in self.memory.get_history():
            messages.append(turn)

        user_content = (
            f"Context Chunks:\n{context_str}\n\n"
            f"User Question: {user_query}\n\n"
            f"Please provide an accurate, grounded answer with clickable markdown citations ([Video Title @ MM:SS](URL))."
        )
        messages.append({"role": "user", "content": user_content})

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": 0.2}
        )
        answer = response["message"]["content"].strip()
        answer = CitationPromptEngine.linkify_citations(answer, top_chunks)

        self.memory.add_user_message(user_query)
        self.memory.add_assistant_message(answer)

        return {
            "query": user_query,
            "expanded_queries": expanded_queries,
            "chunks": top_chunks,
            "answer": answer
        }

    def reset_memory(self):
        self.memory.clear()
