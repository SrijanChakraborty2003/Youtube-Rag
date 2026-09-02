import os
import re
from typing import List, Dict, Any, Optional

def parse_srt_timestamp(ts_str: str) -> float:
    """
    Parses an SRT timestamp string (e.g., '00:01:23,450' or '00:01:23.450')
    into float seconds.
    """
    ts_str = ts_str.strip().replace(',', '.')
    parts = ts_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {ts_str}")
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600.0 + minutes * 60.0 + seconds

def format_seconds_to_timestamp(seconds: float) -> str:
    """
    Formats float seconds into standard HH:MM:SS string.
    """
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def extract_video_id(url: str) -> Optional[str]:
    """
    Extracts the 11-character YouTube video ID from various YouTube URL formats.
    """
    if not url:
        return None
    patterns = [
        r'(?:v=|\/embed\/|\/1080p\/|\/720p\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|\?v=)([^#\&\?]*)"?',
        r'([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            candidate = match.group(1)
            if len(candidate) == 11:
                return candidate
    return None

def parse_srt_file(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parses an SRT subtitle file into a list of cue dictionaries:
    [{'index': int, 'start': float, 'end': float, 'text': str}]
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found at: {srt_path}")

    cues = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                time_line = lines[1]
                if '-->' in time_line:
                    start_str, end_str = time_line.split('-->')
                    start_sec = parse_srt_timestamp(start_str)
                    end_sec = parse_srt_timestamp(end_str)
                    text = " ".join(lines[2:])
                    cues.append({
                        "index": index,
                        "start": start_sec,
                        "end": end_sec,
                        "text": text
                    })
            except Exception:
                continue

    return cues

def create_sliding_window_chunks(
    cues: List[Dict[str, Any]],
    video_metadata: Dict[str, Any],
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Groups SRT cues into sliding window chunks.
    Default: chunk_seconds = 60.0, overlap_seconds = 5.0 (step_seconds = 55.0).
    """
    if not cues:
        return []

    title = video_metadata.get("title", "Untitled Video")
    video_url = video_metadata.get("url", "")
    video_id = extract_video_id(video_url) or "video"

    step_seconds = chunk_seconds - overlap_seconds
    if step_seconds <= 0:
        raise ValueError("chunk_seconds must be greater than overlap_seconds")

    max_end_time = max(cue["end"] for cue in cues)
    chunks = []
    chunk_index = 0

    win_start = 0.0
    while win_start < max_end_time:
        win_end = win_start + chunk_seconds

        # Select cues overlapping with current window
        matching_cues = [
            cue for cue in cues
            if cue["end"] > win_start and cue["start"] < win_end
        ]

        if matching_cues:
            actual_start_sec = matching_cues[0]["start"]
            actual_end_sec = matching_cues[-1]["end"]

            # 1. Build text with inline timestamp tags e.g. "[00:12:50] What did we else have?"
            tagged_cue_parts = []
            for cue in matching_cues:
                cue_ts_str = format_seconds_to_timestamp(cue["start"])
                cue_clean_text = cue["text"].strip()
                if cue_clean_text:
                    tagged_cue_parts.append(f"[{cue_ts_str}] {cue_clean_text}")

            text_with_timestamps = " ".join(tagged_cue_parts)
            text_with_timestamps = re.sub(r'\s+', ' ', text_with_timestamps).strip()

            # 2. Build plain concatenated text
            raw_text = " ".join(cue["text"] for cue in matching_cues)
            raw_text = re.sub(r'\s+', ' ', raw_text).strip()

            if raw_text:
                start_sec_int = int(actual_start_sec)
                if video_id and video_id != "video":
                    timestamp_url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec_int}s"
                elif video_url:
                    sep = "&" if "?" in video_url else "?"
                    timestamp_url = f"{video_url}{sep}t={start_sec_int}s"
                else:
                    timestamp_url = ""

                # 3. Build granular cue list with exact deep links for each cue
                detailed_cues = []
                for cue in matching_cues:
                    c_start_int = int(cue["start"])
                    if video_id and video_id != "video":
                        c_url = f"https://www.youtube.com/watch?v={video_id}&t={c_start_int}s"
                    elif video_url:
                        sep = "&" if "?" in video_url else "?"
                        c_url = f"{video_url}{sep}t={c_start_int}s"
                    else:
                        c_url = ""
                    detailed_cues.append({
                        "start_seconds": round(cue["start"], 2),
                        "end_seconds": round(cue["end"], 2),
                        "start_timestamp": format_seconds_to_timestamp(cue["start"]),
                        "timestamp_url": c_url,
                        "text": cue["text"]
                    })

                chunk_obj = {
                    "chunk_id": f"{video_id}_{chunk_index}",
                    "video_title": title,
                    "video_url": video_url,
                    "start_seconds": round(actual_start_sec, 2),
                    "end_seconds": round(actual_end_sec, 2),
                    "start_timestamp": format_seconds_to_timestamp(actual_start_sec),
                    "end_timestamp": format_seconds_to_timestamp(actual_end_sec),
                    "timestamp_url": timestamp_url,
                    "text": text_with_timestamps,
                    "plain_text": raw_text,
                    "cues": detailed_cues
                }
                chunks.append(chunk_obj)
                chunk_index += 1

        win_start += step_seconds

    return chunks
