import os
import sys
import argparse
from audio_downloader import download_audio
from audio_transcriber import transcribe_audio, save_srt
from rag.ingest import ingest_video

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

def process_track_synchronously(track):
    """
    Transcribes the downloaded WAV file immediately and deletes it to save disk space
    before the next download begins.
    """
    title = track["title"]
    video_folder = track["video_folder"]
    audio_path = track["audio_path"]
    
    print(f"\n[ASR] Starting transcription for: {title}")
    if os.path.exists(audio_path):
        try:
            # Run ASR transcription
            segments = transcribe_audio(
                audio_file=audio_path,
                chunk_seconds=60.0,
                overlap_seconds=1.0
            )
            
            # Save subtitle file
            srt_path = os.path.join(video_folder, "audio.srt")
            save_srt(segments, srt_path)
            print(f"[ASR] Successfully saved SRT to: {srt_path}")
            print(f"[ASR] Parakeet is done processing this file: {title}")
            
            # Run Data Ingestion & Sliding Window Chunking Pipeline (60s window, 5s overlap)
            try:
                ingest_video(video_folder, chunk_seconds=60.0, overlap_seconds=5.0)
            except Exception as ingest_err:
                print(f"[INGEST] Error during chunking ingestion for {title}: {ingest_err}", file=sys.stderr)
            
            # Delete the massive audio WAV file immediately to free C drive space
            try:
                os.remove(audio_path)
                print(f"[ASR] Cleaned up temporary WAV file: {audio_path}")
            except Exception as del_err:
                print(f"[ASR] Warning: Could not delete audio WAV file: {del_err}", file=sys.stderr)
        except Exception as e:
            print(f"[ASR] Error transcribing {title}: {e}", file=sys.stderr)
    else:
        print(f"[ASR] Warning: Audio file not found at {audio_path}", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and Transcribe YouTube links sequentially (WAV + local ASR)")
    parser.add_argument("-o", "--output", default=r"C:\Users\User\Downloads\Rag (2)\Rag\op", help="Output directory for the files")
    parser.add_argument("-c", "--cookies-from-browser", help="Browser to read cookies from (e.g. chrome, edge, firefox, brave)")
    
    args = parser.parse_args()

    # Hardcoded list of YouTube links (can be video or playlist links)
    YOUTUBE_LINKS = [
        "https://www.youtube.com/watch?v=0hgzLDHplYk",
        # Add more video or playlist links here
    ]

    print("Starting sequential pipeline: Downloader -> Transcriber -> WAV Cleanup (Parakeet 0.6b)...")
    
    for idx, link in enumerate(YOUTUBE_LINKS, 1):
        print(f"\n[{idx}/{len(YOUTUBE_LINKS)}] Processing URL: {link}")
        
        # Download audio one by one. The callback transcribes and deletes each WAV file
        # synchronously before yt-dlp proceeds to download the next video.
        download_audio(
            url=link, 
            output_dir=args.output, 
            cookies_from_browser=args.cookies_from_browser,
            on_complete_callback=process_track_synchronously
        )

    print("\nAll downloads and transcriptions completed successfully!")
