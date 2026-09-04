import os
import sys
import json
import re
import tempfile
import yt_dlp
from yt_dlp.networking.impersonate import ImpersonateTarget

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

def sanitize_folder_name(name):
    """
    Removes invalid characters for folder names.
    """
    sanitized = re.sub(r'[\\/*?:"<>|]', "", name)
    return sanitized.strip()

def get_ffmpeg_path():
    """
    Returns the path to ffmpeg executable, utilizing imageio_ffmpeg if installed.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        if os.path.exists(ffmpeg_exe):
            return ffmpeg_exe
    except Exception:
        pass
    import shutil
    ffmpeg_sys = shutil.which("ffmpeg")
    if ffmpeg_sys:
        return ffmpeg_sys
    return None

def download_audio(
    url: str,
    output_dir: str = '.',
    cookies_from_browser: str = None,
    on_complete_callback = None,
    on_progress_callback = None
) -> list:
    """
    Downloads a YouTube video or playlist as audio in WAV format.
    Creates a folder for each video containing the audio file and a metadata JSON file.
    Invokes on_progress_callback(event_data) at each stage.
    Invokes on_complete_callback(track_info) immediately upon finishing each download.
    Returns a list of dictionaries with info about downloaded videos.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ffmpeg_path = get_ffmpeg_path()

    # Options to extract video information (works for both single videos and playlists)
    extract_opts = {
        'extract_flat': True,
        'impersonate': ImpersonateTarget.from_str('chrome'),
        'ignoreerrors': True,
    }
    if ffmpeg_path:
        extract_opts['ffmpeg_location'] = ffmpeg_path
    if cookies_from_browser:
        extract_opts['cookiesfrombrowser'] = (cookies_from_browser,)

    print(f"Extracting information from: {url}")
    with yt_dlp.YoutubeDL(extract_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            print(f"Error extracting info: {e}", file=sys.stderr)
            return []

    if not info:
        print("No information could be retrieved.", file=sys.stderr)
        return []

    # Determine if it's a playlist or a single video
    is_playlist = 'entries' in info
    if is_playlist:
        entries = [entry for entry in info['entries'] if entry]
        playlist_title = sanitize_folder_name(info.get('title', 'Playlist'))
        target_dir = os.path.join(output_dir, playlist_title)
    else:
        entries = [info]
        target_dir = output_dir

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    total_videos = len(entries)
    print(f"Found {total_videos} video(s) to process.")
    print("-" * 50)

    if on_progress_callback:
        on_progress_callback({
            "stage": "discovered",
            "total": total_videos,
            "title": info.get('title', 'Playlist') if is_playlist else entries[0].get('title', 'Video')
        })

    downloaded_tracks = []

    for idx, entry in enumerate(entries, 1):
        title = entry.get('title')
        video_id = entry.get('id')
        
        # Determine the video URL
        video_url = entry.get('url') or entry.get('webpage_url')
        if video_url and not video_url.startswith('http'):
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        elif not video_url and video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
        if not title or not video_url:
            print(f"Skipping entry {idx} due to missing title or URL.")
            continue

        if on_progress_callback:
            on_progress_callback({
                "stage": "downloading",
                "idx": idx,
                "total": total_videos,
                "title": title,
                "url": video_url
            })

        print(f"[{idx}/{total_videos}] Downloading: {title}")


        # Create subfolder named after the sanitized video title inside target_dir
        safe_title = sanitize_folder_name(title)
        video_folder = os.path.join(target_dir, safe_title)
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)

        # Create a unique temporary directory for this video download
        video_temp_dir = tempfile.mkdtemp()

        # Download options for this specific video
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'audio.%(ext)s',
            'paths': {
                'home': os.path.abspath(video_folder),
                'temp': video_temp_dir,
            },
            'impersonate': ImpersonateTarget.from_str('chrome'),
            'postprocessors': [
                {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }
            ],
            'verbose': False,
            'quiet': True,
        }
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = ffmpeg_path
        if cookies_from_browser:
            ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)

        # Perform download
        import shutil
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Save metadata to JSON in the same folder
            metadata = {
                "title": title,
                "url": video_url
            }
            metadata_path = os.path.join(video_folder, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            audio_path = os.path.join(video_folder, "audio.wav")
            print(f"Successfully downloaded audio and saved metadata.json to: {video_folder}")
            track = {
                "title": title,
                "video_folder": video_folder,
                "audio_path": audio_path,
                "idx": idx,
                "total": total_videos,
                "url": video_url
            }
            downloaded_tracks.append(track)
            if on_complete_callback:
                on_complete_callback(track)

        except Exception as e:
            print(f"Error processing {title}: {e}", file=sys.stderr)
        finally:
            # Clean up the unique temporary directory
            shutil.rmtree(video_temp_dir, ignore_errors=True)

    return downloaded_tracks
