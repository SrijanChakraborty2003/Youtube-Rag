import os
import sys
import gc
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
import torch

# Limit PyTorch CPU threads to prevent CPU starvation for other subprocesses (like ffmpeg)
torch.set_num_threads(4)


try:
    import torchaudio
    _HAS_TORCHAUDIO = True
except (ImportError, OSError, Exception) as _ta_err:
    logger.warning("torchaudio import failed (%s); falling back to built-in linear resampler.", _ta_err)
    _HAS_TORCHAUDIO = False

import nemo.collections.asr as nemo_asr

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

logger = logging.getLogger("audio-transcription")
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_GPU_INFER_LOCK = threading.RLock()
_GPU_REQUEST_LOCK = threading.RLock()
 
_asr_model = None
_model_lock = threading.RLock()

def clear_gpu_memory(verbose: bool = False, sync: bool = False) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        if sync:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception as exc:
        if verbose:
            logger.warning("GPU clear failed: %s", exc)

class GPUMemoryGuard:
    def __init__(self, sync: bool = False):
        self.sync = sync

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory(sync=self.sync)

def get_asr_model():
    global _asr_model
    if _asr_model is not None:
        return _asr_model

    with _model_lock:
        if _asr_model is None:
            model_name = "nvidia/parakeet-tdt-0.6b-v2"
            logger.info("Loading ASR model: %s on device: %s", model_name, DEVICE)
            _asr_model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_name,
                map_location=DEVICE,
            )
            _asr_model.eval()

    return _asr_model

def _normalize_peak_(samples: np.ndarray, target_peak: float = 0.95) -> None:
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples *= (target_peak / max_val)

def _write_temp_wav(samples: np.ndarray, sr: int, chunk_idx: int, subtype: str = "PCM_16") -> str:
    temp_file = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_idx}.wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    sf.write(temp_path, samples, sr, subtype=subtype)
    return temp_path

def _safe_transcribe(file_list: List[str], timestamps: bool = True):
    asr_model = get_asr_model()
    with _GPU_INFER_LOCK:
        with GPUMemoryGuard():
            try:
                with torch.inference_mode():
                    return asr_model.transcribe(
                        file_list,
                        timestamps=timestamps,
                        batch_size=1,
                        num_workers=0,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e).lower()
                if "out of memory" in msg:
                    logger.error("GPU OOM during transcription: %s", e)
                    clear_gpu_memory(verbose=True, sync=True)
                if "stream is capturing" in msg or "capture must end" in msg:
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    logger.warning("Detected CUDA capture overlap. Consider enabling exclusive request mode.")
                raise

def _extract_segments(nemo_result, time_offset: float = 0.0) -> List[Dict[str, Any]]:
    if not nemo_result:
        return []

    out = []
    first = nemo_result[0]

    while isinstance(first, list) and len(first) > 0:
        first = first[0]

    if not first:
        return []

    ts_container = first.get("timestamp") if isinstance(first, dict) else getattr(first, "timestamp", None)
    if not ts_container:
        ts_container = first.get("timestep") if isinstance(first, dict) else getattr(first, "timestep", None)

    if ts_container and isinstance(ts_container, dict):
        segments = ts_container.get("segment") or ts_container.get("segments") or []
        for ts in segments:
            if isinstance(ts, dict):
                start = float(ts.get("start", ts.get("start_offset", 0.0))) + time_offset
                end = float(ts.get("end", ts.get("end_offset", start))) + time_offset
                seg = str(ts.get("segment") or ts.get("text") or "").strip()
            else:
                start = float(getattr(ts, "start", 0.0)) + time_offset
                end = float(getattr(ts, "end", start)) + time_offset
                seg = str(getattr(ts, "segment", getattr(ts, "text", ""))).strip()

            if seg:
                out.append({"start": start, "end": end, "segment": seg})

        if not out:
            words = ts_container.get("word") or ts_container.get("words") or []
            for w in words:
                if isinstance(w, dict):
                    start = float(w.get("start", w.get("start_offset", 0.0))) + time_offset
                    end = float(w.get("end", w.get("end_offset", start))) + time_offset
                    seg = str(w.get("word") or w.get("text") or "").strip()
                else:
                    start = float(getattr(w, "start", 0.0)) + time_offset
                    end = float(getattr(w, "end", start)) + time_offset
                    seg = str(getattr(w, "word", getattr(w, "text", ""))).strip()

                if seg:
                    out.append({"start": start, "end": end, "segment": seg})

    if not out:
        text = first.get("text") if isinstance(first, dict) else getattr(first, "text", None)
        if text is not None:
            text_str = str(text).strip()
            if text_str and not text_str.startswith("Hypothesis("):
                out.append({"start": time_offset, "end": time_offset, "segment": text_str})

    return out

def chunked_transcription_streaming(
    audio_file: str,
    target_sr: int = 16000,
    chunk_seconds: float = 45.0,
    overlap_seconds: float = 1.0,
    pcm_subtype: str = "PCM_16",
    cpu_resample_backend: str = "torchaudio",
    exclusive_request: bool = False,
) -> List[Dict[str, Any]]:
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory
        logger.info("Total GPU memory: %.2f GB", total_mem / (1024**3))
        if total_mem < 6 * (1024**3):
            logger.info("GPU memory is low (<6GB); reducing chunk size...")
            chunk_seconds = min(chunk_seconds, 120.0)
        elif total_mem < 8 * (1024**3):
            logger.info("GPU memory is medium (6-8GB); reducing chunk size...")
            chunk_seconds = min(chunk_seconds, 60.0)

    logger.info("Chunk size: %.2fs", chunk_seconds)
    all_segments: List[Dict[str, Any]] = []
    resampler = None

    with sf.SoundFile(audio_file, "r") as f:
        src_sr = f.samplerate
        n_channels = f.channels

        if src_sr != target_sr:
            if _HAS_TORCHAUDIO and cpu_resample_backend == "torchaudio":
                logger.info("Using torchaudio resampler: %d Hz -> %d Hz", src_sr, target_sr)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=src_sr, new_freq=target_sr, dtype=torch.float32
                )
            else:
                logger.info("Using linear resampler: %d Hz -> %d Hz", src_sr, target_sr)
                resampler = "linear"

        logger.info("Source SR: %d | Target SR: %d | Chunk: %.2fs, Overlap: %.2fs", src_sr, target_sr, chunk_seconds, overlap_seconds)
        chunk_frames = int(chunk_seconds * src_sr)
        overlap_frames = int(overlap_seconds * src_sr)
        step_frames = max(1, chunk_frames - overlap_frames)

        start_frame = 0
        chunk_idx = 0

        while start_frame < len(f):
            f.seek(start_frame)
            frames_to_read = min(chunk_frames, len(f) - start_frame)
            samples = f.read(frames_to_read, dtype="float32", always_2d=True)
            if samples.size == 0:
                break

            if n_channels > 1:
                samples = samples.mean(axis=1)
            else:
                samples = samples[:, 0]

            _normalize_peak_(samples)

            if resampler is not None:
                if resampler == "linear":
                    t = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)
                    with torch.inference_mode():
                        t = torch.nn.functional.interpolate(
                            t, size=int(t.shape[-1] * target_sr / src_sr),
                            mode="linear", align_corners=False
                        )
                    samples = t.squeeze(0).squeeze(0).contiguous().cpu().numpy()
                    del t
                else:
                    t = torch.from_numpy(samples)
                    with torch.inference_mode():
                        t = resampler(t)
                    samples = t.contiguous().cpu().numpy()
                    del t

            effective_sr = target_sr if resampler else src_sr
            tmp_path = _write_temp_wav(samples, effective_sr, chunk_idx, subtype=pcm_subtype)
            time_offset = start_frame / float(src_sr)
            
            video_folder_name = os.path.basename(os.path.dirname(audio_file))
            duration_total = len(f) / float(src_sr)
            chunk_end = min(time_offset + chunk_seconds, duration_total)
            print(f"[ASR] Processing segment {chunk_idx + 1} ({time_offset:.2f}s - {chunk_end:.2f}s) of {duration_total:.2f}s total for folder: '{video_folder_name}'")

            try:
                if exclusive_request:
                    with _GPU_REQUEST_LOCK:
                        result = _safe_transcribe([tmp_path], timestamps=True)
                else:
                    result = _safe_transcribe([tmp_path], timestamps=True)
                segs = _extract_segments(result, time_offset=time_offset)
                if segs:
                    all_segments.extend(segs)
            finally:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
                del samples
                clear_gpu_memory()

            start_frame += step_frames
            chunk_idx += 1

    all_segments.sort(key=lambda s: (s["start"], s["end"]))
    return all_segments

def detect_long_segment(
    audio_path: str,
    transcription_segments: List[Dict[str, Any]],
    max_segment_duration: float = 45.0,
    exclusive_request: bool = False,
) -> List[Dict[str, Any]]:
    if not transcription_segments:
        return transcription_segments

    try:
        from pydub import AudioSegment
    except ImportError:
        logger.warning("pydub not installed. Skipping long segment refinement.")
        return transcription_segments

    modified_segments: List[Dict[str, Any]] = []
    full_audio = AudioSegment.from_file(audio_path)

    for segment in transcription_segments:
        seg_start = float(segment["start"])
        seg_end = float(segment["end"])
        seg_dur = seg_end - seg_start

        if seg_dur <= max_segment_duration:
            modified_segments.append(segment)
            continue

        try:
            start_ms = int(seg_start * 1000)
            end_ms = int(seg_end * 1000)
            long_audio = full_audio[start_ms:end_ms]

            forced_chunk_sec = 15
            forced_chunk_ms = forced_chunk_sec * 1000

            sub_segments_all: List[Dict[str, Any]] = []

            for offset_ms in range(0, len(long_audio), forced_chunk_ms):
                chunk_audio = long_audio[offset_ms : offset_ms + forced_chunk_ms]
                temp_filename = f"_tmp_{seg_start:.2f}_{offset_ms}.wav"
                chunk_audio.export(temp_filename, format="wav")

                try:
                    if exclusive_request:
                        with _GPU_REQUEST_LOCK:
                            result = _safe_transcribe([temp_filename], timestamps=True)
                    else:
                        result = _safe_transcribe([temp_filename], timestamps=True)

                    chunk_offset_sec = seg_start + (offset_ms / 1000.0)
                    sub = _extract_segments(result, time_offset=chunk_offset_sec)
                    if sub:
                        sub_segments_all.extend(sub)
                finally:
                    try:
                        Path(temp_filename).unlink(missing_ok=True)
                    except Exception:
                        pass

            if sub_segments_all:
                modified_segments.extend(sub_segments_all)
            else:
                modified_segments.append(segment)

        except Exception as e:
            logger.error("Error refining long segment: %s", e)
            modified_segments.append(segment)

        finally:
            clear_gpu_memory()

    modified_segments.sort(key=lambda x: x["start"])
    return modified_segments

def transcribe_audio(
    audio_file: str,
    target_sr: int = 16000,
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 1.0,
    max_segment_duration: float = 45.0,
    cpu_resample_backend: str = "torchaudio",
    exclusive_request: bool = False,
) -> List[Dict[str, Any]]:
    if not audio_file or not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file was not found: {audio_file}")

    logger.info("Starting local Parakeet transcription: file=%s", audio_file)

    def _run(exclusive: bool):
        with GPUMemoryGuard(sync=True):
            segs = chunked_transcription_streaming(
                audio_file,
                target_sr=target_sr,
                chunk_seconds=chunk_seconds,
                overlap_seconds=overlap_seconds,
                cpu_resample_backend=cpu_resample_backend,
                exclusive_request=exclusive,
            )
            segs = detect_long_segment(
                audio_file,
                segs,
                max_segment_duration=max_segment_duration,
                exclusive_request=exclusive,
            )
        return segs

    try:
        return _run(exclusive_request)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("stream is capturing" in msg or "capture must end" in msg) and not exclusive_request:
            logger.warning("Retrying transcription with exclusive_request=True due to CUDA capture error...")
            return _run(True)
        raise

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    if millis == 1000:
        millis = 0
        secs += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_srt(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments, 1):
            start = format_timestamp(seg['start'])
            end = format_timestamp(seg['end'])
            text = seg['segment'].strip()
            f.write(f"{idx}\n{start} --> {end}\n{text}\n\n")
