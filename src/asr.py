# src/asr.py
from __future__ import annotations
import os
import pathlib
import torch
from faster_whisper import WhisperModel as FWModel
import whisperx
from progress import get_progress_tracker, ProgressCallback

# CT2 (faster-whisper) repos to search locally under ./models/asr/ct2/...
_CT2_REPO_CHOICES: dict[str, list[str]] = {
    "medium.en": [
        "Systran/faster-whisper-medium.en",
        "guillaumekln/faster-whisper-medium.en",
    ],
    "large-v3-turbo": [
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "h2oai/faster-whisper-large-v3-turbo",
    ],
}

def _asr_device() -> str:
    """
    Device for CTranslate2 (faster-whisper). CTranslate2 does NOT support 'mps',
    so we force CPU on Apple Silicon.
    """
    return "cpu"

def _align_device() -> str:
    """Device for PyTorch-based aligner (WhisperX). Prefer MPS on Apple Silicon."""
    return "mps" if torch.backends.mps.is_available() else "cpu"

def _latest_snapshot_dir_any(cache_root: pathlib.Path, repo_ids: list[str]) -> pathlib.Path:
    """
    Given cache_root=./models/asr/ct2 and a list of repo_ids, return the newest
    snapshot directory that exists locally:
      ./models/asr/ct2/models--ORG--REPO/snapshots/<rev>/
    """
    for repo_id in repo_ids:
        safe = f"models--{repo_id.replace('/', '--')}"
        base = cache_root / safe / "snapshots"
        if not base.exists():
            continue
        snaps = [p for p in base.iterdir() if p.is_dir()]
        if snaps:
            snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return snaps[0]
    raise FileNotFoundError(
        f"No CT2 snapshot found under {cache_root} for any of: {repo_ids}. "
        f"Run scripts/download_models.py online once to cache CT2 models."
    )

def transcribe_with_alignment(
    audio_path: str,
    asr_model: str = "medium.en",
    role: str | None = None,
):
    """
    Run ASR using faster-whisper (CT2) on CPU with language='en' and VAD disabled,
    then run WhisperX English alignment on MPS (if available) or CPU.

    Returns a flat list of word dicts:
      [{ 'text': str, 'start': float, 'end': float, 'speaker': role|None }, ...]
    """
    if asr_model not in _CT2_REPO_CHOICES:
        raise ValueError(f"Unknown asr_model: {asr_model}")

    # Initialize progress tracking
    tracker = get_progress_tracker()
    
    # Devices
    asr_device = _asr_device()          # 'cpu' (CTranslate2)
    align_device = _align_device()      # 'mps' if available, else 'cpu'

    # Compute type for CT2 on CPU
    compute_type = "int8"               # fast + memory efficient on CPU

    # Resolve local CT2 model snapshot directory (no network)
    models_root = pathlib.Path(os.getenv("HF_HOME", "./models")).resolve()
    ct2_cache = models_root / "asr" / "ct2"
    local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[asr_model])

    # ---- ASR via faster-whisper directly (bypass whisperx.load_model) ----
    # Load CT2 model from local path
    role_label = f" ({role})" if role else ""
    asr_task = tracker.add_task(f"ASR Transcription{role_label}", stage="asr_transcription")
    
    fw = FWModel(
        str(local_model_dir),     # model_size_or_path (positional)
        device=asr_device,        # 'cpu' (CT2 has no MPS)
        compute_type=compute_type # 'int8' CPU
    )

    # Transcribe: force English, disable VAD filter
    # (We already standardize audio to 16k mono WAV upstream.)
    tracker.update(asr_task, description=f"ASR Transcription{role_label} - Loading audio")
    
    # Get audio duration for progress estimation
    import librosa
    try:
        audio_duration = librosa.get_duration(path=audio_path)
        # Estimate segments based on typical 30-second chunks
        estimated_segments = max(1, int(audio_duration / 30))
    except Exception:
        estimated_segments = 10  # Fallback estimate
    
    segments, info = fw.transcribe(
        audio_path,
        language="en",
        vad_filter=False,               # no VAD; avoid extra deps
        word_timestamps=False,          # WhisperX does alignment—no need here
        beam_size=5,
    )

    tracker.update(asr_task, description=f"ASR Transcription{role_label} - Processing segments")
    
    # Convert generator to list of dicts that WhisperX align() expects
    seg_list = []
    segment_count = 0
    for s in segments:
        # s has .start, .end, .text
        seg_list.append({
            "start": float(s.start) if s.start is not None else 0.0,
            "end": float(s.end) if s.end is not None else 0.0,
            "text": (s.text or "").strip(),
        })
        segment_count += 1
        # Update progress based on segment count
        if estimated_segments > 0:
            progress = (segment_count / estimated_segments) * 80  # 80% of ASR task
            tracker.update(asr_task, description=f"ASR Transcription{role_label} - {segment_count}/{estimated_segments} segments")
    
    tracker.complete_task(asr_task, stage="asr_transcription")

    # ---- WhisperX alignment (English) ----
    align_task = tracker.add_task(f"Alignment{role_label}", stage="alignment")
    tracker.update(align_task, description=f"Alignment{role_label} - Loading model")
    
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device=align_device,             # 'mps' if available, else 'cpu'
        model_dir=str(models_root),
    )

    tracker.update(align_task, description=f"Alignment{role_label} - Processing segments")
    
    aligned = whisperx.align(
        seg_list,
        align_model,
        metadata,
        audio_path,
        device=align_device,
        return_char_alignments=False,
    )

    tracker.complete_task(align_task, stage="alignment")

    # Process word segments
    words_task = tracker.add_task(f"Processing words{role_label}", stage="word_processing")
    words = []
    word_count = 0
    total_words = len(aligned.get("word_segments", []))
    
    for seg in aligned.get("word_segments", []):
        w = seg.get("word")
        if not w:
            continue
        words.append(
            {
                "text": w,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": role if role else None,
            }
        )
        word_count += 1
        # Update progress
        if total_words > 0:
            progress = (word_count / total_words) * 100
            tracker.update(words_task, description=f"Processing words{role_label} - {word_count}/{total_words}")
    
    tracker.complete_task(words_task, stage="word_processing")

    # Optional MPS memory tidy (no-op on CPU)
    try:
        if align_device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass

    return words
