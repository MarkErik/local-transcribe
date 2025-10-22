# src/asr.py
from __future__ import annotations
import os
import pathlib
import torch
from faster_whisper import WhisperModel as FWModel
import whisperx
from progress import get_progress_tracker, ProgressCallback
from logging_config import get_logger, ASRError, ErrorContext, error_context

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

@error_context(reraise=True)
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
    logger = get_logger()
    
    try:
        # Validate inputs
        if asr_model not in _CT2_REPO_CHOICES:
            raise ASRError(f"Unknown ASR model: {asr_model}", model=asr_model)
        
        if not os.path.exists(audio_path):
            raise ASRError(f"Audio file not found: {audio_path}", model=asr_model)
        
        logger.info(f"Starting ASR transcription for {audio_path} using model {asr_model}")
        
        # Initialize progress tracking
        tracker = get_progress_tracker()
        
        # Devices
        asr_device = _asr_device()          # 'cpu' (CTranslate2)
        align_device = _align_device()      # 'mps' if available, else 'cpu'

        logger.info(f"Using devices: ASR={asr_device}, Alignment={align_device}")

        # Compute type for CT2 on CPU
        compute_type = "int8"               # fast + memory efficient on CPU

        # Resolve local CT2 model snapshot directory (no network)
        models_root = pathlib.Path(os.getenv("HF_HOME", "./models")).resolve()
        ct2_cache = models_root / "asr" / "ct2"
        
        try:
            local_model_dir = _latest_snapshot_dir_any(ct2_cache, _CT2_REPO_CHOICES[asr_model])
        except FileNotFoundError as e:
            raise ASRError(f"CT2 model not found: {e}", model=asr_model, cause=e)

        # ---- ASR via faster-whisper directly (bypass whisperx.load_model) ----
        # Load CT2 model from local path
        role_label = f" ({role})" if role else ""
        # Get audio duration for progress estimation
        import librosa
        try:
            audio_duration = librosa.get_duration(path=audio_path)
            # Estimate segments based on typical 30-second chunks
            estimated_segments = max(1, int(audio_duration / 30))
            logger.info(f"Audio duration: {audio_duration:.2f}s, estimated segments: {estimated_segments}")
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}, using fallback estimate")
            estimated_segments = 10  # Fallback estimate
        
        asr_task = tracker.add_task(f"ASR Transcription{role_label}", total=100, stage="asr_transcription")
        
        try:
            logger.debug(f"Loading CT2 model from {local_model_dir}")
            fw = FWModel(
                str(local_model_dir),     # model_size_or_path (positional)
                device=asr_device,        # 'cpu' (CT2 has no MPS)
                compute_type=compute_type # 'int8' CPU
            )
        except Exception as e:
            raise ASRError(f"Failed to load CT2 model: {e}", model=asr_model, cause=e)

        # Transcribe: force English, disable VAD filter
        # (We already standardize audio to 16k mono WAV upstream.)
        tracker.update(asr_task, advance=10, description=f"ASR Transcription{role_label} - Loading audio")
        
        try:
            logger.debug(f"Starting transcription with {asr_model}")
            segments, info = fw.transcribe(
                audio_path,
                language="en",
                vad_filter=False,               # no VAD; avoid extra deps
                word_timestamps=False,          # WhisperX does alignment—no need here
                beam_size=5,
            )
        except Exception as e:
            raise ASRError(f"Transcription failed: {e}", model=asr_model, cause=e)

        tracker.update(asr_task, advance=5, description=f"ASR Transcription{role_label} - Processing segments")
        
        # Convert generator to list of dicts that WhisperX align() expects
        seg_list = []
        segment_count = 0
        
        try:
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
                    progress_percent = (segment_count / estimated_segments) * 70  # 70% of ASR task
                    tracker.update(asr_task, advance=0, description=f"ASR Transcription{role_label} - {segment_count}/{estimated_segments} segments")
                    # Manually set the progress
                    tracker.progress.update(asr_task, completed=10 + 5 + progress_percent)
        except Exception as e:
            raise ASRError(f"Failed to process transcription segments: {e}", model=asr_model, cause=e)
        
        logger.info(f"Processed {segment_count} transcription segments")
        tracker.update(asr_task, advance=15, description=f"ASR Transcription{role_label} - Complete")
        tracker.complete_task(asr_task, stage="asr_transcription")

        # ---- WhisperX alignment (English) ----
        align_task = tracker.add_task(f"Alignment{role_label}", total=100, stage="alignment")
        tracker.update(align_task, advance=20, description=f"Alignment{role_label} - Loading model")
        
        try:
            logger.debug("Loading WhisperX alignment model")
            align_model, metadata = whisperx.load_align_model(
                language_code="en",
                device=align_device,             # 'mps' if available, else 'cpu'
                model_dir=str(models_root),
            )
        except Exception as e:
            raise ASRError(f"Failed to load alignment model: {e}", model=asr_model, cause=e)

        tracker.update(align_task, advance=10, description=f"Alignment{role_label} - Processing segments")
        
        try:
            logger.debug("Running WhisperX alignment")
            aligned = whisperx.align(
                seg_list,
                align_model,
                metadata,
                audio_path,
                device=align_device,
                return_char_alignments=False,
            )
        except Exception as e:
            raise ASRError(f"Alignment failed: {e}", model=asr_model, cause=e)

        tracker.update(align_task, advance=70, description=f"Alignment{role_label} - Complete")
        tracker.complete_task(align_task, stage="alignment")

        # Process word segments
        words_task = tracker.add_task(f"Processing words{role_label}", total=len(aligned.get("word_segments", [])), stage="word_processing")
        words = []
        word_count = 0
        total_words = len(aligned.get("word_segments", []))
        
        logger.info(f"Processing {total_words} word segments")
        
        try:
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
                    tracker.update(words_task, advance=1, description=f"Processing words{role_label} - {word_count}/{total_words}")
        except Exception as e:
            raise ASRError(f"Failed to process word segments: {e}", model=asr_model, cause=e)
        
        logger.info(f"Processed {word_count} words")
        tracker.complete_task(words_task, stage="word_processing")

        # Optional MPS memory tidy (no-op on CPU)
        try:
            if align_device == "mps":
                torch.mps.empty_cache()
                logger.debug("Cleared MPS memory cache")
        except Exception as e:
            logger.warning(f"Failed to clear MPS cache: {e}")

        logger.info(f"ASR transcription completed successfully for {audio_path}")
        return words
        
    except Exception as e:
        if isinstance(e, ASRError):
            logger.error(f"ASR error in {asr_model}: {e}")
            raise
        else:
            logger.error(f"Unexpected error in ASR transcription: {e}")
            raise ASRError(f"Unexpected error: {e}", model=asr_model, cause=e)
