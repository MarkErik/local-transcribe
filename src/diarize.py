# src/diarize.py
from __future__ import annotations
from typing import List, Dict
import os
import warnings

# Aggressively suppress all warnings before importing pyannote
warnings.filterwarnings("ignore")

import torch
import soundfile as sf

# Optional: use torchaudio for resampling if the file isn't 16 kHz (should already be, but this is a safety net)
try:
    import torchaudio
    _HAVE_TORCHAUDIO = True
except Exception:
    _HAVE_TORCHAUDIO = False

from pyannote.audio import Pipeline
from turns import build_turns
from merge import merge_turn_streams
from progress import get_progress_tracker
from logging_config import get_logger, DiarizationError, ErrorContext, error_context

@error_context(reraise=True)
def _load_waveform_mono_32f(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio as float32 and return (waveform [1, T], sample_rate).
    This bypasses torchcodec by using soundfile, avoiding pyannote's built-in decoding.
    """
    logger = get_logger()
    
    try:
        logger.debug(f"Loading audio waveform from {audio_path}")
        
        # soundfile returns numpy array
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        
        if data.size == 0:
            raise DiarizationError(f"Audio file is empty: {audio_path}")
        
        # Ensure mono
        if getattr(data, "ndim", 1) > 1:
            logger.debug(f"Converting {data.shape[1]}-channel audio to mono")
            # Collapse multi-channel to mono
            data = data.mean(axis=1).astype("float32", copy=False)

        # Convert to torch and add channel dim -> [1, T]
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, time)
        
        logger.debug(f"Loaded waveform: shape={waveform.shape}, sample_rate={sr}")
        return waveform, int(sr)
        
    except Exception as e:
        if isinstance(e, DiarizationError):
            raise
        else:
            raise DiarizationError(f"Failed to load audio waveform: {e}", cause=e)


@error_context(reraise=True)
def _maybe_resample(waveform: torch.Tensor, sr: int, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """
    Resample to target_sr if needed (no-op if sr == target_sr).
    Uses torchaudio if available; otherwise returns original.
    """
    logger = get_logger()
    
    try:
        if sr == target_sr:
            logger.debug(f"Audio already at target sample rate {sr}")
            return waveform, sr
            
        if _HAVE_TORCHAUDIO:
            logger.debug(f"Resampling audio from {sr} to {target_sr}")
            # waveform is [1, T] (channel-first), which torchaudio expects
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wf = resampler(waveform)
            logger.debug(f"Resampled waveform: shape={wf.shape}")
            return wf.contiguous(), target_sr
        else:
            logger.warning(f"Torchaudio not available, leaving audio at {sr}")
            # Fallback: leave as-is; pyannote can handle non-16k, but timings are cleanest at 16k.
            return waveform, sr
            
    except Exception as e:
        raise DiarizationError(f"Failed to resample audio: {e}", cause=e)


@error_context(reraise=True)
def diarize_mixed(audio_path: str, words: List[Dict]) -> List[Dict]:
    """
    Diarize a mixed/combined track and assign speakers to words by majority overlap,
    then build readable turns per speaker.

    Parameters
    ----------
    audio_path : str
        Path to the (already standardized) audio file. We load it ourselves to bypass torchcodec.
    words : List[Dict]
        Word-level list from ASR alignment:
          [{ 'text': str, 'start': float, 'end': float, 'speaker': None|str }, ...]

    Returns
    -------
    List[Dict]
        Merged list of turns across speakers, sorted by time:
          [{ 'speaker': str, 'start': float, 'end': float, 'text': str }, ...]
    """
    logger = get_logger()
    
    try:
        # Validate inputs
        if not os.path.exists(audio_path):
            raise DiarizationError(f"Audio file not found: {audio_path}")
        
        if not words:
            raise DiarizationError("No words provided for diarization")
        
        logger.info(f"Starting speaker diarization for {audio_path} with {len(words)} words")
        
        # Initialize progress tracking
        tracker = get_progress_tracker()
        
        # Ensure pyannote/huggingface hub will read token from env if ever needed (usually not for offline)
        token = os.getenv("HUGGINGFACE_TOKEN", "")
        if token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)
            os.environ.setdefault("HF_TOKEN", token)

        cache_dir = os.getenv("PYANNOTE_CACHE", "./models/diarization")

        # Load pipeline (no built-in decoding; we'll pass waveform directly)
        diarize_task = tracker.add_task("Speaker Diarization - Loading pipeline", total=100, stage="diarization")
        
        try:
            logger.debug("Loading pyannote diarization pipeline")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise DiarizationError(f"Failed to load diarization pipeline: {e}", cause=e)

        tracker.update(diarize_task, advance=20, description="Speaker Diarization - Loading audio")
        
        # --- Load audio into memory to bypass torchcodec ---
        try:
            waveform, sr = _load_waveform_mono_32f(audio_path)
            waveform, sr = _maybe_resample(waveform, sr, target_sr=16000)  # safety: standardize to 16k if we can
        except Exception as e:
            raise DiarizationError(f"Failed to load and process audio: {e}", cause=e)

        tracker.update(diarize_task, advance=10, description="Speaker Diarization - Processing audio")
        
        # Run diarization on in-memory audio dict (channel-first waveform)
        try:
            logger.debug("Running speaker diarization")
            diar = pipeline({"waveform": waveform, "sample_rate": sr})
        except Exception as e:
            raise DiarizationError(f"Diarization processing failed: {e}", cause=e)

        tracker.update(diarize_task, advance=20, description="Speaker Diarization - Processing segments")
        
        # Convert annotation to a simple list of segments for overlap computation
        diar_segments = []
        segment_count = 0
        
        try:
            # The diarization result is an Annotation object that can be iterated directly
            for segment, _, label in diar.itertracks(yield_label=True):
                diar_segments.append({"start": float(segment.start), "end": float(segment.end), "label": label})
                segment_count += 1
                tracker.update(diarize_task, advance=1, description=f"Speaker Diarization - Found {segment_count} segments")
        except Exception as e:
            raise DiarizationError(f"Failed to process diarization segments: {e}", cause=e)
        
        logger.info(f"Found {segment_count} diarization segments")
        tracker.update(diarize_task, advance=50, description="Speaker Diarization - Complete")
        tracker.complete_task(diarize_task, stage="diarization")

        if not diar_segments:
            logger.warning("No diarization segments found, using default speaker")
            # Create a single segment covering the entire audio
            if words:
                max_time = max(w.get("end", 0) for w in words)
                diar_segments = [{"start": 0.0, "end": max_time, "label": "Speaker_A"}]
            else:
                return []

        # Helper: compute overlap between [a1,a2] and [b1,b2]
        def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
            return max(0.0, min(a2, b2) - max(a1, b1))

        # Assign each word to the diarization label with maximum temporal overlap
        assign_task = tracker.add_task("Assigning speakers to words", total=len(words), stage="speaker_assignment")
        tagged_words: List[Dict] = []
        word_count = 0
        total_words = len(words)
        
        logger.info(f"Assigning speakers to {total_words} words")
        
        try:
            for w in words:
                w_start = float(w["start"])
                w_end = float(w["end"])
                best_label = None
                best_ov = 0.0
                
                for ds in diar_segments:
                    ov = _overlap(w_start, w_end, ds["start"], ds["end"])
                    if ov > best_ov:
                        best_ov = ov
                        best_label = ds["label"]
                
                new_w = dict(w)
                new_w["speaker"] = best_label or "Speaker_A"  # default fallback if no overlap found
                tagged_words.append(new_w)
                
                word_count += 1
                if total_words > 0:
                    tracker.update(assign_task, advance=1, description=f"Assigning speakers - {word_count}/{total_words} words")
        except Exception as e:
            raise DiarizationError(f"Failed to assign speakers to words: {e}", cause=e)
        
        logger.info(f"Assigned speakers to {word_count} words")
        tracker.complete_task(assign_task, stage="speaker_assignment")

        # Group into turns per speaker, then merge all speakers by time
        turns_task = tracker.add_task("Building conversation turns", total=len(speakers), stage="turn_building")
        
        speakers: dict[str, List[Dict]] = {}
        for w in tagged_words:
            speakers.setdefault(w["speaker"], []).append(w)

        all_turns: List[Dict] = []
        speaker_count = 0
        total_speakers = len(speakers)
        
        logger.info(f"Building turns for {total_speakers} speakers")
        
        try:
            for spk, spk_words in speakers.items():
                all_turns.extend(build_turns(spk_words, speaker_label=spk))
                speaker_count += 1
                tracker.update(turns_task, advance=1, description=f"Building turns - {speaker_count}/{total_speakers} speakers")

            merged_turns = merge_turn_streams(all_turns, [])
        except Exception as e:
            raise DiarizationError(f"Failed to build conversation turns: {e}", cause=e)
        
        logger.info(f"Built {len(merged_turns)} conversation turns")
        tracker.complete_task(turns_task, stage="turn_building")
        
        logger.info(f"Speaker diarization completed successfully for {audio_path}")
        return merged_turns
        
    except Exception as e:
        if isinstance(e, DiarizationError):
            logger.error(f"Diarization error: {e}")
            raise
        else:
            logger.error(f"Unexpected error in diarization: {e}")
            raise DiarizationError(f"Unexpected error: {e}", cause=e)
