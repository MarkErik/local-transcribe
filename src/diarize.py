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
from src.progress import get_progress_tracker
from config import is_debug_enabled, is_info_enabled
from logging_config import get_logger, DiarizationError, ErrorContext, error_context
from cross_talk import detect_basic_cross_talk, assign_words_with_basic_cross_talk, BASIC_CROSS_TALK_CONFIG

DEFAULT_SPEAKER = "UNKNOWN"

@error_context(reraise=True)
def _load_waveform_mono_32f(audio_path: str) -> tuple[torch.Tensor, int]:
    """
    Load audio as float32 and return (waveform [1, T], sample_rate).
    This bypasses torchcodec by using soundfile, avoiding pyannote's built-in decoding.
    """
    logger = get_logger()
    
    try:
        if is_debug_enabled():
            logger.debug(f"Loading audio waveform from {audio_path}")
        
        # soundfile returns numpy array
        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        
        if data.size == 0:
            raise DiarizationError(f"Audio file is empty: {audio_path}")
        
        # Ensure mono
        if getattr(data, "ndim", 1) > 1:
            if is_debug_enabled():
                logger.debug(f"Converting {data.shape[1]}-channel audio to mono")
            # Collapse multi-channel to mono
            data = data.mean(axis=1).astype("float32", copy=False)

        # Convert to torch and add channel dim -> [1, T]
        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, time)
        
        if is_debug_enabled():
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
            if is_debug_enabled():
                logger.debug(f"Audio already at target sample rate {sr}")
            return waveform, sr
            
        if _HAVE_TORCHAUDIO:
            if is_debug_enabled():
                logger.debug(f"Resampling audio from {sr} to {target_sr}")
            # waveform is [1, T] (channel-first), which torchaudio expects
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            wf = resampler(waveform)
            if is_debug_enabled():
                logger.debug(f"Resampled waveform: shape={wf.shape}")
            return wf.contiguous(), target_sr
        else:
            if is_info_enabled():
                logger.warning(f"Torchaudio not available, leaving audio at {sr}")
            # Fallback: leave as-is; pyannote can handle non-16k, but timings are cleanest at 16k.
            return waveform, sr
            
    except Exception as e:
        raise DiarizationError(f"Failed to resample audio: {e}", cause=e)


@error_context(reraise=True)
def diarize_mixed(
    audio_path: str,
    words: List[Dict],
    pipeline=None,
    device: str = "cpu",
    num_speakers: int = 2,
    detect_cross_talk: bool = False,
    cross_talk_config: dict = None
) -> List[Dict]:
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
    pipeline : Pipeline, optional
        Pre-loaded diarization pipeline. If None, a new pipeline will be loaded.
    device : str, optional
        Device to use for processing, by default "cpu"
    num_speakers : int, optional
        Number of speakers to detect, by default 2
    detect_cross_talk : bool, optional
        Whether to enable basic cross-talk detection, by default False
    cross_talk_config : dict, optional
        Configuration for cross-talk detection. If None, uses BASIC_CROSS_TALK_CONFIG

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
        
        if is_info_enabled():
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
        # Use meaningful stages instead of arbitrary percentage
        diarize_task = tracker.add_task("Speaker Diarization", total=4)  # 4 main stages: load, audio, process, complete
        
        try:
            if is_debug_enabled():
                logger.debug("Loading pyannote diarization pipeline")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                cache_dir=cache_dir,
            )
        except Exception as e:
            raise DiarizationError(f"Failed to load diarization pipeline: {e}", cause=e)

        try:
            tracker.update(diarize_task, advance=1, description="Speaker Diarization - Loading audio")
        except Exception as e:
            if is_info_enabled():
                logger.warning(f"Failed to update diarization progress: {e}")
        
        # --- Load audio into memory to bypass torchcodec ---
        try:
            waveform, sr = _load_waveform_mono_32f(audio_path)
            waveform, sr = _maybe_resample(waveform, sr, target_sr=16000)  # safety: standardize to 16k if we can
        except Exception as e:
            raise DiarizationError(f"Failed to load and process audio: {e}", cause=e)

        try:
            tracker.update(diarize_task, advance=1, description="Speaker Diarization - Processing audio")
        except Exception as e:
            if is_info_enabled():
                logger.warning(f"Failed to update diarization audio processing progress: {e}")
        
        # Run diarization on in-memory audio dict (channel-first waveform)
        try:
            if is_debug_enabled():
                logger.debug("Running speaker diarization")
            diar = pipeline({"waveform": waveform, "sample_rate": sr})
        except Exception as e:
            raise DiarizationError(f"Diarization processing failed: {e}", cause=e)

        tracker.update(diarize_task, advance=20, description="Speaker Diarization - Processing segments")
        
        # --- Debugging: Inspect the DiarizeOutput object ---
        if is_debug_enabled():
            logger.debug(f"Type of diar object: {type(diar)}")
            logger.debug(f"Attributes of diar object: {dir(diar)}")
        
        # Inspect the 'speaker_diarization' attribute
        if hasattr(diar, 'speaker_diarization'):
            annotation_obj = diar.speaker_diarization
            if is_debug_enabled():
                logger.debug(f"Type of speaker_diarization: {type(annotation_obj)}")
                logger.debug(f"Attributes of speaker_diarization: {dir(annotation_obj)}")
            # Check for known methods on the annotation object
            if hasattr(annotation_obj, 'itertracks'):
                if is_debug_enabled():
                    logger.debug("speaker_diarization has itertracks method")
            if hasattr(annotation_obj, 'tracks'):
                if is_debug_enabled():
                    logger.debug("speaker_diarization has tracks attribute")
            if hasattr(annotation_obj, 'get_timeline'):
                if is_debug_enabled():
                    logger.debug("speaker_diarization has get_timeline method")
            if hasattr(annotation_obj, 'support'):
                if is_debug_enabled():
                    logger.debug(f"speaker_diarization support: {annotation_obj.support}")
        # --- End Debugging ---

        # Convert annotation to a simple list of segments for overlap computation
        diar_segments = []
        segment_count = 0
        
        try:
            # The diarization result is a DiarizeOutput object, which contains an Annotation object
            # We need to access the 'speaker_diarization' attribute to get the Annotation
            annotation_obj = diar.speaker_diarization
            
            # Try iterating with itertracks (common for Annotation objects)
            try:
                for segment, track, label in annotation_obj.itertracks(yield_label=True):
                    diar_segments.append({"start": float(segment.start), "end": float(segment.end), "label": label})
                    segment_count += 1
                    try:
                        try:
                            tracker.update(diarize_task, advance=1, description=f"Speaker Diarization - Found {segment_count} segments")
                        except Exception as e:
                            if is_info_enabled():
                                logger.warning(f"Failed to update diarization segment progress: {e}")
                    except Exception as e:
                        if is_info_enabled():
                            logger.warning(f"Failed to update diarization segment progress: {e}")
            except AttributeError:
                # If itertracks is not available, try direct iteration if the object supports it
                try:
                    for segment in annotation_obj.support.itersegments():
                        # Assuming labels can be retrieved, e.g., via annotation_obj[segment]
                        label = annotation_obj[segment]
                        diar_segments.append({"start": float(segment.start), "end": float(segment.end), "label": label})
                        segment_count += 1
                        tracker.update(diarize_task, advance=1, description=f"Speaker Diarization - Found {segment_count} segments")
                except Exception as e_inner:
                    raise DiarizationError(f"Failed to process diarization segments using annotation support: {e_inner}", cause=e_inner)
        except Exception as e:
            raise DiarizationError(f"Failed to process diarization segments: {e}", cause=e)
        
        if is_info_enabled():
            logger.info(f"Found {segment_count} diarization segments")
        try:
            tracker.update(diarize_task, advance=1, description="Speaker Diarization - Complete")
            tracker.complete_task(diarize_task)
        except Exception as e:
            if is_info_enabled():
                logger.warning(f"Failed to complete diarization task: {e}")

        if not diar_segments:
            if is_info_enabled():
                logger.warning("No diarization segments found, using default speaker")
            # Create a single segment covering the entire audio
            if words:
                max_time = max(w.get("end", 0) for w in words)
                diar_segments = [{"start": 0.0, "end": max_time, "label": DEFAULT_SPEAKER}]
            else:
                return []

        # Helper: compute overlap between [a1,a2] and [b1,b2]
        def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
            return max(0.0, min(a2, b2) - max(a1, b2))

        # Helper: compute distance to nearest speaker boundary
        def _distance_to_boundary(word_start: float, word_end: float, segments: List[Dict]) -> float:
            """Find the minimum distance from word to any speaker segment boundary."""
            min_distance = float('inf')
            
            for seg in segments:
                # Distance to segment start
                dist_to_start = abs(word_end - seg["start"])
                # Distance to segment end
                dist_to_end = abs(word_start - seg["end"])
                
                min_distance = min(min_distance, dist_to_start, dist_to_end)
            
            return min_distance

        # Helper: apply boundary-aware speaker assignment with buffer zones
        def _assign_speaker_with_boundary_awareness(
            word_start: float,
            word_end: float,
            segments: List[Dict],
            buffer_zone: float = 0.3,
            min_overlap_threshold: float = 0.01
        ) -> tuple[str, float, bool]:
            """
            Assign speaker with boundary awareness and buffer zones.
            
            Returns:
                tuple: (best_speaker, confidence, is_boundary_word)
            """
            # Calculate distance to nearest boundary
            boundary_distance = _distance_to_boundary(word_start, word_end, segments)
            is_boundary_word = boundary_distance < buffer_zone
            
            # Calculate overlaps for all segments - use original logic
            best_speaker = None
            best_overlap = 0.0
            speaker_overlaps = {}
            total_overlap = 0.0
            
            for ds in segments:
                ov = _overlap(word_start, word_end, ds["start"], ds["end"])
                if ov > 0:
                    speaker_overlaps[ds["label"]] = speaker_overlaps.get(ds["label"], 0.0) + ov
                    total_overlap += ov
                    
                # Use original logic: find best speaker by maximum overlap
                if ov > best_overlap:
                    best_overlap = ov
                    best_speaker = ds["label"]
            
            if not best_speaker:
                return DEFAULT_SPEAKER, 0.0, is_boundary_word
            
            # Calculate confidence as absolute overlap (not ratio) for compatibility
            confidence = best_overlap
            
            # For boundary words with very low overlap, apply center-based smoothing
            if is_boundary_word and best_overlap < min_overlap_threshold:
                # Look for neighboring segments that might provide better context
                word_center = (word_start + word_end) / 2
                
                # Find segments that contain the word center
                center_containing_segments = [
                    seg for seg in segments
                    if seg["start"] <= word_center <= seg["end"]
                ]
                
                if center_containing_segments:
                    # Prefer segment that contains word center
                    best_speaker = center_containing_segments[0]["label"]
                    confidence = max(confidence, min_overlap_threshold)  # Ensure minimum confidence
            
            return best_speaker, confidence, is_boundary_word

        # Assign each word to the diarization label with boundary-aware logic
        assign_task = tracker.add_task("Assigning speakers to words", total=len(words))
        tagged_words: List[Dict] = []
        word_count = 0
        total_words = len(words)
        
        if is_info_enabled():
            logger.info(f"Assigning speakers to {total_words} words with boundary-aware logic")
        
        # Configuration for boundary handling
        buffer_zone = 0.3  # 300ms buffer zone around boundaries
        confidence_threshold = 0.05  # Minimum confidence for assignment
        
        boundary_word_count = 0
        low_confidence_count = 0
        
        try:
            for w in words:
                w_start = float(w["start"])
                w_end = float(w["end"])
                
                # Use boundary-aware assignment
                best_speaker, confidence, is_boundary_word = _assign_speaker_with_boundary_awareness(
                    w_start, w_end, diar_segments, buffer_zone, confidence_threshold
                )
                
                if is_boundary_word:
                    boundary_word_count += 1
                
                if confidence < confidence_threshold:
                    low_confidence_count += 1
                    if is_debug_enabled():
                        logger.debug(f"LOW CONFIDENCE BOUNDARY WORD: '{w['text']}' at {w_start:.3f}-{w_end:.3f} "
                                   f"assigned to {best_speaker} (confidence: {confidence:.3f}, boundary_distance: {_distance_to_boundary(w_start, w_end, diar_segments):.3f})")
                
                new_w = dict(w)
                new_w["speaker"] = best_speaker
                new_w["confidence"] = confidence
                new_w["is_boundary_word"] = is_boundary_word
                tagged_words.append(new_w)
                
                word_count += 1
                if total_words > 0:
                    try:
                        tracker.update(assign_task, advance=1, description=f"Assigning speakers - {word_count}/{total_words} words")
                    except Exception as e:
                        if is_info_enabled():
                            logger.warning(f"Failed to update speaker assignment progress: {e}")
        except Exception as e:
            raise DiarizationError(f"Failed to assign speakers to words: {e}", cause=e)
        
        if is_info_enabled():
            logger.info(f"Speaker assignment complete: {boundary_word_count} boundary words, "
                       f"{low_confidence_count} low confidence assignments")
            logger.info(f"Assigned speakers to {word_count} words")
        try:
            tracker.complete_task(assign_task)
        except Exception as e:
            if is_info_enabled():
                logger.warning(f"Failed to complete speaker assignment task: {e}")
        
        # Apply cross-talk detection if enabled
        if detect_cross_talk:
            if is_info_enabled():
                logger.info("Cross-talk detection enabled, processing cross-talk segments")
            
            # Use provided config or default to BASIC_CROSS_TALK_CONFIG
            config = cross_talk_config if cross_talk_config is not None else BASIC_CROSS_TALK_CONFIG
            
            try:
                # Detect cross-talk segments
                cross_talk_task = tracker.add_task("Detecting cross-talk segments", total=2)  # 2 stages: detect, enhance
                if is_debug_enabled():
                    logger.debug("Detecting basic cross-talk segments")
                
                # Validate configuration
                if not isinstance(config, dict):
                    if is_info_enabled():
                        logger.warning(f"Invalid cross-talk config type: {type(config)}, using default")
                    config = BASIC_CROSS_TALK_CONFIG
                
                overlap_threshold = config.get("overlap_threshold", BASIC_CROSS_TALK_CONFIG["overlap_threshold"])
                if not isinstance(overlap_threshold, (int, float)) or overlap_threshold < 0:
                    if is_info_enabled():
                        logger.warning(f"Invalid overlap_threshold: {overlap_threshold}, using default")
                    overlap_threshold = BASIC_CROSS_TALK_CONFIG["overlap_threshold"]
                
                if is_debug_enabled():
                    logger.debug(f"Using overlap threshold: {overlap_threshold}")
                
                # Detect cross-talk segments
                cross_talk_segments = detect_basic_cross_talk(diar_segments, overlap_threshold=overlap_threshold)
                
                if not cross_talk_segments:
                    if is_info_enabled():
                        logger.info("No cross-talk segments detected")
                else:
                    if is_info_enabled():
                        logger.info(f"Detected {len(cross_talk_segments)} cross-talk segments")
                    for i, ct in enumerate(cross_talk_segments[:5]):  # Log first 5 segments for debugging
                        if is_debug_enabled():
                            logger.debug(f"Cross-talk segment {i+1}: {ct['start']:.2f}-{ct['end']:.2f}, speakers: {ct['speakers']}")
                    if len(cross_talk_segments) > 5:
                        if is_debug_enabled():
                            logger.debug(f"... and {len(cross_talk_segments) - 5} more segments")
                
                tracker.update(cross_talk_task, advance=1, description="Cross-talk detection - Segments identified")
                
                # Enhance word assignments with cross-talk information
                if is_debug_enabled():
                    logger.debug("Enhancing word assignments with cross-talk information")
                enhanced_words = assign_words_with_basic_cross_talk(tagged_words, diar_segments, cross_talk_segments)
                
                # Count cross-talk words for logging
                cross_talk_words = sum(1 for w in enhanced_words if w.get("cross_talk", False))
                if is_info_enabled():
                    logger.info(f"Marked {cross_talk_words} words as cross-talk")
                
                # Log confidence statistics if available
                confidences = [w.get("confidence", 1.0) for w in enhanced_words if w.get("cross_talk", False)]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    min_confidence = min(confidences)
                    if is_debug_enabled():
                        logger.debug(f"Cross-talk word confidence - avg: {avg_confidence:.3f}, min: {min_confidence:.3f}")
                
                # Replace tagged_words with enhanced version
                tagged_words = enhanced_words
                
                try:
                    tracker.update(cross_talk_task, advance=1, description="Cross-talk detection - Words enhanced")
                    tracker.complete_task(cross_talk_task)
                except Exception as e:
                    if is_info_enabled():
                        logger.warning(f"Failed to complete cross-talk detection task: {e}")
                
            except Exception as e:
                error_msg = f"Cross-talk detection failed: {e}"
                if is_info_enabled():
                    logger.error(error_msg)
                if is_debug_enabled():
                    logger.debug(f"Error details: {type(e).__name__}: {str(e)}", exc_info=True)
                # Continue with original tagged_words if cross-talk detection fails
                if is_info_enabled():
                    logger.info("Continuing with standard speaker assignment without cross-talk detection")

        # Group into turns per speaker, then merge all speakers by time
        speakers: dict[str, List[Dict]] = {}
        for w in tagged_words:
            speakers.setdefault(w["speaker"], []).append(w)
        
        turns_task = tracker.add_task("Building conversation turns", total=len(speakers))

        all_turns: List[Dict] = []
        speaker_count = 0
        total_speakers = len(speakers)
        
        if is_info_enabled():
            logger.info(f"Building turns for {total_speakers} speakers")
        
        try:
            for spk, spk_words in speakers.items():
                all_turns.extend(build_turns(spk_words, speaker_label=spk))
                speaker_count += 1
                try:
                    tracker.update(turns_task, advance=1, description=f"Building turns - {speaker_count}/{total_speakers} speakers")
                except Exception as e:
                    if is_info_enabled():
                        logger.warning(f"Failed to update turns building progress: {e}")

            merged_turns = merge_turn_streams(all_turns, [])
        except Exception as e:
            raise DiarizationError(f"Failed to build conversation turns: {e}", cause=e)
        
        if is_info_enabled():
            logger.info(f"Built {len(merged_turns)} conversation turns")
        try:
            tracker.complete_task(turns_task)
        except Exception as e:
            if is_info_enabled():
                logger.warning(f"Failed to complete turns building task: {e}")
        
        if is_info_enabled():
            logger.info(f"Speaker diarization completed successfully for {audio_path}")
        return merged_turns
        
    except Exception as e:
        if isinstance(e, DiarizationError):
            if is_info_enabled():
                logger.error(f"Diarization error: {e}")
            raise
        else:
            if is_info_enabled():
                logger.error(f"Unexpected error in diarization: {e}")
            raise DiarizationError(f"Unexpected error: {e}", cause=e)
