# src/diarize.py
from __future__ import annotations
from typing import List, Dict, Optional
import os
import warnings
import json
from pathlib import Path

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


def _save_diarization_stage(
    stage_name: str,
    data: List[Dict],
    output_dir: Optional[Path] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save intermediate diarization stage outputs for debugging and analysis.
    
    Parameters
    ----------
    stage_name : str
        Name of the stage (e.g., "raw_segments", "refined_segments", "tagged_words")
    data : List[Dict]
        Data to save (segments, words, or turns)
    output_dir : Optional[Path]
        Directory to save outputs. If None, uses environment variable or skips.
    metadata : Optional[Dict]
        Additional metadata to include in the output
    """
    if output_dir is None:
        # Check if user wants to save stages
        save_stages = os.getenv("SAVE_DIARIZATION_STAGES", "").lower() in ("1", "true", "yes")
        if not save_stages:
            return
        
        output_dir_str = os.getenv("DIARIZATION_STAGES_DIR", "./diarization_stages")
        output_dir = Path(output_dir_str)
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            "stage": stage_name,
            "data": data,
            "count": len(data),
        }
        
        if metadata:
            output_data["metadata"] = metadata
        
        # Save as JSON
        output_file = output_dir / f"{stage_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Also save as simple text format for easy reading
        txt_file = output_dir / f"{stage_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {stage_name.upper()} ===\n")
            f.write(f"Count: {len(data)}\n\n")
            
            if metadata:
                f.write("Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            for i, item in enumerate(data, 1):
                if "start" in item and "end" in item:
                    duration = item["end"] - item["start"]
                    f.write(f"{i}. [{item['start']:.3f}-{item['end']:.3f}] ({duration:.3f}s)")
                    
                    if "label" in item:
                        f.write(f" {item['label']}")
                    elif "speaker" in item:
                        f.write(f" {item['speaker']}")
                    
                    if "text" in item:
                        f.write(f": {item['text']}")
                    
                    f.write("\n")
                else:
                    f.write(f"{i}. {item}\n")
        
        logger = get_logger()
        logger.debug(f"Saved diarization stage '{stage_name}' to {output_file} and {txt_file}")
        
    except Exception as e:
        logger = get_logger()
        logger.warning(f"Failed to save diarization stage '{stage_name}': {e}")


@error_context(reraise=True)
def _refine_short_segments(
    segments: List[Dict], 
    min_duration: float = 0.8,
    merge_gap: float = 0.3
) -> List[Dict]:
    """
    Refine diarization segments by reassigning short utterances based on context.
    
    Strategy:
    1. Identify short segments (< min_duration)
    2. Check if surrounded by the same speaker → reassign to that speaker
    3. Merge adjacent segments from same speaker if gap is small
    
    Parameters
    ----------
    segments : List[Dict]
        List of diarization segments with 'start', 'end', 'label'
    min_duration : float
        Minimum duration (in seconds) to consider a segment as "short" (default: 0.8s)
    merge_gap : float
        Maximum gap between segments to merge them (default: 0.3s)
    
    Returns
    -------
    List[Dict]
        Refined list of segments
    """
    logger = get_logger()
    
    if not segments or len(segments) < 2:
        return segments
    
    # Sort by start time to ensure proper ordering
    segments = sorted(segments, key=lambda x: x["start"])
    
    # Phase 1: Context-aware reassignment of short segments
    logger.debug(f"Phase 1: Reassigning short segments (< {min_duration}s)")
    reassigned_count = 0
    
    for i in range(len(segments)):
        seg = segments[i]
        duration = seg["end"] - seg["start"]
        
        if duration < min_duration:
            # Check neighbors
            prev_label = segments[i-1]["label"] if i > 0 else None
            next_label = segments[i+1]["label"] if i < len(segments) - 1 else None
            
            # If surrounded by same speaker, reassign to that speaker
            if prev_label and next_label and prev_label == next_label and seg["label"] != prev_label:
                logger.debug(
                    f"Reassigning short segment [{seg['start']:.2f}-{seg['end']:.2f}] "
                    f"({duration:.2f}s) from {seg['label']} to {prev_label}"
                )
                seg["label"] = prev_label
                reassigned_count += 1
            # If at boundary, assign to neighbor
            elif i == 0 and next_label and seg["label"] != next_label and duration < min_duration / 2:
                logger.debug(f"Reassigning first short segment to {next_label}")
                seg["label"] = next_label
                reassigned_count += 1
            elif i == len(segments) - 1 and prev_label and seg["label"] != prev_label and duration < min_duration / 2:
                logger.debug(f"Reassigning last short segment to {prev_label}")
                seg["label"] = prev_label
                reassigned_count += 1
    
    logger.info(f"Reassigned {reassigned_count} short segments based on context")
    
    # Phase 2: Merge adjacent segments from same speaker
    logger.debug(f"Phase 2: Merging adjacent segments with gap < {merge_gap}s")
    merged_segments = []
    current_seg = None
    merge_count = 0
    
    for seg in segments:
        if current_seg is None:
            current_seg = dict(seg)  # Start new segment
        else:
            gap = seg["start"] - current_seg["end"]
            same_speaker = seg["label"] == current_seg["label"]
            
            if same_speaker and gap <= merge_gap:
                # Merge: extend current segment
                logger.debug(
                    f"Merging segments [{current_seg['start']:.2f}-{current_seg['end']:.2f}] "
                    f"and [{seg['start']:.2f}-{seg['end']:.2f}] (gap: {gap:.2f}s)"
                )
                current_seg["end"] = seg["end"]
                merge_count += 1
            else:
                # Finalize current and start new
                merged_segments.append(current_seg)
                current_seg = dict(seg)
    
    # Don't forget the last segment
    if current_seg is not None:
        merged_segments.append(current_seg)
    
    logger.info(f"Merged {merge_count} adjacent segments, resulting in {len(merged_segments)} total segments")
    
    return merged_segments


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
def diarize_mixed(
    audio_path: str, 
    words: List[Dict],
    min_segment_duration: float = 0.8,
    merge_gap: float = 0.3,
    stages_output_dir: Optional[Path] = None
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
    min_segment_duration : float, optional
        Minimum duration (seconds) to consider a segment as "short" for reassignment (default: 0.8)
    merge_gap : float, optional
        Maximum gap (seconds) between segments to merge them (default: 0.3)
    stages_output_dir : Optional[Path], optional
        Directory to save intermediate stage outputs. If None, checks environment variable.

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
        
        # --- Debugging: Inspect the DiarizeOutput object ---
        logger.debug(f"Type of diar object: {type(diar)}")
        logger.debug(f"Attributes of diar object: {dir(diar)}")
        
        # Inspect the 'speaker_diarization' attribute
        if hasattr(diar, 'speaker_diarization'):
            annotation_obj = diar.speaker_diarization
            logger.debug(f"Type of speaker_diarization: {type(annotation_obj)}")
            logger.debug(f"Attributes of speaker_diarization: {dir(annotation_obj)}")
            # Check for known methods on the annotation object
            if hasattr(annotation_obj, 'itertracks'):
                logger.debug("speaker_diarization has itertracks method")
            if hasattr(annotation_obj, 'tracks'):
                logger.debug("speaker_diarization has tracks attribute")
            if hasattr(annotation_obj, 'get_timeline'):
                logger.debug("speaker_diarization has get_timeline method")
            if hasattr(annotation_obj, 'support'):
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
                    tracker.update(diarize_task, advance=1, description=f"Speaker Diarization - Found {segment_count} segments")
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
        
        logger.info(f"Found {segment_count} diarization segments")
        
        # Save Stage 1: Raw segments from diarization pipeline
        _save_diarization_stage(
            "1_raw_segments",
            diar_segments,
            output_dir=stages_output_dir,
            metadata={
                "total_segments": len(diar_segments),
                "audio_path": audio_path,
                "min_segment_duration": min_segment_duration,
                "merge_gap": merge_gap
            }
        )
        
        if not diar_segments:
            logger.warning("No diarization segments found, using default speaker")
            # Create a single segment covering the entire audio
            if words:
                max_time = max(w.get("end", 0) for w in words)
                diar_segments = [{"start": 0.0, "end": max_time, "label": "Speaker_A"}]
                tracker.update(diarize_task, advance=50, description="Speaker Diarization - Complete")
                tracker.complete_task(diarize_task, stage="diarization")
            else:
                tracker.complete_task(diarize_task, stage="diarization")
                return []
        else:
            # Refine short segments with context-aware reassignment
            tracker.update(diarize_task, advance=25, description="Speaker Diarization - Refining short segments")
            
            # Save Stage 2: Pre-refinement (copy for comparison)
            raw_segments_copy = [dict(s) for s in diar_segments]
            
            diar_segments = _refine_short_segments(
                diar_segments, 
                min_duration=min_segment_duration,
                merge_gap=merge_gap
            )
            logger.info(f"Refined segments: {len(diar_segments)} segments after processing")
            
            # Save Stage 3: Refined segments after context-aware reassignment
            _save_diarization_stage(
                "2_refined_segments",
                diar_segments,
                output_dir=stages_output_dir,
                metadata={
                    "total_segments": len(diar_segments),
                    "segments_before_refinement": len(raw_segments_copy),
                    "segments_removed": len(raw_segments_copy) - len(diar_segments),
                    "min_segment_duration": min_segment_duration,
                    "merge_gap": merge_gap
                }
            )
            
            tracker.update(diarize_task, advance=25, description="Speaker Diarization - Complete")
            tracker.complete_task(diarize_task, stage="diarization")

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
        
        # Save Stage 4: Words with assigned speakers
        _save_diarization_stage(
            "3_tagged_words",
            tagged_words,
            output_dir=stages_output_dir,
            metadata={
                "total_words": len(tagged_words),
                "unique_speakers": len(set(w["speaker"] for w in tagged_words))
            }
        )

        # Group into turns per speaker, then merge all speakers by time
        speakers: dict[str, List[Dict]] = {}
        for w in tagged_words:
            speakers.setdefault(w["speaker"], []).append(w)
        
        turns_task = tracker.add_task("Building conversation turns", total=len(speakers), stage="turn_building")

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
        
        # Save Stage 5: Final merged turns
        _save_diarization_stage(
            "4_final_turns",
            merged_turns,
            output_dir=stages_output_dir,
            metadata={
                "total_turns": len(merged_turns),
                "unique_speakers": len(set(t["speaker"] for t in merged_turns)),
                "total_duration": merged_turns[-1]["end"] if merged_turns else 0.0
            }
        )
        
        # Save Stage 6: Per-speaker turns (before merging)
        for spk, spk_words in speakers.items():
            spk_turns = [t for t in all_turns if t["speaker"] == spk]
            _save_diarization_stage(
                f"5_speaker_{spk.replace(' ', '_').lower()}_turns",
                spk_turns,
                output_dir=stages_output_dir,
                metadata={
                    "speaker": spk,
                    "turns": len(spk_turns),
                    "words": len(spk_words)
                }
            )
        
        logger.info(f"Speaker diarization completed successfully for {audio_path}")
        return merged_turns
        
    except Exception as e:
        if isinstance(e, DiarizationError):
            logger.error(f"Diarization error: {e}")
            raise
        else:
            logger.error(f"Unexpected error in diarization: {e}")
            raise DiarizationError(f"Unexpected error: {e}", cause=e)
