#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, WordSegment
from local_transcribe.framework import registry


def _extract_turns_as_dicts(transcript: Any) -> List[Dict]:
    """
    Extract turns as dictionaries from various transcript formats.
    
    Handles:
    - TranscriptFlow (new hierarchical format)
    - List of Turn objects
    - List of HierarchicalTurn objects
    - List of dictionaries
    
    Returns list of dicts with 'speaker', 'start', 'end', 'text' keys.
    """
    # Handle TranscriptFlow
    if hasattr(transcript, 'turns') and hasattr(transcript, 'metadata'):
        # This is a TranscriptFlow object
        turns = transcript.turns
        result = []
        for t in turns:
            # HierarchicalTurn uses primary_speaker
            speaker = getattr(t, 'primary_speaker', None) or getattr(t, 'speaker', 'Unknown')
            result.append({
                "speaker": speaker,
                "start": t.start,
                "end": t.end,
                "text": t.text
            })
        return result
    
    # Handle list of turns
    if isinstance(transcript, list):
        result = []
        for t in transcript:
            if isinstance(t, dict):
                result.append(t)
            elif hasattr(t, 'primary_speaker'):
                # HierarchicalTurn
                result.append({
                    "speaker": t.primary_speaker,
                    "start": t.start,
                    "end": t.end,
                    "text": t.text
                })
            elif hasattr(t, 'speaker'):
                # Turn object
                result.append({
                    "speaker": t.speaker,
                    "start": t.start,
                    "end": t.end,
                    "text": t.text
                })
            else:
                # Unknown format, try to extract what we can
                result.append({
                    "speaker": str(getattr(t, 'speaker', getattr(t, 'primary_speaker', 'Unknown'))),
                    "start": float(getattr(t, 'start', 0)),
                    "end": float(getattr(t, 'end', 0)),
                    "text": str(getattr(t, 'text', ''))
                })
        return result
    
    # Fallback - return empty list
    return []


def render_video(subs_path: str | Path, output_mp4: str | Path, audio_config: Union[str, Path, Dict[str, str]], width: int = 1920, height: int = 1080, word_segments: Optional[List[WordSegment]] = None):
    """
    Create a video with a black background and burned-in subtitles + original audio.
    Requires ffmpeg on PATH. Uses SRT input.
    If word_segments provided and contain [REDACTED] tokens, audio will be blanked during those times.

    Args:
        subs_path: Path to SRT subtitle file
        output_mp4: Output MP4 file path
        audio_config: Can be:
            - Single audio file path (str/Path) for combined_audio mode
            - Dict mapping speaker names to audio file paths for split_audio mode
        width: Video width (default 1920)
        height: Video height (default 1080)
        word_segments: Optional word segments for [REDACTED] audio blanking
    """
    subs_path = Path(subs_path)
    output_mp4 = Path(output_mp4)

    # Collect [REDACTED] time ranges for audio blanking
    mute_ranges = []
    if word_segments:
        for word in word_segments:
            if word.text == "[REDACTED]":
                mute_ranges.append((word.start, word.end))
    
    # Handle different audio configuration formats
    if isinstance(audio_config, (str, Path)):
        # Single audio track (combined_audio mode)
        audio_path = Path(audio_config)
        
        if mute_ranges:
            # Build audio filter to mute [REDACTED] time ranges
            volume_filter = _build_volume_mute_filter(mute_ranges)
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30",
                "-i", str(audio_path),
                "-filter_complex", f"[1:a]{volume_filter}[a];[0:v]subtitles={subs_path.as_posix()}[v]",
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-shortest",
                str(output_mp4),
            ]
        else:
            # No muting needed - original logic
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30",
                "-i", str(audio_path),
                "-vf", f"subtitles={subs_path.as_posix()}",
                "-c:v", "libx264", "-tune", "stillimage",
                "-c:a", "aac", "-shortest",
                str(output_mp4),
            ]
    elif isinstance(audio_config, dict):
        # Multiple audio tracks with speaker names (split_audio mode)
        audio_paths = list(audio_config.values())
        
        if not audio_paths:
            raise ValueError("No audio paths provided in audio_config dictionary")
        
        # Build FFmpeg command for multiple audio tracks
        cmd = ["ffmpeg", "-y"]
        
        # Add video source (black background)
        cmd.extend(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r=30"])
        
        # Add all audio inputs
        for i, audio_path in enumerate(audio_paths):
            cmd.extend(["-i", str(audio_path)])
        
        # Build audio filter complex for merging multiple tracks
        if len(audio_paths) == 1:
            # Single track - check if muting needed
            if mute_ranges:
                volume_filter = _build_volume_mute_filter(mute_ranges)
                cmd.extend(["-filter_complex", f"[1:a]{volume_filter}[a];[0:v]subtitles={subs_path.as_posix()}[v]"])
                cmd.extend(["-map", "[v]", "-map", "[a]"])
            else:
                cmd.extend(["-map", "0:v", "-map", "1:a"])
                cmd.extend(["-vf", f"subtitles={subs_path.as_posix()}"])
        else:
            # Multiple tracks - merge them
            input_labels = [f"[{i+1}:a]" for i in range(len(audio_paths))]
            merge_filter = f"{''.join(input_labels)}amerge=inputs={len(audio_paths)}[amerged]"
            
            if mute_ranges:
                # Add muting after merge
                volume_filter = _build_volume_mute_filter(mute_ranges)
                filter_chain = f"{merge_filter};[amerged]{volume_filter}[a]"
                cmd.extend(["-filter_complex", filter_chain])
                cmd.extend(["-map", "0:v", "-map", "[a]"])
            else:
                cmd.extend(["-filter_complex", merge_filter])
                cmd.extend(["-map", "0:v", "-map", "[amerged]"])
        
        # Add subtitle filter if not already in filter_complex
        if not mute_ranges or len(audio_paths) > 1:
            if "-vf" not in cmd:
                cmd.extend(["-vf", f"subtitles={subs_path.as_posix()}"])
        
        # Add output settings
        cmd.extend(["-c:v", "libx264", "-tune", "stillimage"])
        cmd.extend(["-c:a", "aac", "-shortest"])
        cmd.append(str(output_mp4))
        
    else:
        raise ValueError(f"Unsupported audio_config type: {type(audio_config)}. Expected str, Path, or Dict[str, str]")

    subprocess.run(cmd, check=True)


def _build_volume_mute_filter(mute_ranges: List[tuple[float, float]]) -> str:
    """
    Build FFmpeg volume filter to mute specific time ranges.
    
    Args:
        mute_ranges: List of (start, end) tuples in seconds
        
    Returns:
        FFmpeg volume filter string
        
    Example output: "volume=enable='between(t,1.5,2.3)+between(t,5.7,6.1)':volume=0"
    """
    if not mute_ranges:
        return ""
    
    conditions = [f"between(t,{start:.3f},{end:.3f})" for start, end in mute_ranges]
    enable_expr = "+".join(conditions)
    return f"volume=enable='{enable_expr}':volume=0"


def _group_words_into_cues(words: List[WordSegment], max_words_per_cue: int = 3) -> List[Dict]:
    """
    Group word segments into subtitle cues.
    Replaces [REDACTED] tokens with underscores for display.
    
    Args:
        words: List of WordSegment objects
        max_words_per_cue: Maximum words per subtitle cue
        
    Returns:
        List of cue dictionaries with speaker, start, end, text
    """
    cues = []
    current_cue_words = []
    
    for word in words:
        # If speaker changes, finalize current cue
        if current_cue_words and word.speaker != current_cue_words[0].speaker:
            # Create cue for previous speaker with [REDACTED] → ______
            cue_text = " ".join("______" if w.text == "[REDACTED]" else w.text for w in current_cue_words)
            cue_start = current_cue_words[0].start
            cue_end = current_cue_words[-1].end
            cue_speaker = current_cue_words[0].speaker
            
            cues.append({
                "speaker": cue_speaker,
                "start": cue_start,
                "end": cue_end,
                "text": cue_text
            })
            current_cue_words = []
        
        current_cue_words.append(word)
        
        # Check if we should end the cue
        if len(current_cue_words) >= max_words_per_cue:
            # Create cue with [REDACTED] → ______
            cue_text = " ".join("______" if w.text == "[REDACTED]" else w.text for w in current_cue_words)
            cue_start = current_cue_words[0].start
            cue_end = current_cue_words[-1].end
            cue_speaker = current_cue_words[0].speaker
            
            cues.append({
                "speaker": cue_speaker,
                "start": cue_start,
                "end": cue_end,
                "text": cue_text
            })
            
            current_cue_words = []
    
    # Handle remaining words
    if current_cue_words:
        cue_text = " ".join("______" if w.text == "[REDACTED]" else w.text for w in current_cue_words)
        cue_start = current_cue_words[0].start
        cue_end = current_cue_words[-1].end
        cue_speaker = current_cue_words[0].speaker
        
        cues.append({
            "speaker": cue_speaker,
            "start": cue_start,
            "end": cue_end,
            "text": cue_text
        })
    
    return cues


class VideoWriter(OutputWriter):
    """Output writer for MP4 video with burned-in subtitles."""
    
    @property
    def name(self) -> str:
        return "video"
    
    @property
    def description(self) -> str:
        return "MP4 video with burned-in subtitles and original audio"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".mp4"]
    
    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """Write MP4 video with subtitles.
        
        Args:
            turns: List of conversation turns (or TranscriptFlow)
            output_path: Output MP4 file path
            word_segments: Optional word segments for detailed subtitle timing
            **kwargs: Additional arguments including 'audio_config'
        """
        # Import SRT writer to generate subtitles
        from local_transcribe.providers.file_writers.str_writer import write_srt
        
        # Create temporary SRT file
        srt_path = Path(output_path).with_suffix('.srt')
        
        if word_segments:
            # Generate cues from word segments
            cues = _group_words_into_cues(word_segments)
            write_srt(cues, srt_path)
        else:
            # Fallback to turn-based subtitles
            turn_dicts = _extract_turns_as_dicts(turns)
            write_srt(turn_dicts, srt_path)
        
        try:
            # Get audio configuration from kwargs
            audio_config = kwargs.get('audio_config')
            if audio_config is None:
                raise ValueError("audio_config is required for video generation")
            
            # Render the video (pass word_segments for [REDACTED] audio blanking)
            render_video(srt_path, output_path, audio_config, word_segments=word_segments)
            
        finally:
            # Clean up temporary SRT file
            if srt_path.exists():
                srt_path.unlink()


# Register the video writer with the global registry
registry.register_output_writer(VideoWriter())
