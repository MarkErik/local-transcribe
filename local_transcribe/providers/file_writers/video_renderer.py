#!/usr/bin/env python3
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Dict, Union, Optional
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, WordSegment
from local_transcribe.framework import registry


def render_video(subs_path: str | Path, output_mp4: str | Path, audio_config: Union[str, Path, Dict[str, str]], width: int = 1920, height: int = 1080):
    """
    Create a video with a black background and burned-in subtitles + original audio.
    Requires ffmpeg on PATH. Uses SRT input.

    Args:
        subs_path: Path to SRT subtitle file
        output_mp4: Output MP4 file path
        audio_config: Can be:
            - Single audio file path (str/Path) for combined_audio mode
            - Dict mapping speaker names to audio file paths for split_audio mode
        width: Video width (default 1920)
        height: Video height (default 1080)
    """
    subs_path = Path(subs_path)
    output_mp4 = Path(output_mp4)

    # Handle different audio configuration formats
    if isinstance(audio_config, (str, Path)):
        # Single audio track (combined_audio mode)
        audio_path = Path(audio_config)
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
            # Single track - no mixing needed
            cmd.extend(["-map", "0:v", "-map", "1:a"])
        else:
            # Multiple tracks - merge them
            input_labels = [f"[{i+1}:a]" for i in range(len(audio_paths))]
            merge_filter = f"{''.join(input_labels)}amerge=inputs={len(audio_paths)}[a]"
            cmd.extend(["-filter_complex", merge_filter])
            cmd.extend(["-map", "0:v", "-map", "[a]"])
        
        # Add subtitle filter and output settings
        cmd.extend(["-vf", f"subtitles={subs_path.as_posix()}"])
        cmd.extend(["-c:v", "libx264", "-tune", "stillimage"])
        cmd.extend(["-c:a", "aac", "-shortest"])
        cmd.append(str(output_mp4))
        
    else:
        raise ValueError(f"Unsupported audio_config type: {type(audio_config)}. Expected str, Path, or Dict[str, str]")

    subprocess.run(cmd, check=True)


def _group_words_into_cues(words: List[WordSegment], max_words_per_cue: int = 3) -> List[Dict]:
    """
    Group word segments into subtitle cues.
    
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
            # Create cue for previous speaker
            cue_text = " ".join(w.text for w in current_cue_words)
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
            # Create cue
            cue_text = " ".join(w.text for w in current_cue_words)
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
        cue_text = " ".join(w.text for w in current_cue_words)
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
    
    def write(self, turns: List[Turn], output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """Write MP4 video with subtitles.
        
        Args:
            turns: List of conversation turns
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
            turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
            write_srt(turn_dicts, srt_path)
        
        try:
            # Get audio configuration from kwargs
            audio_config = kwargs.get('audio_config')
            if audio_config is None:
                raise ValueError("audio_config is required for video generation")
            
            # Render the video
            render_video(srt_path, output_path, audio_config)
            
        finally:
            # Clean up temporary SRT file
            if srt_path.exists():
                srt_path.unlink()


# Register the video writer with the global registry
registry.register_output_writer(VideoWriter())
