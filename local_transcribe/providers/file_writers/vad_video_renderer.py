#!/usr/bin/env python3
"""
VAD Video Renderer for creating review videos from VAD-split-audio transcripts.

This module generates MP4 videos with block-based subtitles for the VAD pipeline.
Unlike word-aligned pipelines, VAD transcripts have block-level timestamps,
so we display full blocks of text that appear/disappear with block timing.

Features:
- Full block text display with dynamic font sizing
- Page splitting for long blocks
- Interjection display on dedicated bottom line
- Multiple audio track merging
- ASS subtitle format for precise positioning
"""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from dataclasses import dataclass
from math import ceil

from local_transcribe.framework.plugin_interfaces import OutputWriter
from local_transcribe.framework import registry


# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FRAME_RATE = 30

# Font settings
BASE_FONT_SIZE = 48
MIN_FONT_SIZE = 16
MAX_FONT_SIZE = 64

# Layout settings
MAIN_TEXT_MARGIN_LEFT = 100
MAIN_TEXT_MARGIN_RIGHT = 100
MAIN_TEXT_TOP_Y = 150
MAIN_TEXT_BOTTOM_Y = 850  # Leave room for interjections

INTERJECTION_Y = 950  # Bottom area for interjections
INTERJECTION_FONT_SIZE = 32

# Text layout calculations
USABLE_WIDTH = VIDEO_WIDTH - MAIN_TEXT_MARGIN_LEFT - MAIN_TEXT_MARGIN_RIGHT
MAIN_TEXT_HEIGHT = MAIN_TEXT_BOTTOM_Y - MAIN_TEXT_TOP_Y

# Speaker colors (ASS format: &HAABBGGRR - Alpha, Blue, Green, Red)
SPEAKER_COLORS = {
    0: "&H00FFFFFF",  # White
    1: "&H00FFFF00",  # Cyan
}
INTERJECTION_COLOR = "&H0000FFFF"  # Yellow for interjections


@dataclass
class TextPage:
    """A page of text to display for a portion of a block."""
    text: str
    start_time: float
    end_time: float
    font_size: int
    speaker: str


@dataclass
class InterjectionCue:
    """An interjection to display."""
    text: str
    speaker: str
    start_time: float
    end_time: float


def _estimate_char_width(font_size: int) -> float:
    """Estimate character width for a given font size (monospace approximation)."""
    return font_size * 0.55


def _estimate_line_height(font_size: int) -> float:
    """Estimate line height for a given font size."""
    return font_size * 1.4


def _calculate_text_metrics(text: str, font_size: int) -> tuple[int, int]:
    """
    Calculate how many lines the text will take and chars per line.
    
    Returns:
        Tuple of (num_lines, chars_per_line)
    """
    char_width = _estimate_char_width(font_size)
    chars_per_line = int(USABLE_WIDTH / char_width)
    
    if chars_per_line <= 0:
        chars_per_line = 1
    
    # Count lines needed (word-wrap simulation)
    words = text.split()
    lines = 1
    current_line_len = 0
    
    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_line_len + word_len > chars_per_line:
            lines += 1
            current_line_len = len(word)
        else:
            current_line_len += word_len
    
    return lines, chars_per_line


def _calculate_max_lines(font_size: int) -> int:
    """Calculate maximum lines that fit in the main text area."""
    line_height = _estimate_line_height(font_size)
    return max(1, int(MAIN_TEXT_HEIGHT / line_height))


def calculate_optimal_font_size(text: str) -> int:
    """
    Calculate optimal font size to fit text in display area.
    
    Starts with BASE_FONT_SIZE and reduces until text fits,
    respecting MIN_FONT_SIZE limit.
    
    Args:
        text: The text to display
        
    Returns:
        Optimal font size (between MIN_FONT_SIZE and BASE_FONT_SIZE)
    """
    for size in range(BASE_FONT_SIZE, MIN_FONT_SIZE - 1, -2):
        num_lines, _ = _calculate_text_metrics(text, size)
        max_lines = _calculate_max_lines(size)
        
        if num_lines <= max_lines:
            return size
    
    return MIN_FONT_SIZE


def split_text_into_pages(text: str, duration: float, start_time: float, speaker: str) -> List[TextPage]:
    """
    Split long text into multiple pages that fit on screen.
    
    Each page gets equal time allocation based on word count proportion.
    
    Args:
        text: Full block text
        duration: Total duration of the block in seconds
        start_time: Start time of the block
        speaker: Speaker name
        
    Returns:
        List of TextPage objects
    """
    # First try to fit at minimum font size
    font_size = calculate_optimal_font_size(text)
    num_lines, chars_per_line = _calculate_text_metrics(text, font_size)
    max_lines = _calculate_max_lines(font_size)
    
    # If it fits, return single page
    if num_lines <= max_lines:
        return [TextPage(
            text=text,
            start_time=start_time,
            end_time=start_time + duration,
            font_size=font_size,
            speaker=speaker,
        )]
    
    # Need to split into pages
    words = text.split()
    total_words = len(words)
    
    if total_words == 0:
        return []
    
    # Estimate words per page based on capacity
    chars_per_page = chars_per_line * max_lines
    avg_word_len = sum(len(w) for w in words) / total_words if total_words else 5
    words_per_page = max(1, int(chars_per_page / (avg_word_len + 1)))
    
    pages: List[TextPage] = []
    word_idx = 0
    
    while word_idx < total_words:
        # Get words for this page
        page_words = words[word_idx:word_idx + words_per_page]
        page_text = " ".join(page_words)
        
        # Calculate time for this page (proportional to word count)
        page_word_count = len(page_words)
        page_duration = (page_word_count / total_words) * duration
        
        page_start = start_time if not pages else pages[-1].end_time
        page_end = page_start + page_duration
        
        # Recalculate font size for this page (it might fit better)
        page_font_size = calculate_optimal_font_size(page_text)
        
        pages.append(TextPage(
            text=page_text,
            start_time=page_start,
            end_time=page_end,
            font_size=page_font_size,
            speaker=speaker,
        ))
        
        word_idx += words_per_page
    
    # Ensure last page ends at correct time
    if pages:
        pages[-1].end_time = start_time + duration
    
    return pages


def _format_ass_time(seconds: float) -> str:
    """
    Format time in seconds to ASS time format: H:MM:SS.cc (centiseconds).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def _escape_ass_text(text: str) -> str:
    """
    Escape special characters for ASS subtitle format.
    
    Args:
        text: Raw text
        
    Returns:
        Escaped text safe for ASS
    """
    # ASS special characters that need escaping
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    # Newlines in ASS use \N
    text = text.replace("\n", "\\N")
    return text


def _get_speaker_color(speaker: str, speaker_map: Dict[str, int]) -> str:
    """Get ASS color code for a speaker."""
    if speaker not in speaker_map:
        speaker_map[speaker] = len(speaker_map) % len(SPEAKER_COLORS)
    return SPEAKER_COLORS.get(speaker_map[speaker], SPEAKER_COLORS[0])


def generate_ass_subtitles(
    transcript: Any,
    total_duration: float,
) -> str:
    """
    Generate ASS subtitle content from a TranscriptFlow.
    
    Args:
        transcript: TranscriptFlow object with turns and interjections
        total_duration: Total audio duration in seconds
        
    Returns:
        Complete ASS file content as string
    """
    # Build speaker color map
    speaker_map: Dict[str, int] = {}
    
    # Collect all pages and interjections
    all_pages: List[TextPage] = []
    all_interjections: List[InterjectionCue] = []
    
    # Extract turns from TranscriptFlow
    turns = []
    if hasattr(transcript, 'turns'):
        turns = transcript.turns
    elif isinstance(transcript, list):
        turns = transcript
    
    for turn in turns:
        # Get turn attributes
        if hasattr(turn, 'primary_speaker'):
            speaker = turn.primary_speaker
        elif hasattr(turn, 'speaker'):
            speaker = turn.speaker
        else:
            speaker = "Unknown"
        
        start = getattr(turn, 'start', 0)
        end = getattr(turn, 'end', start + 1)
        text = getattr(turn, 'text', '')
        
        if not text.strip():
            continue
        
        duration = end - start
        
        # Split into pages if needed
        pages = split_text_into_pages(text, duration, start, speaker)
        all_pages.extend(pages)
        
        # Collect interjections
        interjections = getattr(turn, 'interjections', [])
        for ij in interjections:
            ij_speaker = getattr(ij, 'speaker', 'Unknown')
            ij_start = getattr(ij, 'start', start)
            ij_end = getattr(ij, 'end', ij_start + 0.5)
            ij_text = getattr(ij, 'text', '')
            
            if ij_text.strip():
                all_interjections.append(InterjectionCue(
                    text=ij_text,
                    speaker=ij_speaker,
                    start_time=ij_start,
                    end_time=ij_end,
                ))
    
    # Generate ASS content
    ass_content = _generate_ass_header(speaker_map, turns)
    
    # Add main text events
    for page in all_pages:
        color = _get_speaker_color(page.speaker, speaker_map)
        start_ts = _format_ass_time(page.start_time)
        end_ts = _format_ass_time(page.end_time)
        escaped_text = _escape_ass_text(page.text)
        
        # Center the text with dynamic font size
        # Using alignment 8 (top-center) with custom positioning
        style_override = f"{{\\fs{page.font_size}\\c{color}\\an8}}"
        
        ass_content += f"Dialogue: 0,{start_ts},{end_ts},MainText,,0,0,0,,{style_override}{escaped_text}\n"
    
    # Add interjection events
    for ij in all_interjections:
        start_ts = _format_ass_time(ij.start_time)
        end_ts = _format_ass_time(ij.end_time)
        escaped_text = _escape_ass_text(f"[{ij.speaker}]: {ij.text}")
        
        ass_content += f"Dialogue: 1,{start_ts},{end_ts},Interjection,,0,0,0,,{escaped_text}\n"
    
    return ass_content


def _generate_ass_header(speaker_map: Dict[str, int], turns: list) -> str:
    """Generate ASS file header with styles."""
    
    # Build speaker labels for display
    speakers = []
    for turn in turns:
        if hasattr(turn, 'primary_speaker'):
            speaker = turn.primary_speaker
        elif hasattr(turn, 'speaker'):
            speaker = turn.speaker
        else:
            continue
        if speaker not in speakers:
            speakers.append(speaker)
    
    header = f"""[Script Info]
Title: VAD Transcript Video
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: MainText,Arial,{BASE_FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,8,{MAIN_TEXT_MARGIN_LEFT},{MAIN_TEXT_MARGIN_RIGHT},{MAIN_TEXT_TOP_Y},1
Style: Interjection,Arial,{INTERJECTION_FONT_SIZE},{INTERJECTION_COLOR},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,{MAIN_TEXT_MARGIN_LEFT},{MAIN_TEXT_MARGIN_RIGHT},20,1
Style: SpeakerLabel,Arial,28,&H00AAAAAA,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,7,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header


def _check_ffmpeg_filter_available(filter_name: str) -> bool:
    """Check if an FFmpeg filter is available in the current installation."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True,
            text=True,
        )
        # Filter list format: " TSC filtername  V->V  Description"
        # Look for the filter name as a complete word
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 2:
                # Filter name is typically the second column after flags
                if filter_name in parts:
                    return True
        return False
    except Exception:
        return False


def render_vad_video(
    ass_path: Path,
    output_mp4: Path,
    audio_config: Dict[str, str],
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
) -> None:
    """
    Render video with ASS subtitles and merged audio tracks.
    
    Args:
        ass_path: Path to ASS subtitle file
        output_mp4: Output MP4 file path
        audio_config: Dict mapping speaker names to audio file paths
        width: Video width
        height: Video height
    
    Raises:
        RuntimeError: If FFmpeg is not installed or missing required filters
    """
    # Check if the 'ass' filter is available in FFmpeg
    if not _check_ffmpeg_filter_available("ass"):
        raise RuntimeError(
            "FFmpeg 'ass' filter is not available. This filter requires FFmpeg to be "
            "compiled with libass support.\n"
            "On macOS, you can install FFmpeg with libass using:\n"
            "  brew uninstall ffmpeg\n"
            "  brew install ffmpeg --with-libass\n"
            "Or install via Homebrew tap:\n"
            "  brew tap homebrew-ffmpeg/ffmpeg\n"
            "  brew install homebrew-ffmpeg/ffmpeg/ffmpeg --with-libass\n"
            "The ASS subtitle file has been saved and can be used with other tools."
        )
    
    audio_paths = list(audio_config.values())
    
    if not audio_paths:
        raise ValueError("No audio paths provided in audio_config")
    
    # Escape the ASS path for FFmpeg filter syntax
    # FFmpeg filters need special characters escaped with multiple levels:
    # 1. Backslash escape for FFmpeg filter parser: \ : ' [ ]
    # 2. Single quotes around the path to handle spaces and special chars
    # The order matters: first escape special chars, then wrap in quotes
    escaped_ass_path = ass_path.as_posix()
    # Escape backslashes first (before other escapes add more backslashes)
    escaped_ass_path = escaped_ass_path.replace("\\", "\\\\\\\\")
    # Escape single quotes (need to break out of quoting: ' -> '\''  )
    escaped_ass_path = escaped_ass_path.replace("'", "'\\''")
    # Escape colons and brackets for FFmpeg filter syntax
    escaped_ass_path = escaped_ass_path.replace(":", "\\:")
    escaped_ass_path = escaped_ass_path.replace("[", "\\[")
    escaped_ass_path = escaped_ass_path.replace("]", "\\]")
    # Wrap the entire path in single quotes to handle spaces
    escaped_ass_path = f"'{escaped_ass_path}'"
    
    # Build FFmpeg command
    cmd = ["ffmpeg", "-y"]
    
    # Add video source (black background)
    cmd.extend(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={FRAME_RATE}"])
    
    # Add all audio inputs
    for audio_path in audio_paths:
        cmd.extend(["-i", str(audio_path)])
    
    # Build filter complex
    if len(audio_paths) == 1:
        # Single audio track
        filter_complex = f"[0:v]ass={escaped_ass_path}[v]"
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[v]", "-map", "1:a"])
    else:
        # Multiple audio tracks - merge them
        input_labels = [f"[{i+1}:a]" for i in range(len(audio_paths))]
        audio_merge = f"{''.join(input_labels)}amerge=inputs={len(audio_paths)}[a]"
        video_filter = f"[0:v]ass={escaped_ass_path}[v]"
        filter_complex = f"{audio_merge};{video_filter}"
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[v]", "-map", "[a]"])
    
    # Output settings
    cmd.extend([
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-c:a", "aac",
        "-ac", "2",  # Stereo output
        "-shortest",
        str(output_mp4),
    ])
    
    subprocess.run(cmd, check=True)


class VADVideoWriter(OutputWriter):
    """Output writer for MP4 video with block-based subtitles for VAD transcripts."""
    
    @property
    def name(self) -> str:
        return "vad-video"
    
    @property
    def description(self) -> str:
        return "MP4 video with block-based subtitles for VAD transcripts"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".mp4"]
    
    @property
    def supported_modes(self) -> List[str]:
        """VADVideoWriter only supports the VAD split audio mode."""
        return ["vad_split_audio"]
    
    def write(
        self,
        turns: Any,
        output_path: str,
        word_segments: Optional[List[Any]] = None,
        **kwargs
    ) -> None:
        """
        Write MP4 video with block-based subtitles.
        
        Args:
            turns: TranscriptFlow object with turns and interjections
            output_path: Output MP4 file path
            word_segments: Ignored (VAD doesn't have word-level timing)
            **kwargs: Additional arguments including 'audio_config'
        """
        output_mp4 = Path(output_path)
        
        # Get audio configuration
        audio_config = kwargs.get('audio_config')
        if audio_config is None:
            raise ValueError("audio_config is required for VAD video generation")
        
        if not isinstance(audio_config, dict):
            raise ValueError(f"audio_config must be a dict for VAD mode, got {type(audio_config)}")
        
        # Calculate total duration from transcript
        total_duration = self._get_total_duration(turns)
        
        # Generate ASS subtitle file
        ass_content = generate_ass_subtitles(turns, total_duration)
        
        # Write ASS to temp file and render
        ass_path = output_mp4.with_suffix('.ass')
        try:
            ass_path.write_text(ass_content, encoding='utf-8')
            
            # Render the video
            render_vad_video(ass_path, output_mp4, audio_config)
            
        finally:
            # Clean up temp ASS file
            if ass_path.exists():
                ass_path.unlink()
    
    def _get_total_duration(self, transcript: Any) -> float:
        """Get total duration from transcript."""
        turns = []
        if hasattr(transcript, 'turns'):
            turns = transcript.turns
        elif isinstance(transcript, list):
            turns = transcript
        
        if not turns:
            return 0.0
        
        max_end = 0.0
        for turn in turns:
            end = getattr(turn, 'end', 0)
            if end > max_end:
                max_end = end
            # Also check interjections
            for ij in getattr(turn, 'interjections', []):
                ij_end = getattr(ij, 'end', 0)
                if ij_end > max_end:
                    max_end = ij_end
        
        return max_end


# Register the VAD video writer
registry.register_output_writer(VADVideoWriter())
