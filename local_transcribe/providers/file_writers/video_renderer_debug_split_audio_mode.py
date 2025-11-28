#!/usr/bin/env python3
"""
Debug video renderer for split audio mode alignment verification.

This module creates a debug video that displays words one at a time as they are spoken,
synchronized with their alignment timestamps. Useful for visually verifying how closely
the forced alignment matches the actual audio.

Words appear at their start time and remain visible for at least 100ms.
Words spoken in rapid succession (less than 100ms apart) accumulate horizontally.
When horizontal space is exhausted, words wrap to a new row below.
"""

from __future__ import annotations
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from local_transcribe.framework.plugin_interfaces import WordSegment


# Configuration constants
MIN_DISPLAY_DURATION_MS = 120  # Minimum time a word stays visible (milliseconds)
MIN_DISPLAY_DURATION_SEC = MIN_DISPLAY_DURATION_MS / 1000.0

# Video settings
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
FRAME_RATE = 60

# Text layout settings
FONT_SIZE = 54
LEFT_MARGIN = 70
RIGHT_MARGIN = 70
TOP_MARGIN = 300  # Y position for first row of words
ROW_HEIGHT = 80  # Vertical space between rows
WORD_SPACING = 24  # Horizontal space between words
MAX_ROWS = 6  # Maximum number of word rows before wrapping to top

# Approximate character width for positioning (monospace assumption)
# This is a rough estimate; actual width depends on font
CHAR_WIDTH_ESTIMATE = FONT_SIZE * 0.6


@dataclass
class DisplayWord:
    """A word with its display timing and position information."""
    text: str
    start: float  # When the word appears (seconds)
    display_until: float  # When the word disappears (seconds)
    x_position: int  # Horizontal position (pixels from left)
    row: int  # Which row (0-indexed)
    group_id: int  # Words in same rapid-succession group share an ID


@dataclass 
class WordGroup:
    """A group of words spoken in rapid succession."""
    words: List[DisplayWord]
    group_id: int
    start: float  # When first word appears
    end: float  # When last word's display_until expires
    

def estimate_word_width(text: str) -> int:
    """Estimate the pixel width of a word based on character count."""
    return int(len(text) * CHAR_WIDTH_ESTIMATE)


def calculate_display_words(word_segments: List[WordSegment]) -> List[DisplayWord]:
    """
    Calculate display timing and positions for all words.
    
    Words spoken within MIN_DISPLAY_DURATION of each other are grouped horizontally.
    When a gap >= MIN_DISPLAY_DURATION occurs, position resets to left margin.
    When words would exceed right margin, they wrap to the next row.
    
    Args:
        word_segments: List of WordSegment objects with timing info
        
    Returns:
        List of DisplayWord objects with positions calculated
    """
    if not word_segments:
        return []
    
    display_words = []
    current_x = LEFT_MARGIN
    current_row = 0
    current_group_id = 0
    max_x = VIDEO_WIDTH - RIGHT_MARGIN
    
    prev_word_start = None
    
    for i, word in enumerate(word_segments):
        # Calculate display duration (at least MIN_DISPLAY_DURATION)
        actual_duration = word.end - word.start
        display_duration = max(actual_duration, MIN_DISPLAY_DURATION_SEC)
        display_until = word.start + display_duration
        
        # Determine if this starts a new group (gap >= MIN_DISPLAY_DURATION from previous word start)
        if prev_word_start is not None:
            gap_from_prev = word.start - prev_word_start
            if gap_from_prev >= MIN_DISPLAY_DURATION_SEC:
                # New group - reset to left margin
                current_x = LEFT_MARGIN
                current_row = 0
                current_group_id += 1
        
        # Estimate width of this word
        word_width = estimate_word_width(word.text)
        
        # Check if word fits on current row
        if current_x + word_width > max_x:
            # Wrap to next row
            current_row += 1
            current_x = LEFT_MARGIN
            
            # If we've exceeded max rows, wrap back to top
            if current_row >= MAX_ROWS:
                current_row = 0
        
        # Create display word
        display_word = DisplayWord(
            text=word.text,
            start=word.start,
            display_until=display_until,
            x_position=current_x,
            row=current_row,
            group_id=current_group_id
        )
        display_words.append(display_word)
        
        # Update position for next word
        current_x += word_width + WORD_SPACING
        prev_word_start = word.start
    
    return display_words


def get_visible_words_at_time(t: float, display_words: List[DisplayWord]) -> List[DisplayWord]:
    """
    Get all words that should be visible at time t.
    
    A word is visible if: word.start <= t < word.display_until
    
    Args:
        t: Time in seconds
        display_words: List of DisplayWord objects
        
    Returns:
        List of DisplayWord objects visible at time t
    """
    visible = []
    for word in display_words:
        if word.start <= t < word.display_until:
            visible.append(word)
    return visible


def generate_ass_subtitle(
    display_words: List[DisplayWord],
    speaker_name: str,
    total_duration: float
) -> str:
    """
    Generate an ASS (Advanced SubStation Alpha) subtitle file content.
    
    ASS format allows precise positioning and timing of text, which is needed
    for our word-by-word display with specific positions.
    
    Args:
        display_words: List of DisplayWord objects with timing and positions
        speaker_name: Name of the speaker for display
        total_duration: Total duration of the audio in seconds
        
    Returns:
        String content of the ASS subtitle file
    """
    # ASS header with style definitions
    ass_content = f"""[Script Info]
Title: Debug Alignment Video
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,Courier New,{FONT_SIZE},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,7,0,0,0,1
Style: Info,Arial,32,&H00AAAAAA,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,1,20,20,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Add speaker info as a persistent subtitle at the bottom
    end_time = format_ass_time(total_duration)
    ass_content += f"Dialogue: 0,0:00:00.00,{end_time},Info,,0,0,0,,Speaker: {speaker_name}\n"
    
    # Add each word as a positioned dialogue event
    for word in display_words:
        start_time = format_ass_time(word.start)
        end_time = format_ass_time(word.display_until)
        
        # Calculate Y position based on row
        y_pos = TOP_MARGIN + (word.row * ROW_HEIGHT)
        
        # Use ASS positioning tag: {\pos(x,y)}
        # Escape special characters in word text
        escaped_text = escape_ass_text(word.text)
        positioned_text = f"{{\\pos({word.x_position},{y_pos})}}{escaped_text}"
        
        ass_content += f"Dialogue: 1,{start_time},{end_time},Word,,0,0,0,,{positioned_text}\n"
    
    return ass_content


def format_ass_time(seconds: float) -> str:
    """
    Format time in seconds to ASS time format: H:MM:SS.cc (centiseconds).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def escape_ass_text(text: str) -> str:
    """
    Escape special characters for ASS subtitle format.
    
    Args:
        text: Raw text
        
    Returns:
        Escaped text safe for ASS format
    """
    # ASS uses backslash for escapes, and curly braces for tags
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def render_debug_video(
    word_segments: List[WordSegment],
    audio_path: str | Path,
    output_path: str | Path,
    speaker_name: str = "Unknown",
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT
) -> None:
    """
    Create a debug video showing words appearing as they are spoken.
    
    This video is designed for verifying alignment accuracy. Words appear
    at their start timestamp and stay visible for at least 100ms. Words
    spoken in rapid succession stack horizontally, then wrap to new rows.
    
    Args:
        word_segments: List of WordSegment objects with timing info
        audio_path: Path to the speaker's audio file
        output_path: Output MP4 file path
        speaker_name: Name of the speaker for display
        width: Video width (default 1920)
        height: Video height (default 1080)
    """
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    if not word_segments:
        raise ValueError("No word segments provided")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Calculate display words with positions
    display_words = calculate_display_words(word_segments)
    
    # Get total duration from the last word
    total_duration = max(w.display_until for w in display_words)
    
    # Generate ASS subtitle content
    ass_content = generate_ass_subtitle(display_words, speaker_name, total_duration)
    
    # Write ASS file to temp location
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as f:
        f.write(ass_content)
        ass_path = f.name
    
    try:
        # Build FFmpeg command
        # Note: Using ass= filter for ASS subtitles (more precise than subtitles=)
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={FRAME_RATE}",
            "-i", str(audio_path),
            "-vf", f"ass={ass_path}",
            "-c:v", "libx264",
            "-preset", "fast",
            "-tune", "stillimage",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e
        
    finally:
        # Clean up temp ASS file
        if os.path.exists(ass_path):
            os.unlink(ass_path)


def render_debug_video_for_speaker(
    word_segments: List[WordSegment],
    audio_path: str | Path,
    output_dir: str | Path,
    speaker_name: str
) -> Path:
    """
    Convenience function to render a debug video for a single speaker.
    
    Creates output file named: {speaker_name}_debug_alignment.mp4
    
    Args:
        word_segments: List of WordSegment objects for this speaker
        audio_path: Path to the speaker's audio file
        output_dir: Directory to save the output video
        speaker_name: Name of the speaker
        
    Returns:
        Path to the generated video file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize speaker name for filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in speaker_name)
    output_path = output_dir / f"{safe_name}_debug_alignment.mp4"
    
    render_debug_video(
        word_segments=word_segments,
        audio_path=audio_path,
        output_path=output_path,
        speaker_name=speaker_name
    )
    
    return output_path


# Utility function for testing
def create_test_video_from_json(
    json_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    speaker_name: str = "Test Speaker"
) -> None:
    """
    Create a debug video from a JSON file containing word segments.
    
    The JSON file should have a 'words' array with objects containing:
    - text: string
    - start: float (seconds)
    - end: float (seconds)
    - speaker: string (optional)
    
    Args:
        json_path: Path to JSON file with word segments
        audio_path: Path to audio file
        output_path: Output video path
        speaker_name: Speaker name to display
    """
    import json
    
    json_path = Path(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = data.get('words', data)  # Handle both {words: [...]} and [...]
    
    word_segments = [
        WordSegment(
            text=w['text'],
            start=w['start'],
            end=w['end'],
            speaker=w.get('speaker', speaker_name)
        )
        for w in words
    ]
    
    render_debug_video(
        word_segments=word_segments,
        audio_path=audio_path,
        output_path=output_path,
        speaker_name=speaker_name
    )


if __name__ == "__main__":
    # Example usage for testing
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python video_renderer_debug_split_audio_mode.py <json_path> <audio_path> <output_path> [speaker_name]")
        print("")
        print("Creates a debug video showing words appearing as they are spoken,")
        print("synchronized with alignment timestamps.")
        sys.exit(1)
    
    json_path = sys.argv[1]
    audio_path = sys.argv[2]
    output_path = sys.argv[3]
    speaker_name = sys.argv[4] if len(sys.argv) > 4 else "Speaker"
    
    print(f"Creating debug alignment video...")
    print(f"  JSON: {json_path}")
    print(f"  Audio: {audio_path}")
    print(f"  Output: {output_path}")
    print(f"  Speaker: {speaker_name}")
    
    create_test_video_from_json(json_path, audio_path, output_path, speaker_name)
    
    print(f"Done! Video saved to: {output_path}")
