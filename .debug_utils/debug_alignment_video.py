#!/usr/bin/env python3
"""
Debug video renderer for split audio mode alignment verification.

This standalone script creates a debug video that displays words one at a time 
as they are spoken, synchronized with their alignment timestamps. Useful for 
visually verifying how closely the forced alignment matches the actual audio.

Words appear at their start time and remain visible for at least 100ms.
Words spoken in rapid succession (less than 100ms apart) accumulate horizontally.
When horizontal space is exhausted, words wrap to a new row below.

SINGLE-SPEAKER MODE:
    When the JSON contains words from 0-1 speakers (or >2 speakers as fallback),
    words appear across the full width of the screen. Speaker label is shown
    at bottom-left if a valid speaker name exists in the data.

TWO-SPEAKER MODE:
    When exactly 2 unique speakers are detected in the JSON, the screen is split:
    - First speaker (by order of appearance): words on left half, label bottom-left
    - Second speaker: words on right half (cyan colored), label bottom-right
    Each speaker's words accumulate and wrap independently within their half.

Usage:
    python debug_alignment_video.py <json_path> <audio_path> <output_path> [--speaker NAME]

Example:
    python debug_alignment_video.py words.json speaker_audio.wav output.mp4
    python debug_alignment_video.py diarized.json mixed_audio.wav output.mp4 -v
"""

from __future__ import annotations
import subprocess
import tempfile
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


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

# Two-speaker mode layout
CENTER_GAP = 80  # Horizontal gap between left and right speaker areas
CENTER_X = VIDEO_WIDTH // 2
LEFT_SPEAKER_MAX_X = CENTER_X - (CENTER_GAP // 2)  # ~920
RIGHT_SPEAKER_MIN_X = CENTER_X + (CENTER_GAP // 2)  # ~1000
RIGHT_SPEAKER_MAX_X = VIDEO_WIDTH - RIGHT_MARGIN  # ~1850

# Speaker colors (ASS format: &HBBGGRR - note: BGR order, not RGB)
# These are used in two-speaker mode to distinguish speakers
SPEAKER_1_COLOR = "&H00FFFFFF"  # White
SPEAKER_2_COLOR = "&H00FFFF00"  # Cyan (BGR: 00FFFF = RGB: FFFF00 = Cyan)

# Approximate character width for positioning (monospace assumption)
CHAR_WIDTH_ESTIMATE = FONT_SIZE * 0.6


@dataclass
class WordSegment:
    """Represents a transcribed word with timing information."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None


@dataclass
class DisplayWord:
    """A word with its display timing and position information."""
    text: str
    start: float  # When the word appears (seconds)
    display_until: float  # When the word disappears (seconds)
    x_position: int  # Horizontal position (pixels from left)
    row: int  # Which row (0-indexed)
    group_id: int  # Words in same rapid-succession group share an ID
    speaker: Optional[str] = None  # Speaker name for color coding


@dataclass
class SpeakerLayout:
    """Layout boundaries for a speaker's word display area."""
    left_x: int  # Left boundary (pixels)
    right_x: int  # Right boundary (pixels)
    color: str  # ASS color code
    label_x: int  # X position for speaker label
    label_alignment: int  # ASS alignment for label (1=left, 3=right)


def estimate_word_width(text: str) -> int:
    """Estimate the pixel width of a word based on character count."""
    return int(len(text) * CHAR_WIDTH_ESTIMATE)


def detect_speakers(word_segments: List[WordSegment]) -> tuple[List[str], bool]:
    """
    Detect unique speakers in word segments and determine display mode.
    
    Args:
        word_segments: List of WordSegment objects
        
    Returns:
        Tuple of (ordered list of speaker names, is_two_speaker_mode)
        Speaker order is determined by first appearance in transcript.
    """
    seen_speakers = []
    for word in word_segments:
        speaker = word.speaker
        # Treat None, "None", "null", empty string as no speaker
        if speaker and speaker not in ("None", "null", "") and speaker not in seen_speakers:
            seen_speakers.append(speaker)
    
    if len(seen_speakers) == 2:
        return seen_speakers, True
    elif len(seen_speakers) > 2:
        print(f"  Warning: Found {len(seen_speakers)} speakers, falling back to single-speaker mode")
        print(f"  Speakers: {seen_speakers}")
        return seen_speakers, False
    else:
        return seen_speakers, False


def get_speaker_layouts(speakers: List[str], is_two_speaker_mode: bool) -> dict[str, SpeakerLayout]:
    """
    Create layout configurations for each speaker.
    
    Args:
        speakers: List of speaker names (ordered by first appearance)
        is_two_speaker_mode: Whether to use split-screen layout
        
    Returns:
        Dictionary mapping speaker name to SpeakerLayout
    """
    if not is_two_speaker_mode or len(speakers) < 2:
        # Single speaker mode - full width
        default_layout = SpeakerLayout(
            left_x=LEFT_MARGIN,
            right_x=VIDEO_WIDTH - RIGHT_MARGIN,
            color=SPEAKER_1_COLOR,
            label_x=LEFT_MARGIN,
            label_alignment=1  # Left aligned
        )
        if speakers:
            return {speakers[0]: default_layout}
        return {"default": default_layout}
    
    # Two speaker mode - split screen
    return {
        speakers[0]: SpeakerLayout(
            left_x=LEFT_MARGIN,
            right_x=LEFT_SPEAKER_MAX_X,
            color=SPEAKER_1_COLOR,
            label_x=LEFT_MARGIN,
            label_alignment=1  # Left aligned
        ),
        speakers[1]: SpeakerLayout(
            left_x=RIGHT_SPEAKER_MIN_X,
            right_x=RIGHT_SPEAKER_MAX_X,
            color=SPEAKER_2_COLOR,
            label_x=VIDEO_WIDTH - RIGHT_MARGIN,
            label_alignment=3  # Right aligned
        )
    }


def calculate_display_words(
    word_segments: List[WordSegment],
    speaker_layouts: Optional[dict[str, SpeakerLayout]] = None
) -> List[DisplayWord]:
    """
    Calculate display timing and positions for all words.
    
    Words spoken within MIN_DISPLAY_DURATION of each other are grouped horizontally.
    When a gap >= MIN_DISPLAY_DURATION occurs, position resets to left margin.
    When words would exceed right margin, they wrap to the next row.
    
    In two-speaker mode, each speaker has independent position tracking within
    their designated screen area.
    
    Args:
        word_segments: List of WordSegment objects with timing info
        speaker_layouts: Optional dict mapping speaker names to SpeakerLayout objects
        
    Returns:
        List of DisplayWord objects with positions calculated
    """
    if not word_segments:
        return []
    
    # Default layout if none provided
    if speaker_layouts is None:
        speaker_layouts = {
            "default": SpeakerLayout(
                left_x=LEFT_MARGIN,
                right_x=VIDEO_WIDTH - RIGHT_MARGIN,
                color=SPEAKER_1_COLOR,
                label_x=LEFT_MARGIN,
                label_alignment=1
            )
        }
    
    display_words = []
    
    # Track state per speaker for two-speaker mode
    speaker_state: dict[str, dict] = {}
    for speaker_name in speaker_layouts:
        layout = speaker_layouts[speaker_name]
        speaker_state[speaker_name] = {
            "current_x": layout.left_x,
            "current_row": 0,
            "current_group_id": 0,
            "prev_word_start": None,
        }
    
    # Global group ID counter (increments across all speakers)
    global_group_id = 0
    
    for word in word_segments:
        # Calculate display duration (at least MIN_DISPLAY_DURATION)
        actual_duration = word.end - word.start
        display_duration = max(actual_duration, MIN_DISPLAY_DURATION_SEC)
        display_until = word.start + display_duration
        
        # Get speaker's layout and state
        speaker = word.speaker if word.speaker in speaker_layouts else list(speaker_layouts.keys())[0]
        layout = speaker_layouts[speaker]
        state = speaker_state[speaker]
        
        # Determine if this starts a new group for this speaker
        if state["prev_word_start"] is not None:
            gap_from_prev = word.start - state["prev_word_start"]
            if gap_from_prev >= MIN_DISPLAY_DURATION_SEC:
                # New group - reset to speaker's left margin
                state["current_x"] = layout.left_x
                state["current_row"] = 0
                global_group_id += 1
                state["current_group_id"] = global_group_id
        
        # Estimate width of this word
        word_width = estimate_word_width(word.text)
        
        # Check if word fits on current row within speaker's area
        if state["current_x"] + word_width > layout.right_x:
            # Wrap to next row
            state["current_row"] += 1
            state["current_x"] = layout.left_x
            
            # If we've exceeded max rows, wrap back to top
            if state["current_row"] >= MAX_ROWS:
                state["current_row"] = 0
        
        # Create display word
        display_word = DisplayWord(
            text=word.text,
            start=word.start,
            display_until=display_until,
            x_position=state["current_x"],
            row=state["current_row"],
            group_id=state["current_group_id"],
            speaker=speaker
        )
        display_words.append(display_word)
        
        # Update state for next word from this speaker
        state["current_x"] += word_width + WORD_SPACING
        state["prev_word_start"] = word.start
    
    return display_words


def generate_ass_subtitle(
    display_words: List[DisplayWord],
    speakers: List[str],
    speaker_layouts: dict[str, SpeakerLayout],
    total_duration: float,
    is_two_speaker_mode: bool
) -> str:
    """
    Generate an ASS (Advanced SubStation Alpha) subtitle file content.
    
    ASS format allows precise positioning and timing of text, which is needed
    for our word-by-word display with specific positions.
    
    Args:
        display_words: List of DisplayWord objects with timing and positions
        speakers: List of speaker names (ordered by first appearance)
        speaker_layouts: Dict mapping speaker names to their layout config
        total_duration: Total duration of the audio in seconds
        is_two_speaker_mode: Whether we're in split-screen mode
        
    Returns:
        String content of the ASS subtitle file
    """
    # ASS header with style definitions
    # Create styles for each speaker's words (with their color)
    styles_section = f"""[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Word,Courier New,{FONT_SIZE},{SPEAKER_1_COLOR},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,7,0,0,0,1
Style: Info,Arial,32,&H00AAAAAA,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,1,20,20,20,1
Style: InfoRight,Arial,32,&H00AAAAAA,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,1,0,3,20,20,20,1"""
    
    # Add speaker-specific word styles if in two-speaker mode
    if is_two_speaker_mode and len(speakers) >= 2:
        for speaker_name in speakers[:2]:
            if speaker_name in speaker_layouts:
                layout = speaker_layouts[speaker_name]
                style_name = f"Word_{speaker_name.replace(' ', '_')}"
                styles_section += f"\nStyle: {style_name},Courier New,{FONT_SIZE},{layout.color},&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,7,0,0,0,1"
    
    ass_content = f"""[Script Info]
Title: Debug Alignment Video
ScriptType: v4.00+
WrapStyle: 0
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
ScaledBorderAndShadow: yes

{styles_section}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    # Add speaker labels
    end_time = format_ass_time(total_duration)
    
    if is_two_speaker_mode and len(speakers) >= 2:
        # Two speaker mode - labels on left and right
        ass_content += f"Dialogue: 0,0:00:00.00,{end_time},Info,,0,0,0,,Speaker: {speakers[0]}\n"
        ass_content += f"Dialogue: 0,0:00:00.00,{end_time},InfoRight,,0,0,0,,Speaker: {speakers[1]}\n"
    elif speakers:
        # Single speaker mode - only show label if speaker name is valid
        speaker_name = speakers[0] if speakers else None
        if speaker_name and speaker_name not in ("None", "null", ""):
            ass_content += f"Dialogue: 0,0:00:00.00,{end_time},Info,,0,0,0,,Speaker: {speaker_name}\n"
    
    # Add each word as a positioned dialogue event
    for word in display_words:
        start_time = format_ass_time(word.start)
        word_end_time = format_ass_time(word.display_until)
        
        # Calculate Y position based on row
        y_pos = TOP_MARGIN + (word.row * ROW_HEIGHT)
        
        # Determine which style to use based on speaker
        if is_two_speaker_mode and word.speaker in speaker_layouts:
            style_name = f"Word_{word.speaker.replace(' ', '_')}"
        else:
            style_name = "Word"
        
        # Use ASS positioning tag: {\pos(x,y)}
        # Escape special characters in word text
        escaped_text = escape_ass_text(word.text)
        positioned_text = f"{{\\pos({word.x_position},{y_pos})}}{escaped_text}"
        
        ass_content += f"Dialogue: 1,{start_time},{word_end_time},{style_name},,0,0,0,,{positioned_text}\n"
    
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


def load_word_segments_from_json(json_path: Path, speaker_name: str = "Speaker") -> List[WordSegment]:
    """
    Load word segments from a JSON file.
    
    The JSON file should have a 'words' array with objects containing:
    - text: string
    - start: float (seconds)
    - end: float (seconds)
    - speaker: string (optional)
    
    Args:
        json_path: Path to JSON file with word segments
        speaker_name: Default speaker name if not in JSON
        
    Returns:
        List of WordSegment objects
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = data.get('words', data)  # Handle both {words: [...]} and [...]
    
    return [
        WordSegment(
            text=w['text'],
            start=w['start'],
            end=w['end'],
            speaker=w.get('speaker', speaker_name)
        )
        for w in words
    ]


def render_debug_video(
    word_segments: List[WordSegment],
    audio_path: Path,
    output_path: Path,
    speaker_name: str = "Unknown",
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    verbose: bool = False
) -> None:
    """
    Create a debug video showing words appearing as they are spoken.
    
    This video is designed for verifying alignment accuracy. Words appear
    at their start timestamp and stay visible for at least 100ms. Words
    spoken in rapid succession stack horizontally, then wrap to new rows.
    
    In two-speaker mode (when exactly 2 speakers are detected), each speaker's
    words appear on their respective half of the screen with different colors.
    
    Args:
        word_segments: List of WordSegment objects with timing info
        audio_path: Path to the speaker's audio file
        output_path: Output MP4 file path
        speaker_name: Name of the speaker for display (used as fallback)
        width: Video width (default 1920)
        height: Video height (default 1080)
        verbose: Print detailed progress
    """
    if not word_segments:
        raise ValueError("No word segments provided")
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Detect speakers and determine mode
    speakers, is_two_speaker_mode = detect_speakers(word_segments)
    
    # Use command-line speaker name as fallback if no speakers detected
    if not speakers:
        speakers = [speaker_name] if speaker_name and speaker_name not in ("None", "null", "") else []
    
    if verbose:
        if is_two_speaker_mode:
            print(f"  Two-speaker mode: {speakers[0]} (left) vs {speakers[1]} (right)")
        elif speakers:
            print(f"  Single-speaker mode: {speakers[0]}")
        else:
            print(f"  No speaker labels")
    
    # Get layout configuration for speakers
    speaker_layouts = get_speaker_layouts(speakers, is_two_speaker_mode)
    
    # Calculate display words with positions
    display_words = calculate_display_words(word_segments, speaker_layouts)
    
    if verbose:
        print(f"  Processed {len(display_words)} words")
        group_ids = set(w.group_id for w in display_words)
        print(f"  Created {len(group_ids)} display groups")
    
    # Get total duration from the last word
    total_duration = max(w.display_until for w in display_words)
    
    if verbose:
        print(f"  Video duration: {total_duration:.1f}s")
    
    # Generate ASS subtitle content
    ass_content = generate_ass_subtitle(
        display_words, speakers, speaker_layouts, total_duration, is_two_speaker_mode
    )
    
    # Write ASS file to temp location
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as f:
        f.write(ass_content)
        ass_path = f.name
    
    try:
        if verbose:
            print(f"  Running FFmpeg...")
        
        # Build FFmpeg command
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
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        if verbose:
            print(f"  FFmpeg completed successfully")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e
        
    finally:
        # Clean up temp ASS file
        if os.path.exists(ass_path):
            os.unlink(ass_path)


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(
        description="Create a debug video showing words appearing as they are spoken, "
                    "synchronized with alignment timestamps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_alignment_video.py words.json audio.wav output.mp4
  python debug_alignment_video.py words.json audio.m4a video.mp4 --speaker "Participant"
  python debug_alignment_video.py data.json audio.mp3 out.mp4 -v

The JSON file should contain word segments with 'text', 'start', and 'end' fields.
Supported formats:
  {"words": [{"text": "hello", "start": 0.0, "end": 0.5}, ...]}
  [{"text": "hello", "start": 0.0, "end": 0.5}, ...]
"""
    )
    
    parser.add_argument("json_path", type=Path, help="Path to JSON file with word segments")
    parser.add_argument("audio_path", type=Path, help="Path to audio file")
    parser.add_argument("output_path", type=Path, help="Output MP4 file path")
    parser.add_argument("--speaker", "-s", default="Speaker", help="Speaker name to display (default: Speaker)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.json_path.exists():
        print(f"Error: JSON file not found: {args.json_path}")
        return 1
    
    if not args.audio_path.exists():
        print(f"Error: Audio file not found: {args.audio_path}")
        return 1
    
    print(f"Creating debug alignment video...")
    print(f"  JSON: {args.json_path}")
    print(f"  Audio: {args.audio_path}")
    print(f"  Output: {args.output_path}")
    print(f"  Speaker: {args.speaker}")
    print()
    
    try:
        # Load word segments
        word_segments = load_word_segments_from_json(args.json_path, args.speaker)
        print(f"Loaded {len(word_segments)} words from JSON")
        
        # Create output directory if needed
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Render the video
        render_debug_video(
            word_segments=word_segments,
            audio_path=args.audio_path,
            output_path=args.output_path,
            speaker_name=args.speaker,
            verbose=args.verbose
        )
        
        print()
        print(f"Done! Video saved to: {args.output_path}")
        print(f"  File size: {args.output_path.stat().st_size / 1024:.1f} KB")
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
