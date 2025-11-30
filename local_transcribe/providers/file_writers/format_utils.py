#!/usr/bin/env python3
"""
Shared formatting utilities for file writers.

This module provides common formatting functions used across multiple
output writers for the hierarchical TranscriptFlow data structure.
"""

from typing import Optional
import textwrap


def format_timestamp(seconds: float, style: str = "hms") -> str:
    """
    Format seconds into a human-readable timestamp.
    
    Args:
        seconds: Time in seconds
        style: Format style:
            - "hms": 00:00:00.000 (hours:minutes:seconds.milliseconds)
            - "ms": 00:00.000 (minutes:seconds.milliseconds)
            - "seconds": 123.45s (raw seconds with 's' suffix)
            - "compact": 1h 15m 30s (human readable)
    
    Returns:
        Formatted timestamp string
    """
    if seconds < 0:
        seconds = 0.0
    
    if style == "seconds":
        return f"{seconds:.2f}s"
    
    ms = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    
    if style == "hms":
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    elif style == "ms":
        # Include hours in minutes if > 0
        total_m = h * 60 + m
        return f"{total_m:02d}:{s:02d}.{ms:03d}"
    elif style == "compact":
        parts = []
        if h > 0:
            parts.append(f"{h}h")
        if m > 0 or h > 0:
            parts.append(f"{m}m")
        parts.append(f"{s}s")
        return " ".join(parts)
    else:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Human-readable duration (e.g., "1 hour 15 minutes 30 seconds")
    """
    if seconds < 0:
        seconds = 0.0
    
    total_seconds = int(seconds)
    s = total_seconds % 60
    m = (total_seconds // 60) % 60
    h = total_seconds // 3600
    
    parts = []
    if h > 0:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m > 0:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s > 0 or not parts:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    
    return " ".join(parts)


def format_speaker_name(speaker: str, capitalize: bool = True) -> str:
    """
    Format a speaker name for display.
    
    Handles SPEAKER_XX format and applies title case.
    
    Args:
        speaker: Raw speaker name
        capitalize: Whether to capitalize the name
    
    Returns:
        Formatted speaker name
    """
    if not speaker:
        return "Unknown"
    
    # Handle SPEAKER_XX format - keep as is but cleaner
    if speaker.upper().startswith("SPEAKER_"):
        return speaker.upper()
    
    if capitalize:
        return speaker.title()
    
    return speaker


def get_interjection_symbol(interjection_type: str) -> str:
    """
    Get a symbol/emoji for an interjection type.
    
    Args:
        interjection_type: Type of interjection
    
    Returns:
        Symbol representing the interjection type
    """
    symbols = {
        "acknowledgment": "✓",
        "question": "?",
        "reaction": "!",
        "unclear": "·",
        "overlap": "≈"
    }
    return symbols.get(interjection_type.lower(), "·")


def get_interjection_verb(interjection_type: str) -> str:
    """
    Get a verb describing the interjection action.
    
    Args:
        interjection_type: Type of interjection
    
    Returns:
        Verb describing the action (e.g., "acknowledges", "questions")
    """
    verbs = {
        "acknowledgment": "acknowledges",
        "question": "questions",
        "reaction": "reacts",
        "unclear": "interjects",
        "overlap": "overlaps"
    }
    return verbs.get(interjection_type.lower(), "interjects")


def wrap_text(text: str, width: int = 70, initial_indent: str = "", subsequent_indent: str = "") -> str:
    """
    Word wrap text with proper indentation.
    
    Args:
        text: Text to wrap
        width: Maximum line width
        initial_indent: Indent for first line
        subsequent_indent: Indent for subsequent lines
    
    Returns:
        Wrapped text with indentation
    """
    if not text:
        return ""
    
    wrapper = textwrap.TextWrapper(
        width=width,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False
    )
    
    return wrapper.fill(text)


def escape_markdown(text: str) -> str:
    """
    Escape special Markdown characters in text.
    
    Args:
        text: Text to escape
    
    Returns:
        Text with Markdown special characters escaped
    """
    # Characters that need escaping in Markdown
    special_chars = ['\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!', '|']
    
    result = text
    for char in special_chars:
        result = result.replace(char, '\\' + char)
    
    return result


def escape_html(text: str) -> str:
    """
    Escape HTML special characters in text.
    
    Args:
        text: Text to escape
    
    Returns:
        Text with HTML special characters escaped
    """
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#x27;'
    }
    
    result = text
    for char, replacement in replacements.items():
        result = result.replace(char, replacement)
    
    return result


def get_speaker_color(speaker: str, speaker_list: list) -> str:
    """
    Get a consistent color for a speaker based on their position in the speaker list.
    
    Args:
        speaker: Speaker name
        speaker_list: List of all speakers (for consistent ordering)
    
    Returns:
        CSS color string
    """
    # Color palette designed for accessibility and distinction
    colors = [
        "#4A90D9",  # Blue
        "#D94A4A",  # Red  
        "#4AD94A",  # Green
        "#D9A84A",  # Orange
        "#9B4AD9",  # Purple
        "#4AD9D9",  # Cyan
        "#D94A9B",  # Pink
        "#7AD94A",  # Lime
    ]
    
    try:
        idx = speaker_list.index(speaker)
        return colors[idx % len(colors)]
    except ValueError:
        return colors[0]


def calculate_position_percent(time: float, total_duration: float, start_offset: float = 0) -> float:
    """
    Calculate percentage position for timeline visualization.
    
    Args:
        time: Time in seconds
        total_duration: Total duration in seconds
        start_offset: Start time offset in seconds
    
    Returns:
        Position as percentage (0-100)
    """
    if total_duration <= 0:
        return 0.0
    
    adjusted_time = time - start_offset
    return max(0, min(100, (adjusted_time / total_duration) * 100))


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places
    
    Returns:
        Formatted percentage string (e.g., "85.5%")
    """
    return f"{value * 100:.{decimals}f}%"
