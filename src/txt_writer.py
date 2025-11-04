# src/txt_writer.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import logging

from config import is_debug_enabled, is_info_enabled

logger = logging.getLogger(__name__)


def _process_text_for_cross_talk(text: str, turn: Dict) -> str:
    """
    Process text to mark cross-talk words with asterisks.
    
    Parameters
    ----------
    text : str
        The original text to process
    turn : Dict
        Turn dictionary containing word objects with cross-talk information
        
    Returns
    -------
    str
        Text with cross-talk words marked with asterisks
    """
    # If no words in the turn, return original text
    if "words" not in turn or not turn["words"]:
        return text
    
    words = turn["words"]
    processed_words = []
    
    try:
        # Split the original text into words
        text_words = text.split()
        
        # Create a mapping of word positions to cross-talk status
        word_positions = {}
        for i, word_obj in enumerate(words):
            if "text" in word_obj and "cross_talk" in word_obj:
                word_text = word_obj["text"].strip()
                if word_text:  # Only process non-empty words
                    word_positions[i] = word_obj.get("cross_talk", False)
        
        # Process each word in the text
        for i, word in enumerate(text_words):
            # Check if this word position has cross-talk marking
            if i in word_positions and word_positions[i]:
                processed_words.append(f"*{word}*")
            else:
                processed_words.append(word)
        
        return " ".join(processed_words)
    
    except Exception as e:
        if is_info_enabled():
            logger.error(f"Error in _process_text_for_cross_talk: {e}")
        # Return original text if processing fails
        return text


def _fmt_ts(t: float) -> str:
    """Format seconds -> 00:00:00.000 for human-readable timestamps."""
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_timestamped_txt(turns: List[Dict], path: str | Path, mark_cross_talk: bool = False) -> None:
    """Write a text file with [timestamp] Speaker: text format."""
    path = Path(path)
    lines: list[str] = []
    for t in turns:
        ts = _fmt_ts(float(t["start"]))
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()
        
        # Process text for cross-talk marking if requested
        if mark_cross_talk:
            try:
                processed_text = _process_text_for_cross_talk(text, t)
                lines.append(f"[{ts}] {speaker}: {processed_text}")
            except Exception as e:
                if is_info_enabled():
                    logger.error(f"Error processing cross-talk in timestamped text: {e}")
                # Fall back to original text if processing fails
                lines.append(f"[{ts}] {speaker}: {text}")
        else:
            lines.append(f"[{ts}] {speaker}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plain_txt(turns: List[Dict], path: str | Path, mark_cross_talk: bool = False) -> None:
    """Write a plain transcript without timestamps, merging consecutive segments from the same speaker into paragraphs."""
    path = Path(path)
    lines: list[str] = []
    
    if not turns:
        path.write_text("", encoding="utf-8")
        return

    current_speaker = turns[0].get("speaker", "Unknown")
    current_paragraph_text = []

    for t in turns:
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()

        if speaker == current_speaker:
            current_paragraph_text.append(text)
        else:
            # Speaker changed, finalize the previous speaker's paragraph
            if current_paragraph_text:  # Ensure there's text to write
                if mark_cross_talk:
                    try:
                        # Process the paragraph for cross-talk marking
                        paragraph_text = ' '.join(current_paragraph_text)
                        processed_turn = {"text": paragraph_text, "words": []}
                        
                        # Collect all words from the turns in this paragraph
                        for turn in turns:
                            if turn.get("speaker") == current_speaker and "words" in turn:
                                processed_turn["words"].extend(turn["words"])
                        
                        processed_text = _process_text_for_cross_talk(paragraph_text, processed_turn)
                        lines.append(f"{current_speaker}: {processed_text}")
                    except Exception as e:
                        if is_info_enabled():
                            logger.error(f"Error processing cross-talk in plain text paragraph: {e}")
                        # Fall back to original text if processing fails
                        lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")
                else:
                    lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")
            
            # Start a new paragraph for the new speaker
            current_speaker = speaker
            current_paragraph_text = [text]
    
    # After the loop, add the last speaker's paragraph
    if current_paragraph_text:
        if mark_cross_talk:
            try:
                # Process the paragraph for cross-talk marking
                paragraph_text = ' '.join(current_paragraph_text)
                processed_turn = {"text": paragraph_text, "words": []}
                
                # Collect all words from the turns in this paragraph
                for turn in turns:
                    if turn.get("speaker") == current_speaker and "words" in turn:
                        processed_turn["words"].extend(turn["words"])
                
                processed_text = _process_text_for_cross_talk(paragraph_text, processed_turn)
                lines.append(f"{current_speaker}: {processed_text}")
            except Exception as e:
                if is_info_enabled():
                    logger.error(f"Error processing cross-talk in plain text paragraph: {e}")
                # Fall back to original text if processing fails
                lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")
        else:
            lines.append(f"{current_speaker}: {' '.join(current_paragraph_text)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asr_words(words: List[Dict], path: str | Path, mark_cross_talk: bool = False) -> None:
    """Write ASR word-level results as plain text without timestamps or speaker labels."""
    path = Path(path)
    lines: list[str] = []
    for w in words:
        text = w.get("text", "").strip()
        if text:  # Only add non-empty text
            if mark_cross_talk and w.get("cross_talk", False):
                lines.append(f"*{text}*")
            else:
                lines.append(text)
    path.write_text(" ".join(lines) + "\n", encoding="utf-8")


def write_txt(
    turns: List[Dict],
    output_path: str,
    mark_cross_talk: bool = False,
    timestamped: bool = True
) -> None:
    """
    Write transcript text with optional cross-talk marking.
    
    Parameters
    ----------
    turns : List[Dict]
        List of turn dictionaries with speaker, text, and word information
    output_path : str
        Path to the output file
    mark_cross_talk : bool, optional
        Whether to mark cross-talk words with asterisks, by default False
    timestamped : bool, optional
        Whether to include timestamps in the output, by default True
        
    Returns
    -------
    None
    """
    try:
        if timestamped:
            write_timestamped_txt(turns, output_path, mark_cross_talk=mark_cross_talk)
        else:
            write_plain_txt(turns, output_path, mark_cross_talk=mark_cross_talk)
        
        if is_info_enabled():
            logger.info(f"Successfully wrote transcript to {output_path} with cross-talk marking: {mark_cross_talk}")
    
    except Exception as e:
        if is_info_enabled():
            logger.error(f"Error writing transcript to {output_path}: {e}")
        raise
