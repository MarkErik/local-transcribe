#!/usr/bin/env python3
from __future__ import annotations
from typing import List, Dict, Optional
from pathlib import Path
from local_transcribe.lib.logging_config import get_logger, OutputError, error_context
from local_transcribe.framework.plugin_interfaces import WordSegment


def _fmt_ts(t: float) -> str:
    """Format seconds -> SRT timestamp 00:00:00,000."""
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@error_context(reraise=True)
def write_srt(turns: List[Dict], path: str | Path) -> None:
    """
    Write SRT with 'Speaker: text' lines. Ensures non-negative, non-inverted times.
    
    Parameters
    ----------
    turns : List[Dict]
        List of turn dictionaries with speaker, start, end, and text
    path : str | Path
        Output file path
        
    Raises
    ------
    OutputError
        If SRT file writing fails
    """
    logger = get_logger()
    
    try:
        output_path = Path(path)
        
        # Validate inputs
        if not turns:
            raise OutputError("No turns provided for SRT output", output_path=str(output_path))
        
        logger.info(f"Writing SRT file: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines: list[str] = []
        
        for i, t in enumerate(turns, start=1):
            try:
                # Validate turn data
                if not all(key in t for key in ["speaker", "start", "end", "text"]):
                    raise OutputError(
                        f"Turn {i} missing required fields: {t}",
                        output_path=str(output_path)
                    )
                
                start_s = max(0.0, float(t["start"]))
                end_s = max(start_s, float(t["end"]))
                
                # Validate timestamps
                if end_s < start_s:
                    logger.warning(f"Turn {i} has inverted timestamps, fixing: {start_s} -> {end_s}")
                    end_s = start_s + 0.1  # Add minimal duration
                
                start = _fmt_ts(start_s)
                end = _fmt_ts(end_s)
                text = f"{t['speaker']}: {t['text']}".strip()
                
                lines += [str(i), f"{start} --> {end}", text, ""]
                
            except Exception as e:
                raise OutputError(
                    f"Failed to process turn {i} for SRT: {e}",
                    output_path=str(output_path),
                    cause=e
                )
        
        # Write file
        try:
            content = "\n".join(lines) + "\n"
            output_path.write_text(content, encoding="utf-8")
        except Exception as e:
            raise OutputError(
                f"Failed to write SRT file: {e}",
                output_path=str(output_path),
                cause=e
            )
        
        logger.info(f"Successfully wrote SRT file with {len(turns)} turns: {output_path}")
        
    except Exception as e:
        if isinstance(e, OutputError):
            logger.error(f"SRT output error: {e}")
            raise
        else:
            logger.error(f"Unexpected error writing SRT file: {e}")
            raise OutputError(
                f"Unexpected error writing SRT file: {e}",
                output_path=str(path),
                cause=e
            )




# Plugin classes
from local_transcribe.framework.plugin_interfaces import OutputWriter, Turn, registry
from typing import List


class SRTWriter(OutputWriter):
    @property
    def name(self) -> str:
        return "srt"

    @property
    def description(self) -> str:
        return "SRT subtitle format"

    @property
    def supported_formats(self) -> List[str]:
        return [".srt"]

    def write(self, turns: List[Turn], output_path: str, word_segments: Optional[List[WordSegment]] = None) -> None:
        # Convert Turn to dict for compatibility
        turn_dicts = [{"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text} for t in turns]
        write_srt(turn_dicts, output_path)


# Register the SRT writer only
registry.register_output_writer(SRTWriter())
