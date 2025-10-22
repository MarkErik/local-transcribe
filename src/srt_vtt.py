# src/srt_vtt.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
from logging_config import get_logger, OutputError, ErrorContext, error_context


def _fmt_ts(t: float) -> str:
    """Format seconds -> SRT timestamp 00:00:00,000."""
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ts_to_seconds(ts: str) -> float:
    """Inverse of _fmt_ts (accepts 'hh:mm:ss,ms')."""
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


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


@error_context(reraise=True)
def write_vtt(turns: List[Dict], path: str | Path) -> None:
    """
    Write WebVTT with 'Speaker: text' lines.
    
    Parameters
    ----------
    turns : List[Dict]
        List of turn dictionaries with speaker, start, end, and text
    path : str | Path
        Output file path
        
    Raises
    ------
    OutputError
        If VTT file writing fails
    """
    logger = get_logger()
    
    try:
        output_path = Path(path)
        
        # Validate inputs
        if not turns:
            raise OutputError("No turns provided for VTT output", output_path=str(output_path))
        
        logger.info(f"Writing VTT file: {output_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        lines: list[str] = ["WEBVTT", ""]
        
        for i, t in enumerate(turns):
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
                
                # VTT uses '.' as ms separator
                s_hms = _fmt_ts(start_s).replace(",", ".")
                e_hms = _fmt_ts(end_s).replace(",", ".")
                text = f"{t['speaker']}: {t['text']}".strip()
                
                lines += [f"{s_hms} --> {e_hms}", text, ""]
                
            except Exception as e:
                raise OutputError(
                    f"Failed to process turn {i} for VTT: {e}",
                    output_path=str(output_path),
                    cause=e
                )
        
        # Write file
        try:
            content = "\n".join(lines) + "\n"
            output_path.write_text(content, encoding="utf-8")
        except Exception as e:
            raise OutputError(
                f"Failed to write VTT file: {e}",
                output_path=str(output_path),
                cause=e
            )
        
        logger.info(f"Successfully wrote VTT file with {len(turns)} turns: {output_path}")
        
    except Exception as e:
        if isinstance(e, OutputError):
            logger.error(f"VTT output error: {e}")
            raise
        else:
            logger.error(f"Unexpected error writing VTT file: {e}")
            raise OutputError(
                f"Unexpected error writing VTT file: {e}",
                output_path=str(path),
                cause=e
            )
