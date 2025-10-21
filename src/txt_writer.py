# src/txt_writer.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path


def _fmt_ts(t: float) -> str:
    """Format seconds -> 00:00:00.000 for human-readable timestamps."""
    if t < 0:
        t = 0.0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_timestamped_txt(turns: List[Dict], path: str | Path) -> None:
    """Write a text file with [timestamp] Speaker: text format."""
    path = Path(path)
    lines: list[str] = []
    for t in turns:
        ts = _fmt_ts(float(t["start"]))
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()
        lines.append(f"[{ts}] {speaker}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plain_txt(turns: List[Dict], path: str | Path) -> None:
    """Write a plain transcript without timestamps."""
    path = Path(path)
    lines: list[str] = []
    for t in turns:
        speaker = t.get("speaker", "Unknown")
        text = t.get("text", "").strip()
        lines.append(f"{speaker}: {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
