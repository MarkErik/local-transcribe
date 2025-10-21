# src/srt_vtt.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path


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


def write_srt(turns: List[Dict], path: str | Path) -> None:
    """
    Write SRT with 'Speaker: text' lines. Ensures non-negative, non-inverted times.
    """
    path = Path(path)
    lines: list[str] = []
    for i, t in enumerate(turns, start=1):
        start_s = max(0.0, float(t["start"]))
        end_s = max(start_s, float(t["end"]))
        start = _fmt_ts(start_s)
        end = _fmt_ts(end_s)
        text = f"{t['speaker']}: {t['text']}".strip()
        lines += [str(i), f"{start} --> {end}", text, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_vtt(turns: List[Dict], path: str | Path) -> None:
    """
    Write WebVTT with 'Speaker: text' lines.
    """
    path = Path(path)
    lines: list[str] = ["WEBVTT", ""]
    for t in turns:
        start_s = max(0.0, float(t["start"]))
        end_s = max(start_s, float(t["end"]))
        # VTT uses '.' as ms separator
        s_hms = _fmt_ts(start_s).replace(",", ".")
        e_hms = _fmt_ts(end_s).replace(",", ".")
        text = f"{t['speaker']}: {t['text']}".strip()
        lines += [f"{s_hms} --> {e_hms}", text, ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
