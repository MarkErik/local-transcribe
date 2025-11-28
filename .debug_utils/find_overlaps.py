#!/usr/bin/env python3
"""
Utility script to detect overlapping word segments in a JSON word segments file.

What it does:
- Reads a JSON file (expected to contain a top-level key 'words' which is a list of objects)
- Detects words where `start` is before the previous word's `end` (overlap)
- Attempts to map words back to their line numbers in the original JSON file
  by scanning the file sequentially for the JSON representation of the word's text.
- Writes a human-readable report to a text file (default: <input>-overlaps.txt)

Usage:
    python .debug_utils/find_overlaps.py samples/participant_word_segments_deidentified.json

Options:
    --context N     Number of surrounding words to include around each overlap (default 2)
    --out FILE      Output report file; if not provided, a default is used.

Notes:
- Line mapping uses simple sequential scan of the file to match `"text": <value>`
  so it should be robust for the usual formatted JSON where each word object is
  written in sequence.
"""

import argparse
import json
from pathlib import Path
import re
import sys


def find_text_line_indices(lines, words):
    """
    Given the file lines and parsed words list, locate the line number where the
    `"text": <value>` pair occurs for each word sequentially. We return a list
    of line indices (1-based) for each word. If a word can't be found, we add
    None at that index.

    This performs a sequential (not global) search so duplicate words are
    matched in order of appearance.
    """
    line_idx = 0
    indices = []

    # Pre-join lines for fast searching? We'll scan line by line so we can return line numbers.
    for w in words:
        text_repr = json.dumps(w.get("text", ""))  # json.dumps ensures the same escaping
        pattern = f'"text": {text_repr}'
        found = False

        # Search from current position forward
        while line_idx < len(lines):
            if pattern in lines[line_idx]:
                indices.append(line_idx + 1)  # 1-based
                found = True
                break
            line_idx += 1

        if not found:
            indices.append(None)

    return indices


def find_object_start_line(lines, text_line_index):
    """
    Given the line index (1-based) where the "text": entry was found, try to
    find the start of its JSON object by searching upward for a line containing '{'.

    Returns the 1-based line index of the object start, or the original line
    if not found.
    """
    if text_line_index is None:
        return None

    li = text_line_index - 1  # 0-based
    while li >= 0:
        if "{" in lines[li]:
            return li + 1
        li -= 1
    return text_line_index


def find_overlaps(words):
    """
    Return list of overlap reports (tuples) for pairs where words[i].start < words[i-1].end
    Each return entry is a dict with details about overlapping pair.
    """
    overlaps = []
    prev = None
    for i, w in enumerate(words):
        if prev is not None:
            try:
                if w.get("start") is not None and prev.get("end") is not None and w["start"] < prev["end"]:
                    overlaps.append({
                        "index_prev": i - 1,
                        "index_curr": i,
                        "prev": prev,
                        "curr": w,
                    })
            except Exception as e:
                # If any keys missing or not comparable, skip
                pass
        prev = w
    return overlaps


def report_overlaps(json_path: Path, context=2, out_path: Path = None):
    if out_path is None:
        out_path = json_path.parent / (json_path.stem + "-overlaps.txt")

    # Read file
    with json_path.open("r", encoding="utf8") as fh:
        lines = fh.readlines()
        fh.seek(0)
        try:
            data = json.load(open(json_path, "r", encoding="utf8"))
        except Exception as e:
            print(f"Unable to parse JSON from {json_path}: {e}")
            raise

    # Accept top-level key 'words' or just use root list
    if isinstance(data, dict) and "words" in data:
        words = data["words"]
    elif isinstance(data, list):
        words = data
    else:
        raise ValueError("JSON format not recognized: expected dict with 'words' or top-level list.")

    text_line_indices = find_text_line_indices(lines, words)
    object_start_lines = [find_object_start_line(lines, li) for li in text_line_indices]

    overlaps = find_overlaps(words)

    # Report
    with out_path.open("w", encoding="utf8") as out:
        out.write(f"Overlap report for: {json_path}\n")
        out.write(f"Total words: {len(words)}\n")
        out.write(f"Overlaps found: {len(overlaps)}\n")
        out.write("\n")

        for idx, ov in enumerate(overlaps, start=1):
            i_prev = ov["index_prev"]
            i_curr = ov["index_curr"]

            # Compute context slice
            start_idx = max(0, i_prev - context)
            end_idx = min(len(words), i_curr + context + 1)

            out.write(f"Overlap #{idx}\n")
            out.write(f"  Pair indices: {i_prev} -> {i_curr}\n")
            out.write(f"  Prev line: {object_start_lines[i_prev]} (text line: {text_line_indices[i_prev]})\n")
            out.write(f"  Curr line: {object_start_lines[i_curr]} (text line: {text_line_indices[i_curr]})\n")
            out.write("  --- Context ---\n")
            for j in range(start_idx, end_idx):
                w = words[j]
                w_text = w.get("text", "")
                w_start = w.get("start")
                w_end = w.get("end")
                w_speaker = w.get("speaker")
                obj_line = object_start_lines[j]
                text_line = text_line_indices[j]
                prefix = "*" if j in (i_prev, i_curr) else " "
                out.write(f"  {prefix} idx={j} text={w_text!r} start={w_start} end={w_end} speaker={w_speaker} obj_line={obj_line} text_line={text_line}\n")
            out.write("\n")

    print(f"Saved overlaps to: {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=Path, help="Path to the input JSON file")
    parser.add_argument("--context", type=int, default=2, help="Surrounding words to include in the report")
    parser.add_argument("--out", type=Path, default=None, help="Output report file path")
    args = parser.parse_args(argv)

    try:
        out_path = report_overlaps(args.json_file, context=args.context, out_path=args.out)
    except Exception as e:
        print(f"Error while reporting overlaps: {e}")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
