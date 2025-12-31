"""
CLI interface for transcript comparison.

Usage:
    uv run python -m tools.transcript_compare.cli compare file_a.json file_b.json
    uv run python -m tools.transcript_compare.cli compare file_a.json file_b.json --audio audio.m4a
    uv run python -m tools.transcript_compare.cli web
    uv run python -m tools.transcript_compare.cli web --port 5050
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .extractor import extract_from_file
from .diff_engine import compute_diff, generate_unified_diff, DiffType


def format_percentage(value: float) -> str:
    """Format a percentage value."""
    return f"{value:.1f}%"


def print_header(text: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}\n")


def print_statistics(result) -> None:
    """Print comparison statistics."""
    print_header("📊 COMPARISON STATISTICS")
    
    print(f"  Transcript A: {result.total_words_a:,} words")
    print(f"  Transcript B: {result.total_words_b:,} words")
    print()
    print(f"  ✓ Matching words:    {result.matching_words:,}")
    print(f"  + Inserted (in B):   {result.inserted_words:,}")
    print(f"  - Deleted (in A):    {result.deleted_words:,}")
    print(f"  ~ Replaced (A→B):    {result.replaced_words_a:,} → {result.replaced_words_b:,}")
    print()
    print(f"  Similarity:          {format_percentage(result.similarity_ratio * 100)}")
    print(f"  Word Error Rate:     {format_percentage(result.word_error_rate * 100)}")


def print_substitutions(result, max_items: int = 15) -> None:
    """Print common word substitutions."""
    if not result.common_substitutions:
        return
    
    print_header("🔄 COMMON SUBSTITUTIONS", "-")
    print(f"  {'Word in A':<20} {'Word in B':<20} {'Count':>8}")
    print(f"  {'-'*20} {'-'*20} {'-'*8}")
    
    for word_a, word_b, count in result.common_substitutions[:max_items]:
        print(f"  {word_a:<20} {word_b:<20} {count:>8}")


def print_unique_words(result, max_items: int = 15) -> None:
    """Print words unique to each transcript."""
    print_header("📝 WORDS UNIQUE TO EACH TRANSCRIPT", "-")
    
    if result.unique_to_a:
        print("  Words only in Transcript A:")
        for word, count in result.unique_to_a[:max_items]:
            print(f"    - {word} ({count}x)")
    
    print()
    
    if result.unique_to_b:
        print("  Words only in Transcript B:")
        for word, count in result.unique_to_b[:max_items]:
            print(f"    + {word} ({count}x)")


def print_diff_segments(result, max_segments: int = 30, context_words: int = 3) -> None:
    """Print individual diff segments with context."""
    print_header("🔍 DETAILED DIFFERENCES", "-")
    
    diff_count = 0
    for i, seg in enumerate(result.segments):
        if seg.diff_type == DiffType.EQUAL:
            continue
        
        diff_count += 1
        if diff_count > max_segments:
            remaining = sum(1 for s in result.segments[i:] if s.diff_type != DiffType.EQUAL)
            print(f"\n  ... and {remaining} more differences")
            break
        
        # Get context
        before_a = result.words_a[max(0, seg.position_a - context_words):seg.position_a]
        after_a = result.words_a[seg.position_a + seg.length_a:seg.position_a + seg.length_a + context_words]
        
        before_b = result.words_b[max(0, seg.position_b - context_words):seg.position_b]
        after_b = result.words_b[seg.position_b + seg.length_b:seg.position_b + seg.length_b + context_words]
        
        type_symbol = {
            DiffType.INSERT: "+",
            DiffType.DELETE: "-",
            DiffType.REPLACE: "~",
        }[seg.diff_type]
        
        print(f"\n  [{diff_count}] {seg.diff_type.value.upper()} at position A:{seg.position_a}, B:{seg.position_b}")
        
        if seg.diff_type == DiffType.REPLACE:
            print(f"      A: ...{' '.join(before_a)} [{' '.join(seg.words_a)}] {' '.join(after_a)}...")
            print(f"      B: ...{' '.join(before_b)} [{' '.join(seg.words_b)}] {' '.join(after_b)}...")
        elif seg.diff_type == DiffType.DELETE:
            print(f"      A: ...{' '.join(before_a)} [{' '.join(seg.words_a)}] {' '.join(after_a)}...")
            print(f"      B: (missing)")
        elif seg.diff_type == DiffType.INSERT:
            print(f"      A: (missing)")
            print(f"      B: ...{' '.join(before_b)} [{' '.join(seg.words_b)}] {' '.join(after_b)}...")


def cmd_compare(args) -> int:
    """Run transcript comparison."""
    try:
        # Load transcripts
        print(f"Loading transcript A: {args.file_a}")
        transcript_a = extract_from_file(args.file_a)
        print(f"  → {transcript_a.word_count:,} words ({transcript_a.format_type})")
        
        print(f"Loading transcript B: {args.file_b}")
        transcript_b = extract_from_file(args.file_b)
        print(f"  → {transcript_b.word_count:,} words ({transcript_b.format_type})")
        
        # Run comparison
        print("\nComparing transcripts...")
        result = compute_diff(transcript_a.words, transcript_b.words)
        
        # Print results
        print_statistics(result)
        print_substitutions(result)
        print_unique_words(result)
        
        if args.detailed:
            print_diff_segments(result, max_segments=args.max_diffs)
        
        if args.unified:
            print_header("📋 UNIFIED DIFF VIEW", "-")
            print(generate_unified_diff(result))
        
        # Summary
        print_header("✅ COMPARISON COMPLETE")
        print(f"  Files compared: {Path(args.file_a).name} vs {Path(args.file_b).name}")
        print(f"  Similarity: {format_percentage(result.similarity_ratio * 100)}")
        
        total_diffs = result.inserted_words + result.deleted_words + max(result.replaced_words_a, result.replaced_words_b)
        print(f"  Total differences: {total_diffs:,} words")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error during comparison: {e}", file=sys.stderr)
        return 1


def cmd_web(args) -> int:
    """Start the web interface."""
    from .web_app import run_server
    
    print(f"Starting web interface on port {args.port}...")
    
    if args.file_a or args.file_b:
        # Pre-load files if specified
        from .web_app import current_state
        
        if args.file_a:
            print(f"Pre-loading transcript A: {args.file_a}")
            current_state["transcript_a"] = extract_from_file(args.file_a)
        
        if args.file_b:
            print(f"Pre-loading transcript B: {args.file_b}")
            current_state["transcript_b"] = extract_from_file(args.file_b)
        
        if args.audio:
            print(f"Pre-loading audio: {args.audio}")
            current_state["audio_file"] = str(Path(args.audio).absolute())
    
    run_server(host=args.host, port=args.port, debug=args.debug)
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="transcript-compare",
        description="Compare and analyze transcription differences",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two transcript files",
    )
    compare_parser.add_argument("file_a", help="First transcript file (JSON)")
    compare_parser.add_argument("file_b", help="Second transcript file (JSON)")
    compare_parser.add_argument(
        "-d", "--detailed",
        action="store_true",
        help="Show detailed diff segments",
    )
    compare_parser.add_argument(
        "-u", "--unified",
        action="store_true",
        help="Show unified diff format",
    )
    compare_parser.add_argument(
        "--max-diffs",
        type=int,
        default=30,
        help="Maximum number of diff segments to show (default: 30)",
    )
    
    # Web command
    web_parser = subparsers.add_parser(
        "web",
        help="Start the web interface",
    )
    web_parser.add_argument(
        "-p", "--port",
        type=int,
        default=5050,
        help="Port to run the server on (default: 5050)",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Run in debug mode (default: True)",
    )
    web_parser.add_argument(
        "--file-a",
        help="Pre-load transcript A",
    )
    web_parser.add_argument(
        "--file-b",
        help="Pre-load transcript B",
    )
    web_parser.add_argument(
        "--audio",
        help="Pre-load audio file",
    )
    
    args = parser.parse_args()
    
    if args.command == "compare":
        return cmd_compare(args)
    elif args.command == "web":
        return cmd_web(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
