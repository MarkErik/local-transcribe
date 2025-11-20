#!/usr/bin/env python3
"""
Simple script to compare two transcript files with line numbers and words.
Each file should have CSV format: Line,Word
"""

import csv
import argparse
from typing import Dict, List, Tuple


def load_transcript(file_path: str) -> Dict[int, str]:
    """Load a transcript file into a dictionary mapping line numbers to words."""
    transcript = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                line_num = int(row['Line'])
                word = row['Word']
                transcript[line_num] = word
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        exit(1)
    return transcript


def compare_transcripts(transcript1: Dict[int, str], transcript2: Dict[int, str]) -> Tuple[List[int], List[Tuple[int, str, str]]]:
    """
    Compare two transcripts and return differences.
    
    Returns:
        Tuple containing:
        - List of line numbers present in transcript1 but not in transcript2
        - List of tuples (line_number, word1, word2) where words differ
    """
    missing_lines = []
    differing_words = []
    
    # Check for lines in transcript1 but not in transcript2
    all_lines = set(transcript1.keys()).union(set(transcript2.keys()))
    
    for line_num in sorted(all_lines):
        if line_num not in transcript1:
            # Line missing from transcript1
            continue
        elif line_num not in transcript2:
            # Line missing from transcript2
            missing_lines.append(line_num)
        else:
            # Compare words
            word1 = transcript1[line_num]
            word2 = transcript2[line_num]
            if word1 != word2:
                differing_words.append((line_num, word1, word2))
    
    return missing_lines, differing_words


def print_comparison_results(missing_lines: List[int], differing_words: List[Tuple[int, str, str]]):
    """Print the comparison results in a readable format."""
    if not missing_lines and not differing_words:
        print("✅ The transcripts are identical!")
        return
    
    print("\n📊 COMPARISON RESULTS:")
    print("=" * 50)
    
    if missing_lines:
        print(f"\n❌ Lines present in first transcript but missing in second ({len(missing_lines)} lines):")
        for line_num in missing_lines[:10]:  # Show first 10
            print(f"  Line {line_num}")
        if len(missing_lines) > 10:
            print(f"  ... and {len(missing_lines) - 10} more")
    
    if differing_words:
        print(f"\n🔍 Lines with different words ({len(differing_words)} lines):")
        for line_num, word1, word2 in differing_words[:10]:  # Show first 10
            print(f"  Line {line_num}: '{word1}' → '{word2}'")
        if len(differing_words) > 10:
            print(f"  ... and {len(differing_words) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Compare two transcript files")
    parser.add_argument("file1", help="Path to first transcript file")
    parser.add_argument("file2", help="Path to second transcript file")
    
    args = parser.parse_args()
    
    print(f"Loading transcripts...")
    transcript1 = load_transcript(args.file1)
    transcript2 = load_transcript(args.file2)
    
    print(f"Transcript 1: {len(transcript1)} lines")
    print(f"Transcript 2: {len(transcript2)} lines")
    
    missing_lines, differing_words = compare_transcripts(transcript1, transcript2)
    print_comparison_results(missing_lines, differing_words)


if __name__ == "__main__":
    main()
