#!/usr/bin/env python3
"""
Build turns from multiple participant word segment JSON files.

This script takes multiple word segment JSON files (one per participant/speaker),
processes them through the SplitAudioTurnBuilder, and outputs a hierarchical
turn JSON file.

Usage:
    python build_turns_from_segments.py \
        --input file1.json file2.json [file3.json ...] \
        --output turns.json \
        --llm-url http://localhost:8080

Example:
    python build_turns_from_segments.py \
        --input transcript-sample/interviewer_word_segments_deidentified.json \
                transcript-sample/participant_word_segments_deidentified.json \
        --output transcript-sample/turns_output.json \
        --llm-url http://100.84.208.72:8083
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from local_transcribe.framework.plugin_interfaces import WordSegment
from local_transcribe.processing.turn_building.split_audio_llm_turn_builder import SplitAudioTurnBuilder


def load_word_segments(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load word segments from a JSON file.
    
    Expected format:
    {
        "metadata": {...},
        "words": [
            {"text": "hello", "start": 0.0, "end": 0.5, "speaker": "Speaker1"},
            ...
        ]
    }
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of word segment dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'words' in data:
        return data['words']
    elif isinstance(data, list):
        # Support flat list format as well
        return data
    else:
        raise ValueError(f"Unrecognized format in {file_path}. Expected 'words' key or a list.")


def dict_to_word_segment(word_dict: Dict[str, Any]) -> WordSegment:
    """Convert a dictionary to a WordSegment object."""
    return WordSegment(
        text=word_dict['text'],
        start=word_dict['start'],
        end=word_dict['end'],
        speaker=word_dict.get('speaker', 'Unknown')
    )


def build_output(
    transcript_flow,
    builder: SplitAudioTurnBuilder,
    input_files: List[Path],
    processing_time: float,
    llm_url: str,
    time_range: tuple = None
) -> Dict[str, Any]:
    """
    Build the output dictionary from the TranscriptFlow.
    
    Args:
        transcript_flow: The TranscriptFlow object from the turn builder
        builder: The SplitAudioTurnBuilder instance (for stats)
        input_files: List of input file paths
        processing_time: Time taken to process in seconds
        llm_url: The LLM URL used
        time_range: Optional (start, end) time range if filtered
        
    Returns:
        Dictionary ready for JSON serialization
    """
    output = {
        'metadata': {
            'input_files': [str(f) for f in input_files],
            'processing_time_seconds': round(processing_time, 2),
            'llm_url': llm_url,
            'llm_stats': builder.llm_stats,
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
        'summary': {
            'total_turns': transcript_flow.total_turns,
            'total_interjections': transcript_flow.total_interjections,
            'conversation_metrics': transcript_flow.conversation_metrics,
        },
        'turns': []
    }
    
    if time_range:
        output['metadata']['time_range'] = {
            'start_seconds': time_range[0],
            'end_seconds': time_range[1]
        }
    
    for turn in transcript_flow.turns:
        turn_data = {
            'turn_id': turn.turn_id,
            'primary_speaker': turn.primary_speaker,
            'start': turn.start,
            'end': turn.end,
            'duration': turn.duration,
            'word_count': turn.word_count,
            'text': turn.text,
            'turn_type': turn.turn_type,
            'flow_continuity': turn.flow_continuity,
            'interjections': []
        }
        
        for ij in turn.interjections:
            turn_data['interjections'].append({
                'speaker': ij.speaker,
                'start': ij.start,
                'end': ij.end,
                'text': ij.text,
                'interjection_type': ij.interjection_type,
                'confidence': ij.confidence,
                'classification_method': ij.classification_method,
            })
        
        output['turns'].append(turn_data)
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Build turns from multiple participant word segment JSON files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with two input files
  python build_turns_from_segments.py \\
      --input interviewer.json participant.json \\
      --output turns.json \\
      --llm-url http://localhost:8080

  # Process only the first 60 seconds
  python build_turns_from_segments.py \\
      --input interviewer.json participant.json \\
      --output turns_first_minute.json \\
      --llm-url http://localhost:8080 \\
      --end-time 60

  # Process a specific time range
  python build_turns_from_segments.py \\
      --input interviewer.json participant.json \\
      --output turns_excerpt.json \\
      --llm-url http://localhost:8080 \\
      --start-time 120 --end-time 180
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        type=Path,
        help='One or more word segment JSON files (one per speaker/participant)'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        type=Path,
        help='Output JSON file for the resulting turns'
    )
    
    parser.add_argument(
        '-u', '--llm-url',
        required=True,
        help='URL for the LLM server (e.g., http://localhost:8080)'
    )
    
    parser.add_argument(
        '--start-time',
        type=float,
        default=0.0,
        help='Start time in seconds (filter words before this time)'
    )
    
    parser.add_argument(
        '--end-time',
        type=float,
        default=None,
        help='End time in seconds (filter words after this time)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input files exist
    for input_file in args.input:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Load all word segments
    print(f"Loading word segments from {len(args.input)} files...")
    all_words = []
    
    for input_file in args.input:
        print(f"  Loading: {input_file}")
        words = load_word_segments(input_file)
        print(f"    Found {len(words)} words")
        all_words.extend(words)
    
    print(f"Total words loaded: {len(all_words)}")
    
    # Convert to WordSegment objects
    word_segments = [dict_to_word_segment(w) for w in all_words]
    
    # Apply time range filter if specified
    time_range = None
    if args.start_time > 0 or args.end_time is not None:
        original_count = len(word_segments)
        
        if args.end_time is not None:
            word_segments = [w for w in word_segments if w.start >= args.start_time and w.end <= args.end_time]
            time_range = (args.start_time, args.end_time)
            print(f"Filtered to time range [{args.start_time:.1f}s - {args.end_time:.1f}s]: {len(word_segments)} words (from {original_count})")
        else:
            word_segments = [w for w in word_segments if w.start >= args.start_time]
            time_range = (args.start_time, None)
            print(f"Filtered to start time >= {args.start_time:.1f}s: {len(word_segments)} words (from {original_count})")
    
    if not word_segments:
        print("Error: No word segments remain after filtering. Check your time range.", file=sys.stderr)
        sys.exit(1)
    
    # Initialize the turn builder
    print(f"\nInitializing SplitAudioTurnBuilder...")
    print(f"LLM endpoint: {args.llm_url}")
    
    builder = SplitAudioTurnBuilder(llm_url=args.llm_url)
    
    # Build turns
    print(f"\nBuilding turns...")
    start_time = time.time()
    
    transcript_flow = builder.build_turns(
        word_segments,
        llm_url=args.llm_url
    )
    
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    print(f"Processing time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Total turns: {transcript_flow.total_turns}")
    print(f"Total interjections: {transcript_flow.total_interjections}")
    print(f"LLM stats: {builder.llm_stats}")
    
    # Build and save output
    output = build_output(
        transcript_flow,
        builder,
        args.input,
        elapsed,
        args.llm_url,
        time_range
    )
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nOutput saved to: {args.output}")
    
    # Show preview if verbose
    if args.verbose:
        print(f"\n{'='*60}")
        print("TURN PREVIEW:")
        print(f"{'='*60}")
        for turn in transcript_flow.turns[:10]:  # Show first 10 turns
            print(f"\nTURN {turn.turn_id}: {turn.primary_speaker} [{turn.start:.1f}-{turn.end:.1f}s]")
            preview = turn.text[:80] + '...' if len(turn.text) > 80 else turn.text
            print(f'  "{preview}"')
            if turn.interjections:
                for ij in turn.interjections:
                    print(f'  -> [{ij.start:.1f}s] {ij.speaker}: "{ij.text}" ({ij.classification_method})')
        
        if len(transcript_flow.turns) > 10:
            print(f"\n... and {len(transcript_flow.turns) - 10} more turns")


if __name__ == '__main__':
    main()
