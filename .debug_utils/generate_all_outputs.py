#!/usr/bin/env python3
"""
Generate all output formats from diarized word segments.

This script processes the diarized word segments through the turn building
pipeline and then generates all available output formats.
"""

import argparse
import json
from pathlib import Path

from local_transcribe.processing.turn_building import (
    build_turns_combined_audio,
    TurnBuilderConfig
)
from local_transcribe.providers.file_writers.annotated_markdown_writer import write_annotated_markdown
from local_transcribe.providers.file_writers.dialogue_script_writer import write_dialogue_script
from local_transcribe.providers.file_writers.html_timeline_writer import write_html_timeline
from local_transcribe.providers.file_writers.txt_writer import (
    write_timestamped_txt,
    write_plain_txt,
    _extract_turns_as_dicts as txt_extract_turns
)
from local_transcribe.providers.file_writers.json_writer import (
    _extract_turns_as_dicts as json_extract_turns
)


def main():
    parser = argparse.ArgumentParser(description="Generate all output formats from diarized word segments.")
    parser.add_argument("input_file", help="Path to the JSON file containing diarized word segments")
    parser.add_argument("--out", default="samples/all_outputs", help="Output directory for generated files")
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        return
    
    print(f"Loading word segments from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    words = data.get('words', [])
    metadata = data.get('metadata', {})
    
    print(f"Loaded {len(words)} words")
    
    # Configure turn builder
    config = TurnBuilderConfig(
        max_interjection_duration=2.0,
        max_interjection_words=5,
        max_gap_to_merge_turns=3.0
    )
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run turn building
    print("\n" + "="*60)
    print("Running turn building pipeline...")
    print("="*60 + "\n")
    
    transcript_flow = build_turns_combined_audio(
        words=words,
        intermediate_dir=output_dir,
        config=config
    )
    
    print(f"\nTranscript Flow: {transcript_flow}")
    
    # Generate all output formats
    print("\n" + "="*60)
    print("Generating output files...")
    print("="*60)
    
    # 1. Full TranscriptFlow JSON (hierarchical format)
    json_path = output_dir / "transcript_flow.json"
    with open(json_path, 'w') as f:
        json.dump(transcript_flow.to_dict(), f, indent=2)
    print(f"✓ Full transcript flow JSON: {json_path}")
    
    # 2. Simple turns JSON (flat format)
    simple_json_path = output_dir / "turns_simple.json"
    turns_dicts = json_extract_turns(transcript_flow)
    with open(simple_json_path, 'w') as f:
        json.dump({"turns": turns_dicts}, f, indent=2)
    print(f"✓ Simple turns JSON: {simple_json_path}")
    
    # 3. Annotated Markdown (rich hierarchical format)
    md_path = output_dir / "transcript_annotated.md"
    write_annotated_markdown(transcript_flow, md_path)
    print(f"✓ Annotated Markdown: {md_path}")
    
    # 4. Dialogue Script (screenplay style)
    script_path = output_dir / "transcript_script.txt"
    write_dialogue_script(transcript_flow, script_path)
    print(f"✓ Dialogue Script: {script_path}")
    
    # 5. Interactive HTML Timeline
    html_path = output_dir / "transcript_timeline.html"
    write_html_timeline(transcript_flow, html_path)
    print(f"✓ HTML Timeline: {html_path}")
    
    # 6. Timestamped TXT
    ts_txt_path = output_dir / "transcript_timestamped.txt"
    turns_for_txt = txt_extract_turns(transcript_flow)
    write_timestamped_txt(turns_for_txt, ts_txt_path)
    print(f"✓ Timestamped TXT: {ts_txt_path}")
    
    # 7. Plain TXT (merged paragraphs)
    plain_txt_path = output_dir / "transcript_plain.txt"
    write_plain_txt(turns_for_txt, plain_txt_path)
    print(f"✓ Plain TXT: {plain_txt_path}")
    
    # 8. Word segments JSON (original format with words)
    word_segments_path = output_dir / "word_segments.json"
    with open(word_segments_path, 'w') as f:
        json.dump({
            "metadata": {
                "total_words": len(words),
                "format_version": "1.0",
                "source": str(input_file)
            },
            "words": words
        }, f, indent=2)
    print(f"✓ Word Segments JSON: {word_segments_path}")
    
    # Summary
    print("\n" + "="*60)
    print("OUTPUT FILES GENERATED")
    print("="*60)
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nFiles:")
    for f in sorted(output_dir.glob("*")):
        if f.is_file() and not f.name.startswith('.'):
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
    
    print("\n" + "="*60)
    print("FORMAT DESCRIPTIONS")
    print("="*60)
    print("""
1. transcript_flow.json      - Full hierarchical JSON with interjections embedded
2. turns_simple.json         - Simplified flat JSON with just speaker/start/end/text
3. transcript_annotated.md   - Rich Markdown with statistics and inline interjections
4. transcript_script.txt     - Screenplay-style dialogue format
5. transcript_timeline.html  - Interactive HTML timeline visualization
6. transcript_timestamped.txt- Simple text with [HH:MM:SS.mmm] timestamps
7. transcript_plain.txt      - Clean text with speaker labels, no timestamps
8. word_segments.json        - Original word-level segments with timing
""")


if __name__ == "__main__":
    main()
