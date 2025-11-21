#!/usr/bin/env python3
"""
Improved transcript comparison tool using the TranscriptComparisonEngine.

This script compares two transcript CSV files with Line and Word columns,
providing detailed analysis including semantic similarity, timing alignment,
and quality metrics.
"""

import argparse
import sys
from pathlib import Path

# Add the local_transcribe module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from local_transcribe.transcript_comparison.engine import TranscriptComparisonEngine
from local_transcribe.transcript_comparison.data_structures import ComparisonConfig


def main():
    parser = argparse.ArgumentParser(
        description="Compare two transcript CSV files with detailed analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument("file1", help="Path to first transcript CSV file")
    parser.add_argument("file2", help="Path to second transcript CSV file")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output file path (if not specified, prints to stdout)")
    parser.add_argument("--format", "-f", choices=["text", "json", "html", "csv"],
                       default="text", help="Output format")
    
    # Comparison options
    parser.add_argument("--source1", default="reference", help="Name of first transcript source")
    parser.add_argument("--source2", default="hypothesis", help="Name of second transcript source")
    parser.add_argument("--alignment", choices=["standard", "hierarchical", "line"],
                       default="standard", help="Alignment strategy to use")
    
    # Configuration options
    parser.add_argument("--semantic-threshold", type=float, default=0.8,
                       help="Semantic similarity threshold (0.0-1.0)")
    parser.add_argument("--timing-tolerance", type=float, default=2.0,
                       help="Timing tolerance in seconds")
    parser.add_argument("--max-differences", type=int, default=100,
                       help="Maximum number of differences to show")
    parser.add_argument("--context-window", type=int, default=2,
                       help="Size of context window in segments")
    
    # Verbosity options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed alignment information")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Only show essential information")
    
    args = parser.parse_args()
    
    # Create comparison configuration
    config = ComparisonConfig(
        semantic_threshold=args.semantic_threshold,
        timing_tolerance=args.timing_tolerance,
        max_differences=args.max_differences,
        context_window_size=args.context_window,
        include_context=args.verbose
    )
    
    # Initialize the comparison engine
    engine = TranscriptComparisonEngine(config)
    
    try:
        print(f"Loading transcripts...")
        
        # Choose comparison method based on alignment strategy
        if args.alignment == "line":
            print(f"Using line number alignment strategy...")
            comparison_result = engine.compare_with_line_alignment(
                args.file1, args.file2, args.source1, args.source2
            )
        elif args.alignment == "hierarchical":
            print(f"Using hierarchical alignment strategy...")
            # Set hierarchical alignment in config
            config.use_hierarchical_alignment = True
            comparison_result = engine.compare_csv_files(
                args.file1, args.file2, args.source1, args.source2
            )
        else:  # standard
            print(f"Using standard alignment strategy...")
            comparison_result = engine.compare_csv_files(
                args.file1, args.file2, args.source1, args.source2
            )
        
        # Format and output results
        formatted_results = engine.format_results(comparison_result, args.format)
        
        if args.output:
            # Save to file
            engine.save_results(comparison_result, args.output, args.format)
            print(f"Results saved to: {args.output}")
        else:
            # Print to stdout
            print("\n" + "="*70)
            print(formatted_results)
        
        # Print summary if not quiet
        if not args.quiet:
            print("\n" + "-"*50)
            summary = comparison_result.summary
            metrics = comparison_result.quality_metrics
            
            print(f"SUMMARY:")
            print(f"  Total segments: {summary.total_segments}")
            print(f"  Matched segments: {summary.matched_segments} ({summary.matched_segments/max(summary.total_segments,1)*100:.1f}%)")
            print(f"  Substitutions: {summary.substituted_segments}")
            print(f"  Insertions: {summary.inserted_segments}")
            print(f"  Deletions: {summary.deleted_segments}")
            print(f"  Overall similarity: {summary.overall_similarity*100:.1f}%")
            print(f"  Word Error Rate: {metrics.word_error_rate*100:.1f}%")
            print(f"  Overall Quality: {metrics.overall_quality*100:.1f}%")
            
            if summary.total_segments == 0:
                print("\n⚠️  No segments found in transcripts. Please check the input files.")
            elif summary.matched_segments == summary.total_segments and metrics.word_error_rate < 0.01:
                print("\n✅ The transcripts are nearly identical!")
            elif metrics.overall_quality > 0.9:
                print("\n👍 The transcripts are very similar!")
            elif metrics.overall_quality > 0.7:
                print("\n👌 The transcripts are quite similar.")
            elif metrics.overall_quality > 0.5:
                print("\n🤔 The transcripts have moderate differences.")
            else:
                print("\n❌ The transcripts are significantly different.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
