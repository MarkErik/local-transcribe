#!/usr/bin/env python3
"""
Transcription Comparison Utility

Compares two JSON transcription files with timestamps and speakers.
Useful for evaluating different transcription methods (e.g., different VAD segmenters).

Usage:
    python compare_transcriptions.py file1.json file2.json [options]
"""

import json
import argparse
import sys
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
import difflib


@dataclass
class WordSegment:
    """Represents a word segment with timing and speaker info."""
    text: str
    start: float
    end: float
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ComparisonResult:
    """Results of comparing two transcription files."""
    # Basic stats
    total_words_1: int
    total_words_2: int
    duration_1: float
    duration_2: float
    speakers_1: set
    speakers_2: set

    # Text metrics
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    exact_matches: int
    word_overlap_f1: float

    # Timing metrics
    mean_timing_diff: float
    median_timing_diff: float
    timing_std_dev: float
    boundary_alignment_rate: float

    # Speaker metrics
    speaker_agreement_rate: float
    speaker_transition_matches: int

    # Segmentation metrics
    segment_count_1: int
    segment_count_2: int
    avg_words_per_segment_1: float
    avg_words_per_segment_2: float


class TranscriptionComparator:
    """Compares two transcription JSON files."""

    def __init__(self, timing_tolerance: float = 0.5, text_similarity_threshold: float = 0.8):
        """
        Initialize comparator.

        Args:
            timing_tolerance: Maximum time difference (seconds) for word alignment
            text_similarity_threshold: Minimum text similarity for word matching
        """
        self.timing_tolerance = timing_tolerance
        self.text_similarity_threshold = text_similarity_threshold

    def load_transcription(self, filepath: str) -> List[WordSegment]:
        """Load transcription from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        words = []
        for word_data in data.get('words', []):
            words.append(WordSegment(
                text=word_data['text'],
                start=word_data['start'],
                end=word_data['end'],
                speaker=word_data.get('speaker')
            ))

        return words

    def calculate_basic_stats(self, words1: List[WordSegment], words2: List[WordSegment]) -> Dict[str, Any]:
        """Calculate basic statistics for both transcriptions."""
        def get_stats(words):
            if not words:
                return {
                    'total_words': 0,
                    'duration': 0.0,
                    'speakers': set(),
                    'segment_count': 0,
                    'avg_words_per_segment': 0.0
                }

            total_words = len(words)
            duration = words[-1].end - words[0].start if words else 0.0
            speakers = set(w.speaker for w in words if w.speaker)

            # Calculate segments (groups of consecutive words by same speaker)
            segments = []
            current_segment = []
            current_speaker = None

            for word in words:
                if word.speaker != current_speaker:
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = [word]
                    current_speaker = word.speaker
                else:
                    current_segment.append(word)

            if current_segment:
                segments.append(current_segment)

            avg_words_per_segment = total_words / len(segments) if segments else 0.0

            return {
                'total_words': total_words,
                'duration': duration,
                'speakers': speakers,
                'segment_count': len(segments),
                'avg_words_per_segment': avg_words_per_segment
            }

        stats1 = get_stats(words1)
        stats2 = get_stats(words2)

        return {
            'stats1': stats1,
            'stats2': stats2,
            'word_count_diff': abs(stats1['total_words'] - stats2['total_words']),
            'duration_diff': abs(stats1['duration'] - stats2['duration']),
            'speaker_overlap': len(stats1['speakers'] & stats2['speakers']),
            'unique_speakers_1': len(stats1['speakers'] - stats2['speakers']),
            'unique_speakers_2': len(stats2['speakers'] - stats1['speakers'])
        }

    def align_words(self, words1: List[WordSegment], words2: List[WordSegment]) -> List[Tuple[Optional[WordSegment], Optional[WordSegment]]]:
        """
        Align words between two transcriptions using dynamic programming.
        Returns list of (word1, word2) pairs, where one may be None for insertions/deletions.
        """
        n, m = len(words1), len(words2)

        # Simple alignment based on timing proximity and text similarity
        alignments = []

        i, j = 0, 0
        while i < n and j < m:
            w1, w2 = words1[i], words2[j]

            # Check if words can be aligned
            time_diff = abs(w1.start - w2.start)
            text_sim = difflib.SequenceMatcher(None, w1.text.lower(), w2.text.lower()).ratio()

            if time_diff <= self.timing_tolerance and text_sim >= self.text_similarity_threshold:
                # Good alignment
                alignments.append((w1, w2))
                i += 1
                j += 1
            elif w1.start < w2.start - self.timing_tolerance:
                # w1 is too early, skip it (deletion)
                alignments.append((w1, None))
                i += 1
            else:
                # w2 is too early, skip it (insertion)
                alignments.append((None, w2))
                j += 1

        # Add remaining words
        while i < n:
            alignments.append((words1[i], None))
            i += 1
        while j < m:
            alignments.append((None, words2[j]))
            j += 1

        return alignments

    def calculate_text_metrics(self, alignments: List[Tuple[Optional[WordSegment], Optional[WordSegment]]]) -> Dict[str, Any]:
        """Calculate text-based metrics from word alignments."""
        substitutions = 0
        deletions = 0
        insertions = 0
        exact_matches = 0
        total_chars_1 = 0
        total_chars_2 = 0
        char_errors = 0

        for w1, w2 in alignments:
            if w1 and w2:
                # Both present - check for match
                if w1.text == w2.text:
                    exact_matches += 1
                else:
                    substitutions += 1

                # Character-level comparison
                total_chars_1 += len(w1.text)
                total_chars_2 += len(w2.text)
                char_errors += sum(1 for a, b in zip(w1.text, w2.text) if a != b)
                char_errors += abs(len(w1.text) - len(w2.text))  # Length differences

            elif w1 and not w2:
                deletions += 1
                total_chars_1 += len(w1.text)
            elif not w1 and w2:
                insertions += 1
                total_chars_2 += len(w2.text)

        # Calculate WER and CER
        total_words_1 = sum(1 for w1, w2 in alignments if w1 is not None)
        wer = (substitutions + deletions + insertions) / total_words_1 if total_words_1 > 0 else 0.0
        cer = char_errors / total_chars_1 if total_chars_1 > 0 else 0.0

        # Word overlap F1 (treating alignments as predictions)
        true_positives = exact_matches
        false_positives = insertions
        false_negatives = deletions + substitutions

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'wer': wer,
            'cer': cer,
            'exact_matches': exact_matches,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'word_overlap_f1': f1,
            'precision': precision,
            'recall': recall
        }

    def calculate_timing_metrics(self, alignments: List[Tuple[Optional[WordSegment], Optional[WordSegment]]]) -> Dict[str, Any]:
        """Calculate timing-based metrics from aligned words."""
        timing_diffs = []

        for w1, w2 in alignments:
            if w1 and w2:
                # Both words present - compare timing
                start_diff = w1.start - w2.start
                end_diff = w1.end - w2.end
                timing_diffs.extend([start_diff, end_diff])

        if not timing_diffs:
            return {
                'mean_timing_diff': 0.0,
                'median_timing_diff': 0.0,
                'timing_std_dev': 0.0,
                'boundary_alignment_rate': 0.0
            }

        mean_diff = statistics.mean(timing_diffs)
        median_diff = statistics.median(timing_diffs)
        std_dev = statistics.stdev(timing_diffs) if len(timing_diffs) > 1 else 0.0

        # Boundary alignment rate (percentage of boundaries within tolerance)
        aligned_boundaries = sum(1 for diff in timing_diffs if abs(diff) <= self.timing_tolerance)
        boundary_alignment_rate = aligned_boundaries / len(timing_diffs)

        return {
            'mean_timing_diff': mean_diff,
            'median_timing_diff': median_diff,
            'timing_std_dev': std_dev,
            'boundary_alignment_rate': boundary_alignment_rate
        }

    def calculate_speaker_metrics(self, alignments: List[Tuple[Optional[WordSegment], Optional[WordSegment]]]) -> Dict[str, Any]:
        """Calculate speaker consistency metrics."""
        total_aligned = 0
        speaker_agreements = 0
        transition_matches = 0

        prev_speaker_1 = None
        prev_speaker_2 = None

        for w1, w2 in alignments:
            if w1 and w2:
                total_aligned += 1

                # Speaker agreement
                if w1.speaker == w2.speaker:
                    speaker_agreements += 1

                # Speaker transitions
                if w1.speaker != prev_speaker_1 and prev_speaker_1 is not None:
                    if w2.speaker != prev_speaker_2 and prev_speaker_2 is not None:
                        transition_matches += 1

                prev_speaker_1 = w1.speaker
                prev_speaker_2 = w2.speaker

        speaker_agreement_rate = speaker_agreements / total_aligned if total_aligned > 0 else 0.0

        return {
            'speaker_agreement_rate': speaker_agreement_rate,
            'speaker_transition_matches': transition_matches,
            'total_aligned_words': total_aligned
        }

    def compare(self, file1: str, file2: str) -> ComparisonResult:
        """Compare two transcription files and return results."""
        # Load transcriptions
        words1 = self.load_transcription(file1)
        words2 = self.load_transcription(file2)

        # Basic stats
        basic_stats = self.calculate_basic_stats(words1, words2)

        # Align words
        alignments = self.align_words(words1, words2)

        # Calculate metrics
        text_metrics = self.calculate_text_metrics(alignments)
        timing_metrics = self.calculate_timing_metrics(alignments)
        speaker_metrics = self.calculate_speaker_metrics(alignments)

        return ComparisonResult(
            # Basic stats
            total_words_1=basic_stats['stats1']['total_words'],
            total_words_2=basic_stats['stats2']['total_words'],
            duration_1=basic_stats['stats1']['duration'],
            duration_2=basic_stats['stats2']['duration'],
            speakers_1=basic_stats['stats1']['speakers'],
            speakers_2=basic_stats['stats2']['speakers'],

            # Text metrics
            wer=text_metrics['wer'],
            cer=text_metrics['cer'],
            exact_matches=text_metrics['exact_matches'],
            word_overlap_f1=text_metrics['word_overlap_f1'],

            # Timing metrics
            mean_timing_diff=timing_metrics['mean_timing_diff'],
            median_timing_diff=timing_metrics['median_timing_diff'],
            timing_std_dev=timing_metrics['timing_std_dev'],
            boundary_alignment_rate=timing_metrics['boundary_alignment_rate'],

            # Speaker metrics
            speaker_agreement_rate=speaker_metrics['speaker_agreement_rate'],
            speaker_transition_matches=speaker_metrics['speaker_transition_matches'],

            # Segmentation metrics
            segment_count_1=basic_stats['stats1']['segment_count'],
            segment_count_2=basic_stats['stats2']['segment_count'],
            avg_words_per_segment_1=basic_stats['stats1']['avg_words_per_segment'],
            avg_words_per_segment_2=basic_stats['stats2']['avg_words_per_segment']
        )


def print_comparison_report(result: ComparisonResult, file1: str, file2: str):
    """Print a formatted comparison report."""
    print("=" * 80)
    print("TRANSCRIPTION COMPARISON REPORT")
    print("=" * 80)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print()

    print("BASIC STATISTICS")
    print("-" * 40)
    print(f"Total words (File 1):     {result.total_words_1:,}")
    print(f"Total words (File 2):     {result.total_words_2:,}")
    print(f"Duration (File 1):        {result.duration_1:.2f}s")
    print(f"Duration (File 2):        {result.duration_2:.2f}s")
    print(f"Speakers (File 1):        {len(result.speakers_1)}")
    print(f"Speakers (File 2):        {len(result.speakers_2)}")
    print()

    print("TEXT METRICS")
    print("-" * 40)
    print(f"Word Error Rate (WER):    {result.wer:.1%}")
    print(f"Character Error Rate:     {result.cer:.1%}")
    print(f"Exact word matches:       {result.exact_matches:,}")
    print(f"Word overlap F1-score:    {result.word_overlap_f1:.1%}")
    print()

    print("TIMING METRICS")
    print("-" * 40)
    print(f"Mean timing difference:   {result.mean_timing_diff:.3f}s")
    print(f"Median timing difference: {result.median_timing_diff:.3f}s")
    print(f"Timing std deviation:     {result.timing_std_dev:.3f}s")
    print(f"Boundary alignment rate:  {result.boundary_alignment_rate:.1%}")
    print()

    print("SPEAKER METRICS")
    print("-" * 40)
    print(f"Speaker agreement rate:   {result.speaker_agreement_rate:.1%}")
    print(f"Speaker transitions:      {result.speaker_transition_matches}")
    print()

    print("SEGMENTATION METRICS")
    print("-" * 40)
    print(f"Segments (File 1):        {result.segment_count_1}")
    print(f"Segments (File 2):        {result.segment_count_2}")
    print(f"Avg words/segment (1):    {result.avg_words_per_segment_1:.1f}")
    print(f"Avg words/segment (2):    {result.avg_words_per_segment_2:.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two transcription JSON files")
    parser.add_argument("file1", help="First transcription JSON file")
    parser.add_argument("file2", help="Second transcription JSON file")
    parser.add_argument("--timing-tolerance", type=float, default=0.5,
                       help="Maximum timing difference for word alignment (seconds)")
    parser.add_argument("--text-threshold", type=float, default=0.8,
                       help="Minimum text similarity for word matching (0-1)")
    parser.add_argument("--json-output", help="Save detailed results to JSON file")

    args = parser.parse_args()

    comparator = TranscriptionComparator(
        timing_tolerance=args.timing_tolerance,
        text_similarity_threshold=args.text_threshold
    )

    try:
        result = comparator.compare(args.file1, args.file2)
        print_comparison_report(result, args.file1, args.file2)

        if args.json_output:
            # Convert result to dict for JSON serialization
            result_dict = {
                'files': {'file1': args.file1, 'file2': args.file2},
                'parameters': {
                    'timing_tolerance': args.timing_tolerance,
                    'text_similarity_threshold': args.text_threshold
                },
                'metrics': {
                    'basic_stats': {
                        'total_words_1': result.total_words_1,
                        'total_words_2': result.total_words_2,
                        'duration_1': result.duration_1,
                        'duration_2': result.duration_2,
                        'speakers_1': list(result.speakers_1),
                        'speakers_2': list(result.speakers_2)
                    },
                    'text_metrics': {
                        'wer': result.wer,
                        'cer': result.cer,
                        'exact_matches': result.exact_matches,
                        'word_overlap_f1': result.word_overlap_f1
                    },
                    'timing_metrics': {
                        'mean_timing_diff': result.mean_timing_diff,
                        'median_timing_diff': result.median_timing_diff,
                        'timing_std_dev': result.timing_std_dev,
                        'boundary_alignment_rate': result.boundary_alignment_rate
                    },
                    'speaker_metrics': {
                        'speaker_agreement_rate': result.speaker_agreement_rate,
                        'speaker_transition_matches': result.speaker_transition_matches
                    },
                    'segmentation_metrics': {
                        'segment_count_1': result.segment_count_1,
                        'segment_count_2': result.segment_count_2,
                        'avg_words_per_segment_1': result.avg_words_per_segment_1,
                        'avg_words_per_segment_2': result.avg_words_per_segment_2
                    }
                }
            }

            with open(args.json_output, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Detailed results saved to {args.json_output}")

    except Exception as e:
        print(f"Error comparing transcriptions: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()