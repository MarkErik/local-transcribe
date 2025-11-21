"""
Output formatting module.

This module formats comparison results for output in various formats,
including JSON, human-readable text, HTML, and CSV.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, TextIO

from .data_structures import (
    ComparisonResult, AlignedPair, Difference, QualityMetrics, ComparisonSummary
)


class ComparisonResultFormatter:
    """Formats comparison results for output in various formats."""
    
    def __init__(self, config):
        """Initialize the formatter with configuration."""
        self.config = config
    
    def format_results(self, comparison_result: ComparisonResult, 
                      output_format: str = "text") -> str:
        """
        Format comparison results in the specified format.
        
        Args:
            comparison_result: Result of transcript comparison
            output_format: Output format ("json", "text", "html", "csv")
            
        Returns:
            Formatted output as a string
        """
        if output_format.lower() == "json":
            return self._format_json(comparison_result)
        elif output_format.lower() == "text":
            return self._format_text(comparison_result)
        elif output_format.lower() == "html":
            return self._format_html(comparison_result)
        elif output_format.lower() == "csv":
            return self._format_csv(comparison_result)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _format_json(self, comparison_result: ComparisonResult) -> str:
        """Format comparison results as JSON."""
        # Convert data structures to JSON-serializable format
        # Get source information safely
        ref_source = "unknown"
        hyp_source = "unknown"
        
        if (comparison_result.alignment_result.aligned_pairs and
            len(comparison_result.alignment_result.aligned_pairs) > 0):
            first_pair = comparison_result.alignment_result.aligned_pairs[0]
            if first_pair.reference_segment:
                ref_source = first_pair.reference_segment.features.get("source", "unknown")
            if first_pair.hypothesis_segment:
                hyp_source = first_pair.hypothesis_segment.features.get("source", "unknown")
        
        json_data = {
            "comparison_metadata": {
                "reference_source": ref_source,
                "hypothesis_source": hyp_source,
                "comparison_date": datetime.now().isoformat(),
                "comparison_config": {
                    "semantic_threshold": self.config.semantic_threshold,
                    "fuzzy_threshold": self.config.fuzzy_threshold,
                    "timing_tolerance": self.config.timing_tolerance
                }
            },
            "summary": {
                "total_segments": comparison_result.summary.total_segments,
                "matched_segments": comparison_result.summary.matched_segments,
                "substituted_segments": comparison_result.summary.substituted_segments,
                "inserted_segments": comparison_result.summary.inserted_segments,
                "deleted_segments": comparison_result.summary.deleted_segments,
                "timing_differences": comparison_result.summary.timing_differences,
                "overall_similarity": comparison_result.summary.overall_similarity
            },
            "quality_metrics": {
                "word_error_rate": comparison_result.quality_metrics.word_error_rate,
                "match_error_rate": comparison_result.quality_metrics.match_error_rate,
                "semantic_similarity": comparison_result.quality_metrics.semantic_similarity,
                "timing_accuracy": comparison_result.quality_metrics.timing_accuracy,
                "fluency_score": comparison_result.quality_metrics.fluency_score,
                "confidence_score": comparison_result.quality_metrics.confidence_score,
                "overall_quality": comparison_result.quality_metrics.overall_quality
            },
            "differences": [self._difference_to_dict(diff) for diff in comparison_result.differences],
            "alignment_details": [self._aligned_pair_to_dict(pair) for pair in comparison_result.alignment_result.aligned_pairs]
        }
        
        return json.dumps(json_data, indent=2)
    
    def _difference_to_dict(self, difference: Difference) -> Dict:
        """Convert a Difference object to a dictionary."""
        return {
            "type": difference.type,
            "severity": difference.severity,
            "reference_segment": self._segment_to_dict(difference.reference_segment) if difference.reference_segment else None,
            "hypothesis_segment": self._segment_to_dict(difference.hypothesis_segment) if difference.hypothesis_segment else None,
            "description": difference.description,
            "context": [self._segment_to_dict(seg) for seg in difference.context],
            "suggestions": difference.suggestions
        }
    
    def _aligned_pair_to_dict(self, pair: AlignedPair) -> Dict:
        """Convert an AlignedPair object to a dictionary."""
        return {
            "reference_segment": self._segment_to_dict(pair.reference_segment) if pair.reference_segment else None,
            "hypothesis_segment": self._segment_to_dict(pair.hypothesis_segment) if pair.hypothesis_segment else None,
            "similarity_score": pair.similarity_score,
            "alignment_type": pair.alignment_type,
            "timing_difference": pair.timing_difference,
            "confidence": pair.confidence
        }
    
    def _segment_to_dict(self, segment) -> Dict:
        """Convert a TextSegment object to a dictionary."""
        return {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "speaker": segment.speaker,
            "segment_id": segment.segment_id,
            "confidence": segment.confidence,
            "features": segment.features
        }
    
    def _format_text(self, comparison_result: ComparisonResult) -> str:
        """Format comparison results as human-readable text."""
        lines = []
        
        # Header
        lines.append("TRANSCRIPT COMPARISON REPORT")
        lines.append("=" * 50)
        lines.append("")
        
        # Metadata - Get source information safely
        ref_source = "unknown"
        hyp_source = "unknown"
        
        if (comparison_result.alignment_result.aligned_pairs and
            len(comparison_result.alignment_result.aligned_pairs) > 0):
            first_pair = comparison_result.alignment_result.aligned_pairs[0]
            if first_pair.reference_segment:
                ref_source = first_pair.reference_segment.features.get("source", "unknown")
            if first_pair.hypothesis_segment:
                hyp_source = first_pair.hypothesis_segment.features.get("source", "unknown")
        
        lines.append(f"Reference Source: {ref_source}")
        lines.append(f"Hypothesis Source: {hyp_source}")
        lines.append(f"Comparison Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 10)
        summary = comparison_result.summary
        lines.append(f"Total Segments: {summary.total_segments}")
        lines.append(f"Matched Segments: {summary.matched_segments} ({summary.matched_segments/summary.total_segments*100:.1f}%)" if summary.total_segments > 0 else "Matched Segments: 0 (0.0%)")
        lines.append(f"Substituted Segments: {summary.substituted_segments} ({summary.substituted_segments/summary.total_segments*100:.1f}%)" if summary.total_segments > 0 else "Substituted Segments: 0 (0.0%)")
        lines.append(f"Inserted Segments: {summary.inserted_segments} ({summary.inserted_segments/summary.total_segments*100:.1f}%)" if summary.total_segments > 0 else "Inserted Segments: 0 (0.0%)")
        lines.append(f"Deleted Segments: {summary.deleted_segments} ({summary.deleted_segments/summary.total_segments*100:.1f}%)" if summary.total_segments > 0 else "Deleted Segments: 0 (0.0%)")
        lines.append(f"Overall Similarity: {summary.overall_similarity*100:.1f}%")
        lines.append("")
        
        # Quality Metrics
        lines.append("QUALITY METRICS")
        lines.append("-" * 20)
        metrics = comparison_result.quality_metrics
        lines.append(f"Word Error Rate: {metrics.word_error_rate*100:.1f}%")
        lines.append(f"Match Error Rate: {metrics.match_error_rate*100:.1f}%")
        lines.append(f"Semantic Similarity: {metrics.semantic_similarity*100:.1f}%")
        lines.append(f"Timing Accuracy: {metrics.timing_accuracy*100:.1f}%")
        lines.append(f"Fluency Score: {metrics.fluency_score*100:.1f}%")
        lines.append(f"Overall Quality: {metrics.overall_quality*100:.1f}%")
        lines.append("")
        
        # Differences
        if comparison_result.differences:
            lines.append("DIFFERENCES")
            lines.append("-" * 15)
            
            for i, diff in enumerate(comparison_result.differences[:self.config.max_differences], 1):
                lines.append(f"{i}. {diff.severity.upper()} {diff.type.upper()}")
                
                if diff.reference_segment:
                    lines.append(f"   Reference: \"{diff.reference_segment.text}\" ({self._format_time(diff.reference_segment.start)}-{self._format_time(diff.reference_segment.end)})")
                
                if diff.hypothesis_segment:
                    lines.append(f"   Hypothesis: \"{diff.hypothesis_segment.text}\" ({self._format_time(diff.hypothesis_segment.start)}-{self._format_time(diff.hypothesis_segment.end)})")
                
                if diff.context:
                    context_text = " ".join(seg.text for seg in diff.context)
                    lines.append(f"   Context: \"...{context_text}...\"")
                
                lines.append(f"   Description: {diff.description}")
                
                if diff.suggestions:
                    lines.append("   Suggestions:")
                    for suggestion in diff.suggestions:
                        lines.append(f"     - {suggestion}")
                
                lines.append("")
        
        # Alignment Details
        if self.config.include_context:
            lines.append("ALIGNMENT DETAILS")
            lines.append("-" * 22)
            
            for i, pair in enumerate(comparison_result.alignment_result.aligned_pairs[:20], 1):  # Limit to first 20 for readability
                lines.append(f"Segment {i}:")
                
                if pair.reference_segment:
                    lines.append(f"  Reference: \"{pair.reference_segment.text}\" ({self._format_time(pair.reference_segment.start)}-{self._format_time(pair.reference_segment.end)})")
                
                if pair.hypothesis_segment:
                    lines.append(f"  Hypothesis: \"{pair.hypothesis_segment.text}\" ({self._format_time(pair.hypothesis_segment.start)}-{self._format_time(pair.hypothesis_segment.end)})")
                
                lines.append(f"  Type: {pair.alignment_type.replace('_', ' ').title()}")
                lines.append(f"  Similarity: {pair.similarity_score*100:.1f}%")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes}:{seconds:02d}"
    
    def _format_html(self, comparison_result: ComparisonResult) -> str:
        """Format comparison results as HTML."""
        # Get source information safely
        ref_source = "unknown"
        hyp_source = "unknown"
        
        if (comparison_result.alignment_result.aligned_pairs and
            len(comparison_result.alignment_result.aligned_pairs) > 0):
            first_pair = comparison_result.alignment_result.aligned_pairs[0]
            if first_pair.reference_segment:
                ref_source = first_pair.reference_segment.features.get("source", "unknown")
            if first_pair.hypothesis_segment:
                hyp_source = first_pair.hypothesis_segment.features.get("source", "unknown")
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Transcript Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .exact {{ background-color: #d4edda; }}
        .semantic {{ background-color: #d1ecf1; }}
        .substitution {{ background-color: #fff3cd; }}
        .insertion {{ background-color: #f8d7da; }}
        .deletion {{ background-color: #f8d7da; }}
        .severity-minor {{ color: #28a745; }}
        .severity-moderate {{ color: #ffc107; }}
        .severity-major {{ color: #dc3545; }}
        .metrics {{ display: flex; flex-wrap: wrap; }}
        .metric {{ margin: 10px; padding: 15px; border-radius: 5px; background-color: #f8f9fa; width: 200px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; }}
        .metric-label {{ font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <h1>Transcript Comparison Report</h1>
    
    <h2>Metadata</h2>
    <table>
        <tr><th>Property</th><th>Value</th></tr>
        <tr><td>Reference Source</td><td>{ref_source}</td></tr>
        <tr><td>Hypothesis Source</td><td>{hyp_source}</td></tr>
        <tr><td>Comparison Date</td><td>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
    </table>
    
    <h2>Summary</h2>
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.total_segments}</div>
            <div class="metric-label">Total Segments</div>
        </div>
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.matched_segments}</div>
            <div class="metric-label">Matched Segments</div>
        </div>
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.substituted_segments}</div>
            <div class="metric-label">Substitutions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.inserted_segments}</div>
            <div class="metric-label">Insertions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.deleted_segments}</div>
            <div class="metric-label">Deletions</div>
        </div>
        <div class="metric">
            <div class="metric-value">{comparison_result.summary.overall_similarity*100:.1f}%</div>
            <div class="metric-label">Overall Similarity</div>
        </div>
    </div>
    
    <h2>Quality Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Word Error Rate</td><td>{comparison_result.quality_metrics.word_error_rate*100:.1f}%</td></tr>
        <tr><td>Match Error Rate</td><td>{comparison_result.quality_metrics.match_error_rate*100:.1f}%</td></tr>
        <tr><td>Semantic Similarity</td><td>{comparison_result.quality_metrics.semantic_similarity*100:.1f}%</td></tr>
        <tr><td>Timing Accuracy</td><td>{comparison_result.quality_metrics.timing_accuracy*100:.1f}%</td></tr>
        <tr><td>Fluency Score</td><td>{comparison_result.quality_metrics.fluency_score*100:.1f}%</td></tr>
        <tr><td>Overall Quality</td><td>{comparison_result.quality_metrics.overall_quality*100:.1f}%</td></tr>
    </table>
    
    <h2>Differences</h2>
    <table>
        <tr><th>Type</th><th>Severity</th><th>Reference</th><th>Hypothesis</th><th>Description</th></tr>
"""
        
        for diff in comparison_result.differences[:self.config.max_differences]:
            ref_text = diff.reference_segment.text if diff.reference_segment else ""
            hyp_text = diff.hypothesis_segment.text if diff.hypothesis_segment else ""
            
            html += f"""
        <tr>
            <td>{diff.type.title()}</td>
            <td class="severity-{diff.severity}">{diff.severity.title()}</td>
            <td>{ref_text}</td>
            <td>{hyp_text}</td>
            <td>{diff.description}</td>
        </tr>
"""
        
        html += """
    </table>
    
    <h2>Alignment Details</h2>
    <table>
        <tr><th>Reference</th><th>Hypothesis</th><th>Type</th><th>Similarity</th></tr>
"""
        
        for pair in comparison_result.alignment_result.aligned_pairs[:20]:  # Limit to first 20 for readability
            ref_text = pair.reference_segment.text if pair.reference_segment else ""
            hyp_text = pair.hypothesis_segment.text if pair.hypothesis_segment else ""
            
            html += f"""
        <tr class="{pair.alignment_type}">
            <td>{ref_text}</td>
            <td>{hyp_text}</td>
            <td>{pair.alignment_type.replace('_', ' ').title()}</td>
            <td>{pair.similarity_score*100:.1f}%</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        return html
    
    def _format_csv(self, comparison_result: ComparisonResult) -> str:
        """Format comparison results as CSV."""
        lines = []
        
        # Header
        lines.append("type,severity,reference_text,hypothesis_text,reference_start,reference_end,hypothesis_start,hypothesis_end,similarity_score,description")
        
        # Process differences
        for diff in comparison_result.differences:
            ref_text = diff.reference_segment.text.replace('"', '""') if diff.reference_segment else ""
            hyp_text = diff.hypothesis_segment.text.replace('"', '""') if diff.hypothesis_segment else ""
            ref_start = diff.reference_segment.start if diff.reference_segment else ""
            ref_end = diff.reference_segment.end if diff.reference_segment else ""
            hyp_start = diff.hypothesis_segment.start if diff.hypothesis_segment else ""
            hyp_end = diff.hypothesis_segment.end if diff.hypothesis_segment else ""
            
            lines.append(f'"{diff.type}","{diff.severity}","{ref_text}","{hyp_text}",{ref_start},{ref_end},{hyp_start},{hyp_end},"{diff.description}"')
        
        # Process aligned pairs
        for pair in comparison_result.alignment_result.aligned_pairs:
            ref_text = pair.reference_segment.text.replace('"', '""') if pair.reference_segment else ""
            hyp_text = pair.hypothesis_segment.text.replace('"', '""') if pair.hypothesis_segment else ""
            ref_start = pair.reference_segment.start if pair.reference_segment else ""
            ref_end = pair.reference_segment.end if pair.reference_segment else ""
            hyp_start = pair.hypothesis_segment.start if pair.hypothesis_segment else ""
            hyp_end = pair.hypothesis_segment.end if pair.hypothesis_segment else ""
            
            lines.append(f'"{pair.alignment_type}","match","{ref_text}","{hyp_text}",{ref_start},{ref_end},{hyp_start},{hyp_end},{pair.similarity_score},""')
        
        return "\n".join(lines)
    
    def write_to_file(self, comparison_result: ComparisonResult, 
                     output_path: str, output_format: str = "text") -> None:
        """
        Write formatted comparison results to a file.
        
        Args:
            comparison_result: Result of transcript comparison
            output_path: Path to output file
            output_format: Output format ("json", "text", "html", "csv")
        """
        formatted_output = self.format_results(comparison_result, output_format)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
    
    def write_to_stream(self, comparison_result: ComparisonResult, 
                       stream: TextIO, output_format: str = "text") -> None:
        """
        Write formatted comparison results to a stream.
        
        Args:
            comparison_result: Result of transcript comparison
            stream: Output stream (e.g., sys.stdout)
            output_format: Output format ("json", "text", "html", "csv")
        """
        formatted_output = self.format_results(comparison_result, output_format)
        stream.write(formatted_output)


class CSVOutputFormatter:
    """Specialized formatter for CSV output with compatibility to existing system."""
    
    def __init__(self, config):
        """Initialize the CSV formatter with configuration."""
        self.config = config
    
    def format_to_csv(self, comparison_result: ComparisonResult) -> str:
        """Format comparison result to CSV format compatible with existing system."""
        output_lines = []
        
        # Add header
        output_lines.append("line_number,word,difference_type,similarity_score,context")
        
        # Process aligned pairs
        for pair in comparison_result.alignment_result.aligned_pairs:
            if pair.reference_segment:
                line_number = pair.reference_segment.features.get("csv_line_number", "")
                word = pair.reference_segment.text
                diff_type = pair.alignment_type
                similarity = pair.similarity_score
                
                # Get context (previous and next words)
                context = self._get_context(pair, comparison_result)
                
                output_lines.append(f"{line_number},{word},{diff_type},{similarity},\"{context}\"")
        
        # Process unaligned reference segments (deletions)
        for segment in comparison_result.alignment_result.unaligned_reference:
            line_number = segment.features.get("csv_line_number", "")
            word = segment.text
            context = self._get_segment_context(segment, comparison_result)
            
            output_lines.append(f"{line_number},{word},deletion,0.0,\"{context}\"")
        
        # Process unaligned hypothesis segments (insertions)
        for segment in comparison_result.alignment_result.unaligned_hypothesis:
            line_number = segment.features.get("csv_line_number", "")
            word = segment.text
            context = self._get_segment_context(segment, comparison_result)
            
            output_lines.append(f"{line_number},{word},insertion,0.0,\"{context}\"")
        
        return "\n".join(output_lines)
    
    def _get_context(self, aligned_pair: AlignedPair, comparison_result: ComparisonResult) -> str:
        """Get context words for an aligned pair."""
        if not aligned_pair.reference_segment:
            return ""
        
        # Find the index of this segment in the original transcript
        target_index = -1
        for i, pair in enumerate(comparison_result.alignment_result.aligned_pairs):
            if pair == aligned_pair:
                target_index = i
                break
        
        if target_index == -1:
            return ""
        
        # Get surrounding segments
        context_segments = []
        start_idx = max(0, target_index - self.config.context_window_size)
        end_idx = min(len(comparison_result.alignment_result.aligned_pairs) - 1, 
                      target_index + self.config.context_window_size)
        
        for i in range(start_idx, end_idx + 1):
            if i != target_index:  # Skip the current segment
                pair = comparison_result.alignment_result.aligned_pairs[i]
                if pair.reference_segment:
                    context_segments.append(pair.reference_segment.text)
        
        return " ".join(context_segments)
    
    def _get_segment_context(self, segment, comparison_result: ComparisonResult) -> str:
        """Get context words for an unaligned segment."""
        # For unaligned segments, find the closest aligned segments
        context_segments = []
        
        # Add some segments from the aligned pairs
        for pair in comparison_result.alignment_result.aligned_pairs[:self.config.context_window_size]:
            if pair.reference_segment:
                context_segments.append(pair.reference_segment.text)
        
        return " ".join(context_segments)