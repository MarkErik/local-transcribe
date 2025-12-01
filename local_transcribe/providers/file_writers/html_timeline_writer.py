#!/usr/bin/env python3
"""
Interactive HTML Timeline Writer for hierarchical transcripts.

This writer produces a self-contained HTML file with an interactive
timeline visualization of the conversation, showing turns as colored
bars and interjections as markers.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path
import json

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow
from local_transcribe.providers.file_writers.format_utils import (
    format_timestamp,
    format_duration,
    format_speaker_name,
    get_interjection_symbol,
    get_interjection_verb,
    escape_html,
    get_speaker_color,
    calculate_position_percent,
    format_percentage
)


def write_html_timeline(transcript: TranscriptFlow, path: str | Path) -> None:
    """
    Write a TranscriptFlow as an interactive HTML timeline.
    
    Args:
        transcript: TranscriptFlow object with hierarchical turn structure
        path: Output file path
    """
    path = Path(path)
    
    # Extract data from TranscriptFlow
    if not hasattr(transcript, 'turns') or not hasattr(transcript, 'metadata'):
        raise ValueError("Expected TranscriptFlow object with 'turns' and 'metadata' attributes")
    
    turns = transcript.turns
    metadata = transcript.metadata
    conversation_metrics = getattr(transcript, 'conversation_metrics', {})
    speaker_statistics = getattr(transcript, 'speaker_statistics', {})
    
    # Calculate timeline parameters
    if turns:
        timeline_start = min(getattr(t, 'start', 0) for t in turns)
        timeline_end = max(getattr(t, 'end', 0) for t in turns)
    else:
        timeline_start = 0
        timeline_end = 0
    
    total_duration = timeline_end - timeline_start if timeline_end > timeline_start else 1
    speakers = metadata.get('speakers', [])
    
    # Build HTML
    html = _build_html_document(
        turns=turns,
        metadata=metadata,
        conversation_metrics=conversation_metrics,
        speaker_statistics=speaker_statistics,
        speakers=speakers,
        timeline_start=timeline_start,
        total_duration=total_duration
    )
    
    path.write_text(html, encoding="utf-8")


def _build_html_document(
    turns: List[Any],
    metadata: dict,
    conversation_metrics: dict,
    speaker_statistics: dict,
    speakers: List[str],
    timeline_start: float,
    total_duration: float
) -> str:
    """Build the complete HTML document."""
    
    # Generate turn data for JavaScript
    turns_data = []
    for turn in turns:
        turn_dict = {
            "id": getattr(turn, 'turn_id', 0),
            "speaker": getattr(turn, 'primary_speaker', 'Unknown'),
            "start": getattr(turn, 'start', 0),
            "end": getattr(turn, 'end', 0),
            "text": getattr(turn, 'text', ''),
            "wordCount": getattr(turn, 'word_count', 0),
            "speakingRate": getattr(turn, 'speaking_rate', 0),
            "flowContinuity": getattr(turn, 'flow_continuity', 1.0),
            "turnType": getattr(turn, 'turn_type', 'monologue'),
            "interjections": []
        }
        
        for ij in getattr(turn, 'interjections', []):
            turn_dict["interjections"].append({
                "speaker": getattr(ij, 'speaker', 'Unknown'),
                "start": getattr(ij, 'start', 0),
                "end": getattr(ij, 'end', 0),
                "text": getattr(ij, 'text', ''),
                "type": getattr(ij, 'interjection_type', 'unclear')
            })
        
        turns_data.append(turn_dict)
    
    # Generate speaker colors
    speaker_colors = {s: get_speaker_color(s, speakers) for s in speakers}
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Timeline</title>
    <style>
{_get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Conversation Timeline</h1>
            <div class="metadata">
                <div class="meta-item">
                    <span class="meta-label">Duration:</span>
                    <span class="meta-value">{format_duration(total_duration)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Total Turns:</span>
                    <span class="meta-value">{conversation_metrics.get('total_turns', len(turns))}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Interjections:</span>
                    <span class="meta-value">{conversation_metrics.get('total_interjections', 0)}</span>
                </div>
                <div class="meta-item">
                    <span class="meta-label">Avg Flow:</span>
                    <span class="meta-value">{format_percentage(conversation_metrics.get('avg_flow_continuity', 1.0))}</span>
                </div>
            </div>
            <div class="speaker-legend">
                {_generate_speaker_legend_html(speakers, speaker_colors)}
            </div>
        </header>
        
        <div class="timeline-wrapper">
            <div class="time-axis">
                {_generate_time_axis_html(timeline_start, total_duration)}
            </div>
            <div class="timeline-container" id="timeline">
                {_generate_timeline_html(turns, speakers, speaker_colors, timeline_start, total_duration)}
            </div>
        </div>
        
        <div class="turn-details" id="turn-details">
            <div class="details-placeholder">
                Click on a turn to see details
            </div>
        </div>
        
        <div class="statistics">
            <h2>Speaker Statistics</h2>
            <div class="stats-grid">
                {_generate_speaker_stats_html(speaker_statistics, speaker_colors)}
            </div>
        </div>
    </div>
    
    <script>
        const turnsData = {json.dumps(turns_data)};
        const speakerColors = {json.dumps(speaker_colors)};
        const timelineStart = {timeline_start};
        const totalDuration = {total_duration};
        
{_get_javascript()}
    </script>
</body>
</html>"""


def _get_css_styles() -> str:
    """Return the CSS styles for the HTML document."""
    return """
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 15px;
        }
        
        h2 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .metadata {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 15px;
        }
        
        .meta-item {
            display: flex;
            gap: 8px;
        }
        
        .meta-label {
            font-weight: 600;
            color: #666;
        }
        
        .meta-value {
            color: #333;
        }
        
        .speaker-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
        
        .timeline-wrapper {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .time-axis {
            display: flex;
            justify-content: space-between;
            padding: 0 10px;
            margin-bottom: 10px;
            font-size: 12px;
            color: #666;
        }
        
        .timeline-container {
            position: relative;
            height: 120px;
            background: #f8f9fa;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .speaker-track {
            position: absolute;
            left: 0;
            right: 0;
            height: 50px;
        }
        
        .turn-bar {
            position: absolute;
            height: 40px;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            font-size: 11px;
            color: white;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }
        
        .turn-bar:hover {
            transform: scaleY(1.1);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            z-index: 10;
        }
        
        .turn-bar.selected {
            outline: 3px solid #333;
            z-index: 20;
        }
        
        .interjection-marker {
            position: absolute;
            width: 12px;
            height: 12px;
            background: #fff;
            border: 2px solid;
            border-radius: 50%;
            transform: translateX(-50%);
            cursor: pointer;
            z-index: 5;
        }
        
        .interjection-marker:hover {
            transform: translateX(-50%) scale(1.3);
        }
        
        .turn-details {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            min-height: 150px;
        }
        
        .details-placeholder {
            color: #999;
            text-align: center;
            padding: 40px;
        }
        
        .details-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #eee;
        }
        
        .details-speaker {
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .details-time {
            color: #666;
            font-size: 0.9em;
        }
        
        .details-text {
            margin-bottom: 15px;
            line-height: 1.8;
        }
        
        .details-interjections {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 15px;
        }
        
        .details-interjections h4 {
            margin-bottom: 10px;
            color: #666;
        }
        
        .interjection-item {
            display: flex;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 0.9em;
        }
        
        .interjection-time {
            color: #666;
            white-space: nowrap;
        }
        
        .interjection-speaker {
            font-weight: 600;
        }
        
        .details-metrics {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            font-size: 0.9em;
            color: #666;
        }
        
        .metric {
            display: flex;
            gap: 5px;
        }
        
        .metric-label {
            font-weight: 600;
        }
        
        .statistics {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .speaker-card {
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
        }
        
        .speaker-card-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .speaker-card-color {
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }
        
        .speaker-card-name {
            font-weight: 600;
        }
        
        .speaker-card-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 0.9em;
        }
        
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.85em;
        }
        
        .stat-value {
            font-weight: 600;
        }
        
        @media print {
            .timeline-container {
                height: auto !important;
            }
            .turn-bar:hover {
                transform: none;
            }
        }
    """


def _get_javascript() -> str:
    """Return the JavaScript for interactivity."""
    return """
        // Initialize timeline
        document.querySelectorAll('.turn-bar').forEach(bar => {
            bar.addEventListener('click', function() {
                const turnId = parseInt(this.dataset.turnId);
                selectTurn(turnId);
            });
        });
        
        document.querySelectorAll('.interjection-marker').forEach(marker => {
            marker.addEventListener('click', function(e) {
                e.stopPropagation();
                const turnId = parseInt(this.dataset.turnId);
                selectTurn(turnId);
            });
        });
        
        function selectTurn(turnId) {
            // Remove previous selection
            document.querySelectorAll('.turn-bar.selected').forEach(el => {
                el.classList.remove('selected');
            });
            
            // Add selection to current
            const bar = document.querySelector(`.turn-bar[data-turn-id="${turnId}"]`);
            if (bar) {
                bar.classList.add('selected');
            }
            
            // Find turn data
            const turn = turnsData.find(t => t.id === turnId);
            if (turn) {
                showTurnDetails(turn);
            }
        }
        
        function showTurnDetails(turn) {
            const detailsEl = document.getElementById('turn-details');
            const speakerColor = speakerColors[turn.speaker] || '#666';
            
            let interjectionHtml = '';
            if (turn.interjections && turn.interjections.length > 0) {
                interjectionHtml = `
                    <div class="details-interjections">
                        <h4>Interjections (${turn.interjections.length})</h4>
                        ${turn.interjections.map(ij => `
                            <div class="interjection-item">
                                <span class="interjection-time">[${formatTime(ij.start)}]</span>
                                <span class="interjection-speaker" style="color: ${speakerColors[ij.speaker] || '#666'}">${ij.speaker}:</span>
                                <span>"${escapeHtml(ij.text)}"</span>
                                <span style="color: #999">(${ij.type})</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
            
            detailsEl.innerHTML = `
                <div class="details-header">
                    <span class="details-speaker" style="color: ${speakerColor}">Turn ${turn.id}: ${turn.speaker}</span>
                    <span class="details-time">${formatTime(turn.start)} - ${formatTime(turn.end)}</span>
                </div>
                <div class="details-text">${escapeHtml(turn.text)}</div>
                ${interjectionHtml}
                <div class="details-metrics">
                    <div class="metric">
                        <span class="metric-label">Words:</span>
                        <span>${turn.wordCount}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Speaking Rate:</span>
                        <span>${turn.speakingRate.toFixed(1)} wpm</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Flow Continuity:</span>
                        <span>${(turn.flowContinuity * 100).toFixed(1)}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Turn Type:</span>
                        <span>${turn.turnType}</span>
                    </div>
                </div>
            `;
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = (seconds % 60).toFixed(2);
            return `${mins}:${secs.padStart(5, '0')}`;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    """


def _generate_speaker_legend_html(speakers: List[str], colors: dict) -> str:
    """Generate the speaker legend HTML."""
    items = []
    for speaker in speakers:
        color = colors.get(speaker, '#666')
        items.append(f'''
            <div class="legend-item">
                <div class="legend-color" style="background: {color}"></div>
                <span>{escape_html(format_speaker_name(speaker))}</span>
            </div>
        ''')
    return '\n'.join(items)


def _generate_time_axis_html(start: float, duration: float) -> str:
    """Generate time axis markers."""
    markers = []
    # Generate 5-6 time markers
    num_markers = 6
    for i in range(num_markers):
        time = start + (duration * i / (num_markers - 1))
        mins = int(time // 60)
        secs = int(time % 60)
        markers.append(f'<span>{mins}:{secs:02d}</span>')
    return '\n'.join(markers)


def _generate_timeline_html(
    turns: List[Any],
    speakers: List[str],
    colors: dict,
    timeline_start: float,
    total_duration: float
) -> str:
    """Generate the timeline visualization HTML."""
    html_parts = []
    
    # Create tracks for each speaker
    track_height = 50
    track_spacing = 10
    
    for idx, speaker in enumerate(speakers):
        top = idx * (track_height + track_spacing) + 5
        html_parts.append(f'<div class="speaker-track" style="top: {top}px">')
        
        # Add turns for this speaker
        for turn in turns:
            if getattr(turn, 'primary_speaker', '') != speaker:
                continue
            
            turn_id = getattr(turn, 'turn_id', 0)
            start = getattr(turn, 'start', 0)
            end = getattr(turn, 'end', 0)
            
            left = calculate_position_percent(start, total_duration, timeline_start)
            width = calculate_position_percent(end, total_duration, timeline_start) - left
            width = max(width, 0.5)  # Minimum width for visibility
            
            color = colors.get(speaker, '#666')
            
            # Truncate text for display
            text = getattr(turn, 'text', '')[:30]
            if len(getattr(turn, 'text', '')) > 30:
                text += '...'
            
            html_parts.append(f'''
                <div class="turn-bar" 
                     data-turn-id="{turn_id}"
                     style="left: {left:.2f}%; width: {width:.2f}%; background: {color};"
                     title="Turn {turn_id}: {escape_html(text)}">
                    {turn_id}
                </div>
            ''')
            
            # Add interjection markers
            for ij in getattr(turn, 'interjections', []):
                ij_start = getattr(ij, 'start', 0)
                ij_speaker = getattr(ij, 'speaker', '')
                ij_text = getattr(ij, 'text', '')
                ij_color = colors.get(ij_speaker, '#666')
                
                ij_left = calculate_position_percent(ij_start, total_duration, timeline_start)
                
                # Position marker below the turn bar
                html_parts.append(f'''
                    <div class="interjection-marker"
                         data-turn-id="{turn_id}"
                         style="left: {ij_left:.2f}%; top: 42px; border-color: {ij_color};"
                         title="{escape_html(ij_speaker)}: {escape_html(ij_text)}">
                    </div>
                ''')
        
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def _generate_speaker_stats_html(speaker_statistics: dict, colors: dict) -> str:
    """Generate speaker statistics cards."""
    cards = []
    for speaker, stats in speaker_statistics.items():
        color = colors.get(speaker, '#666')
        cards.append(f'''
            <div class="speaker-card">
                <div class="speaker-card-header">
                    <div class="speaker-card-color" style="background: {color}"></div>
                    <span class="speaker-card-name">{escape_html(format_speaker_name(speaker))}</span>
                </div>
                <div class="speaker-card-stats">
                    <div class="stat-item">
                        <span class="stat-label">Turns</span>
                        <span class="stat-value">{stats.get('total_turns', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Words</span>
                        <span class="stat-value">{stats.get('total_words', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg Duration</span>
                        <span class="stat-value">{stats.get('avg_turn_duration', 0):.1f}s</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Avg WPM</span>
                        <span class="stat-value">{stats.get('avg_speaking_rate', 0):.0f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Interjections</span>
                        <span class="stat-value">{stats.get('total_interjections', 0)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Flow</span>
                        <span class="stat-value">{format_percentage(stats.get('avg_flow_continuity', 1.0))}</span>
                    </div>
                </div>
            </div>
        ''')
    return '\n'.join(cards)


class HtmlTimelineWriter(OutputWriter):
    """Writer for interactive HTML timeline visualization."""
    
    @property
    def name(self) -> str:
        return "html-timeline"
    
    @property
    def description(self) -> str:
        return "Interactive HTML timeline visualization with turn details"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".html"]
    
    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to interactive HTML timeline.
        
        Args:
            turns: TranscriptFlow object
            output_path: Path to write the output file
            word_segments: Optional word segments (not used for this format)
            **kwargs: Additional options
        """
        write_html_timeline(turns, output_path)


# Register the writer
registry.register_output_writer(HtmlTimelineWriter())
