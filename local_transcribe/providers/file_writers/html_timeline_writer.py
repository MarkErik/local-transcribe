#!/usr/bin/env python3
"""
Interactive HTML Conversation Viewer for hierarchical transcripts.

This writer produces a self-contained HTML file with an elegant,
easy-to-read conversation view with timeline navigation.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path
import json
import html

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow
from local_transcribe.providers.file_writers.format_utils import (
    format_timestamp,
    format_duration,
    format_speaker_name,
    escape_html,
    get_speaker_color,
    calculate_position_percent,
    format_percentage
)


def write_html_timeline(transcript: TranscriptFlow, path: str | Path) -> None:
    """
    Write a TranscriptFlow as an interactive HTML conversation viewer.
    
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
    html_content = _build_html_document(
        turns=turns,
        metadata=metadata,
        conversation_metrics=conversation_metrics,
        speaker_statistics=speaker_statistics,
        speakers=speakers,
        timeline_start=timeline_start,
        total_duration=total_duration
    )
    
    path.write_text(html_content, encoding="utf-8")


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
    
    # Generate speaker colors - using a refined palette
    speaker_colors = {}
    color_palette = [
        {"bg": "#E3F2FD", "text": "#1565C0", "accent": "#1976D2"},  # Blue
        {"bg": "#F3E5F5", "text": "#7B1FA2", "accent": "#9C27B0"},  # Purple
        {"bg": "#E8F5E9", "text": "#2E7D32", "accent": "#43A047"},  # Green
        {"bg": "#FFF3E0", "text": "#E65100", "accent": "#FB8C00"},  # Orange
        {"bg": "#FCE4EC", "text": "#C2185B", "accent": "#E91E63"},  # Pink
        {"bg": "#E0F7FA", "text": "#00838F", "accent": "#00ACC1"},  # Cyan
    ]
    for idx, speaker in enumerate(speakers):
        speaker_colors[speaker] = color_palette[idx % len(color_palette)]
    
    # Generate conversation HTML
    conversation_html = _generate_conversation_html(turns, speaker_colors, speakers, timeline_start)
    
    # Generate mini-timeline HTML  
    timeline_html = _generate_mini_timeline_html(turns, speaker_colors, timeline_start, total_duration)
    
    # Generate speaker legend
    legend_html = _generate_speaker_legend_html(speakers, speaker_colors, speaker_statistics)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Conversation Transcript</title>
    <style>
{_get_css_styles()}
    </style>
</head>
<body>
    <div class="app-container">
        <!-- Header -->
        <header class="header">
            <div class="header-content">
                <h1>Conversation Transcript</h1>
                <div class="header-meta">
                    <span class="meta-badge">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                        {_format_duration_compact(total_duration)}
                    </span>
                    <span class="meta-badge">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                        {conversation_metrics.get('total_turns', len(turns))} turns
                    </span>
                    <span class="meta-badge">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                        </svg>
                        {len(speakers)} speakers
                    </span>
                </div>
            </div>
            <div class="header-controls">
                <div class="search-container">
                    <svg class="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="M21 21l-4.35-4.35"></path>
                    </svg>
                    <input type="text" id="search-input" placeholder="Search transcript..." />
                    <span id="search-results" class="search-results"></span>
                </div>
                <button id="theme-toggle" class="icon-btn" title="Toggle dark mode">
                    <svg class="sun-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                    </svg>
                    <svg class="moon-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                    </svg>
                </button>
                <button id="stats-toggle" class="icon-btn" title="Toggle statistics">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <line x1="18" y1="20" x2="18" y2="10"></line>
                        <line x1="12" y1="20" x2="12" y2="4"></line>
                        <line x1="6" y1="20" x2="6" y2="14"></line>
                    </svg>
                </button>
            </div>
        </header>
        
        <!-- Speaker Legend -->
        <div class="speaker-legend">
            {legend_html}
        </div>
        
        <!-- Statistics Panel (hidden by default) -->
        <div id="stats-panel" class="stats-panel hidden">
            {_generate_stats_panel_html(speaker_statistics, speaker_colors, conversation_metrics)}
        </div>
        
        <!-- Mini Timeline -->
        <div class="timeline-wrapper">
            <div class="timeline-time-labels">
                <span>0:00</span>
                <span>{_format_time_label(total_duration / 4)}</span>
                <span>{_format_time_label(total_duration / 2)}</span>
                <span>{_format_time_label(total_duration * 3 / 4)}</span>
                <span>{_format_time_label(total_duration)}</span>
            </div>
            <div class="timeline-track" id="timeline-track">
                {timeline_html}
                <div class="timeline-cursor" id="timeline-cursor"></div>
            </div>
        </div>
        
        <!-- Main Conversation View -->
        <main class="conversation-container" id="conversation">
            {conversation_html}
        </main>
        
        <!-- Keyboard shortcuts hint -->
        <div class="shortcuts-hint" id="shortcuts-hint">
            <span>↑↓ Navigate</span>
            <span>/ Search</span>
            <span>Esc Close</span>
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


def _format_duration_compact(seconds: float) -> str:
    """Format duration as compact string like '1h 23m' or '45m 30s'."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def _format_time_label(seconds: float) -> str:
    """Format time for timeline labels."""
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes}:{secs:02d}"


def _get_css_styles() -> str:
    """Return the CSS styles for the HTML document."""
    return """
        :root {
            /* Light theme */
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-tertiary: #f1f5f9;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --accent-color: #3b82f6;
            --accent-light: #dbeafe;
            --highlight-bg: #fef3c7;
            --interjection-bg: rgba(0, 0, 0, 0.04);
        }
        
        [data-theme="dark"] {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.3);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
            --accent-color: #60a5fa;
            --accent-light: #1e3a5f;
            --highlight-bg: #78350f;
            --interjection-bg: rgba(255, 255, 255, 0.05);
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        html {
            scroll-behavior: smooth;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }
        
        .app-container {
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            max-width: 1000px;
            margin: 0 auto;
            padding: 0 16px;
        }
        
        /* Header */
        .header {
            position: sticky;
            top: 0;
            z-index: 100;
            background: var(--bg-primary);
            border-bottom: 1px solid var(--border-color);
            padding: 12px 0;
            margin-bottom: 8px;
        }
        
        .header-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .header h1 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
        }
        
        .header-meta {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }
        
        .meta-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.8rem;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            padding: 4px 10px;
            border-radius: 16px;
        }
        
        .meta-badge svg {
            opacity: 0.7;
        }
        
        .header-controls {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .search-container {
            position: relative;
            flex: 1;
            max-width: 300px;
        }
        
        .search-icon {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            pointer-events: none;
        }
        
        #search-input {
            width: 100%;
            padding: 8px 12px 8px 40px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 0.9rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        
        #search-input:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px var(--accent-light);
        }
        
        #search-input::placeholder {
            color: var(--text-muted);
        }
        
        .search-results {
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .icon-btn {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-secondary);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .icon-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }
        
        .icon-btn.active {
            background: var(--accent-light);
            border-color: var(--accent-color);
            color: var(--accent-color);
        }
        
        /* Theme toggle icons */
        [data-theme="dark"] .sun-icon { display: block; }
        [data-theme="dark"] .moon-icon { display: none; }
        :root .sun-icon { display: none; }
        :root .moon-icon { display: block; }
        
        /* Speaker Legend */
        .speaker-legend {
            display: flex;
            gap: 16px;
            padding: 12px 0;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .speaker-chip {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .speaker-chip:hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md);
        }
        
        .speaker-chip.dimmed {
            opacity: 0.4;
        }
        
        .speaker-chip .speaker-stats {
            font-size: 0.75rem;
            opacity: 0.8;
        }
        
        /* Stats Panel */
        .stats-panel {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        
        .stats-panel.hidden {
            display: none;
        }
        
        .stat-card {
            padding: 12px;
            border-radius: 8px;
            background: var(--bg-secondary);
        }
        
        .stat-card-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }
        
        .stat-card-color {
            width: 12px;
            height: 12px;
            border-radius: 4px;
        }
        
        .stat-card-name {
            font-weight: 600;
            font-size: 0.9rem;
        }
        
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        
        .stat-item {
            font-size: 0.8rem;
        }
        
        .stat-label {
            color: var(--text-muted);
            display: block;
        }
        
        .stat-value {
            font-weight: 600;
            color: var(--text-primary);
        }
        
        /* Timeline */
        .timeline-wrapper {
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 12px 16px;
            margin-bottom: 16px;
        }
        
        .timeline-time-labels {
            display: flex;
            justify-content: space-between;
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            padding: 0 2px;
        }
        
        .timeline-track {
            position: relative;
            height: 32px;
            background: var(--bg-tertiary);
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
        }
        
        .timeline-segment {
            position: absolute;
            top: 4px;
            height: 24px;
            border-radius: 4px;
            transition: opacity 0.2s, transform 0.2s;
            cursor: pointer;
        }
        
        .timeline-segment:hover {
            opacity: 0.9;
            transform: scaleY(1.1);
        }
        
        .timeline-segment.active {
            outline: 2px solid var(--text-primary);
            outline-offset: 1px;
            z-index: 10;
        }
        
        .timeline-cursor {
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: var(--accent-color);
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }
        
        .timeline-track:hover .timeline-cursor {
            opacity: 1;
        }
        
        /* Conversation */
        .conversation-container {
            flex: 1;
            padding-bottom: 80px;
        }
        
        .turn-block {
            margin-bottom: 4px;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .turn-card {
            display: flex;
            gap: 12px;
            padding: 16px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            transition: box-shadow 0.2s, border-color 0.2s;
            cursor: pointer;
        }
        
        .turn-card:hover {
            box-shadow: var(--shadow-sm);
        }
        
        .turn-card.highlighted {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px var(--accent-light);
        }
        
        .turn-card.search-match {
            background: var(--highlight-bg);
        }
        
        .turn-avatar {
            flex-shrink: 0;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 1rem;
        }
        
        .turn-content {
            flex: 1;
            min-width: 0;
        }
        
        .turn-header {
            display: flex;
            align-items: baseline;
            gap: 8px;
            margin-bottom: 6px;
            flex-wrap: wrap;
        }
        
        .turn-speaker {
            font-weight: 600;
            font-size: 0.95rem;
        }
        
        .turn-time {
            font-size: 0.8rem;
            color: var(--text-muted);
        }
        
        .turn-badges {
            display: flex;
            gap: 6px;
            margin-left: auto;
        }
        
        .turn-badge {
            font-size: 0.7rem;
            padding: 2px 8px;
            border-radius: 10px;
            background: var(--bg-tertiary);
            color: var(--text-secondary);
        }
        
        .turn-text {
            font-size: 0.95rem;
            line-height: 1.7;
            color: var(--text-primary);
        }
        
        .turn-text mark {
            background: var(--highlight-bg);
            padding: 0 2px;
            border-radius: 2px;
        }
        
        /* Interjections */
        .interjections-container {
            margin-top: 12px;
            padding: 12px;
            background: var(--interjection-bg);
            border-radius: 8px;
            border-left: 3px solid var(--border-color);
        }
        
        .interjections-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .interjection-item {
            display: flex;
            align-items: flex-start;
            gap: 8px;
            padding: 6px 0;
            font-size: 0.9rem;
        }
        
        .interjection-item + .interjection-item {
            border-top: 1px solid var(--border-color);
        }
        
        .interjection-speaker {
            font-weight: 600;
            white-space: nowrap;
        }
        
        .interjection-text {
            color: var(--text-secondary);
        }
        
        .interjection-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            white-space: nowrap;
            margin-left: auto;
        }
        
        .interjection-type {
            font-size: 0.7rem;
            padding: 1px 6px;
            border-radius: 8px;
            background: var(--bg-tertiary);
            color: var(--text-muted);
        }
        
        /* Shortcuts hint */
        .shortcuts-hint {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 16px;
            padding: 8px 16px;
            background: var(--bg-primary);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            box-shadow: var(--shadow-lg);
            font-size: 0.75rem;
            color: var(--text-muted);
            opacity: 0.8;
            transition: opacity 0.3s;
        }
        
        .shortcuts-hint:hover {
            opacity: 1;
        }
        
        .shortcuts-hint span {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        /* Responsive */
        @media (max-width: 640px) {
            .app-container {
                padding: 0 8px;
            }
            
            .header-content {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .header-controls {
                width: 100%;
            }
            
            .search-container {
                max-width: none;
            }
            
            .speaker-legend {
                gap: 8px;
            }
            
            .turn-card {
                padding: 12px;
            }
            
            .turn-avatar {
                width: 32px;
                height: 32px;
                font-size: 0.85rem;
            }
            
            .shortcuts-hint {
                display: none;
            }
        }
        
        /* Print styles */
        @media print {
            .header, .timeline-wrapper, .shortcuts-hint, .speaker-legend {
                display: none !important;
            }
            
            .conversation-container {
                padding: 0;
            }
            
            .turn-card {
                page-break-inside: avoid;
                box-shadow: none;
                border: 1px solid #ddd;
            }
        }
    """


def _get_javascript() -> str:
    """Return the JavaScript for interactivity."""
    return """
        // State
        let currentTurnIndex = -1;
        let searchMatches = [];
        let currentMatchIndex = -1;
        let activeSpeakers = new Set(Object.keys(speakerColors));
        
        // Elements
        const conversation = document.getElementById('conversation');
        const timelineTrack = document.getElementById('timeline-track');
        const timelineCursor = document.getElementById('timeline-cursor');
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        const themeToggle = document.getElementById('theme-toggle');
        const statsToggle = document.getElementById('stats-toggle');
        const statsPanel = document.getElementById('stats-panel');
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Check for saved theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme === 'dark' || (!savedTheme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                document.documentElement.setAttribute('data-theme', 'dark');
            }
            
            setupEventListeners();
        });
        
        function setupEventListeners() {
            // Turn cards
            document.querySelectorAll('.turn-card').forEach((card, index) => {
                card.addEventListener('click', () => selectTurn(index));
            });
            
            // Timeline segments
            document.querySelectorAll('.timeline-segment').forEach((segment, index) => {
                segment.addEventListener('click', (e) => {
                    e.stopPropagation();
                    selectTurn(index);
                });
            });
            
            // Timeline track click
            timelineTrack.addEventListener('click', (e) => {
                const rect = timelineTrack.getBoundingClientRect();
                const percent = (e.clientX - rect.left) / rect.width;
                const targetTime = timelineStart + (percent * totalDuration);
                jumpToTime(targetTime);
            });
            
            // Timeline cursor
            timelineTrack.addEventListener('mousemove', (e) => {
                const rect = timelineTrack.getBoundingClientRect();
                const percent = ((e.clientX - rect.left) / rect.width) * 100;
                timelineCursor.style.left = percent + '%';
            });
            
            // Search
            searchInput.addEventListener('input', debounce(handleSearch, 200));
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    if (e.shiftKey) {
                        navigateSearch(-1);
                    } else {
                        navigateSearch(1);
                    }
                }
                if (e.key === 'Escape') {
                    clearSearch();
                    searchInput.blur();
                }
            });
            
            // Theme toggle
            themeToggle.addEventListener('click', toggleTheme);
            
            // Stats toggle
            statsToggle.addEventListener('click', toggleStats);
            
            // Speaker chips
            document.querySelectorAll('.speaker-chip').forEach(chip => {
                chip.addEventListener('click', () => toggleSpeaker(chip.dataset.speaker));
            });
            
            // Keyboard navigation
            document.addEventListener('keydown', handleKeyboard);
        }
        
        function selectTurn(index) {
            if (index < 0 || index >= turnsData.length) return;
            
            // Update current index
            currentTurnIndex = index;
            
            // Remove previous highlights
            document.querySelectorAll('.turn-card.highlighted, .timeline-segment.active').forEach(el => {
                el.classList.remove('highlighted', 'active');
            });
            
            // Highlight current turn
            const turnCards = document.querySelectorAll('.turn-card');
            const timelineSegments = document.querySelectorAll('.timeline-segment');
            
            if (turnCards[index]) {
                turnCards[index].classList.add('highlighted');
                turnCards[index].scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            if (timelineSegments[index]) {
                timelineSegments[index].classList.add('active');
            }
        }
        
        function jumpToTime(targetTime) {
            // Find the turn that contains this time
            for (let i = 0; i < turnsData.length; i++) {
                const turn = turnsData[i];
                if (turn.start <= targetTime && turn.end >= targetTime) {
                    selectTurn(i);
                    return;
                }
                if (turn.start > targetTime) {
                    selectTurn(Math.max(0, i - 1));
                    return;
                }
            }
            selectTurn(turnsData.length - 1);
        }
        
        function handleSearch() {
            const query = searchInput.value.trim().toLowerCase();
            
            // Clear previous matches
            searchMatches = [];
            currentMatchIndex = -1;
            document.querySelectorAll('.turn-card.search-match').forEach(el => {
                el.classList.remove('search-match');
            });
            document.querySelectorAll('.turn-text').forEach(el => {
                el.innerHTML = el.textContent;
            });
            
            if (!query) {
                searchResults.textContent = '';
                return;
            }
            
            // Find matches
            const turnCards = document.querySelectorAll('.turn-card');
            turnsData.forEach((turn, index) => {
                if (turn.text.toLowerCase().includes(query)) {
                    searchMatches.push(index);
                    turnCards[index].classList.add('search-match');
                    
                    // Highlight text
                    const textEl = turnCards[index].querySelector('.turn-text');
                    if (textEl) {
                        const regex = new RegExp(`(${escapeRegex(query)})`, 'gi');
                        textEl.innerHTML = textEl.textContent.replace(regex, '<mark>$1</mark>');
                    }
                }
            });
            
            // Update results count
            if (searchMatches.length > 0) {
                searchResults.textContent = `${searchMatches.length} found`;
                currentMatchIndex = 0;
                selectTurn(searchMatches[0]);
            } else {
                searchResults.textContent = 'No results';
            }
        }
        
        function navigateSearch(direction) {
            if (searchMatches.length === 0) return;
            
            currentMatchIndex = (currentMatchIndex + direction + searchMatches.length) % searchMatches.length;
            searchResults.textContent = `${currentMatchIndex + 1}/${searchMatches.length}`;
            selectTurn(searchMatches[currentMatchIndex]);
        }
        
        function clearSearch() {
            searchInput.value = '';
            handleSearch();
        }
        
        function toggleTheme() {
            const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
            document.documentElement.setAttribute('data-theme', isDark ? '' : 'dark');
            localStorage.setItem('theme', isDark ? 'light' : 'dark');
        }
        
        function toggleStats() {
            statsPanel.classList.toggle('hidden');
            statsToggle.classList.toggle('active');
        }
        
        function toggleSpeaker(speaker) {
            const chip = document.querySelector(`.speaker-chip[data-speaker="${speaker}"]`);
            const isActive = activeSpeakers.has(speaker);
            
            if (isActive) {
                activeSpeakers.delete(speaker);
                chip.classList.add('dimmed');
            } else {
                activeSpeakers.add(speaker);
                chip.classList.remove('dimmed');
            }
            
            // Update visibility
            document.querySelectorAll('.turn-block').forEach((block, index) => {
                const turn = turnsData[index];
                block.style.display = activeSpeakers.has(turn.speaker) ? '' : 'none';
            });
            
            document.querySelectorAll('.timeline-segment').forEach((segment, index) => {
                const turn = turnsData[index];
                segment.style.opacity = activeSpeakers.has(turn.speaker) ? '' : '0.2';
            });
        }
        
        function handleKeyboard(e) {
            // Don't capture if typing in search
            if (document.activeElement === searchInput && e.key !== 'Escape') {
                return;
            }
            
            switch (e.key) {
                case 'ArrowDown':
                case 'j':
                    e.preventDefault();
                    selectTurn(Math.min(currentTurnIndex + 1, turnsData.length - 1));
                    break;
                case 'ArrowUp':
                case 'k':
                    e.preventDefault();
                    selectTurn(Math.max(currentTurnIndex - 1, 0));
                    break;
                case '/':
                    e.preventDefault();
                    searchInput.focus();
                    break;
                case 'Escape':
                    clearSearch();
                    document.querySelectorAll('.turn-card.highlighted').forEach(el => {
                        el.classList.remove('highlighted');
                    });
                    document.querySelectorAll('.timeline-segment.active').forEach(el => {
                        el.classList.remove('active');
                    });
                    currentTurnIndex = -1;
                    break;
                case 'Home':
                    e.preventDefault();
                    selectTurn(0);
                    break;
                case 'End':
                    e.preventDefault();
                    selectTurn(turnsData.length - 1);
                    break;
            }
        }
        
        // Utility functions
        function debounce(fn, ms) {
            let timeout;
            return function(...args) {
                clearTimeout(timeout);
                timeout = setTimeout(() => fn.apply(this, args), ms);
            };
        }
        
        function escapeRegex(string) {
            return string.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
        }
        
        function formatTime(seconds) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}:${secs.toString().padStart(2, '0')}`;
        }
    """


def _generate_conversation_html(
    turns: List[Any],
    speaker_colors: dict,
    speakers: List[str],
    timeline_start: float
) -> str:
    """Generate the main conversation view HTML."""
    html_parts = []
    
    for turn in turns:
        turn_id = getattr(turn, 'turn_id', 0)
        speaker = getattr(turn, 'primary_speaker', 'Unknown')
        start = getattr(turn, 'start', 0)
        end = getattr(turn, 'end', 0)
        text = getattr(turn, 'text', '')
        word_count = getattr(turn, 'word_count', 0)
        speaking_rate = getattr(turn, 'speaking_rate', 0)
        interjections = getattr(turn, 'interjections', [])
        
        colors = speaker_colors.get(speaker, {"bg": "#f0f0f0", "text": "#333", "accent": "#666"})
        initials = _get_speaker_initials(speaker)
        
        # Format time
        duration = end - start
        time_str = f"{_format_time_label(start)} – {_format_time_label(end)}"
        
        # Build interjections HTML if any
        interjections_html = ""
        if interjections:
            ij_items = []
            for ij in sorted(interjections, key=lambda x: getattr(x, 'start', 0)):
                ij_speaker = getattr(ij, 'speaker', 'Unknown')
                ij_text = getattr(ij, 'text', '')
                ij_start = getattr(ij, 'start', 0)
                ij_type = getattr(ij, 'interjection_type', 'unclear')
                ij_colors = speaker_colors.get(ij_speaker, {"bg": "#f0f0f0", "text": "#333", "accent": "#666"})
                
                ij_items.append(f'''
                    <div class="interjection-item">
                        <span class="interjection-speaker" style="color: {ij_colors['text']}">{escape_html(format_speaker_name(ij_speaker))}:</span>
                        <span class="interjection-text">"{escape_html(ij_text)}"</span>
                        <span class="interjection-type">{ij_type}</span>
                        <span class="interjection-time">{_format_time_label(ij_start)}</span>
                    </div>
                ''')
            
            interjections_html = f'''
                <div class="interjections-container">
                    <div class="interjections-label">Interjections ({len(interjections)})</div>
                    {''.join(ij_items)}
                </div>
            '''
        
        html_parts.append(f'''
            <div class="turn-block" data-turn-id="{turn_id}">
                <div class="turn-card">
                    <div class="turn-avatar" style="background: {colors['bg']}; color: {colors['text']}">
                        {initials}
                    </div>
                    <div class="turn-content">
                        <div class="turn-header">
                            <span class="turn-speaker" style="color: {colors['text']}">{escape_html(format_speaker_name(speaker))}</span>
                            <span class="turn-time">{time_str} ({duration:.1f}s, {word_count} words)</span>
                        </div>
                        <div class="turn-text">{escape_html(text)}</div>
                        {interjections_html}
                    </div>
                </div>
            </div>
        ''')
    
    return '\n'.join(html_parts)


def _generate_mini_timeline_html(
    turns: List[Any],
    speaker_colors: dict,
    timeline_start: float,
    total_duration: float
) -> str:
    """Generate the mini timeline HTML."""
    html_parts = []
    
    for turn in turns:
        turn_id = getattr(turn, 'turn_id', 0)
        speaker = getattr(turn, 'primary_speaker', 'Unknown')
        start = getattr(turn, 'start', 0)
        end = getattr(turn, 'end', 0)
        
        colors = speaker_colors.get(speaker, {"bg": "#f0f0f0", "text": "#333", "accent": "#666"})
        
        left = calculate_position_percent(start, total_duration, timeline_start)
        width = calculate_position_percent(end, total_duration, timeline_start) - left
        width = max(width, 0.3)  # Minimum width for visibility
        
        html_parts.append(f'''
            <div class="timeline-segment" 
                 data-turn-id="{turn_id}"
                 style="left: {left:.2f}%; width: {width:.2f}%; background: {colors['accent']};"
                 title="Turn {turn_id}: {escape_html(format_speaker_name(speaker))}">
            </div>
        ''')
    
    return '\n'.join(html_parts)


def _generate_speaker_legend_html(
    speakers: List[str],
    speaker_colors: dict,
    speaker_statistics: dict
) -> str:
    """Generate the speaker legend HTML."""
    items = []
    for speaker in speakers:
        colors = speaker_colors.get(speaker, {"bg": "#f0f0f0", "text": "#333", "accent": "#666"})
        stats = speaker_statistics.get(speaker, {})
        turn_count = stats.get('total_turns', 0)
        word_count = stats.get('total_words', 0)
        
        items.append(f'''
            <div class="speaker-chip" 
                 data-speaker="{escape_html(speaker)}"
                 style="background: {colors['bg']}; color: {colors['text']}">
                <span>{escape_html(format_speaker_name(speaker))}</span>
                <span class="speaker-stats">{turn_count} turns, {word_count} words</span>
            </div>
        ''')
    return '\n'.join(items)


def _generate_stats_panel_html(
    speaker_statistics: dict,
    speaker_colors: dict,
    conversation_metrics: dict
) -> str:
    """Generate the statistics panel HTML."""
    cards = []
    
    for speaker, stats in speaker_statistics.items():
        colors = speaker_colors.get(speaker, {"bg": "#f0f0f0", "text": "#333", "accent": "#666"})
        
        cards.append(f'''
            <div class="stat-card">
                <div class="stat-card-header">
                    <div class="stat-card-color" style="background: {colors['accent']}"></div>
                    <span class="stat-card-name">{escape_html(format_speaker_name(speaker))}</span>
                </div>
                <div class="stat-grid">
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
                        <span class="stat-label">Speaking Rate</span>
                        <span class="stat-value">{stats.get('avg_speaking_rate', 0):.0f} wpm</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Interjections</span>
                        <span class="stat-value">{stats.get('total_interjections', 0)}</span>
                    </div>
                </div>
            </div>
        ''')
    
    return '\n'.join(cards)


def _get_speaker_initials(speaker: str) -> str:
    """Get initials from speaker name."""
    name = format_speaker_name(speaker)
    words = name.split()
    if len(words) >= 2:
        return (words[0][0] + words[1][0]).upper()
    elif words:
        return words[0][:2].upper()
    return "??"


class HtmlTimelineWriter(OutputWriter):
    """Writer for interactive HTML conversation visualization."""
    
    @property
    def name(self) -> str:
        return "html-timeline"
    
    @property
    def description(self) -> str:
        return "Interactive HTML conversation viewer with timeline navigation"
    
    @property
    def supported_formats(self) -> List[str]:
        return [".html"]
    
    def write(self, turns: TranscriptFlow, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to interactive HTML conversation viewer.
        
        Args:
            turns: TranscriptFlow object
            output_path: Path to write the output file
            word_segments: Optional word segments (not used for this format)
            **kwargs: Additional options
        """
        write_html_timeline(turns, output_path)


# Register the writer
registry.register_output_writer(HtmlTimelineWriter())
