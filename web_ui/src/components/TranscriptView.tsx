/**
 * Transcript view component that displays HierarchicalTurns with speaker colors.
 * 
 * Supports block-level highlighting synchronized with audio playback.
 * Supports PII highlighting when piiHighlightEnabled is true.
 */

import { useRef, useEffect, useMemo, useCallback } from 'react';
import type { TranscriptTurn, PIIReplacement } from '../api';
import { useDeIdentificationStore } from '../store';

export interface TranscriptViewProps {
  /** Array of transcript turns */
  turns: TranscriptTurn[];
  /** Current playback time in seconds */
  currentTime?: number;
  /** Called when user clicks on a turn to seek */
  onSeek?: (time: number) => void;
  /** Auto-scroll to follow playback */
  autoScroll?: boolean;
  /** Selected turn ID (for editing) */
  selectedTurnId?: number | null;
  /** Called when a turn is selected */
  onTurnSelect?: (turnId: number) => void;
  /** Called when a word is selected for editing */
  onWordSelect?: (turnId: number, wordIndex: number, wordText: string) => void;
  /** PII replacements for highlighting (optional, can also use store) */
  piiReplacements?: PIIReplacement[];
}

// Speaker colors
const SPEAKER_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  Interviewer: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-300 dark:border-blue-700',
    text: 'text-blue-700 dark:text-blue-300',
  },
  Participant: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-300 dark:border-green-700',
    text: 'text-green-700 dark:text-green-300',
  },
};

function getSpeakerColors(speaker: string) {
  // Check for exact match first
  if (SPEAKER_COLORS[speaker]) {
    return SPEAKER_COLORS[speaker];
  }
  
  // Check for partial match
  const lowerSpeaker = speaker.toLowerCase();
  if (lowerSpeaker.includes('interviewer') || lowerSpeaker.includes('ma')) {
    return SPEAKER_COLORS.Interviewer;
  }
  if (lowerSpeaker.includes('participant') || lowerSpeaker.includes('p')) {
    return SPEAKER_COLORS.Participant;
  }
  
  // Default colors
  return {
    bg: 'bg-gray-50 dark:bg-gray-800',
    border: 'border-gray-300 dark:border-gray-700',
    text: 'text-gray-700 dark:text-gray-300',
  };
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Get the PII highlight class for a word based on its redaction status.
 */
function getPIIHighlightClass(
  wordIndex: number,
  turnId: number,
  piiReplacements: PIIReplacement[],
  piiHighlightEnabled: boolean
): string {
  if (!piiHighlightEnabled || !piiReplacements.length) return '';
  
  // Find matching replacement (not overridden)
  const replacement = piiReplacements.find(
    r => r.turn_id === turnId && r.word_index === wordIndex && !r.is_override
  );
  
  if (!replacement) {
    // Check if there's an override (restored)
    const override = piiReplacements.find(
      r => r.turn_id === turnId && r.word_index === wordIndex && r.is_override
    );
    if (override) {
      return 'bg-gray-100 line-through text-gray-500';
    }
    return '';
  }
  
  if (replacement.is_manual) {
    return 'bg-red-100 border-b-2 border-red-400';
  }
  
  if (replacement.pass_number === 2) {
    return 'bg-orange-100 border-b-2 border-orange-400';
  }
  
  // Default: first pass
  return 'bg-yellow-100 border-b-2 border-yellow-400';
}

interface TurnItemProps {
  turn: TranscriptTurn;
  isActive: boolean;
  isSelected: boolean;
  onSeek: (time: number) => void;
  onSelect: (turnId: number) => void;
  onWordSelect?: (turnId: number, wordIndex: number, wordText: string) => void;
  piiReplacements: PIIReplacement[];
  piiHighlightEnabled: boolean;
}

function TurnItem({ 
  turn, 
  isActive, 
  isSelected, 
  onSeek, 
  onSelect,
  onWordSelect,
  piiReplacements,
  piiHighlightEnabled,
}: TurnItemProps) {
  const colors = getSpeakerColors(turn.primary_speaker);
  
  const handleClick = useCallback(() => {
    onSelect(turn.turn_id);
    // Also seek audio to this turn's start time
    onSeek(turn.start);
  }, [turn.turn_id, onSelect, turn.start, onSeek]);
  
  const handleTimestampClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onSeek(turn.start);
  }, [turn.start, onSeek]);

  return (
    <div
      className={`
        group relative rounded-lg border p-3 mb-2 cursor-pointer transition-all duration-200
        ${colors.bg} ${colors.border}
        ${isActive ? 'ring-2 ring-indigo-500 ring-offset-2 dark:ring-offset-gray-900' : ''}
        ${isSelected ? 'ring-2 ring-yellow-400 ring-offset-2 dark:ring-offset-gray-900' : ''}
        hover:shadow-md
      `}
      onClick={handleClick}
      data-turn-id={turn.turn_id}
      data-start-time={turn.start}
      data-end-time={turn.end}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-1">
        <span className={`font-medium text-sm ${colors.text}`}>
          {turn.primary_speaker}
        </span>
        <button
          onClick={handleTimestampClick}
          className="text-xs text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 font-mono"
          title="Click to seek"
        >
          {formatTimestamp(turn.start)}
        </button>
      </div>
      
      {/* Turn text - with optional word-level PII highlighting */}
      <div className="text-gray-800 dark:text-gray-200 text-sm leading-relaxed">
        {turn.words && turn.words.length > 0 && (piiHighlightEnabled || onWordSelect) ? (
          // Render word-by-word for PII highlighting or word selection
          turn.words.map((word, idx) => {
            const highlightClass = getPIIHighlightClass(
              idx,
              turn.turn_id,
              piiReplacements,
              piiHighlightEnabled
            );
            
            // Find if this word was redacted to show tooltip
            const replacement = piiReplacements.find(
              r => r.turn_id === turn.turn_id && r.word_index === idx && !r.is_override
            );
            
            const handleWordClick = onWordSelect ? (e: React.MouseEvent) => {
              e.stopPropagation();
              onWordSelect(turn.turn_id, idx, word.word);
            } : undefined;
            
            return (
              <span
                key={idx}
                onClick={handleWordClick}
                className={`
                  ${highlightClass}
                  ${onWordSelect ? 'cursor-pointer hover:bg-blue-100 dark:hover:bg-blue-900/30' : ''}
                  px-0.5 rounded
                `}
                title={replacement ? `Original: ${replacement.original_text}` : undefined}
              >
                {word.word}{idx < turn.words!.length - 1 ? ' ' : ''}
              </span>
            );
          })
        ) : (
          // Render plain text when no PII highlighting needed
          turn.text
        )}
      </div>
      
      {/* Interjections */}
      {turn.interjections && turn.interjections.length > 0 && (
        <div className="mt-2 space-y-1">
          {turn.interjections.map((interjection, idx) => {
            const interjectionColors = getSpeakerColors(interjection.speaker);
            return (
              <div
                key={idx}
                className={`
                  inline-block rounded px-2 py-0.5 text-xs
                  ${interjectionColors.bg} ${interjectionColors.text}
                  border ${interjectionColors.border}
                `}
              >
                <span className="font-medium">[{interjection.speaker}]</span> {interjection.text}
              </div>
            );
          })}
        </div>
      )}
      
      {/* Turn ID indicator (subtle) */}
      <div className="absolute top-1 right-1 text-[10px] text-gray-400 dark:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity">
        #{turn.turn_id}
      </div>
    </div>
  );
}

export function TranscriptView({
  turns,
  currentTime = 0,
  onSeek,
  autoScroll = true,
  selectedTurnId,
  onTurnSelect,
  onWordSelect,
  piiReplacements: propPiiReplacements,
}: TranscriptViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeTurnRef = useRef<HTMLDivElement | null>(null);
  
  // Get PII state from store (can be overridden by props)
  const { piiReplacements: storePiiReplacements, piiHighlightEnabled } = useDeIdentificationStore();
  const piiReplacements = propPiiReplacements ?? storePiiReplacements;
  
  // Find the currently active turn based on playback time
  const activeTurnId = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].start <= currentTime) {
        return turns[i].turn_id;
      }
    }
    return turns[0]?.turn_id ?? null;
  }, [turns, currentTime]);
  
  // Auto-scroll to active turn
  useEffect(() => {
    if (!autoScroll || !activeTurnRef.current || !containerRef.current) return;
    
    const container = containerRef.current;
    const activeTurn = activeTurnRef.current;
    
    // Calculate if the active turn is visible
    const containerRect = container.getBoundingClientRect();
    const turnRect = activeTurn.getBoundingClientRect();
    
    const isVisible = 
      turnRect.top >= containerRect.top &&
      turnRect.bottom <= containerRect.bottom;
    
    if (!isVisible) {
      activeTurn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeTurnId, autoScroll]);
  
  const handleSeek = useCallback((time: number) => {
    onSeek?.(time);
  }, [onSeek]);
  
  const handleSelect = useCallback((turnId: number) => {
    onTurnSelect?.(turnId);
  }, [onTurnSelect]);

  if (turns.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-500 dark:text-gray-400">
        No transcript available
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-full overflow-y-auto p-4 space-y-1"
    >
      {turns.map((turn) => {
        const isActive = turn.turn_id === activeTurnId;
        const isSelected = turn.turn_id === selectedTurnId;
        
        return (
          <div
            key={turn.turn_id}
            ref={isActive ? (el) => { activeTurnRef.current = el; } : undefined}
          >
            <TurnItem
              turn={turn}
              isActive={isActive}
              isSelected={isSelected}
              onSeek={handleSeek}
              onSelect={handleSelect}
              onWordSelect={onWordSelect}
              piiReplacements={piiReplacements}
              piiHighlightEnabled={piiHighlightEnabled}
            />
          </div>
        );
      })}
    </div>
  );
}

export default TranscriptView;
