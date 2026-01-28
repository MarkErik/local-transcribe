/**
 * Transcript view component that displays HierarchicalTurns with speaker colors.
 * 
 * Supports block-level highlighting synchronized with audio playback.
 */

import { useRef, useEffect, useMemo, useCallback } from 'react';
import type { TranscriptTurn } from '../api';

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

interface TurnItemProps {
  turn: TranscriptTurn;
  isActive: boolean;
  isSelected: boolean;
  onSeek: (time: number) => void;
  onSelect: (turnId: number) => void;
}

function TurnItem({ turn, isActive, isSelected, onSeek, onSelect }: TurnItemProps) {
  const colors = getSpeakerColors(turn.primary_speaker);
  
  const handleClick = useCallback(() => {
    onSelect(turn.turn_id);
  }, [turn.turn_id, onSelect]);
  
  const handleTimestampClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onSeek(turn.start_time);
  }, [turn.start_time, onSeek]);

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
      data-start-time={turn.start_time}
      data-end-time={turn.end_time}
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
          {formatTimestamp(turn.start_time)}
        </button>
      </div>
      
      {/* Turn text */}
      <p className="text-gray-800 dark:text-gray-200 text-sm leading-relaxed">
        {turn.text}
      </p>
      
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
}: TranscriptViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const activeTurnRef = useRef<HTMLDivElement | null>(null);
  
  // Find the currently active turn based on playback time
  const activeTurnId = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].start_time <= currentTime) {
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
            />
          </div>
        );
      })}
    </div>
  );
}

export default TranscriptView;
