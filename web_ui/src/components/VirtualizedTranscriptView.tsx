/**
 * Virtualized Transcript View Component
 * 
 * High-performance transcript view using simple windowing for large transcripts.
 * Only renders turns near the current scroll position for better performance.
 */

import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import type { TranscriptTurn, PIIReplacement } from '../api';
import { useDeIdentificationStore } from '../store';
import { EditableTurn } from './WordEditor';

export interface VirtualizedTranscriptViewProps {
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
  /** Items to render above/below viewport */
  overscan?: number;
  /** Estimated height per row */
  estimatedRowHeight?: number;
  /** Enable editing mode */
  editingEnabled?: boolean;
  /** Called when a word is changed */
  onWordChange?: (turnId: number, wordIndex: number, newText: string) => void;
  /** Called when a word is deleted */
  onWordDelete?: (turnId: number, wordIndex: number) => void;
  /** Called when text is inserted before a word */
  onWordInsert?: (turnId: number, wordIndex: number, text: string) => void;
  /** Called when speaker is changed */
  onSpeakerChange?: (turnId: number, newSpeaker: string) => void;
  /** Selected word index for highlighting */
  selectedWordIndex?: number | null;
  /** Set of edited word indices per turn */
  editedWordIndices?: Map<number, Set<number>>;
}

// Speaker colors
const SPEAKER_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  Interviewer: {
    bg: 'bg-blue-50',
    border: 'border-blue-300',
    text: 'text-blue-700',
  },
  Participant: {
    bg: 'bg-green-50',
    border: 'border-green-300',
    text: 'text-green-700',
  },
};

function getSpeakerColors(speaker: string) {
  if (SPEAKER_COLORS[speaker]) {
    return SPEAKER_COLORS[speaker];
  }
  
  const lowerSpeaker = speaker.toLowerCase();
  if (lowerSpeaker.includes('interviewer') || lowerSpeaker.includes('ma')) {
    return SPEAKER_COLORS.Interviewer;
  }
  if (lowerSpeaker.includes('participant') || lowerSpeaker.includes('p')) {
    return SPEAKER_COLORS.Participant;
  }
  
  return {
    bg: 'bg-gray-50',
    border: 'border-gray-300',
    text: 'text-gray-700',
  };
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function getPIIHighlightClass(
  wordIndex: number,
  turnId: number,
  piiReplacements: PIIReplacement[],
  piiHighlightEnabled: boolean
): string {
  if (!piiHighlightEnabled || !piiReplacements.length) return '';
  
  const replacement = piiReplacements.find(
    r => r.turn_id === turnId && r.word_index === wordIndex && !r.is_override
  );
  
  if (!replacement) {
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
  
  return 'bg-yellow-100 border-b-2 border-yellow-400';
}

interface WindowedTurnItemProps {
  turn: TranscriptTurn;
  isActive: boolean;
  isSelected: boolean;
  onSeek: (time: number) => void;
  onSelect: (turnId: number) => void;
  onWordSelect?: (turnId: number, wordIndex: number, wordText: string) => void;
  piiReplacements: PIIReplacement[];
  piiHighlightEnabled: boolean;
}

function WindowedTurnItem({ 
  turn, 
  isActive, 
  isSelected, 
  onSeek, 
  onSelect,
  onWordSelect,
  piiReplacements,
  piiHighlightEnabled,
}: WindowedTurnItemProps) {
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
    <div className="px-4 py-1">
      <div
        className={`
          group relative rounded-lg border p-3 cursor-pointer transition-all duration-200
          ${colors.bg} ${colors.border}
          ${isActive ? 'ring-2 ring-indigo-500 ring-offset-2' : ''}
          ${isSelected ? 'ring-2 ring-yellow-400 ring-offset-2' : ''}
          hover:shadow-md
        `}
        onClick={handleClick}
        data-turn-id={turn.turn_id}
      >
        {/* Header row */}
        <div className="flex items-center justify-between mb-1">
          <span className={`font-medium text-sm ${colors.text}`}>
            {turn.primary_speaker}
          </span>
          <button
            onClick={handleTimestampClick}
            className="text-xs text-gray-500 hover:text-indigo-600 font-mono"
            title="Click to seek"
          >
            {formatTimestamp(turn.start)}
          </button>
        </div>
        
        {/* Turn text */}
        <div className="text-gray-800 text-sm leading-relaxed">
          {turn.words && turn.words.length > 0 && (piiHighlightEnabled || onWordSelect) ? (
            turn.words.map((word, idx) => {
              const highlightClass = getPIIHighlightClass(
                idx,
                turn.turn_id,
                piiReplacements,
                piiHighlightEnabled
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
                    ${onWordSelect ? 'cursor-pointer hover:bg-blue-100' : ''}
                    px-0.5 rounded
                  `}
                >
                  {word.word}{idx < turn.words!.length - 1 ? ' ' : ''}
                </span>
              );
            })
          ) : (
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
        
        {/* Turn ID indicator */}
        <div className="absolute top-1 right-1 text-[10px] text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity">
          #{turn.turn_id}
        </div>
      </div>
    </div>
  );
}

export function VirtualizedTranscriptView({
  turns,
  currentTime = 0,
  onSeek,
  autoScroll = true,
  selectedTurnId,
  onTurnSelect,
  onWordSelect,
  overscan = 20,
  estimatedRowHeight = 100,
  editingEnabled = false,
  onWordChange,
  onWordDelete,
  onWordInsert,
  onSpeakerChange,
  selectedWordIndex,
  editedWordIndices,
}: VirtualizedTranscriptViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(800);
  
  // Get PII state from store
  const { piiReplacements, piiHighlightEnabled } = useDeIdentificationStore();
  
  // Find active turn
  const activeTurnIndex = useMemo(() => {
    for (let i = turns.length - 1; i >= 0; i--) {
      if (turns[i].start <= currentTime) {
        return i;
      }
    }
    return 0;
  }, [turns, currentTime]);
  
  const activeTurnId = turns[activeTurnIndex]?.turn_id ?? null;
  
  // Calculate visible window
  const { startIndex, endIndex, totalHeight, offsetY } = useMemo(() => {
    const totalHeight = turns.length * estimatedRowHeight;
    const startIndex = Math.max(0, Math.floor(scrollTop / estimatedRowHeight) - overscan);
    const visibleCount = Math.ceil(containerHeight / estimatedRowHeight);
    const endIndex = Math.min(turns.length - 1, startIndex + visibleCount + overscan * 2);
    const offsetY = startIndex * estimatedRowHeight;
    
    return { startIndex, endIndex, totalHeight, offsetY };
  }, [turns.length, scrollTop, containerHeight, estimatedRowHeight, overscan]);
  
  // Get visible turns
  const visibleTurns = useMemo(() => {
    return turns.slice(startIndex, endIndex + 1);
  }, [turns, startIndex, endIndex]);
  
  // Handle scroll
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);
  
  // Update container height on resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height);
      }
    });
    
    resizeObserver.observe(container);
    setContainerHeight(container.clientHeight);
    
    return () => resizeObserver.disconnect();
  }, []);
  
  // Auto-scroll to active turn
  useEffect(() => {
    if (!autoScroll || !containerRef.current) return;
    
    const container = containerRef.current;
    const targetScroll = activeTurnIndex * estimatedRowHeight - containerHeight / 2 + estimatedRowHeight / 2;
    const clampedScroll = Math.max(0, Math.min(targetScroll, (turns.length * estimatedRowHeight) - containerHeight));
    
    // Only scroll if significantly different
    if (Math.abs(container.scrollTop - clampedScroll) > containerHeight / 2) {
      container.scrollTo({
        top: clampedScroll,
        behavior: 'smooth',
      });
    }
  }, [activeTurnIndex, autoScroll, estimatedRowHeight, containerHeight, turns.length]);
  
  const handleSeek = useCallback((time: number) => {
    onSeek?.(time);
  }, [onSeek]);
  
  const handleSelect = useCallback((turnId: number) => {
    onTurnSelect?.(turnId);
  }, [onTurnSelect]);

  if (turns.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-500">
        No transcript available
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      className="h-full overflow-auto"
    >
      <div
        style={{
          height: totalHeight,
          position: 'relative',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: offsetY,
            left: 0,
            right: 0,
          }}
        >
          {visibleTurns.map((turn) => {
            const isActive = turn.turn_id === activeTurnId;
            const isSelected = turn.turn_id === selectedTurnId;
            
            return editingEnabled ? (
              <div key={turn.turn_id} className="px-4 py-1">
                <EditableTurn
                  turnId={turn.turn_id}
                  speaker={turn.primary_speaker}
                  words={turn.words || []}
                  startTime={turn.start}
                  endTime={turn.end}
                  isActive={isActive}
                  isSelected={isSelected}
                  selectedWordIndex={isSelected ? selectedWordIndex : null}
                  editedWordIndices={editedWordIndices?.get(turn.turn_id) || new Set()}
                  onTurnClick={handleSelect}
                  onWordClick={(turnId, wordIdx) => onWordSelect?.(turnId, wordIdx, turn.words![wordIdx].word)}
                  onWordChange={onWordChange}
                  onWordDelete={onWordDelete}
                  onWordInsert={onWordInsert}
                  onSpeakerChange={onSpeakerChange}
                />
              </div>
            ) : (
              <WindowedTurnItem
                key={turn.turn_id}
                turn={turn}
                isActive={isActive}
                isSelected={isSelected}
                onSeek={handleSeek}
                onSelect={handleSelect}
                onWordSelect={onWordSelect}
                piiReplacements={piiReplacements}
                piiHighlightEnabled={piiHighlightEnabled}
              />
            );
            );
          })}
        </div>
      </div>
      
      {/* Performance indicator */}
      <div className="fixed bottom-4 right-4 bg-gray-800 text-white text-xs px-2 py-1 rounded opacity-50 pointer-events-none">
        Showing {visibleTurns.length} of {turns.length} turns
      </div>
    </div>
  );
}

export default VirtualizedTranscriptView;
