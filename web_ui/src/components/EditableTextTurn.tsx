/**
 * Editable Text Turn Component
 * 
 * Block-level text editing for transcript turns using contentEditable.
 * Like a word processor - double-click to edit, click away to finish.
 * Tracks all changes with full undo/redo support.
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import type { TranscriptTurn } from '../api';

export interface EditableTextTurnProps {
  /** The turn to display/edit */
  turn: TranscriptTurn;
  /** Whether this turn is currently active in playback */
  isActive?: boolean;
  /** Whether this turn is selected */
  isSelected?: boolean;
  /** Called when turn is clicked (for selection) */
  onTurnClick?: (turnId: number) => void;
  /** Called when turn text is edited */
  onTextChange?: (turnId: number, newText: string, oldText: string) => void;
  /** Called when speaker is changed */
  onSpeakerChange?: (turnId: number, newSpeaker: string) => void;
  /** Called when seeking to a time */
  onSeek?: (time: number) => void;
}

// Speaker colors
const SPEAKER_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  Interviewer: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-l-blue-400 dark:border-l-blue-600',
    text: 'text-blue-700 dark:text-blue-300',
  },
  Participant: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-l-green-400 dark:border-l-green-600',
    text: 'text-green-700 dark:text-green-300',
  },
};

function getSpeakerColors(speaker: string) {
  const lowerSpeaker = speaker.toLowerCase();
  if (lowerSpeaker.includes('interviewer') || lowerSpeaker.includes('ma')) {
    return SPEAKER_COLORS.Interviewer;
  }
  if (lowerSpeaker.includes('participant') || lowerSpeaker.includes('p')) {
    return SPEAKER_COLORS.Participant;
  }
  
  return {
    bg: 'bg-gray-50 dark:bg-gray-800',
    border: 'border-l-gray-400 dark:border-l-gray-600',
    text: 'text-gray-700 dark:text-gray-300',
  };
}

function formatTimestamp(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function EditableTextTurn({
  turn,
  isActive = false,
  isSelected = false,
  onTurnClick,
  onTextChange,
  onSpeakerChange,
  onSeek,
}: EditableTextTurnProps) {
  const [isEditingText, setIsEditingText] = useState(false);
  const [isEditingSpeaker, setIsEditingSpeaker] = useState(false);
  const [editHistory, setEditHistory] = useState<string[]>([turn.text]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [speakerValue, setSpeakerValue] = useState(turn.primary_speaker);
  
  const textEditRef = useRef<HTMLDivElement>(null);
  const speakerInputRef = useRef<HTMLInputElement>(null);
  const lastTextRef = useRef(turn.text);
  
  const colors = getSpeakerColors(turn.primary_speaker);
  
  // Update speaker value when turn changes
  useEffect(() => {
    setSpeakerValue(turn.primary_speaker);
  }, [turn.primary_speaker]);
  
  // Update text history when turn text changes externally
  useEffect(() => {
    if (turn.text !== lastTextRef.current && !isEditingText) {
      lastTextRef.current = turn.text;
      setEditHistory([turn.text]);
      setHistoryIndex(0);
    }
  }, [turn.text, isEditingText]);
  
  // Focus text editor when editing starts
  useEffect(() => {
    if (isEditingText && textEditRef.current) {
      textEditRef.current.focus();
      // Place cursor at end
      const range = document.createRange();
      const sel = window.getSelection();
      range.selectNodeContents(textEditRef.current);
      range.collapse(false);
      sel?.removeAllRanges();
      sel?.addRange(range);
    }
  }, [isEditingText]);
  
  // Focus speaker input when editing starts
  useEffect(() => {
    if (isEditingSpeaker && speakerInputRef.current) {
      speakerInputRef.current.focus();
      speakerInputRef.current.select();
    }
  }, [isEditingSpeaker]);
  
  // Handle click outside to exit edit mode
  useEffect(() => {
    if (!isEditingText) return;
    
    const handleClickOutside = (e: MouseEvent) => {
      if (textEditRef.current && !textEditRef.current.contains(e.target as Node)) {
        finishTextEditing();
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isEditingText]);
  
  const handleTurnClick = useCallback((e: React.MouseEvent) => {
    if (isEditingText || isEditingSpeaker) return;
    e.stopPropagation();
    onTurnClick?.(turn.turn_id);
  }, [isEditingText, isEditingSpeaker, onTurnClick, turn.turn_id]);
  
  const handleTurnDoubleClick = useCallback((e: React.MouseEvent) => {
    if (isEditingSpeaker) return;
    e.stopPropagation();
    setIsEditingText(true);
  }, [isEditingSpeaker]);
  
  const handleTimestampClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onSeek?.(turn.start);
  }, [turn.start, onSeek]);
  
  const finishTextEditing = useCallback(() => {
    if (!textEditRef.current) return;
    
    const newText = textEditRef.current.innerText.trim();
    const oldText = lastTextRef.current;
    
    if (newText !== oldText && newText) {
      lastTextRef.current = newText;
      onTextChange?.(turn.turn_id, newText, oldText);
      
      // Add to history
      setEditHistory(prev => {
        const newHistory = prev.slice(0, historyIndex + 1);
        newHistory.push(newText);
        return newHistory;
      });
      setHistoryIndex(prev => prev + 1);
    }
    
    setIsEditingText(false);
  }, [turn.turn_id, onTextChange, historyIndex]);
  
  const handleTextKeyDown = useCallback((e: React.KeyboardEvent) => {
    // Cmd/Ctrl + Z for undo
    if ((e.metaKey || e.ctrlKey) && e.key === 'z' && !e.shiftKey) {
      e.preventDefault();
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        if (textEditRef.current) {
          textEditRef.current.innerText = editHistory[newIndex];
        }
      }
      return;
    }
    
    // Cmd/Ctrl + Shift + Z or Cmd/Ctrl + Y for redo
    if (((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'z') || 
        ((e.metaKey || e.ctrlKey) && e.key === 'y')) {
      e.preventDefault();
      if (historyIndex < editHistory.length - 1) {
        const newIndex = historyIndex + 1;
        setHistoryIndex(newIndex);
        if (textEditRef.current) {
          textEditRef.current.innerText = editHistory[newIndex];
        }
      }
      return;
    }
    
    // Escape to cancel
    if (e.key === 'Escape') {
      e.preventDefault();
      if (textEditRef.current) {
        textEditRef.current.innerText = lastTextRef.current;
      }
      setIsEditingText(false);
      return;
    }
  }, [editHistory, historyIndex]);
  
  // Track changes for undo/redo
  const handleTextInput = useCallback(() => {
    // Changes are tracked via keyboard shortcuts (Cmd+Z/Cmd+Shift+Z)
    // We could add auto-history snapshots here for typing pauses if needed
  }, []);
  
  const handleSpeakerDoubleClick = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditingSpeaker(true);
    setSpeakerValue(turn.primary_speaker);
  }, [turn.primary_speaker]);
  
  const handleSpeakerBlur = useCallback(() => {
    if (speakerValue !== turn.primary_speaker && speakerValue.trim()) {
      onSpeakerChange?.(turn.turn_id, speakerValue.trim());
    }
    setIsEditingSpeaker(false);
  }, [speakerValue, turn.primary_speaker, turn.turn_id, onSpeakerChange]);
  
  const handleSpeakerKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSpeakerBlur();
    } else if (e.key === 'Escape') {
      setSpeakerValue(turn.primary_speaker);
      setIsEditingSpeaker(false);
    }
  }, [handleSpeakerBlur, turn.primary_speaker]);
  
  return (
    <div
      className={`
        group relative rounded-lg border-l-4 p-3 mb-2 transition-all duration-200
        ${colors.bg} ${colors.border}
        ${isActive ? 'ring-2 ring-indigo-500 ring-offset-2 dark:ring-offset-gray-900' : ''}
        ${isSelected ? 'ring-2 ring-yellow-400 ring-offset-2 dark:ring-offset-gray-900' : ''}
        ${isEditingText ? 'ring-2 ring-blue-500 ring-offset-2 dark:ring-offset-gray-900 shadow-lg' : 'cursor-pointer hover:shadow-md'}
      `}
      onClick={handleTurnClick}
      onDoubleClick={handleTurnDoubleClick}
      data-turn-id={turn.turn_id}
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        {isEditingSpeaker ? (
          <input
            ref={speakerInputRef}
            type="text"
            value={speakerValue}
            onChange={(e) => setSpeakerValue(e.target.value)}
            onBlur={handleSpeakerBlur}
            onKeyDown={handleSpeakerKeyDown}
            className={`font-medium text-sm ${colors.text} bg-transparent outline-none border-b-2 border-current px-1`}
            onClick={(e) => e.stopPropagation()}
          />
        ) : (
          <span
            className={`font-medium text-sm ${colors.text} cursor-pointer hover:underline px-1`}
            onDoubleClick={handleSpeakerDoubleClick}
            title="Double-click to change speaker"
          >
            {turn.primary_speaker}
          </span>
        )}
        <button
          onClick={handleTimestampClick}
          className="text-xs text-gray-500 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 font-mono"
          title="Click to seek"
        >
          {formatTimestamp(turn.start)}
        </button>
      </div>
      
      {/* Turn text - editable */}
      <div
        ref={textEditRef}
        contentEditable={isEditingText}
        suppressContentEditableWarning
        onKeyDown={handleTextKeyDown}
        onInput={handleTextInput}
        className={`
          text-gray-800 dark:text-gray-200 text-sm leading-relaxed
          ${isEditingText ? 'outline-none bg-white dark:bg-gray-700 p-2 rounded border border-blue-300 dark:border-blue-600' : ''}
        `}
        style={{ 
          minHeight: isEditingText ? '60px' : 'auto',
          whiteSpace: 'pre-wrap',
        }}
      >
        {turn.text}
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
                  inline-block rounded px-2 py-0.5 text-xs mr-2
                  ${interjectionColors.bg} ${interjectionColors.text}
                  border ${interjectionColors.border.replace('border-l-', 'border-')}
                `}
              >
                <span className="font-medium">[{interjection.speaker}]</span> {interjection.text}
              </div>
            );
          })}
        </div>
      )}
      
      {/* Edit mode indicator */}
      {isEditingText && (
        <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 flex items-center space-x-3">
          <span>
            <kbd className="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-600 dark:text-gray-100 dark:border-gray-500">
              ⌘Z
            </kbd> Undo
          </span>
          <span>
            <kbd className="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-600 dark:text-gray-100 dark:border-gray-500">
              ⌘⇧Z
            </kbd> Redo
          </span>
          <span>
            <kbd className="px-1.5 py-0.5 text-xs font-semibold text-gray-800 bg-gray-100 border border-gray-200 rounded dark:bg-gray-600 dark:text-gray-100 dark:border-gray-500">
              Esc
            </kbd> Cancel
          </span>
          <span className="text-gray-400">Click outside to save</span>
        </div>
      )}
      
      {/* Turn ID indicator (subtle) */}
      <div className="absolute top-1 right-1 text-[10px] text-gray-400 dark:text-gray-600 opacity-0 group-hover:opacity-100 transition-opacity">
        #{turn.turn_id}
      </div>
    </div>
  );
}

export default EditableTextTurn;
