/**
 * Word editing component - inline editor for individual words
 * 
 * Provides click-to-edit functionality with audio loop playback.
 */

import { useState, useRef, useEffect, useCallback } from 'react';

interface Word {
  word: string;
  start: number;
  end: number;
  confidence?: number;
}

export interface WordEditorProps {
  /** Word to edit */
  word: Word;
  /** Index of word in the turn */
  wordIndex: number;
  /** Turn ID containing this word */
  turnId: number;
  /** Whether this word is selected */
  isSelected?: boolean;
  /** Whether this word has been edited */
  isEdited?: boolean;
  /** Called when word is clicked */
  onClick?: (wordIndex: number) => void;
  /** Called when word text is changed */
  onChange?: (wordIndex: number, newText: string) => void;
  /** Called when word is deleted */
  onDelete?: (wordIndex: number) => void;
  /** Called when user wants to insert before this word */
  onInsertBefore?: (wordIndex: number, text: string) => void;
}

export function WordEditor({
  word,
  wordIndex,
  isSelected = false,
  isEdited = false,
  onClick,
  onChange,
  onDelete,
  onInsertBefore,
}: WordEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState(word.word);
  const [showInsertPoint, setShowInsertPoint] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  
  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);
  
  // Reset edit value when word changes
  useEffect(() => {
    setEditValue(word.word);
  }, [word.word]);
  
  const handleClick = useCallback(() => {
    onClick?.(wordIndex);
  }, [onClick, wordIndex]);
  
  const handleDoubleClick = useCallback(() => {
    setIsEditing(true);
    setEditValue(word.word);
  }, [word.word]);
  
  const handleBlur = useCallback(() => {
    if (editValue !== word.word && editValue.trim()) {
      onChange?.(wordIndex, editValue.trim());
    }
    setIsEditing(false);
  }, [editValue, word.word, wordIndex, onChange]);
  
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleBlur();
    } else if (e.key === 'Escape') {
      setEditValue(word.word);
      setIsEditing(false);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        onDelete?.(wordIndex);
        setIsEditing(false);
      }
    }
  }, [handleBlur, word.word, wordIndex, onDelete]);
  
  const handleInsertClick = useCallback(() => {
    const text = prompt('Enter word(s) to insert:');
    if (text?.trim()) {
      onInsertBefore?.(wordIndex, text.trim());
    }
  }, [wordIndex, onInsertBefore]);
  
  // Base styles
  const baseClasses = "relative inline-block px-0.5 rounded cursor-pointer transition-all";
  
  // Selection/edit state styles
  const stateClasses = isEditing
    ? "bg-blue-100 ring-2 ring-blue-500"
    : isSelected
    ? "bg-yellow-100 ring-2 ring-yellow-400"
    : isEdited
    ? "bg-green-50 border-b border-dashed border-green-400"
    : "hover:bg-gray-100";
  
  // Confidence indicator (dim low-confidence words)
  const confidenceClasses = word.confidence && word.confidence < 0.7
    ? "opacity-70"
    : "";
  
  if (isEditing) {
    return (
      <span className={`${baseClasses} ${stateClasses}`}>
        <input
          ref={inputRef}
          type="text"
          value={editValue}
          onChange={(e) => setEditValue(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          className="bg-transparent outline-none min-w-[2ch] w-auto text-inherit"
          style={{ width: `${Math.max(editValue.length, 1)}ch` }}
        />
      </span>
    );
  }
  
  return (
    <>
      {/* Insert point indicator (shown on hover between words) */}
      <span
        className="inline-block w-0 overflow-visible cursor-pointer group"
        onMouseEnter={() => setShowInsertPoint(true)}
        onMouseLeave={() => setShowInsertPoint(false)}
        onClick={handleInsertClick}
      >
        {showInsertPoint && (
          <span className="absolute -ml-1 px-0.5 text-blue-500 text-xs bg-blue-100 rounded">
            +
          </span>
        )}
      </span>
      
      <span
        className={`${baseClasses} ${stateClasses} ${confidenceClasses}`}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        title={`${word.start.toFixed(2)}s - ${word.end.toFixed(2)}s${
          word.confidence ? ` (${(word.confidence * 100).toFixed(0)}% confidence)` : ''
        }`}
      >
        {word.word}
      </span>
    </>
  );
}

// ==============================================================================
// Editable Turn Component
// ==============================================================================

interface EditableTurnProps {
  turnId: number;
  speaker: string;
  words: Word[];
  startTime: number;
  endTime: number;
  isActive?: boolean;
  isSelected?: boolean;
  selectedWordIndex?: number | null;
  editedWordIndices?: Set<number>;
  onTurnClick?: (turnId: number) => void;
  onWordClick?: (turnId: number, wordIndex: number) => void;
  onWordChange?: (turnId: number, wordIndex: number, newText: string) => void;
  onWordDelete?: (turnId: number, wordIndex: number) => void;
  onWordInsert?: (turnId: number, wordIndex: number, text: string) => void;
  onSpeakerChange?: (turnId: number, newSpeaker: string) => void;
}

export function EditableTurn({
  turnId,
  speaker,
  words,
  startTime,
  isActive = false,
  isSelected = false,
  selectedWordIndex,
  editedWordIndices = new Set(),
  onTurnClick,
  onWordClick,
  onWordChange,
  onWordDelete,
  onWordInsert,
  onSpeakerChange,
}: EditableTurnProps) {
  const [isEditingSpeaker, setIsEditingSpeaker] = useState(false);
  const [speakerValue, setSpeakerValue] = useState(speaker);
  const speakerInputRef = useRef<HTMLInputElement>(null);
  
  // Focus speaker input when editing
  useEffect(() => {
    if (isEditingSpeaker && speakerInputRef.current) {
      speakerInputRef.current.focus();
      speakerInputRef.current.select();
    }
  }, [isEditingSpeaker]);
  
  const handleTurnClick = useCallback(() => {
    onTurnClick?.(turnId);
  }, [onTurnClick, turnId]);
  
  const handleSpeakerDoubleClick = useCallback(() => {
    setIsEditingSpeaker(true);
    setSpeakerValue(speaker);
  }, [speaker]);
  
  const handleSpeakerBlur = useCallback(() => {
    if (speakerValue !== speaker && speakerValue.trim()) {
      onSpeakerChange?.(turnId, speakerValue.trim());
    }
    setIsEditingSpeaker(false);
  }, [speakerValue, speaker, turnId, onSpeakerChange]);
  
  const handleSpeakerKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSpeakerBlur();
    } else if (e.key === 'Escape') {
      setSpeakerValue(speaker);
      setIsEditingSpeaker(false);
    }
  }, [handleSpeakerBlur, speaker]);
  
  // Format timestamp
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  // Speaker color based on name
  const isInterviewer = speaker.toLowerCase().includes('interviewer');
  const speakerColor = isInterviewer ? 'text-blue-600' : 'text-green-600';
  const borderColor = isInterviewer ? 'border-l-blue-400' : 'border-l-green-400';
  
  // Active/selected styles
  const activeClass = isActive ? 'ring-2 ring-orange-400' : '';
  const selectedClass = isSelected ? 'ring-2 ring-yellow-400 bg-yellow-50' : '';
  
  return (
    <div
      className={`p-3 border-l-4 ${borderColor} bg-white rounded shadow-sm mb-2 
                  cursor-pointer transition-all hover:shadow-md ${activeClass} ${selectedClass}`}
      onClick={handleTurnClick}
    >
      {/* Header: Speaker + Timestamp */}
      <div className="flex items-center justify-between mb-1">
        {isEditingSpeaker ? (
          <input
            ref={speakerInputRef}
            type="text"
            value={speakerValue}
            onChange={(e) => setSpeakerValue(e.target.value)}
            onBlur={handleSpeakerBlur}
            onKeyDown={handleSpeakerKeyDown}
            className={`font-semibold text-sm ${speakerColor} bg-transparent outline-none 
                       border-b-2 border-current`}
          />
        ) : (
          <span
            className={`font-semibold text-sm ${speakerColor} cursor-pointer 
                       hover:underline`}
            onDoubleClick={handleSpeakerDoubleClick}
            title="Double-click to change speaker"
          >
            {speaker}
          </span>
        )}
        <span className="text-xs text-gray-400 font-mono">
          {formatTime(startTime)}
        </span>
      </div>
      
      {/* Words */}
      <div className="leading-relaxed">
        {words.map((word, idx) => (
          <WordEditor
            key={`${turnId}-${idx}`}
            word={word}
            wordIndex={idx}
            turnId={turnId}
            isSelected={selectedWordIndex === idx}
            isEdited={editedWordIndices.has(idx)}
            onClick={(wordIdx) => onWordClick?.(turnId, wordIdx)}
            onChange={(wordIdx, newText) => onWordChange?.(turnId, wordIdx, newText)}
            onDelete={(wordIdx) => onWordDelete?.(turnId, wordIdx)}
            onInsertBefore={(wordIdx, text) => onWordInsert?.(turnId, wordIdx, text)}
          />
        ))}
      </div>
    </div>
  );
}

export default WordEditor;
