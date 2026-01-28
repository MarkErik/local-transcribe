/**
 * Turn Actions component
 * 
 * Provides a toolbar for turn-level operations:
 * - Change speaker
 * - Merge with previous/next turn
 * - Split turn at cursor
 * - Toggle interjection status
 */

import { useState, useCallback } from 'react';

export interface TurnActionsProps {
  /** Turn ID */
  turnId: number;
  /** Current speaker */
  speaker: string;
  /** Previous turn ID (for merge) */
  previousTurnId?: number | null;
  /** Next turn ID (for merge) */
  nextTurnId?: number | null;
  /** Whether this is an interjection */
  isInterjection?: boolean;
  /** Parent turn ID if this is an interjection */
  parentTurnId?: number | null;
  /** Called to change speaker */
  onSpeakerChange?: (turnId: number, newSpeaker: string) => void;
  /** Called to merge turns */
  onMerge?: (firstTurnId: number, secondTurnId: number) => void;
  /** Called to split turn at word index */
  onSplit?: (turnId: number, wordIndex: number) => void;
  /** Called to toggle interjection status */
  onToggleInterjection?: (turnId: number, targetTurnId?: number) => void;
  /** Currently selected word index (for split) */
  selectedWordIndex?: number | null;
  /** Whether toolbar is visible */
  isVisible?: boolean;
}

const SPEAKERS = ['Interviewer', 'Participant'];

export function TurnActions({
  turnId,
  speaker,
  previousTurnId,
  nextTurnId,
  isInterjection = false,
  parentTurnId,
  onSpeakerChange,
  onMerge,
  onSplit,
  onToggleInterjection,
  selectedWordIndex,
  isVisible = true,
}: TurnActionsProps) {
  const [showSpeakerMenu, setShowSpeakerMenu] = useState(false);
  
  const handleSpeakerChange = useCallback((newSpeaker: string) => {
    onSpeakerChange?.(turnId, newSpeaker);
    setShowSpeakerMenu(false);
  }, [turnId, onSpeakerChange]);
  
  const handleMergePrevious = useCallback(() => {
    if (previousTurnId != null) {
      onMerge?.(previousTurnId, turnId);
    }
  }, [previousTurnId, turnId, onMerge]);
  
  const handleMergeNext = useCallback(() => {
    if (nextTurnId != null) {
      onMerge?.(turnId, nextTurnId);
    }
  }, [turnId, nextTurnId, onMerge]);
  
  const handleSplit = useCallback(() => {
    if (selectedWordIndex != null && selectedWordIndex > 0) {
      onSplit?.(turnId, selectedWordIndex);
    }
  }, [turnId, selectedWordIndex, onSplit]);
  
  const handleToggleInterjection = useCallback(() => {
    if (isInterjection && parentTurnId != null) {
      // Converting interjection to primary turn
      onToggleInterjection?.(turnId);
    } else if (!isInterjection && previousTurnId != null) {
      // Converting primary turn to interjection (default to previous turn as parent)
      onToggleInterjection?.(turnId, previousTurnId);
    }
  }, [turnId, isInterjection, parentTurnId, previousTurnId, onToggleInterjection]);
  
  if (!isVisible) return null;
  
  return (
    <div className="flex items-center gap-1 p-1 bg-gray-100 dark:bg-gray-700 rounded text-xs">
      {/* Speaker selector */}
      <div className="relative">
        <button
          className="px-2 py-1 bg-white dark:bg-gray-600 rounded border border-gray-300 
                     dark:border-gray-500 hover:bg-gray-50 dark:hover:bg-gray-500
                     flex items-center gap-1"
          onClick={() => setShowSpeakerMenu(!showSpeakerMenu)}
          title="Change speaker"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
          <span className="hidden sm:inline">{speaker}</span>
        </button>
        
        {showSpeakerMenu && (
          <div className="absolute top-full left-0 mt-1 bg-white dark:bg-gray-800 rounded 
                          shadow-lg border border-gray-200 dark:border-gray-700 z-10 py-1 min-w-[120px]">
            {SPEAKERS.map((s) => (
              <button
                key={s}
                className={`w-full px-3 py-1 text-left hover:bg-gray-100 dark:hover:bg-gray-700
                           ${s === speaker ? 'font-medium text-blue-600 dark:text-blue-400' : ''}`}
                onClick={() => handleSpeakerChange(s)}
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </div>
      
      {/* Merge with previous */}
      <button
        className="px-2 py-1 bg-white dark:bg-gray-600 rounded border border-gray-300 
                   dark:border-gray-500 hover:bg-gray-50 dark:hover:bg-gray-500
                   disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={handleMergePrevious}
        disabled={previousTurnId == null}
        title="Merge with previous turn"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
        </svg>
      </button>
      
      {/* Merge with next */}
      <button
        className="px-2 py-1 bg-white dark:bg-gray-600 rounded border border-gray-300 
                   dark:border-gray-500 hover:bg-gray-50 dark:hover:bg-gray-500
                   disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={handleMergeNext}
        disabled={nextTurnId == null}
        title="Merge with next turn"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      {/* Split turn */}
      <button
        className="px-2 py-1 bg-white dark:bg-gray-600 rounded border border-gray-300 
                   dark:border-gray-500 hover:bg-gray-50 dark:hover:bg-gray-500
                   disabled:opacity-50 disabled:cursor-not-allowed"
        onClick={handleSplit}
        disabled={selectedWordIndex == null || selectedWordIndex <= 0}
        title="Split turn at selected word"
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                d="M8 7h12M8 12h12M8 17h12M4 7v.01M4 12v.01M4 17v.01" />
        </svg>
      </button>
      
      {/* Toggle interjection */}
      <button
        className={`px-2 py-1 rounded border 
                    ${isInterjection 
                      ? 'bg-purple-100 dark:bg-purple-900 border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-300' 
                      : 'bg-white dark:bg-gray-600 border-gray-300 dark:border-gray-500'}
                    hover:bg-purple-50 dark:hover:bg-purple-800
                    disabled:opacity-50 disabled:cursor-not-allowed`}
        onClick={handleToggleInterjection}
        disabled={!isInterjection && previousTurnId == null}
        title={isInterjection ? "Convert to primary turn" : "Convert to interjection"}
      >
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
        </svg>
      </button>
    </div>
  );
}

export default TurnActions;
