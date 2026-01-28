/**
 * Find and Replace component
 * 
 * Provides bulk find & replace functionality across the transcript.
 */

import { useState, useCallback, useEffect, useMemo } from 'react';

export interface FindReplaceMatch {
  turnId: number;
  wordIndex: number;
  word: string;
  context: string; // Surrounding text for preview
}

export interface FindReplaceProps {
  /** Whether the panel is visible */
  isOpen: boolean;
  /** Called to close the panel */
  onClose: () => void;
  /** Transcript turns for searching */
  turns: Array<{
    turn_id: number;
    text: string;
    words?: Array<{ word: string }>;
  }>;
  /** Called when replacing a single match */
  onReplace: (turnId: number, wordIndex: number, newValue: string) => void;
  /** Called when replacing all matches */
  onReplaceAll: (matches: FindReplaceMatch[], newValue: string) => void;
  /** Called to highlight a match in the transcript */
  onHighlightMatch?: (turnId: number, wordIndex: number) => void;
}

export function FindReplace({
  isOpen,
  onClose,
  turns,
  onReplace,
  onReplaceAll,
  onHighlightMatch,
}: FindReplaceProps) {
  const [findText, setFindText] = useState('');
  const [replaceText, setReplaceText] = useState('');
  const [caseSensitive, setCaseSensitive] = useState(false);
  const [wholeWord, setWholeWord] = useState(false);
  const [currentMatchIndex, setCurrentMatchIndex] = useState(0);
  
  // Find all matches
  const matches = useMemo(() => {
    if (!findText.trim()) return [];
    
    const results: FindReplaceMatch[] = [];
    const searchTerm = caseSensitive ? findText : findText.toLowerCase();
    
    for (const turn of turns) {
      const words = turn.words || [];
      
      for (let i = 0; i < words.length; i++) {
        const word = words[i].word;
        const compareWord = caseSensitive ? word : word.toLowerCase();
        
        let isMatch = false;
        if (wholeWord) {
          isMatch = compareWord === searchTerm;
        } else {
          isMatch = compareWord.includes(searchTerm);
        }
        
        if (isMatch) {
          // Build context (3 words before and after)
          const contextStart = Math.max(0, i - 3);
          const contextEnd = Math.min(words.length, i + 4);
          const contextWords = words.slice(contextStart, contextEnd).map(w => w.word);
          const highlightIndex = i - contextStart;
          
          // Mark the match in context
          contextWords[highlightIndex] = `**${contextWords[highlightIndex]}**`;
          
          results.push({
            turnId: turn.turn_id,
            wordIndex: i,
            word: word,
            context: contextWords.join(' '),
          });
        }
      }
    }
    
    return results;
  }, [turns, findText, caseSensitive, wholeWord]);
  
  // Reset current match when search changes
  useEffect(() => {
    setCurrentMatchIndex(0);
  }, [findText, caseSensitive, wholeWord]);
  
  // Highlight current match
  useEffect(() => {
    if (matches.length > 0 && currentMatchIndex < matches.length) {
      const match = matches[currentMatchIndex];
      onHighlightMatch?.(match.turnId, match.wordIndex);
    }
  }, [matches, currentMatchIndex, onHighlightMatch]);
  
  const handlePrevious = useCallback(() => {
    setCurrentMatchIndex((prev) => 
      prev > 0 ? prev - 1 : matches.length - 1
    );
  }, [matches.length]);
  
  const handleNext = useCallback(() => {
    setCurrentMatchIndex((prev) => 
      prev < matches.length - 1 ? prev + 1 : 0
    );
  }, [matches.length]);
  
  const handleReplaceCurrent = useCallback(() => {
    if (matches.length > 0 && currentMatchIndex < matches.length) {
      const match = matches[currentMatchIndex];
      onReplace(match.turnId, match.wordIndex, replaceText);
      // Move to next match after replace
      if (currentMatchIndex >= matches.length - 1) {
        setCurrentMatchIndex(Math.max(0, matches.length - 2));
      }
    }
  }, [matches, currentMatchIndex, replaceText, onReplace]);
  
  const handleReplaceAll = useCallback(() => {
    if (matches.length > 0 && replaceText !== undefined) {
      const confirmMsg = `Replace ${matches.length} occurrence(s) of "${findText}" with "${replaceText}"?`;
      if (window.confirm(confirmMsg)) {
        onReplaceAll(matches, replaceText);
      }
    }
  }, [matches, findText, replaceText, onReplaceAll]);
  
  // Close on Escape
  useEffect(() => {
    if (!isOpen) return;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
      // F3 or Enter for next match
      if (e.key === 'F3' || (e.key === 'Enter' && !e.shiftKey)) {
        e.preventDefault();
        handleNext();
      }
      // Shift+F3 or Shift+Enter for previous match
      if (e.key === 'F3' && e.shiftKey) {
        e.preventDefault();
        handlePrevious();
      }
    };
    
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, handleNext, handlePrevious]);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed top-4 right-4 z-50 bg-white dark:bg-gray-800 rounded-lg shadow-xl 
                    border border-gray-200 dark:border-gray-700 w-96">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b 
                      border-gray-200 dark:border-gray-700">
        <h3 className="font-medium text-sm">Find and Replace</h3>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      {/* Search inputs */}
      <div className="p-4 space-y-3">
        {/* Find input */}
        <div>
          <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Find</label>
          <div className="flex gap-2">
            <input
              type="text"
              value={findText}
              onChange={(e) => setFindText(e.target.value)}
              placeholder="Search text..."
              className="flex-1 px-3 py-1.5 text-sm border rounded 
                        dark:bg-gray-700 dark:border-gray-600
                        focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              autoFocus
            />
            <span className="text-xs text-gray-400 self-center whitespace-nowrap">
              {matches.length > 0 
                ? `${currentMatchIndex + 1} of ${matches.length}` 
                : '0 results'}
            </span>
          </div>
        </div>
        
        {/* Replace input */}
        <div>
          <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Replace</label>
          <input
            type="text"
            value={replaceText}
            onChange={(e) => setReplaceText(e.target.value)}
            placeholder="Replace with..."
            className="w-full px-3 py-1.5 text-sm border rounded 
                      dark:bg-gray-700 dark:border-gray-600
                      focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        
        {/* Options */}
        <div className="flex gap-4 text-xs">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={caseSensitive}
              onChange={(e) => setCaseSensitive(e.target.checked)}
              className="rounded text-blue-500"
            />
            <span>Case sensitive</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={wholeWord}
              onChange={(e) => setWholeWord(e.target.checked)}
              className="rounded text-blue-500"
            />
            <span>Whole word</span>
          </label>
        </div>
        
        {/* Current match preview */}
        {matches.length > 0 && currentMatchIndex < matches.length && (
          <div className="bg-gray-50 dark:bg-gray-900 rounded p-2 text-xs">
            <div className="text-gray-500 dark:text-gray-400 mb-1">
              Turn #{matches[currentMatchIndex].turnId}
            </div>
            <div className="text-gray-700 dark:text-gray-300">
              {matches[currentMatchIndex].context.split('**').map((part, i) => 
                i % 2 === 1 
                  ? <mark key={i} className="bg-yellow-200 dark:bg-yellow-800 px-0.5">{part}</mark>
                  : <span key={i}>{part}</span>
              )}
            </div>
          </div>
        )}
        
        {/* Action buttons */}
        <div className="flex gap-2 pt-2">
          <button
            onClick={handlePrevious}
            disabled={matches.length === 0}
            className="px-3 py-1.5 text-sm border rounded hover:bg-gray-100 
                      dark:hover:bg-gray-700 disabled:opacity-50"
          >
            ← Prev
          </button>
          <button
            onClick={handleNext}
            disabled={matches.length === 0}
            className="px-3 py-1.5 text-sm border rounded hover:bg-gray-100 
                      dark:hover:bg-gray-700 disabled:opacity-50"
          >
            Next →
          </button>
          <button
            onClick={handleReplaceCurrent}
            disabled={matches.length === 0}
            className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded 
                      hover:bg-blue-600 disabled:opacity-50"
          >
            Replace
          </button>
          <button
            onClick={handleReplaceAll}
            disabled={matches.length === 0}
            className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded 
                      hover:bg-blue-600 disabled:opacity-50"
          >
            Replace All
          </button>
        </div>
      </div>
    </div>
  );
}

export default FindReplace;
