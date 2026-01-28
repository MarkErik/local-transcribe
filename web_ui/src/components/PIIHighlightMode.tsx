/**
 * PII Highlight Mode component.
 * 
 * Provides a toggle to enable/disable PII highlighting in the transcript view
 * and renders a legend showing the meaning of different highlight colors.
 */

import React from 'react';
import { useDeIdentificationStore } from '../store';

interface PIIHighlightModeProps {
  className?: string;
}

export function PIIHighlightMode({ className = '' }: PIIHighlightModeProps) {
  const { piiHighlightEnabled, togglePIIHighlight, piiReplacements } = useDeIdentificationStore();
  
  // Count replacements by type
  const firstPassCount = piiReplacements.filter(r => r.pass_number === 1 && !r.is_override).length;
  const secondPassCount = piiReplacements.filter(r => r.pass_number === 2 && !r.is_override).length;
  const manualCount = piiReplacements.filter(r => r.is_manual && !r.is_override).length;
  const overrideCount = piiReplacements.filter(r => r.is_override).length;
  
  return (
    <div className={`flex items-center gap-4 ${className}`}>
      {/* Toggle Button */}
      <button
        onClick={togglePIIHighlight}
        className={`
          flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium
          transition-colors duration-150
          ${piiHighlightEnabled 
            ? 'bg-purple-100 text-purple-700 border border-purple-300' 
            : 'bg-gray-100 text-gray-600 border border-gray-300 hover:bg-gray-200'
          }
        `}
        title={piiHighlightEnabled ? 'Hide PII highlights' : 'Show PII highlights'}
      >
        <svg 
          className="w-4 h-4" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          {piiHighlightEnabled ? (
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" 
            />
          ) : (
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth="2" 
              d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" 
            />
          )}
        </svg>
        PII Highlights
      </button>
      
      {/* Legend (shown when highlight mode is enabled) */}
      {piiHighlightEnabled && (
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-yellow-200 border border-yellow-400"></span>
            <span className="text-gray-600">First Pass ({firstPassCount})</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-orange-200 border border-orange-400"></span>
            <span className="text-gray-600">Second Pass ({secondPassCount})</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-red-200 border border-red-400"></span>
            <span className="text-gray-600">Manual ({manualCount})</span>
          </span>
          {overrideCount > 0 && (
            <span className="flex items-center gap-1">
              <span className="w-3 h-3 rounded bg-gray-200 border border-gray-400 line-through"></span>
              <span className="text-gray-600">Restored ({overrideCount})</span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Helper function to get the highlight class for a word based on PII status.
 */
export function getPIIHighlightClass(
  word: string,
  wordIndex: number,
  turnId: number,
  piiReplacements: Array<{
    word_index?: number;
    turn_id?: number;
    pass_number?: number;
    is_manual: boolean;
    is_override: boolean;
  }>,
  piiHighlightEnabled: boolean
): string {
  if (!piiHighlightEnabled) return '';
  
  // Find matching replacement
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

/**
 * Word component with PII highlighting support.
 */
interface PIIWordProps {
  word: string;
  wordIndex: number;
  turnId: number;
  originalText?: string;
  isRedacted?: boolean;
  onClick?: () => void;
  className?: string;
}

export function PIIWord({
  word,
  wordIndex,
  turnId,
  originalText,
  isRedacted,
  onClick,
  className = '',
}: PIIWordProps) {
  const { piiReplacements, piiHighlightEnabled } = useDeIdentificationStore();
  
  const highlightClass = getPIIHighlightClass(
    word,
    wordIndex,
    turnId,
    piiReplacements,
    piiHighlightEnabled
  );
  
  return (
    <span
      onClick={onClick}
      className={`
        cursor-pointer hover:bg-blue-50 px-0.5 rounded
        ${highlightClass}
        ${className}
      `}
      title={isRedacted && originalText ? `Original: ${originalText}` : undefined}
    >
      {word}
    </span>
  );
}

export default PIIHighlightMode;
