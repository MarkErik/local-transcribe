/**
 * Edit History component
 * 
 * Shows a chronological list of edits made to the transcript
 * with the ability to revert to previous states.
 */

import { useState, useCallback, useMemo } from 'react';
import type { Edit } from '../api/client';

export interface EditHistoryProps {
  /** List of edits */
  edits: Edit[];
  /** Whether the panel is visible */
  isOpen: boolean;
  /** Called to close the panel */
  onClose: () => void;
  /** Called to revert to a specific edit (undo all edits after it) */
  onRevertTo: (editId: number) => void;
  /** Called to jump to the location of an edit */
  onJumpTo?: (turnId: number, wordIndex?: number) => void;
}

const EDIT_TYPE_LABELS: Record<string, string> = {
  word_change: 'Changed word',
  word_insert: 'Inserted word',
  word_delete: 'Deleted word',
  speaker_change: 'Changed speaker',
  merge_words: 'Merged words',
  split_word: 'Split word',
  toggle_interjection: 'Toggled interjection',
  insert_annotation: 'Added annotation',
  turn_merge: 'Merged turns',
  turn_split: 'Split turn',
};

const EDIT_TYPE_ICONS: Record<string, string> = {
  word_change: '✏️',
  word_insert: '➕',
  word_delete: '🗑️',
  speaker_change: '👤',
  merge_words: '🔗',
  split_word: '✂️',
  toggle_interjection: '💬',
  insert_annotation: '📝',
  turn_merge: '⬆️',
  turn_split: '⬇️',
};

function formatTime(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return isoString;
  }
}

function formatDate(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return '';
  }
}

interface EditItemProps {
  edit: Edit;
  isFirst: boolean;
  showDate: boolean;
  onRevertTo: (editId: number) => void;
  onJumpTo?: (turnId: number, wordIndex?: number) => void;
}

function EditItem({ edit, isFirst, showDate, onRevertTo, onJumpTo }: EditItemProps) {
  const [showRevert, setShowRevert] = useState(false);
  
  const handleJump = useCallback(() => {
    if (edit.turn_id != null) {
      onJumpTo?.(edit.turn_id, edit.start_index ?? undefined);
    }
  }, [edit, onJumpTo]);
  
  const handleRevert = useCallback(() => {
    if (window.confirm(`Revert all edits after this point? This will undo ${isFirst ? 'this edit' : 'multiple edits'}.`)) {
      onRevertTo(edit.id);
    }
  }, [edit.id, isFirst, onRevertTo]);
  
  const description = useMemo(() => {
    const parts: string[] = [];
    
    if (edit.original_value) {
      parts.push(`"${edit.original_value}"`);
    }
    if (edit.new_value) {
      parts.push(`→ "${edit.new_value}"`);
    }
    if (edit.turn_id != null) {
      parts.push(`in turn #${edit.turn_id}`);
    }
    
    return parts.join(' ');
  }, [edit]);
  
  return (
    <div
      className="group relative"
      onMouseEnter={() => setShowRevert(true)}
      onMouseLeave={() => setShowRevert(false)}
    >
      {/* Date separator */}
      {showDate && (
        <div className="sticky top-0 bg-gray-100 dark:bg-gray-900 px-3 py-1 text-xs 
                       text-gray-500 dark:text-gray-400 font-medium">
          {formatDate(edit.created_at)}
        </div>
      )}
      
      <div className="flex items-start gap-3 px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700/50">
        {/* Icon */}
        <span className="text-base" title={edit.edit_type}>
          {EDIT_TYPE_ICONS[edit.edit_type] || '📝'}
        </span>
        
        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {EDIT_TYPE_LABELS[edit.edit_type] || edit.edit_type}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {formatTime(edit.created_at)}
            </span>
          </div>
          
          {description && (
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-0.5 truncate">
              {description}
            </p>
          )}
        </div>
        
        {/* Actions */}
        <div className={`flex items-center gap-1 ${showRevert ? 'opacity-100' : 'opacity-0'} transition-opacity`}>
          {edit.turn_id != null && (
            <button
              onClick={handleJump}
              className="p-1 text-gray-400 hover:text-blue-500 rounded"
              title="Jump to location"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                      d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </button>
          )}
          <button
            onClick={handleRevert}
            className="p-1 text-gray-400 hover:text-red-500 rounded"
            title="Revert to this point"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

export function EditHistory({
  edits,
  isOpen,
  onClose,
  onRevertTo,
  onJumpTo,
}: EditHistoryProps) {
  const [filterType, setFilterType] = useState<string>('all');
  
  // Group edits by date
  const filteredEdits = useMemo(() => {
    if (filterType === 'all') return edits;
    return edits.filter(e => e.edit_type === filterType);
  }, [edits, filterType]);
  
  // Track which edits need date separators
  const editDates = useMemo(() => {
    const dates = new Map<number, boolean>();
    let lastDate = '';
    
    for (const edit of filteredEdits) {
      const date = formatDate(edit.created_at);
      dates.set(edit.id, date !== lastDate);
      lastDate = date;
    }
    
    return dates;
  }, [filteredEdits]);
  
  const uniqueEditTypes = useMemo(() => {
    const types = new Set(edits.map(e => e.edit_type));
    return Array.from(types);
  }, [edits]);
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed right-4 top-16 bottom-4 w-80 z-40 bg-white dark:bg-gray-800 
                    rounded-lg shadow-xl border border-gray-200 dark:border-gray-700
                    flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b 
                      border-gray-200 dark:border-gray-700">
        <div>
          <h3 className="font-medium">Edit History</h3>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {edits.length} edit{edits.length !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      {/* Filter */}
      {uniqueEditTypes.length > 1 && (
        <div className="px-3 py-2 border-b border-gray-200 dark:border-gray-700">
          <select
            value={filterType}
            onChange={(e) => setFilterType(e.target.value)}
            className="w-full text-sm px-2 py-1 border rounded dark:bg-gray-700 dark:border-gray-600"
          >
            <option value="all">All edit types</option>
            {uniqueEditTypes.map((type) => (
              <option key={type} value={type}>
                {EDIT_TYPE_ICONS[type]} {EDIT_TYPE_LABELS[type] || type}
              </option>
            ))}
          </select>
        </div>
      )}
      
      {/* Edit list */}
      <div className="flex-1 overflow-y-auto">
        {filteredEdits.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400 dark:text-gray-500">
            No edits yet
          </div>
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-gray-700/50">
            {filteredEdits.map((edit, index) => (
              <EditItem
                key={edit.id}
                edit={edit}
                isFirst={index === filteredEdits.length - 1}
                showDate={editDates.get(edit.id) || false}
                onRevertTo={onRevertTo}
                onJumpTo={onJumpTo}
              />
            ))}
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="px-4 py-2 border-t border-gray-200 dark:border-gray-700 
                      text-xs text-gray-500 dark:text-gray-400">
        Click revert icon to undo edits
      </div>
    </div>
  );
}

export default EditHistory;
