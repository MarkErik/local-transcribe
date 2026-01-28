/**
 * Edit Toolbar component
 * 
 * Provides undo/redo buttons, save status, and editing controls.
 */

import { useCallback, useEffect } from 'react';
import { useEditStore } from '../store';
import { createEdit, deleteEdit, rerunJob, listEdits } from '../api/client';

interface EditToolbarProps {
  jobId: string;
  stageName: string;
  onRerunRequested?: (newJobId: string) => void;
}

export function EditToolbar({ jobId, stageName, onRerunRequested }: EditToolbarProps) {
  const {
    isDirty,
    isSaving,
    lastSaveError,
    pendingEdits,
    savedEdits,
    undoStack,
    redoStack,
    undo,
    redo,
    addPendingEdit,
    markEditSaved,
    setIsSaving,
    setSaveError,
    setCurrentJob,
    setSavedEdits,
  } = useEditStore();
  
  // Load existing edits on mount
  useEffect(() => {
    setCurrentJob(jobId, stageName);
    
    listEdits(jobId, stageName)
      .then(setSavedEdits)
      .catch(err => setSaveError(err.message));
  }, [jobId, stageName, setCurrentJob, setSavedEdits, setSaveError]);
  
  // Auto-save pending edits
  useEffect(() => {
    if (pendingEdits.length === 0 || isSaving) return;
    
    const timer = setTimeout(async () => {
      setIsSaving(true);
      setSaveError(null);
      
      try {
        for (const pending of pendingEdits) {
          const saved = await createEdit(jobId, {
            stage_name: stageName,
            edit_type: pending.edit_type,
            turn_id: pending.turn_id,
            start_index: pending.start_index,
            end_index: pending.end_index,
            original_value: pending.original_value,
            new_value: pending.new_value,
          });
          markEditSaved(pending.id, saved);
        }
      } catch (err) {
        setSaveError(err instanceof Error ? err.message : 'Save failed');
      } finally {
        setIsSaving(false);
      }
    }, 2000); // 2 second debounce
    
    return () => clearTimeout(timer);
  }, [pendingEdits, isSaving, jobId, stageName, markEditSaved, setIsSaving, setSaveError]);
  
  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Undo: Ctrl+Z / Cmd+Z
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }
      // Redo: Ctrl+Shift+Z / Cmd+Shift+Z
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) {
        e.preventDefault();
        handleRedo();
      }
      // Also support Ctrl+Y for redo
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') {
        e.preventDefault();
        handleRedo();
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);
  
  const handleUndo = useCallback(async () => {
    const undone = undo();
    if (undone && 'id' in undone && typeof undone.id === 'number') {
      // It's a saved edit - delete from server
      try {
        await deleteEdit(jobId, undone.id);
      } catch (err) {
        console.error('Failed to undo on server:', err);
      }
    }
  }, [undo, jobId]);
  
  const handleRedo = useCallback(() => {
    const redone = redo();
    if (redone) {
      // Re-add as pending edit (will auto-save)
      addPendingEdit({
        edit_type: redone.edit_type,
        turn_id: redone.turn_id,
        start_index: redone.start_index,
        end_index: redone.end_index,
        original_value: redone.original_value,
        new_value: redone.new_value,
      });
    }
  }, [redo, addPendingEdit]);
  
  const handleRerun = useCallback(async () => {
    if (!confirm('Re-run pipeline from this stage? This will create a new job with your edits applied.')) {
      return;
    }
    
    try {
      const result = await rerunJob(jobId, stageName);
      onRerunRequested?.(result.job_id);
    } catch (err) {
      alert(`Failed to rerun: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  }, [jobId, stageName, onRerunRequested]);
  
  const canUndo = pendingEdits.length > 0 || undoStack.length > 0;
  const canRedo = redoStack.length > 0;
  const hasEdits = savedEdits.length > 0 || pendingEdits.length > 0;
  
  return (
    <div className="flex items-center gap-4 px-4 py-2 bg-gray-50 border-b">
      {/* Undo/Redo buttons */}
      <div className="flex items-center gap-1">
        <button
          onClick={handleUndo}
          disabled={!canUndo}
          className="p-1.5 rounded hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
          title="Undo (Ctrl+Z)"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
          </svg>
        </button>
        <button
          onClick={handleRedo}
          disabled={!canRedo}
          className="p-1.5 rounded hover:bg-gray-200 disabled:opacity-40 disabled:cursor-not-allowed"
          title="Redo (Ctrl+Shift+Z)"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d="M21 10h-10a8 8 0 00-8 8v2M21 10l-6 6m6-6l-6-6" />
          </svg>
        </button>
      </div>
      
      {/* Divider */}
      <div className="w-px h-6 bg-gray-300" />
      
      {/* Save status indicator */}
      <div className="flex items-center gap-2 text-sm">
        {isSaving ? (
          <>
            <span className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
            <span className="text-gray-500">Saving...</span>
          </>
        ) : lastSaveError ? (
          <>
            <span className="w-2 h-2 bg-red-500 rounded-full" />
            <span className="text-red-600" title={lastSaveError}>Save failed</span>
          </>
        ) : isDirty ? (
          <>
            <span className="w-2 h-2 bg-yellow-400 rounded-full" />
            <span className="text-gray-500">Unsaved changes</span>
          </>
        ) : (
          <>
            <span className="w-2 h-2 bg-green-500 rounded-full" />
            <span className="text-gray-500">All changes saved</span>
          </>
        )}
      </div>
      
      {/* Edit count */}
      {hasEdits && (
        <>
          <div className="w-px h-6 bg-gray-300" />
          <span className="text-sm text-gray-500">
            {savedEdits.length + pendingEdits.length} edit{savedEdits.length + pendingEdits.length !== 1 ? 's' : ''}
          </span>
        </>
      )}
      
      {/* Spacer */}
      <div className="flex-1" />
      
      {/* Rerun button */}
      {hasEdits && (
        <button
          onClick={handleRerun}
          className="px-3 py-1.5 bg-blue-500 text-white text-sm font-medium rounded 
                     hover:bg-blue-600 transition-colors"
        >
          Re-run Pipeline
        </button>
      )}
    </div>
  );
}

export default EditToolbar;
