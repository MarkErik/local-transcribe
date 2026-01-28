import { create } from 'zustand';
import { Edit, EditType } from '../api/client';

interface UploadState {
  // Upload progress per file (by temp ID)
  uploads: Record<string, {
    filename: string;
    progress: number;
    status: 'pending' | 'uploading' | 'complete' | 'error';
    fileId?: string;
    error?: string;
  }>;
  setUpload: (id: string, data: Partial<UploadState['uploads'][string]>) => void;
  clearUpload: (id: string) => void;
  clearAll: () => void;
}

export const useUploadStore = create<UploadState>((set) => ({
  uploads: {},
  
  setUpload: (id, data) =>
    set((state) => ({
      uploads: {
        ...state.uploads,
        [id]: { ...state.uploads[id], ...data },
      },
    })),
  
  clearUpload: (id) =>
    set((state) => {
      const { [id]: _, ...rest } = state.uploads;
      return { uploads: rest };
    }),
  
  clearAll: () => set({ uploads: {} }),
}));

interface JobProgressState {
  // Progress data per job
  jobs: Record<string, {
    status: string;
    currentStage?: string;
    blockProgress?: {
      current: number;
      total: number;
      speaker: string;
    };
    completedStages: string[];
    error?: string;
  }>;
  setJobProgress: (jobId: string, data: Partial<JobProgressState['jobs'][string]>) => void;
  clearJob: (jobId: string) => void;
}

export const useJobProgressStore = create<JobProgressState>((set) => ({
  jobs: {},
  
  setJobProgress: (jobId, data) =>
    set((state) => ({
      jobs: {
        ...state.jobs,
        [jobId]: { 
          ...state.jobs[jobId],
          completedStages: state.jobs[jobId]?.completedStages || [],
          ...data,
        },
      },
    })),
  
  clearJob: (jobId) =>
    set((state) => {
      const { [jobId]: _, ...rest } = state.jobs;
      return { jobs: rest };
    }),
}));

// ==============================================================================
// Edit Store - manages transcript editing state with undo/redo
// ==============================================================================

interface PendingEdit {
  id: string;  // Local temp ID before save
  edit_type: EditType;
  turn_id: number;
  start_index?: number;
  end_index?: number;
  original_value?: string;
  new_value?: string;
}

interface EditState {
  // Current job being edited
  currentJobId: string | null;
  currentStage: string | null;
  
  // Saved edits (from server)
  savedEdits: Edit[];
  
  // Pending edits (not yet saved)
  pendingEdits: PendingEdit[];
  
  // Undo stack (edit IDs to delete for undo)
  undoStack: number[];
  
  // Redo stack (edits to recreate for redo)
  redoStack: PendingEdit[];
  
  // Selection state
  selectedTurnId: number | null;
  selectedWordIndex: number | null;
  
  // Dirty flag (unsaved changes)
  isDirty: boolean;
  
  // Auto-save state
  isSaving: boolean;
  lastSaveError: string | null;
  
  // Actions
  setCurrentJob: (jobId: string, stage: string) => void;
  setSavedEdits: (edits: Edit[]) => void;
  addPendingEdit: (edit: Omit<PendingEdit, 'id'>) => void;
  markEditSaved: (tempId: string, savedEdit: Edit) => void;
  undo: () => PendingEdit | Edit | null;
  redo: () => PendingEdit | null;
  setSelection: (turnId: number | null, wordIndex?: number | null) => void;
  clearSelection: () => void;
  setIsSaving: (saving: boolean) => void;
  setSaveError: (error: string | null) => void;
  reset: () => void;
}

export const useEditStore = create<EditState>((set, get) => ({
  currentJobId: null,
  currentStage: null,
  savedEdits: [],
  pendingEdits: [],
  undoStack: [],
  redoStack: [],
  selectedTurnId: null,
  selectedWordIndex: null,
  isDirty: false,
  isSaving: false,
  lastSaveError: null,
  
  setCurrentJob: (jobId, stage) => set({
    currentJobId: jobId,
    currentStage: stage,
    savedEdits: [],
    pendingEdits: [],
    undoStack: [],
    redoStack: [],
    isDirty: false,
  }),
  
  setSavedEdits: (edits) => set({
    savedEdits: edits,
    undoStack: edits.map(e => e.id),
  }),
  
  addPendingEdit: (edit) => {
    const tempId = `pending-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    set((state) => ({
      pendingEdits: [...state.pendingEdits, { ...edit, id: tempId }],
      redoStack: [], // Clear redo stack on new edit
      isDirty: true,
    }));
  },
  
  markEditSaved: (tempId, savedEdit) => set((state) => ({
    pendingEdits: state.pendingEdits.filter(e => e.id !== tempId),
    savedEdits: [...state.savedEdits, savedEdit],
    undoStack: [...state.undoStack, savedEdit.id],
    isDirty: state.pendingEdits.length > 1, // Still dirty if more pending
  })),
  
  undo: () => {
    const state = get();
    
    // First undo pending edits
    if (state.pendingEdits.length > 0) {
      const lastPending = state.pendingEdits[state.pendingEdits.length - 1];
      set({
        pendingEdits: state.pendingEdits.slice(0, -1),
        redoStack: [...state.redoStack, lastPending],
        isDirty: state.pendingEdits.length > 1,
      });
      return lastPending;
    }
    
    // Then undo saved edits
    if (state.undoStack.length > 0) {
      const lastEditId = state.undoStack[state.undoStack.length - 1];
      const lastEdit = state.savedEdits.find(e => e.id === lastEditId);
      if (lastEdit) {
        set({
          undoStack: state.undoStack.slice(0, -1),
          savedEdits: state.savedEdits.filter(e => e.id !== lastEditId),
          redoStack: [...state.redoStack, {
            id: `redo-${lastEdit.id}`,
            edit_type: lastEdit.edit_type,
            turn_id: lastEdit.turn_id || 0,
            start_index: lastEdit.start_index,
            end_index: lastEdit.end_index,
            original_value: lastEdit.original_value,
            new_value: lastEdit.new_value,
          }],
        });
        return lastEdit;
      }
    }
    
    return null;
  },
  
  redo: () => {
    const state = get();
    if (state.redoStack.length === 0) return null;
    
    const editToRedo = state.redoStack[state.redoStack.length - 1];
    set({
      redoStack: state.redoStack.slice(0, -1),
      pendingEdits: [...state.pendingEdits, editToRedo],
      isDirty: true,
    });
    return editToRedo;
  },
  
  setSelection: (turnId, wordIndex = null) => set({
    selectedTurnId: turnId,
    selectedWordIndex: wordIndex,
  }),
  
  clearSelection: () => set({
    selectedTurnId: null,
    selectedWordIndex: null,
  }),
  
  setIsSaving: (saving) => set({ isSaving: saving }),
  
  setSaveError: (error) => set({ lastSaveError: error }),
  
  reset: () => set({
    currentJobId: null,
    currentStage: null,
    savedEdits: [],
    pendingEdits: [],
    undoStack: [],
    redoStack: [],
    selectedTurnId: null,
    selectedWordIndex: null,
    isDirty: false,
    isSaving: false,
    lastSaveError: null,
  }),
}));
