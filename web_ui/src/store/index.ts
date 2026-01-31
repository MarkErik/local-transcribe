import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Edit, EditType } from '../api/client';

// ==============================================================================
// Settings Store - persisted user preferences
// ==============================================================================

// Available output formats
export const OUTPUT_FORMATS = {
  'turns-json': 'Turns JSON (structured dialogue data)',
  'timestamped-txt': 'Timestamped Text (readable format with times)',
  'plain-txt': 'Plain Text (just the dialogue)',
  'dialogue-script': 'Dialogue Script (screenplay style)',
  'markdown': 'Markdown (formatted with headers)',
  'srt': 'SRT Subtitles (for video)',
} as const;

export type OutputFormatKey = keyof typeof OUTPUT_FORMATS;

export interface AppSettings {
  // LLM Server URLs
  deIdentificationUrl: string;
  postProcessingUrl: string;
  remoteTranscriptionUrl: string;
  
  // Default job options
  defaultEnableDeIdentification: boolean;
  defaultEnableCleanup: boolean;
  defaultOutputFormats: OutputFormatKey[];
  
  // Transcription settings
  defaultTranscriberProvider: string;
  defaultTranscriberModel: string;
}

interface SettingsState extends AppSettings {
  // Actions
  setDeIdentificationUrl: (url: string) => void;
  setPostProcessingUrl: (url: string) => void;
  setRemoteTranscriptionUrl: (url: string) => void;
  setDefaultEnableDeIdentification: (enabled: boolean) => void;
  setDefaultEnableCleanup: (enabled: boolean) => void;
  setDefaultOutputFormats: (formats: OutputFormatKey[]) => void;
  setDefaultTranscriberProvider: (provider: string) => void;
  setDefaultTranscriberModel: (model: string) => void;
  updateSettings: (settings: Partial<AppSettings>) => void;
  resetToDefaults: () => void;
}

const DEFAULT_SETTINGS: AppSettings = {
  deIdentificationUrl: 'http://100.84.208.72:8080',
  postProcessingUrl: 'http://100.84.208.72:8080',
  remoteTranscriptionUrl: 'http://100.84.208.72:7070',
  defaultEnableDeIdentification: true,
  defaultEnableCleanup: false,
  defaultOutputFormats: ['turns-json', 'timestamped-txt'],
  defaultTranscriberProvider: 'granite',
  defaultTranscriberModel: 'granite-8b',
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      
      setDeIdentificationUrl: (url) => set({ deIdentificationUrl: url }),
      setPostProcessingUrl: (url) => set({ postProcessingUrl: url }),
      setRemoteTranscriptionUrl: (url) => set({ remoteTranscriptionUrl: url }),
      setDefaultEnableDeIdentification: (enabled) => set({ defaultEnableDeIdentification: enabled }),
      setDefaultEnableCleanup: (enabled) => set({ defaultEnableCleanup: enabled }),
      setDefaultOutputFormats: (formats) => set({ defaultOutputFormats: formats }),
      setDefaultTranscriberProvider: (provider) => set({ defaultTranscriberProvider: provider }),
      setDefaultTranscriberModel: (model) => set({ defaultTranscriberModel: model }),
      
      updateSettings: (settings) => set((state) => ({ ...state, ...settings })),
      
      resetToDefaults: () => set(DEFAULT_SETTINGS),
    }),
    {
      name: 'local-transcribe-settings',
    }
  )
);

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
    currentSubstage?: string;
    substageMetadata?: Record<string, unknown>;
    blockProgress?: {
      current: number;
      total: number;
    };
    completedStages: string[];
    error?: string;
  }>;
  setJobProgress: (jobId: string, data: Partial<JobProgressState['jobs'][string]> | ((prev: JobProgressState['jobs'][string]) => Partial<JobProgressState['jobs'][string]>)) => void;
  clearJob: (jobId: string) => void;
}

export const useJobProgressStore = create<JobProgressState>((set) => ({
  jobs: {},
  
  setJobProgress: (jobId, data) =>
    set((state) => {
      // Support both direct Partial objects and functional updates
      let partialData: Partial<JobProgressState['jobs'][string]>;
      
      if (typeof data === 'function') {
        const prev = state.jobs[jobId];
        partialData = data(prev || { status: '', completedStages: [] });
      } else {
        partialData = data;
      }
      
      return {
        jobs: {
          ...state.jobs,
          [jobId]: {
            ...state.jobs[jobId],
            completedStages: state.jobs[jobId]?.completedStages || [],
            status: state.jobs[jobId]?.status || 'running',
            ...partialData,
          },
        },
      };
    }),
  
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
  target_turn_id?: number;
  annotation_type?: string;
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
            target_turn_id: lastEdit.target_turn_id,
            annotation_type: lastEdit.annotation_type,
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

// ==============================================================================
// UI State Store - manages panel visibility and UI state
// ==============================================================================

interface UIState {
  // Panel visibility
  showFindReplace: boolean;
  showEditHistory: boolean;
  showKeyboardHelp: boolean;
  
  // Find/Replace state
  findText: string;
  replaceText: string;
  
  // Annotation menu state
  annotationMenuPosition: { x: number; y: number } | null;
  annotationMenuTurnId: number | null;
  annotationMenuWordIndex: number | null;
  
  // Auto-scroll
  autoScroll: boolean;
  
  // Actions
  toggleFindReplace: () => void;
  toggleEditHistory: () => void;
  toggleKeyboardHelp: () => void;
  setFindText: (text: string) => void;
  setReplaceText: (text: string) => void;
  openAnnotationMenu: (position: { x: number; y: number }, turnId: number, wordIndex: number) => void;
  closeAnnotationMenu: () => void;
  toggleAutoScroll: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  showFindReplace: false,
  showEditHistory: false,
  showKeyboardHelp: false,
  findText: '',
  replaceText: '',
  annotationMenuPosition: null,
  annotationMenuTurnId: null,
  annotationMenuWordIndex: null,
  autoScroll: true,
  
  toggleFindReplace: () => set((state) => ({ showFindReplace: !state.showFindReplace })),
  toggleEditHistory: () => set((state) => ({ showEditHistory: !state.showEditHistory })),
  toggleKeyboardHelp: () => set((state) => ({ showKeyboardHelp: !state.showKeyboardHelp })),
  
  setFindText: (text) => set({ findText: text }),
  setReplaceText: (text) => set({ replaceText: text }),
  
  openAnnotationMenu: (position, turnId, wordIndex) => set({
    annotationMenuPosition: position,
    annotationMenuTurnId: turnId,
    annotationMenuWordIndex: wordIndex,
  }),
  
  closeAnnotationMenu: () => set({
    annotationMenuPosition: null,
    annotationMenuTurnId: null,
    annotationMenuWordIndex: null,
  }),
  
  toggleAutoScroll: () => set((state) => ({ autoScroll: !state.autoScroll })),
}));

// ==============================================================================
// De-identification Store - manages PII review and highlight state
// ==============================================================================

import { DiscoveredName, PIIReplacement } from '../api/client';

interface DeIdentificationState {
  // Current job
  currentJobId: string | null;
  
  // First/second pass status
  firstPassComplete: boolean;
  secondPassComplete: boolean;
  
  // Discovered names from first pass
  discoveredNames: DiscoveredName[];
  
  // PII replacements (audit trail)
  piiReplacements: PIIReplacement[];
  
  // UI state
  piiHighlightEnabled: boolean;
  isLoading: boolean;
  
  // Actions
  setCurrentJobId: (jobId: string | null) => void;
  setFirstPassComplete: (complete: boolean) => void;
  setSecondPassComplete: (complete: boolean) => void;
  setDiscoveredNames: (names: DiscoveredName[]) => void;
  updateNameInclusion: (name: string, include: boolean) => void;
  addDiscoveredName: (name: DiscoveredName) => void;
  removeDiscoveredName: (name: string) => void;
  setPIIReplacements: (replacements: PIIReplacement[]) => void;
  addPIIReplacement: (replacement: PIIReplacement) => void;
  togglePIIHighlight: () => void;
  setLoading: (loading: boolean) => void;
  reset: () => void;
}

export const useDeIdentificationStore = create<DeIdentificationState>((set) => ({
  currentJobId: null,
  firstPassComplete: false,
  secondPassComplete: false,
  discoveredNames: [],
  piiReplacements: [],
  piiHighlightEnabled: false,
  isLoading: false,
  
  setCurrentJobId: (jobId) => set({ currentJobId: jobId }),
  
  setFirstPassComplete: (complete) => set({ firstPassComplete: complete }),
  
  setSecondPassComplete: (complete) => set({ secondPassComplete: complete }),
  
  setDiscoveredNames: (names) => set({ discoveredNames: names }),
  
  updateNameInclusion: (name, include) =>
    set((state) => ({
      discoveredNames: state.discoveredNames.map((n) =>
        n.name === name ? { ...n, include } : n
      ),
    })),
  
  addDiscoveredName: (name) =>
    set((state) => ({
      discoveredNames: [...state.discoveredNames, name],
    })),
  
  removeDiscoveredName: (name) =>
    set((state) => ({
      discoveredNames: state.discoveredNames.filter((n) => n.name !== name),
    })),
  
  setPIIReplacements: (replacements) => set({ piiReplacements: replacements }),
  
  addPIIReplacement: (replacement) =>
    set((state) => ({
      piiReplacements: [...state.piiReplacements, replacement],
    })),
  
  togglePIIHighlight: () => set((state) => ({ piiHighlightEnabled: !state.piiHighlightEnabled })),
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  reset: () => set({
    currentJobId: null,
    firstPassComplete: false,
    secondPassComplete: false,
    discoveredNames: [],
    piiReplacements: [],
    piiHighlightEnabled: false,
    isLoading: false,
  }),
}));
