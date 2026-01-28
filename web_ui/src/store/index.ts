import { create } from 'zustand';

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
