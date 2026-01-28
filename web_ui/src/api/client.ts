/**
 * API client for the Local Transcribe backend.
 */

const API_BASE = '/api';

// Types
export interface UploadInitResponse {
  upload_id: string;
  chunk_size: number;
  total_chunks: number;
}

export interface ChunkUploadResponse {
  bytes_received: number;
  chunks_received: number;
  next_chunk: number;
}

export interface UploadCompleteResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  stored_path: string;
}

export interface JobCreateRequest {
  interviewer_file_id: string;
  participant_file_id: string;
  mode?: string;
  options?: {
    enable_de_identification?: boolean;
    enable_cleanup?: boolean;
    output_formats?: string[];
  };
}

export interface JobCreateResponse {
  job_id: string;
  status: string;
}

export interface Job {
  id: string;
  status: string;
  mode: string;
  config?: Record<string, unknown>;
  created_at?: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  output_dir?: string;
  interviewer_file_id?: string;
  participant_file_id?: string;
  duration_seconds?: number;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
  offset: number;
  limit: number;
}

export interface TranscriptTurn {
  turn_id: number;
  primary_speaker: string;
  text: string;
  start_time: number;
  end_time: number;
  words?: Array<{
    word: string;
    start_time: number;
    end_time: number;
  }>;
  interjections?: Array<{
    speaker: string;
    text: string;
    start_time: number;
    end_time: number;
  }>;
  source_block_ids?: number[];
}

export interface Transcript {
  turns: TranscriptTurn[];
  metadata?: Record<string, unknown>;
  conversation_metrics?: Record<string, unknown>;
  speaker_statistics?: Record<string, unknown>;
}

// API Functions

/**
 * Initialize a chunked file upload.
 */
export async function initUpload(
  filename: string,
  sizeBytes: number,
  contentType: string = 'audio/m4a'
): Promise<UploadInitResponse> {
  const response = await fetch(`${API_BASE}/files/upload/init`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename,
      size_bytes: sizeBytes,
      content_type: contentType,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to initialize upload');
  }
  
  return response.json();
}

/**
 * Upload a single chunk of a file.
 */
export async function uploadChunk(
  uploadId: string,
  chunkNum: number,
  chunk: Blob
): Promise<ChunkUploadResponse> {
  const formData = new FormData();
  formData.append('file', chunk);
  
  const response = await fetch(
    `${API_BASE}/files/upload/${uploadId}/chunk/${chunkNum}`,
    {
      method: 'POST',
      body: formData,
    }
  );
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload chunk');
  }
  
  return response.json();
}

/**
 * Complete a chunked upload.
 */
export async function completeUpload(
  uploadId: string
): Promise<UploadCompleteResponse> {
  const response = await fetch(
    `${API_BASE}/files/upload/${uploadId}/complete`,
    { method: 'POST' }
  );
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to complete upload');
  }
  
  return response.json();
}

/**
 * Upload a file with chunking and progress callback.
 */
export async function uploadFile(
  file: File,
  onProgress?: (percent: number) => void
): Promise<UploadCompleteResponse> {
  // Initialize upload
  const initResponse = await initUpload(file.name, file.size, file.type);
  const { upload_id, chunk_size, total_chunks } = initResponse;
  
  // Upload chunks
  for (let i = 0; i < total_chunks; i++) {
    const start = i * chunk_size;
    const end = Math.min(start + chunk_size, file.size);
    const chunk = file.slice(start, end);
    
    await uploadChunk(upload_id, i, chunk);
    
    if (onProgress) {
      onProgress(((i + 1) / total_chunks) * 100);
    }
  }
  
  // Complete upload
  return completeUpload(upload_id);
}

/**
 * Create a new transcription job.
 */
export async function createJob(
  request: JobCreateRequest
): Promise<JobCreateResponse> {
  const response = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create job');
  }
  
  return response.json();
}

/**
 * List all jobs.
 */
export async function listJobs(
  status?: string,
  limit: number = 100,
  offset: number = 0
): Promise<JobListResponse> {
  const params = new URLSearchParams();
  if (status) params.set('status', status);
  params.set('limit', limit.toString());
  params.set('offset', offset.toString());
  
  const response = await fetch(`${API_BASE}/jobs?${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to list jobs');
  }
  
  return response.json();
}

/**
 * Get a job by ID.
 */
export async function getJob(jobId: string): Promise<Job> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Job not found');
  }
  
  return response.json();
}

/**
 * Get transcript for a job.
 */
export async function getTranscript(
  jobId: string,
  stage?: string
): Promise<Transcript> {
  const params = stage ? `?stage=${stage}` : '';
  const response = await fetch(`${API_BASE}/jobs/${jobId}/transcript${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get transcript');
  }
  
  return response.json();
}

/**
 * Subscribe to job progress events via SSE.
 */
export function subscribeToJobProgress(
  jobId: string,
  onEvent: (event: { type: string; data: Record<string, unknown> }) => void,
  onError?: (error: Error) => void
): () => void {
  const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/progress`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent({ type: 'message', data });
    } catch {
      // Ignore parsing errors
    }
  };
  
  eventSource.addEventListener('stage_start', (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    onEvent({ type: 'stage_start', data });
  });
  
  eventSource.addEventListener('block_progress', (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    onEvent({ type: 'block_progress', data });
  });
  
  eventSource.addEventListener('stage_complete', (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    onEvent({ type: 'stage_complete', data });
  });
  
  eventSource.addEventListener('job_complete', (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    onEvent({ type: 'job_complete', data });
    eventSource.close();
  });
  
  eventSource.addEventListener('job_error', (event) => {
    const data = JSON.parse((event as MessageEvent).data);
    onEvent({ type: 'job_error', data });
    eventSource.close();
  });
  
  eventSource.onerror = () => {
    if (onError) {
      onError(new Error('Connection lost'));
    }
  };
  
  // Return cleanup function
  return () => {
    eventSource.close();
  };
}

/**
 * Check API health.
 */
export async function checkHealth(): Promise<{
  status: string;
  version: string;
  database: string;
}> {
  const response = await fetch(`${API_BASE}/health`);
  
  if (!response.ok) {
    throw new Error('API is not healthy');
  }
  
  return response.json();
}
