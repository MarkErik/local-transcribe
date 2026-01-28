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
  start: number;
  end: number;
  words?: Array<{
    word: string;
    start: number;
    end: number;
  }>;
  interjections?: Array<{
    speaker: string;
    text: string;
    start: number;
    end: number;
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
 * Delete a job and all associated data.
 * Only works for completed, failed, or cancelled jobs.
 */
export async function deleteJob(jobId: string): Promise<{ status: string; job_id: string }> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete job');
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
 * Stage information
 */
export interface TranscriptStage {
  stage: string;
  file: string;
  has_edits: boolean;
}

/**
 * Get available transcript stages for a job.
 */
export async function getAvailableStages(jobId: string): Promise<TranscriptStage[]> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/transcript/stages`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get stages');
  }
  
  const data = await response.json();
  return data.stages;
}

/**
 * Get audio URL for a file ID.
 */
export function getAudioUrl(fileId: string): string {
  return `${API_BASE}/files/${fileId}/audio`;
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

// ==============================================================================
// Edit Types and Functions
// ==============================================================================

export type EditType = 
  | 'word_change'
  | 'word_insert'
  | 'word_delete'
  | 'speaker_change'
  | 'merge_words'
  | 'split_word'
  | 'toggle_interjection'
  | 'insert_annotation'
  | 'turn_merge'
  | 'turn_split';

export interface EditCreateRequest {
  stage_name: string;
  edit_type: EditType;
  turn_id: number;
  start_index?: number;
  end_index?: number;
  original_value?: string;
  new_value?: string;
  target_turn_id?: number;
  annotation_type?: string;
}

export interface Edit {
  id: number;
  job_id: string;
  stage_name: string;
  edit_type: EditType;
  turn_id?: number;
  start_index?: number;
  end_index?: number;
  original_value?: string;
  new_value?: string;
  target_turn_id?: number;
  annotation_type?: string;
  created_at: string;
}

/**
 * Create an edit for a transcript.
 */
export async function createEdit(
  jobId: string,
  edit: EditCreateRequest
): Promise<Edit> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/edits`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(edit),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create edit');
  }
  
  return response.json();
}

/**
 * List all edits for a job.
 */
export async function listEdits(
  jobId: string,
  stage?: string
): Promise<Edit[]> {
  const params = stage ? `?stage=${stage}` : '';
  const response = await fetch(`${API_BASE}/jobs/${jobId}/edits${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to list edits');
  }
  
  const data = await response.json();
  return data.edits;
}

/**
 * Delete an edit (for undo).
 */
export async function deleteEdit(
  jobId: string,
  editId: number
): Promise<void> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/edits/${editId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to delete edit');
  }
}

/**
 * Re-run a job from checkpoint with edits applied.
 */
export async function rerunJob(
  jobId: string,
  startStage?: string
): Promise<JobCreateResponse> {
  const params = startStage ? `?start_stage=${startStage}` : '';
  const response = await fetch(`${API_BASE}/jobs/${jobId}/rerun${params}`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to rerun job');
  }
  
  return response.json();
}

// ==============================================================================
// De-identification API
// ==============================================================================

export interface DiscoveredName {
  name: string;
  source_speaker?: string;
  occurrences: number;
  include: boolean;
}

export interface DeIdentificationStatus {
  job_id: string;
  first_pass_complete: boolean;
  second_pass_complete: boolean;
  discovered_names_count: number;
  reviewed_names_count?: number;
  total_pii_replacements: number;
  manual_redactions: number;
  overrides: number;
}

export interface FirstPassResponse {
  job_id: string;
  discovered_names: DiscoveredName[];
  first_pass_complete: boolean;
  total_names: number;
  message: string;
}

export interface NameListResponse {
  job_id: string;
  discovered_names: DiscoveredName[];
  reviewed_names?: DiscoveredName[];
  first_pass_complete: boolean;
  second_pass_complete: boolean;
}

export interface SecondPassResponse {
  job_id: string;
  second_pass_complete: boolean;
  total_replacements: number;
  message: string;
}

export interface PIIReplacement {
  id: number;
  job_id: string;
  speaker?: string;
  original_text: string;
  replacement_text: string;
  word_index?: number;
  turn_id?: number;
  pass_number?: number;
  is_manual: boolean;
  is_override: boolean;
  timestamp_start?: number;
  created_at: string;
}

export interface PIIReplacementsListResponse {
  job_id: string;
  replacements: PIIReplacement[];
  total: number;
}

/**
 * Get de-identification status for a job.
 */
export async function getDeIdentificationStatus(jobId: string): Promise<DeIdentificationStatus> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/status`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get de-identification status');
  }
  
  return response.json();
}

/**
 * Run first pass of de-identification.
 */
export async function runFirstPass(jobId: string): Promise<FirstPassResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/first-pass`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to run first pass');
  }
  
  return response.json();
}

/**
 * Get discovered names for a job.
 */
export async function getDiscoveredNames(jobId: string): Promise<NameListResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/names`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get discovered names');
  }
  
  return response.json();
}

/**
 * Update name list before second pass.
 */
export async function updateNameList(
  jobId: string,
  names: DiscoveredName[]
): Promise<NameListResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/names`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to update name list');
  }
  
  return response.json();
}

/**
 * Run second pass of de-identification.
 */
export async function runSecondPass(jobId: string): Promise<SecondPassResponse> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/second-pass`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to run second pass');
  }
  
  return response.json();
}

/**
 * Get PII replacements (audit trail) for a job.
 */
export async function getPIIReplacements(
  jobId: string,
  speaker?: string,
  passNumber?: number
): Promise<PIIReplacementsListResponse> {
  const params = new URLSearchParams();
  if (speaker) params.append('speaker', speaker);
  if (passNumber !== undefined) params.append('pass_number', passNumber.toString());
  
  const queryString = params.toString();
  const url = `${API_BASE}/jobs/${jobId}/de-identify/replacements${queryString ? '?' + queryString : ''}`;
  
  const response = await fetch(url);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get PII replacements');
  }
  
  return response.json();
}

/**
 * Manually redact text as PII.
 */
export async function createPIIRedaction(
  jobId: string,
  turnId: number,
  startIndex: number,
  originalText: string,
  replacementText: string = '[NAME]',
  endIndex?: number,
  speaker?: string
): Promise<PIIReplacement> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/redact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      turn_id: turnId,
      start_index: startIndex,
      end_index: endIndex ?? startIndex,
      original_text: originalText,
      replacement_text: replacementText,
      speaker,
    }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to create redaction');
  }
  
  return response.json();
}

/**
 * Restore (un-redact) previously redacted PII.
 */
export async function restorePII(
  jobId: string,
  replacementId: number
): Promise<PIIReplacement> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/de-identify/restore`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ replacement_id: replacementId }),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to restore PII');
  }
  
  return response.json();
}
// ==============================================================================
// Export API
// ==============================================================================

export interface ExportFormat {
  id: string;
  name: string;
  description: string;
  extension: string;
}

export interface ExportFormatsResponse {
  formats: ExportFormat[];
}

export interface ExportOptions {
  include_timestamps?: boolean;
  include_speaker_labels?: boolean;
  include_interjections?: boolean;
  include_metadata?: boolean;
}

export interface ExportRequest {
  format: string;
  stage?: string;
  options?: ExportOptions;
}

/**
 * Get available export formats for a job.
 */
export async function getExportFormats(jobId: string): Promise<ExportFormat[]> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/export/formats`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get export formats');
  }
  
  const data: ExportFormatsResponse = await response.json();
  return data.formats;
}

/**
 * Export transcript and trigger download.
 */
export async function exportTranscript(
  jobId: string,
  request: ExportRequest
): Promise<Blob> {
  const response = await fetch(`${API_BASE}/jobs/${jobId}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to export transcript');
  }
  
  return response.blob();
}

/**
 * Get direct download URL for an export format.
 */
export function getExportUrl(jobId: string, format: string, stage?: string): string {
  const params = stage ? `?stage=${stage}` : '';
  return `${API_BASE}/jobs/${jobId}/export/${format}${params}`;
}

export interface DiffSegment {
  type: 'equal' | 'insert' | 'delete' | 'replace';
  words_a: string[];
  words_b: string[];
  position_a: number;
  position_b: number;
}

export interface DiffData {
  total_words_a: number;
  total_words_b: number;
  matching_words: number;
  inserted_words: number;
  deleted_words: number;
  similarity_ratio: number;
  word_error_rate: number;
  segments: DiffSegment[];
  error?: string;
}

export interface ComparisonData {
  stage_a: string;
  stage_b: string;
  transcript_a: Transcript;
  transcript_b: Transcript;
  diff: DiffData;
}

/**
 * Get comparison data between two pipeline stages.
 */
export async function getComparisonData(
  jobId: string,
  stageA: string,
  stageB: string
): Promise<ComparisonData> {
  const params = new URLSearchParams({
    stage_a: stageA,
    stage_b: stageB,
  });
  
  const response = await fetch(`${API_BASE}/jobs/${jobId}/compare?${params}`);
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get comparison data');
  }
  
  return response.json();
}