"""
Pydantic models for API request/response schemas.

These models define the contract between the frontend and backend API.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


# ==============================================================================
# File Upload Schemas
# ==============================================================================

class UploadInitRequest(BaseModel):
    """Request to initialize a chunked file upload."""
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., ge=1, description="Total file size in bytes")
    content_type: str = Field(default="audio/m4a", description="MIME type of the file")


class UploadInitResponse(BaseModel):
    """Response after initializing an upload."""
    upload_id: str = Field(..., description="Unique upload identifier")
    chunk_size: int = Field(..., description="Size of each chunk in bytes")
    total_chunks: int = Field(..., description="Expected number of chunks")


class ChunkUploadResponse(BaseModel):
    """Response after uploading a chunk."""
    bytes_received: int = Field(..., description="Total bytes received so far")
    chunks_received: int = Field(..., description="Number of chunks received")
    next_chunk: int = Field(..., description="Expected next chunk number")


class UploadCompleteResponse(BaseModel):
    """Response after completing an upload."""
    file_id: str = Field(..., description="Unique file identifier (same as upload_id)")
    filename: str = Field(..., description="Original filename")
    size_bytes: int = Field(..., description="Final file size")
    stored_path: str = Field(..., description="Server-side storage path")


class UploadStatusResponse(BaseModel):
    """Response for upload status check."""
    upload_id: str
    filename: str
    status: str  # pending, uploading, complete, failed
    size_bytes: Optional[int] = None
    chunks_received: int = 0
    total_chunks: int = 0
    percent_complete: float = 0.0


# ==============================================================================
# Job Schemas
# ==============================================================================

class JobOptions(BaseModel):
    """Options for job execution."""
    enable_de_identification: bool = Field(default=True, description="Enable PII de-identification")
    enable_cleanup: bool = Field(default=False, description="Enable LLM transcript cleanup")
    output_formats: List[str] = Field(
        default=["turns-json", "timestamped-txt"],
        description="Output formats to generate"
    )
    remote_transcriber_url: Optional[str] = Field(default=None, description="URL for remote transcription server")
    llm_de_identifier_url: Optional[str] = Field(default=None, description="URL for de-identification LLM server")
    llm_transcript_cleanup_url: Optional[str] = Field(default=None, description="URL for transcript cleanup LLM server")
    transcriber_provider: Optional[str] = Field(default=None, description="Transcriber provider to use")
    transcriber_model: Optional[str] = Field(default=None, description="Transcriber model to use")


class JobCreateRequest(BaseModel):
    """Request to create a new transcription job."""
    interviewer_file_id: str = Field(..., description="File ID for interviewer audio")
    participant_file_id: str = Field(..., description="File ID for participant audio")
    name: Optional[str] = Field(None, description="Optional name for the transcription")
    mode: str = Field(default="vad_split_audio", description="Pipeline mode")
    options: JobOptions = Field(default_factory=JobOptions, description="Job options")


class JobCreateResponse(BaseModel):
    """Response after creating a job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(default="pending", description="Initial job status")


class JobResponse(BaseModel):
    """Full job details response."""
    id: str
    status: str
    mode: str
    name: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    interviewer_file_id: Optional[str] = None
    participant_file_id: Optional[str] = None
    # Computed fields
    duration_seconds: Optional[float] = None


class JobListResponse(BaseModel):
    """Response for listing jobs."""
    jobs: List[JobResponse]
    total: int
    offset: int
    limit: int


# ==============================================================================
# Edit Schemas
# ==============================================================================

class EditCreateRequest(BaseModel):
    """Request to save a transcript edit."""
    stage_name: str = Field(..., description="Stage the edit applies to")
    edit_type: str = Field(
        ..., 
        description="Type of edit: word_change, word_insert, word_delete, speaker_change, merge_words, split_word, toggle_interjection, insert_annotation, turn_merge, turn_split, pii_redact, pii_unredact"
    )
    turn_id: int = Field(..., description="Turn ID being edited")
    start_index: Optional[int] = Field(None, description="Start word index (inclusive)")
    end_index: Optional[int] = Field(None, description="End word index (inclusive)")
    original_value: Optional[str] = Field(None, description="Original text/value")
    new_value: Optional[str] = Field(None, description="New text/value")
    # Additional fields for advanced edits
    target_turn_id: Optional[int] = Field(None, description="Target turn ID for turn operations")
    annotation_type: Optional[str] = Field(None, description="Type of annotation (e.g., laughter, pause, inaudible)")


class EditResponse(BaseModel):
    """Response for a saved edit."""
    id: int
    job_id: str
    stage_name: str
    edit_type: str
    turn_id: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    original_value: Optional[str] = None
    new_value: Optional[str] = None
    target_turn_id: Optional[int] = None
    annotation_type: Optional[str] = None
    created_at: str


# ==============================================================================
# Progress Event Schemas (SSE)
# ==============================================================================

class ProgressEvent(BaseModel):
    """Base class for progress events."""
    event_type: str
    job_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StageStartEvent(ProgressEvent):
    """Event emitted when a pipeline stage starts."""
    event_type: Literal["stage_start"] = "stage_start"
    stage: str
    message: str


class BlockProgressEvent(ProgressEvent):
    """Event emitted for per-block progress during VAD transcription."""
    event_type: Literal["block_progress"] = "block_progress"
    stage: str
    current: int
    total: int


class StageCompleteEvent(ProgressEvent):
    """Event emitted when a pipeline stage completes."""
    event_type: Literal["stage_complete"] = "stage_complete"
    stage: str
    duration_s: float
    summary: Optional[Dict[str, Any]] = None


class JobCompleteEvent(ProgressEvent):
    """Event emitted when a job completes successfully."""
    event_type: Literal["job_complete"] = "job_complete"
    status: Literal["completed"] = "completed"
    output_path: str


class JobErrorEvent(ProgressEvent):
    """Event emitted when a job fails."""
    event_type: Literal["job_error"] = "job_error"
    error: str
    stage: Optional[str] = None


# Union type for all progress events
ProgressEventUnion = Union[
    StageStartEvent,
    BlockProgressEvent, 
    StageCompleteEvent,
    JobCompleteEvent,
    JobErrorEvent,
]


# ==============================================================================
# Health Check
# ==============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str
    database: str = "connected"
