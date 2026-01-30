"""Pydantic models for request/response schemas and database entities."""

from web_api.models.schemas import (
    # File upload
    UploadInitRequest,
    UploadInitResponse,
    ChunkUploadResponse,
    UploadCompleteResponse,
    UploadStatusResponse,
    
    # Jobs
    JobCreateRequest,
    JobCreateResponse,
    JobResponse,
    JobListResponse,
    
    # Edits
    EditCreateRequest,
    EditResponse,
    
    # Progress events
    ProgressEvent,
    StageStartEvent,
    BlockProgressEvent,
    StageCompleteEvent,
    JobCompleteEvent,
    JobErrorEvent,
    
    # Health
    HealthResponse,
)

# Database entity dataclasses and enums
from web_api.models.entities import (
    JobStatus,
    UploadStatus,
    Job,
    UploadedFile,
    Edit,
    DeIdentificationState,
    PIIReplacement,
)

__all__ = [
    # Pydantic schemas
    "UploadInitRequest",
    "UploadInitResponse", 
    "ChunkUploadResponse",
    "UploadCompleteResponse",
    "UploadStatusResponse",
    "JobCreateRequest",
    "JobCreateResponse",
    "JobResponse",
    "JobListResponse",
    "EditCreateRequest",
    "EditResponse",
    "ProgressEvent",
    "StageStartEvent",
    "BlockProgressEvent",
    "StageCompleteEvent",
    "JobCompleteEvent",
    "JobErrorEvent",
    "HealthResponse",
    # Database entities
    "JobStatus",
    "UploadStatus",
    "Job",
    "UploadedFile",
    "Edit",
    "DeIdentificationState",
    "PIIReplacement",
]
