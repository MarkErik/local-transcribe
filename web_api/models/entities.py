"""
Database entity models (dataclasses) and enums.

These represent the core data structures stored in the database.
Factory methods (from_row) provide a single source of truth for
converting SQLite rows to entity objects, reducing code duplication.
"""

import json
import sqlite3
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Union


class JobStatus(str, Enum):
    """Status of a transcription job."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadStatus(str, Enum):
    """Status of a file upload."""
    PENDING = "pending"
    UPLOADING = "uploading"
    COMPLETE = "complete"
    FAILED = "failed"


# Type alias for SQLite rows (can be sqlite3.Row or dict-like)
RowType = Union[sqlite3.Row, Dict[str, Any]]


@dataclass
class Job:
    """Represents a transcription job."""
    id: str
    status: JobStatus
    mode: str
    config_json: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    output_dir: Optional[str] = None
    interviewer_file_id: Optional[str] = None
    participant_file_id: Optional[str] = None
    
    @classmethod
    def from_row(cls, row: RowType) -> "Job":
        """Create a Job from a SQLite row or dict.
        
        This factory method provides a single source of truth for
        converting database rows to Job objects, reducing code duplication.
        """
        return cls(
            id=row["id"],
            status=JobStatus(row["status"]),
            mode=row["mode"],
            config_json=row["config_json"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
            output_dir=row["output_dir"],
            interviewer_file_id=row["interviewer_file_id"],
            participant_file_id=row["participant_file_id"],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "mode": self.mode,
            "config": json.loads(self.config_json) if self.config_json else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "output_dir": self.output_dir,
            "interviewer_file_id": self.interviewer_file_id,
            "participant_file_id": self.participant_file_id,
        }


@dataclass
class UploadedFile:
    """Represents an uploaded file."""
    id: str
    original_filename: str
    stored_path: str
    size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    upload_status: UploadStatus = UploadStatus.PENDING
    created_at: Optional[str] = None
    chunks_received: int = 0
    total_chunks: int = 0
    
    @classmethod
    def from_row(cls, row: RowType) -> "UploadedFile":
        """Create an UploadedFile from a SQLite row or dict."""
        return cls(
            id=row["id"],
            original_filename=row["original_filename"],
            stored_path=row["stored_path"],
            size_bytes=row["size_bytes"],
            content_type=row["content_type"],
            upload_status=UploadStatus(row["upload_status"]),
            created_at=row["created_at"],
            chunks_received=row["chunks_received"],
            total_chunks=row["total_chunks"],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "original_filename": self.original_filename,
            "stored_path": self.stored_path,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "upload_status": self.upload_status.value if isinstance(self.upload_status, UploadStatus) else self.upload_status,
            "created_at": self.created_at,
            "chunks_received": self.chunks_received,
            "total_chunks": self.total_chunks,
        }


@dataclass 
class Edit:
    """Represents a transcript edit."""
    id: Optional[int]
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
    created_at: Optional[str] = None
    
    @classmethod
    def from_row(cls, row: RowType) -> "Edit":
        """Create an Edit from a SQLite row or dict."""
        # Handle optional columns that may not exist in older schema versions
        target_turn_id = row["target_turn_id"] if "target_turn_id" in row.keys() else None
        annotation_type = row["annotation_type"] if "annotation_type" in row.keys() else None
        
        return cls(
            id=row["id"],
            job_id=row["job_id"],
            stage_name=row["stage_name"],
            edit_type=row["edit_type"],
            turn_id=row["turn_id"],
            start_index=row["start_index"],
            end_index=row["end_index"],
            original_value=row["original_value"],
            new_value=row["new_value"],
            target_turn_id=target_turn_id,
            annotation_type=annotation_type,
            created_at=row["created_at"],
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "stage_name": self.stage_name,
            "edit_type": self.edit_type,
            "turn_id": self.turn_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "target_turn_id": self.target_turn_id,
            "annotation_type": self.annotation_type,
            "created_at": self.created_at,
        }


@dataclass
class DeIdentificationState:
    """Represents the de-identification state for a job (two-pass progress)."""
    job_id: str
    first_pass_complete: bool = False
    second_pass_complete: bool = False
    discovered_names_json: Optional[str] = None
    reviewed_names_json: Optional[str] = None
    first_pass_segments_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "first_pass_complete": self.first_pass_complete,
            "second_pass_complete": self.second_pass_complete,
            "discovered_names": json.loads(self.discovered_names_json) if self.discovered_names_json else [],
            "reviewed_names": json.loads(self.reviewed_names_json) if self.reviewed_names_json else None,
            "first_pass_segments_path": self.first_pass_segments_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class PIIReplacement:
    """Represents a PII replacement (audit trail entry)."""
    id: Optional[int]
    job_id: str
    speaker: Optional[str]
    original_text: str
    replacement_text: str = "[NAME]"
    word_index: Optional[int] = None
    turn_id: Optional[int] = None
    pass_number: Optional[int] = None  # 1, 2, or None for manual
    is_manual: bool = False
    is_override: bool = False
    timestamp_start: Optional[float] = None
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "speaker": self.speaker,
            "original_text": self.original_text,
            "replacement_text": self.replacement_text,
            "word_index": self.word_index,
            "turn_id": self.turn_id,
            "pass_number": self.pass_number,
            "is_manual": self.is_manual,
            "is_override": self.is_override,
            "timestamp_start": self.timestamp_start,
            "created_at": self.created_at,
        }
