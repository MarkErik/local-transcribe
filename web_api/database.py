"""
SQLite database connection and schema management.

Provides async-compatible database operations for jobs, files, and edits.
"""

import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum

from web_api.config import get_config


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


# SQL Schema definitions
SCHEMA_SQL = """
-- Jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    mode TEXT NOT NULL DEFAULT 'vad_split_audio',
    config_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    output_dir TEXT,
    interviewer_file_id TEXT,
    participant_file_id TEXT
);

-- Uploaded files table
CREATE TABLE IF NOT EXISTS uploaded_files (
    id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    size_bytes INTEGER,
    content_type TEXT,
    upload_status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunks_received INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0
);

-- Edits table
CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    edit_type TEXT NOT NULL,
    turn_id INTEGER,
    start_index INTEGER,
    end_index INTEGER,
    original_value TEXT,
    new_value TEXT,
    target_turn_id INTEGER,
    annotation_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

-- De-identification state table (tracks two-pass progress)
CREATE TABLE IF NOT EXISTS de_identification_state (
    job_id TEXT PRIMARY KEY,
    first_pass_complete BOOLEAN DEFAULT FALSE,
    second_pass_complete BOOLEAN DEFAULT FALSE,
    discovered_names_json TEXT,
    reviewed_names_json TEXT,
    first_pass_segments_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

-- PII replacements table (audit trail for all redactions)
CREATE TABLE IF NOT EXISTS pii_replacements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    speaker TEXT,
    original_text TEXT NOT NULL,
    replacement_text TEXT DEFAULT '[NAME]',
    word_index INTEGER,
    turn_id INTEGER,
    pass_number INTEGER,
    is_manual BOOLEAN DEFAULT FALSE,
    is_override BOOLEAN DEFAULT FALSE,
    timestamp_start REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);

-- Index for faster job lookups
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_edits_job_id ON edits(job_id);
CREATE INDEX IF NOT EXISTS idx_uploaded_files_status ON uploaded_files(upload_status);
CREATE INDEX IF NOT EXISTS idx_pii_replacements_job_id ON pii_replacements(job_id);
"""


class Database:
    """SQLite database manager."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        self.db_path = db_path or get_config().database_path
        self._ensure_schema()
    
    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Job operations
    
    def create_job(self, job: Job) -> Job:
        """Create a new job record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, status, mode, config_json, created_at, 
                                  interviewer_file_id, participant_file_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.status.value if isinstance(job.status, JobStatus) else job.status,
                    job.mode,
                    job.config_json,
                    job.created_at or datetime.now(timezone.utc).isoformat(),
                    job.interviewer_file_id,
                    job.participant_file_id,
                )
            )
            conn.commit()
        return job
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row:
                return Job(
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
        return None
    
    def update_job_status(
        self, 
        job_id: str, 
        status: JobStatus,
        error_message: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        """Update job status and related fields."""
        with self._get_connection() as conn:
            now = datetime.now(timezone.utc).isoformat()
            
            if status == JobStatus.RUNNING:
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                    (status.value, now, job_id)
                )
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                conn.execute(
                    """
                    UPDATE jobs SET status = ?, completed_at = ?, 
                                    error_message = ?, output_dir = ?
                    WHERE id = ?
                    """,
                    (status.value, now, error_message, output_dir, job_id)
                )
            else:
                conn.execute(
                    "UPDATE jobs SET status = ? WHERE id = ?",
                    (status.value, job_id)
                )
            conn.commit()
    
    def list_jobs(
        self, 
        status: Optional[JobStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with optional filtering."""
        with self._get_connection() as conn:
            if status:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs WHERE status = ? 
                    ORDER BY created_at DESC LIMIT ? OFFSET ?
                    """,
                    (status.value, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?
                    """,
                    (limit, offset)
                ).fetchall()
            
            return [
                Job(
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
                for row in rows
            ]
    
    # File operations
    
    def create_uploaded_file(self, file: UploadedFile) -> UploadedFile:
        """Create a new uploaded file record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO uploaded_files 
                (id, original_filename, stored_path, size_bytes, content_type, 
                 upload_status, created_at, chunks_received, total_chunks)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file.id,
                    file.original_filename,
                    file.stored_path,
                    file.size_bytes,
                    file.content_type,
                    file.upload_status.value if isinstance(file.upload_status, UploadStatus) else file.upload_status,
                    file.created_at or datetime.now(timezone.utc).isoformat(),
                    file.chunks_received,
                    file.total_chunks,
                )
            )
            conn.commit()
        return file
    
    def get_uploaded_file(self, file_id: str) -> Optional[UploadedFile]:
        """Get an uploaded file by ID."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM uploaded_files WHERE id = ?", (file_id,)
            ).fetchone()
            if row:
                return UploadedFile(
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
        return None
    
    def update_upload_progress(
        self, 
        file_id: str, 
        chunks_received: int,
        upload_status: Optional[UploadStatus] = None,
    ) -> None:
        """Update upload progress."""
        with self._get_connection() as conn:
            if upload_status:
                conn.execute(
                    """
                    UPDATE uploaded_files 
                    SET chunks_received = ?, upload_status = ?
                    WHERE id = ?
                    """,
                    (chunks_received, upload_status.value, file_id)
                )
            else:
                conn.execute(
                    "UPDATE uploaded_files SET chunks_received = ? WHERE id = ?",
                    (chunks_received, file_id)
                )
            conn.commit()
    
    def update_upload_complete(
        self, 
        file_id: str, 
        stored_path: str,
        size_bytes: int,
    ) -> None:
        """Mark upload as complete with final path and size."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE uploaded_files 
                SET upload_status = ?, stored_path = ?, size_bytes = ?
                WHERE id = ?
                """,
                (UploadStatus.COMPLETE.value, stored_path, size_bytes, file_id)
            )
            conn.commit()
    
    # Edit operations
    
    def create_edit(self, edit: Edit) -> Edit:
        """Create a new edit record."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO edits 
                (job_id, stage_name, edit_type, turn_id, start_index, end_index,
                 original_value, new_value, target_turn_id, annotation_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edit.job_id,
                    edit.stage_name,
                    edit.edit_type,
                    edit.turn_id,
                    edit.start_index,
                    edit.end_index,
                    edit.original_value,
                    edit.new_value,
                    edit.target_turn_id,
                    edit.annotation_type,
                    edit.created_at or datetime.now(timezone.utc).isoformat(),
                )
            )
            edit.id = cursor.lastrowid
            conn.commit()
        return edit
    
    def get_edits_for_job(
        self, 
        job_id: str, 
        stage_name: Optional[str] = None,
    ) -> List[Edit]:
        """Get all edits for a job, optionally filtered by stage."""
        with self._get_connection() as conn:
            if stage_name:
                rows = conn.execute(
                    """
                    SELECT * FROM edits 
                    WHERE job_id = ? AND stage_name = ?
                    ORDER BY created_at ASC
                    """,
                    (job_id, stage_name)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM edits WHERE job_id = ? ORDER BY created_at ASC
                    """,
                    (job_id,)
                ).fetchall()
            
            return [
                Edit(
                    id=row["id"],
                    job_id=row["job_id"],
                    stage_name=row["stage_name"],
                    edit_type=row["edit_type"],
                    turn_id=row["turn_id"],
                    start_index=row["start_index"],
                    end_index=row["end_index"],
                    original_value=row["original_value"],
                    new_value=row["new_value"],
                    target_turn_id=row["target_turn_id"] if "target_turn_id" in row.keys() else None,
                    annotation_type=row["annotation_type"] if "annotation_type" in row.keys() else None,
                    created_at=row["created_at"],
                )
                for row in rows
            ]
    
    def delete_edit(self, edit_id: int) -> bool:
        """Delete an edit by ID. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM edits WHERE id = ?", (edit_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def delete_edits_after(self, job_id: str, edit_id: int) -> int:
        """Delete all edits for a job with id > edit_id (for undo)."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM edits WHERE job_id = ? AND id > ?",
                (job_id, edit_id)
            )
            conn.commit()
            return cursor.rowcount
    
    def cleanup_stale_uploads(self, hours: int = 24) -> int:
        """Delete uploads that have been pending/uploading for too long."""
        with self._get_connection() as conn:
            # SQLite datetime arithmetic
            cursor = conn.execute(
                """
                DELETE FROM uploaded_files 
                WHERE upload_status IN ('pending', 'uploading')
                AND datetime(created_at) < datetime('now', '-' || ? || ' hours')
                """,
                (hours,)
            )
            conn.commit()
            return cursor.rowcount
    
    # De-identification state operations
    
    def get_de_identification_state(self, job_id: str) -> Optional[DeIdentificationState]:
        """Get de-identification state for a job."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM de_identification_state WHERE job_id = ?",
                (job_id,)
            ).fetchone()
            
            if row is None:
                return None
            
            return DeIdentificationState(
                job_id=row["job_id"],
                first_pass_complete=bool(row["first_pass_complete"]),
                second_pass_complete=bool(row["second_pass_complete"]),
                discovered_names_json=row["discovered_names_json"],
                reviewed_names_json=row["reviewed_names_json"],
                first_pass_segments_path=row["first_pass_segments_path"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
    
    def create_de_identification_state(self, state: DeIdentificationState) -> DeIdentificationState:
        """Create de-identification state record."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO de_identification_state 
                (job_id, first_pass_complete, second_pass_complete, discovered_names_json,
                 reviewed_names_json, first_pass_segments_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.job_id,
                    state.first_pass_complete,
                    state.second_pass_complete,
                    state.discovered_names_json,
                    state.reviewed_names_json,
                    state.first_pass_segments_path,
                    state.created_at or datetime.now(timezone.utc).isoformat(),
                )
            )
            conn.commit()
        return state
    
    def update_de_identification_state(
        self,
        job_id: str,
        first_pass_complete: Optional[bool] = None,
        second_pass_complete: Optional[bool] = None,
        discovered_names_json: Optional[str] = None,
        reviewed_names_json: Optional[str] = None,
        first_pass_segments_path: Optional[str] = None,
    ) -> Optional[DeIdentificationState]:
        """Update de-identification state fields."""
        updates = []
        params = []
        
        if first_pass_complete is not None:
            updates.append("first_pass_complete = ?")
            params.append(first_pass_complete)
        if second_pass_complete is not None:
            updates.append("second_pass_complete = ?")
            params.append(second_pass_complete)
        if discovered_names_json is not None:
            updates.append("discovered_names_json = ?")
            params.append(discovered_names_json)
        if reviewed_names_json is not None:
            updates.append("reviewed_names_json = ?")
            params.append(reviewed_names_json)
        if first_pass_segments_path is not None:
            updates.append("first_pass_segments_path = ?")
            params.append(first_pass_segments_path)
        
        if not updates:
            return self.get_de_identification_state(job_id)
        
        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(job_id)
        
        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE de_identification_state SET {', '.join(updates)} WHERE job_id = ?",
                params
            )
            conn.commit()
        
        return self.get_de_identification_state(job_id)
    
    def upsert_de_identification_state(self, state: DeIdentificationState) -> DeIdentificationState:
        """Create or update de-identification state."""
        existing = self.get_de_identification_state(state.job_id)
        if existing:
            return self.update_de_identification_state(
                job_id=state.job_id,
                first_pass_complete=state.first_pass_complete,
                second_pass_complete=state.second_pass_complete,
                discovered_names_json=state.discovered_names_json,
                reviewed_names_json=state.reviewed_names_json,
                first_pass_segments_path=state.first_pass_segments_path,
            )
        return self.create_de_identification_state(state)
    
    # PII replacement operations
    
    def create_pii_replacement(self, replacement: PIIReplacement) -> PIIReplacement:
        """Create a new PII replacement record."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO pii_replacements 
                (job_id, speaker, original_text, replacement_text, word_index, turn_id,
                 pass_number, is_manual, is_override, timestamp_start, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    replacement.job_id,
                    replacement.speaker,
                    replacement.original_text,
                    replacement.replacement_text,
                    replacement.word_index,
                    replacement.turn_id,
                    replacement.pass_number,
                    replacement.is_manual,
                    replacement.is_override,
                    replacement.timestamp_start,
                    replacement.created_at or datetime.now(timezone.utc).isoformat(),
                )
            )
            replacement.id = cursor.lastrowid
            conn.commit()
        return replacement
    
    def get_pii_replacements_for_job(
        self,
        job_id: str,
        speaker: Optional[str] = None,
        pass_number: Optional[int] = None,
        include_overrides: bool = True,
    ) -> List[PIIReplacement]:
        """Get PII replacements for a job with optional filters."""
        query = "SELECT * FROM pii_replacements WHERE job_id = ?"
        params = [job_id]
        
        if speaker:
            query += " AND speaker = ?"
            params.append(speaker)
        if pass_number is not None:
            query += " AND pass_number = ?"
            params.append(pass_number)
        if not include_overrides:
            query += " AND is_override = FALSE"
        
        query += " ORDER BY created_at ASC"
        
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            
            return [
                PIIReplacement(
                    id=row["id"],
                    job_id=row["job_id"],
                    speaker=row["speaker"],
                    original_text=row["original_text"],
                    replacement_text=row["replacement_text"],
                    word_index=row["word_index"],
                    turn_id=row["turn_id"],
                    pass_number=row["pass_number"],
                    is_manual=bool(row["is_manual"]),
                    is_override=bool(row["is_override"]),
                    timestamp_start=row["timestamp_start"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]
    
    def delete_pii_replacement(self, replacement_id: int) -> bool:
        """Delete a PII replacement by ID. Returns True if deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM pii_replacements WHERE id = ?", (replacement_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def bulk_create_pii_replacements(self, replacements: List[PIIReplacement]) -> List[PIIReplacement]:
        """Bulk insert PII replacements for efficiency."""
        if not replacements:
            return []
        
        with self._get_connection() as conn:
            for replacement in replacements:
                cursor = conn.execute(
                    """
                    INSERT INTO pii_replacements 
                    (job_id, speaker, original_text, replacement_text, word_index, turn_id,
                     pass_number, is_manual, is_override, timestamp_start, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        replacement.job_id,
                        replacement.speaker,
                        replacement.original_text,
                        replacement.replacement_text,
                        replacement.word_index,
                        replacement.turn_id,
                        replacement.pass_number,
                        replacement.is_manual,
                        replacement.is_override,
                        replacement.timestamp_start,
                        replacement.created_at or datetime.now(timezone.utc).isoformat(),
                    )
                )
                replacement.id = cursor.lastrowid
            conn.commit()
        return replacements


# Global database instance
_db: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db


def init_database(db_path: Optional[Path] = None) -> Database:
    """Initialize the database (call on startup)."""
    global _db
    _db = Database(db_path)
    return _db
