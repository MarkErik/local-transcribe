#!/usr/bin/env python3
"""
Tests for database module functionality.

These tests ensure that the database operations work correctly before
and after refactoring. They cover:
- Dataclass serialization (Job, UploadedFile, Edit, etc.)
- CRUD operations for all entities
- Database schema integrity
- Edge cases and error handling
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

# Add parent directory to path for local_transcribe imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_api.database import (
    Database,
    init_database,
    get_database,
    Job,
    JobStatus,
    UploadedFile,
    UploadStatus,
    Edit,
    DeIdentificationState,
    PIIReplacement,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test.db"
        db = Database(db_path)
        yield db


@pytest.fixture
def sample_job():
    """Create a sample Job object."""
    return Job(
        id="test-job-123",
        status=JobStatus.PENDING,
        mode="vad_split_audio",
        config_json=json.dumps({"transcriber": "granite"}),
        created_at=datetime.now(timezone.utc).isoformat(),
        interviewer_file_id="file-int-1",
        participant_file_id="file-part-1",
    )


@pytest.fixture
def sample_uploaded_file():
    """Create a sample UploadedFile object."""
    return UploadedFile(
        id="upload-123",
        original_filename="test_audio.m4a",
        stored_path="/uploads/test_audio.m4a",
        size_bytes=10000,
        content_type="audio/mp4",
        upload_status=UploadStatus.PENDING,
        created_at=datetime.now(timezone.utc).isoformat(),
        chunks_received=0,
        total_chunks=5,
    )


@pytest.fixture
def sample_edit():
    """Create a sample Edit object."""
    return Edit(
        id=None,
        job_id="test-job-123",
        stage_name="manual_edit",
        edit_type="word_change",
        turn_id=1,
        start_index=0,
        end_index=0,
        original_value="hello",
        new_value="Hello",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ==============================================================================
# Dataclass Serialization Tests
# ==============================================================================

class TestJobDataclass:
    """Test Job dataclass functionality."""

    def test_job_to_dict(self, sample_job):
        """Test Job.to_dict() serialization."""
        result = sample_job.to_dict()
        
        assert result["id"] == "test-job-123"
        assert result["status"] == "pending"
        assert result["mode"] == "vad_split_audio"
        assert result["config"]["transcriber"] == "granite"
        assert result["interviewer_file_id"] == "file-int-1"
        assert result["participant_file_id"] == "file-part-1"

    def test_job_status_enum_serialization(self):
        """Test that JobStatus enum serializes correctly."""
        job = Job(id="test", status=JobStatus.RUNNING, mode="test")
        result = job.to_dict()
        assert result["status"] == "running"

    def test_job_with_null_config(self):
        """Test Job with no config_json."""
        job = Job(id="test", status=JobStatus.PENDING, mode="test", config_json=None)
        result = job.to_dict()
        assert result["config"] is None


class TestUploadedFileDataclass:
    """Test UploadedFile dataclass functionality."""

    def test_uploaded_file_to_dict(self, sample_uploaded_file):
        """Test UploadedFile.to_dict() serialization."""
        result = sample_uploaded_file.to_dict()
        
        assert result["id"] == "upload-123"
        assert result["original_filename"] == "test_audio.m4a"
        assert result["stored_path"] == "/uploads/test_audio.m4a"
        assert result["size_bytes"] == 10000
        assert result["upload_status"] == "pending"
        assert result["chunks_received"] == 0
        assert result["total_chunks"] == 5

    def test_upload_status_enum_serialization(self):
        """Test that UploadStatus enum serializes correctly."""
        file = UploadedFile(
            id="test",
            original_filename="test.m4a",
            stored_path="/test",
            upload_status=UploadStatus.COMPLETE,
        )
        result = file.to_dict()
        assert result["upload_status"] == "complete"


class TestEditDataclass:
    """Test Edit dataclass functionality."""

    def test_edit_to_dict(self, sample_edit):
        """Test Edit.to_dict() serialization."""
        result = sample_edit.to_dict()
        
        assert result["job_id"] == "test-job-123"
        assert result["stage_name"] == "manual_edit"
        assert result["edit_type"] == "word_change"
        assert result["turn_id"] == 1
        assert result["original_value"] == "hello"
        assert result["new_value"] == "Hello"

    def test_edit_with_annotation_type(self):
        """Test Edit with annotation_type field."""
        edit = Edit(
            id=1,
            job_id="test",
            stage_name="annotation",
            edit_type="add_annotation",
            turn_id=1,
            annotation_type="[LAUGH]",
        )
        result = edit.to_dict()
        assert result["annotation_type"] == "[LAUGH]"


class TestDeIdentificationStateDataclass:
    """Test DeIdentificationState dataclass functionality."""

    def test_de_identification_state_to_dict(self):
        """Test DeIdentificationState.to_dict() serialization."""
        state = DeIdentificationState(
            job_id="test-job",
            first_pass_complete=True,
            second_pass_complete=False,
            discovered_names_json='[{"name": "John", "occurrences": 3}]',
        )
        result = state.to_dict()
        
        assert result["job_id"] == "test-job"
        assert result["first_pass_complete"] is True
        assert result["second_pass_complete"] is False
        assert result["discovered_names"][0]["name"] == "John"


class TestPIIReplacementDataclass:
    """Test PIIReplacement dataclass functionality."""

    def test_pii_replacement_to_dict(self):
        """Test PIIReplacement.to_dict() serialization."""
        replacement = PIIReplacement(
            id=1,
            job_id="test-job",
            speaker="Interviewer",
            original_text="John",
            replacement_text="[NAME]",
            word_index=5,
            turn_id=2,
            pass_number=1,
            is_manual=False,
            is_override=False,
        )
        result = replacement.to_dict()
        
        assert result["id"] == 1
        assert result["original_text"] == "John"
        assert result["replacement_text"] == "[NAME]"
        assert result["pass_number"] == 1
        assert result["is_manual"] is False


# ==============================================================================
# Job CRUD Tests
# ==============================================================================

class TestJobCRUD:
    """Test Job CRUD operations."""

    def test_create_job(self, temp_db, sample_job):
        """Test creating a new job."""
        created = temp_db.create_job(sample_job)
        
        assert created.id == sample_job.id
        assert created.status == sample_job.status

    def test_get_job(self, temp_db, sample_job):
        """Test retrieving a job by ID."""
        temp_db.create_job(sample_job)
        
        retrieved = temp_db.get_job(sample_job.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_job.id
        assert retrieved.status == JobStatus.PENDING
        assert retrieved.mode == "vad_split_audio"

    def test_get_job_not_found(self, temp_db):
        """Test retrieving a non-existent job returns None."""
        result = temp_db.get_job("non-existent-id")
        assert result is None

    def test_update_job_status_running(self, temp_db, sample_job):
        """Test updating job status to RUNNING."""
        temp_db.create_job(sample_job)
        
        temp_db.update_job_status(sample_job.id, JobStatus.RUNNING)
        
        retrieved = temp_db.get_job(sample_job.id)
        assert retrieved.status == JobStatus.RUNNING
        assert retrieved.started_at is not None

    def test_update_job_status_completed(self, temp_db, sample_job):
        """Test updating job status to COMPLETED."""
        temp_db.create_job(sample_job)
        
        temp_db.update_job_status(
            sample_job.id, 
            JobStatus.COMPLETED, 
            output_dir="/output/test"
        )
        
        retrieved = temp_db.get_job(sample_job.id)
        assert retrieved.status == JobStatus.COMPLETED
        assert retrieved.completed_at is not None
        assert retrieved.output_dir == "/output/test"

    def test_update_job_status_failed(self, temp_db, sample_job):
        """Test updating job status to FAILED with error message."""
        temp_db.create_job(sample_job)
        
        temp_db.update_job_status(
            sample_job.id,
            JobStatus.FAILED,
            error_message="Test error",
        )
        
        retrieved = temp_db.get_job(sample_job.id)
        assert retrieved.status == JobStatus.FAILED
        assert retrieved.error_message == "Test error"

    def test_list_jobs(self, temp_db):
        """Test listing all jobs."""
        for i in range(3):
            job = Job(id=f"job-{i}", status=JobStatus.PENDING, mode="test")
            temp_db.create_job(job)
        
        jobs = temp_db.list_jobs()
        
        assert len(jobs) == 3

    def test_list_jobs_filter_by_status(self, temp_db):
        """Test listing jobs filtered by status."""
        temp_db.create_job(Job(id="job-1", status=JobStatus.PENDING, mode="test"))
        temp_db.create_job(Job(id="job-2", status=JobStatus.RUNNING, mode="test"))
        temp_db.create_job(Job(id="job-3", status=JobStatus.COMPLETED, mode="test"))
        
        pending_jobs = temp_db.list_jobs(status=JobStatus.PENDING)
        
        assert len(pending_jobs) == 1
        assert pending_jobs[0].id == "job-1"

    def test_list_jobs_pagination(self, temp_db):
        """Test listing jobs with pagination."""
        for i in range(10):
            job = Job(id=f"job-{i:02d}", status=JobStatus.PENDING, mode="test")
            temp_db.create_job(job)
        
        first_page = temp_db.list_jobs(limit=5, offset=0)
        second_page = temp_db.list_jobs(limit=5, offset=5)
        
        assert len(first_page) == 5
        assert len(second_page) == 5
        assert first_page[0].id != second_page[0].id

    def test_delete_job(self, temp_db, sample_job):
        """Test deleting a job."""
        temp_db.create_job(sample_job)
        
        result = temp_db.delete_job(sample_job.id)
        
        assert result is True
        assert temp_db.get_job(sample_job.id) is None

    def test_delete_job_not_found(self, temp_db):
        """Test deleting a non-existent job."""
        result = temp_db.delete_job("non-existent")
        assert result is False


# ==============================================================================
# UploadedFile CRUD Tests
# ==============================================================================

class TestUploadedFileCRUD:
    """Test UploadedFile CRUD operations."""

    def test_create_uploaded_file(self, temp_db, sample_uploaded_file):
        """Test creating an uploaded file record."""
        created = temp_db.create_uploaded_file(sample_uploaded_file)
        
        assert created.id == sample_uploaded_file.id

    def test_get_uploaded_file(self, temp_db, sample_uploaded_file):
        """Test retrieving an uploaded file by ID."""
        temp_db.create_uploaded_file(sample_uploaded_file)
        
        retrieved = temp_db.get_uploaded_file(sample_uploaded_file.id)
        
        assert retrieved is not None
        assert retrieved.id == sample_uploaded_file.id
        assert retrieved.original_filename == "test_audio.m4a"
        assert retrieved.total_chunks == 5

    def test_get_uploaded_file_not_found(self, temp_db):
        """Test retrieving a non-existent file returns None."""
        result = temp_db.get_uploaded_file("non-existent")
        assert result is None

    def test_update_upload_progress(self, temp_db, sample_uploaded_file):
        """Test updating upload progress."""
        temp_db.create_uploaded_file(sample_uploaded_file)
        
        temp_db.update_upload_progress(sample_uploaded_file.id, chunks_received=3)
        
        retrieved = temp_db.get_uploaded_file(sample_uploaded_file.id)
        assert retrieved.chunks_received == 3

    def test_update_upload_progress_with_status(self, temp_db, sample_uploaded_file):
        """Test updating upload progress with status change."""
        temp_db.create_uploaded_file(sample_uploaded_file)
        
        temp_db.update_upload_progress(
            sample_uploaded_file.id,
            chunks_received=5,
            upload_status=UploadStatus.UPLOADING,
        )
        
        retrieved = temp_db.get_uploaded_file(sample_uploaded_file.id)
        assert retrieved.chunks_received == 5
        assert retrieved.upload_status == UploadStatus.UPLOADING

    def test_update_upload_complete(self, temp_db, sample_uploaded_file):
        """Test marking upload as complete."""
        temp_db.create_uploaded_file(sample_uploaded_file)
        
        temp_db.update_upload_complete(
            sample_uploaded_file.id,
            stored_path="/final/path/audio.m4a",
            size_bytes=15000,
        )
        
        retrieved = temp_db.get_uploaded_file(sample_uploaded_file.id)
        assert retrieved.upload_status == UploadStatus.COMPLETE
        assert retrieved.stored_path == "/final/path/audio.m4a"
        assert retrieved.size_bytes == 15000


# ==============================================================================
# Edit CRUD Tests
# ==============================================================================

class TestEditCRUD:
    """Test Edit CRUD operations."""

    def test_create_edit(self, temp_db, sample_job, sample_edit):
        """Test creating an edit."""
        temp_db.create_job(sample_job)
        
        created = temp_db.create_edit(sample_edit)
        
        assert created.id is not None
        assert created.job_id == sample_edit.job_id

    def test_get_edits_for_job(self, temp_db, sample_job):
        """Test retrieving all edits for a job."""
        temp_db.create_job(sample_job)
        
        for i in range(3):
            edit = Edit(
                id=None,
                job_id=sample_job.id,
                stage_name="test",
                edit_type="word_change",
                turn_id=i,
            )
            temp_db.create_edit(edit)
        
        edits = temp_db.get_edits_for_job(sample_job.id)
        
        assert len(edits) == 3

    def test_get_edits_filter_by_stage(self, temp_db, sample_job):
        """Test retrieving edits filtered by stage."""
        temp_db.create_job(sample_job)
        
        temp_db.create_edit(Edit(id=None, job_id=sample_job.id, stage_name="stage1", edit_type="change"))
        temp_db.create_edit(Edit(id=None, job_id=sample_job.id, stage_name="stage2", edit_type="change"))
        temp_db.create_edit(Edit(id=None, job_id=sample_job.id, stage_name="stage1", edit_type="change"))
        
        edits = temp_db.get_edits_for_job(sample_job.id, stage_name="stage1")
        
        assert len(edits) == 2

    def test_delete_edit(self, temp_db, sample_job, sample_edit):
        """Test deleting an edit."""
        temp_db.create_job(sample_job)
        created = temp_db.create_edit(sample_edit)
        
        result = temp_db.delete_edit(created.id)
        
        assert result is True
        edits = temp_db.get_edits_for_job(sample_job.id)
        assert len(edits) == 0

    def test_delete_edits_after(self, temp_db, sample_job):
        """Test deleting edits after a certain ID."""
        temp_db.create_job(sample_job)
        
        edit_ids = []
        for i in range(5):
            edit = Edit(id=None, job_id=sample_job.id, stage_name="test", edit_type="change")
            created = temp_db.create_edit(edit)
            edit_ids.append(created.id)
        
        deleted_count = temp_db.delete_edits_after(sample_job.id, edit_ids[2])
        
        assert deleted_count == 2
        remaining = temp_db.get_edits_for_job(sample_job.id)
        assert len(remaining) == 3


# ==============================================================================
# DeIdentificationState CRUD Tests
# ==============================================================================

class TestDeIdentificationStateCRUD:
    """Test DeIdentificationState CRUD operations."""

    def test_create_de_identification_state(self, temp_db, sample_job):
        """Test creating de-identification state."""
        temp_db.create_job(sample_job)
        
        state = DeIdentificationState(
            job_id=sample_job.id,
            first_pass_complete=False,
        )
        created = temp_db.create_de_identification_state(state)
        
        assert created.job_id == sample_job.id

    def test_get_de_identification_state(self, temp_db, sample_job):
        """Test retrieving de-identification state."""
        temp_db.create_job(sample_job)
        state = DeIdentificationState(job_id=sample_job.id, first_pass_complete=True)
        temp_db.create_de_identification_state(state)
        
        retrieved = temp_db.get_de_identification_state(sample_job.id)
        
        assert retrieved is not None
        assert retrieved.first_pass_complete is True

    def test_get_de_identification_state_not_found(self, temp_db, sample_job):
        """Test retrieving non-existent state returns None."""
        temp_db.create_job(sample_job)
        
        result = temp_db.get_de_identification_state(sample_job.id)
        assert result is None

    def test_update_de_identification_state(self, temp_db, sample_job):
        """Test updating de-identification state."""
        temp_db.create_job(sample_job)
        state = DeIdentificationState(job_id=sample_job.id)
        temp_db.create_de_identification_state(state)
        
        temp_db.update_de_identification_state(
            job_id=sample_job.id,
            first_pass_complete=True,
            discovered_names_json='[{"name": "Test"}]',
        )
        
        retrieved = temp_db.get_de_identification_state(sample_job.id)
        assert retrieved.first_pass_complete is True
        assert '"name": "Test"' in retrieved.discovered_names_json

    def test_upsert_de_identification_state_create(self, temp_db, sample_job):
        """Test upsert creates state if not exists."""
        temp_db.create_job(sample_job)
        
        state = DeIdentificationState(job_id=sample_job.id, first_pass_complete=True)
        result = temp_db.upsert_de_identification_state(state)
        
        assert result.first_pass_complete is True

    def test_upsert_de_identification_state_update(self, temp_db, sample_job):
        """Test upsert updates state if exists."""
        temp_db.create_job(sample_job)
        initial = DeIdentificationState(job_id=sample_job.id, first_pass_complete=False)
        temp_db.create_de_identification_state(initial)
        
        updated = DeIdentificationState(job_id=sample_job.id, first_pass_complete=True)
        result = temp_db.upsert_de_identification_state(updated)
        
        assert result.first_pass_complete is True


# ==============================================================================
# PIIReplacement CRUD Tests
# ==============================================================================

class TestPIIReplacementCRUD:
    """Test PIIReplacement CRUD operations."""

    def test_create_pii_replacement(self, temp_db, sample_job):
        """Test creating a PII replacement."""
        temp_db.create_job(sample_job)
        
        replacement = PIIReplacement(
            id=None,
            job_id=sample_job.id,
            speaker="Interviewer",
            original_text="John",
            replacement_text="[NAME]",
            pass_number=1,
        )
        created = temp_db.create_pii_replacement(replacement)
        
        assert created.id is not None

    def test_get_pii_replacements_for_job(self, temp_db, sample_job):
        """Test retrieving PII replacements for a job."""
        temp_db.create_job(sample_job)
        
        for i, name in enumerate(["John", "Jane", "Bob"]):
            replacement = PIIReplacement(
                id=None,
                job_id=sample_job.id,
                speaker="Interviewer",
                original_text=name,
                pass_number=1,
            )
            temp_db.create_pii_replacement(replacement)
        
        replacements = temp_db.get_pii_replacements_for_job(sample_job.id)
        
        assert len(replacements) == 3

    def test_get_pii_replacements_filter_by_speaker(self, temp_db, sample_job):
        """Test filtering PII replacements by speaker."""
        temp_db.create_job(sample_job)
        
        temp_db.create_pii_replacement(PIIReplacement(
            id=None, job_id=sample_job.id, speaker="A", original_text="John"
        ))
        temp_db.create_pii_replacement(PIIReplacement(
            id=None, job_id=sample_job.id, speaker="B", original_text="Jane"
        ))
        
        replacements = temp_db.get_pii_replacements_for_job(sample_job.id, speaker="A")
        
        assert len(replacements) == 1
        assert replacements[0].original_text == "John"

    def test_get_pii_replacements_filter_by_pass(self, temp_db, sample_job):
        """Test filtering PII replacements by pass number."""
        temp_db.create_job(sample_job)
        
        temp_db.create_pii_replacement(PIIReplacement(
            id=None, job_id=sample_job.id, speaker="Interviewer", original_text="John", pass_number=1
        ))
        temp_db.create_pii_replacement(PIIReplacement(
            id=None, job_id=sample_job.id, speaker="Interviewer", original_text="Jane", pass_number=2
        ))
        
        replacements = temp_db.get_pii_replacements_for_job(sample_job.id, pass_number=1)
        
        assert len(replacements) == 1

    def test_delete_pii_replacement(self, temp_db, sample_job):
        """Test deleting a PII replacement."""
        temp_db.create_job(sample_job)
        replacement = PIIReplacement(
            id=None, job_id=sample_job.id, speaker="Interviewer", original_text="John"
        )
        created = temp_db.create_pii_replacement(replacement)
        
        result = temp_db.delete_pii_replacement(created.id)
        
        assert result is True
        remaining = temp_db.get_pii_replacements_for_job(sample_job.id)
        assert len(remaining) == 0

    def test_bulk_create_pii_replacements(self, temp_db, sample_job):
        """Test bulk creating PII replacements."""
        temp_db.create_job(sample_job)
        
        replacements = [
            PIIReplacement(id=None, job_id=sample_job.id, speaker="Interviewer", original_text=name)
            for name in ["John", "Jane", "Bob", "Alice"]
        ]
        
        created = temp_db.bulk_create_pii_replacements(replacements)
        
        assert len(created) == 4
        assert all(r.id is not None for r in created)


# ==============================================================================
# Schema Integrity Tests
# ==============================================================================

class TestSchemaIntegrity:
    """Test database schema integrity."""

    def test_job_cascade_delete(self, temp_db, sample_job):
        """Test that deleting a job removes related edits."""
        temp_db.create_job(sample_job)
        temp_db.create_edit(Edit(
            id=None, job_id=sample_job.id, stage_name="test", edit_type="change"
        ))
        
        temp_db.delete_job(sample_job.id)
        
        edits = temp_db.get_edits_for_job(sample_job.id)
        assert len(edits) == 0

    def test_job_cascade_delete_pii(self, temp_db, sample_job):
        """Test that deleting a job removes PII replacements."""
        temp_db.create_job(sample_job)
        temp_db.create_pii_replacement(PIIReplacement(
            id=None, job_id=sample_job.id, speaker="Interviewer", original_text="Test"
        ))
        
        temp_db.delete_job(sample_job.id)
        
        replacements = temp_db.get_pii_replacements_for_job(sample_job.id)
        assert len(replacements) == 0

    def test_job_cascade_delete_de_id_state(self, temp_db, sample_job):
        """Test that deleting a job removes de-identification state."""
        temp_db.create_job(sample_job)
        temp_db.create_de_identification_state(DeIdentificationState(job_id=sample_job.id))
        
        temp_db.delete_job(sample_job.id)
        
        state = temp_db.get_de_identification_state(sample_job.id)
        assert state is None


# ==============================================================================
# Run tests directly
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
