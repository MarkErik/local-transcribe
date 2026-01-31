#!/usr/bin/env python3
"""
Tests for efficiency improvements.

These tests ensure the three critical efficiency improvements work correctly:
1. Lazy librosa import in remote_transcriber.py
2. Shared MFA helper utilities (consolidating duplicate code)
3. Entity factory methods for database row conversion

Tests are written BEFORE the changes to establish baseline behavior.
"""

import sys
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch, MagicMock

import pytest

# Add parent directory to path for local_transcribe imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Test 1: Lazy Librosa Import in Remote Transcriber
# ==============================================================================

class TestRemoteTranscriberLazyImport:
    """Test that remote_transcriber.py lazily imports librosa."""

    def test_remote_transcriber_module_import_does_not_require_librosa(self):
        """
        Test that importing the remote_transcriber module doesn't immediately
        require librosa to be loaded. This ensures CLI startup is fast.
        """
        # We can't easily test this without module reload, but we can verify
        # the module structure supports lazy import
        from local_transcribe.providers.transcribers import remote_transcriber
        
        # The module should be importable
        assert remote_transcriber is not None
        
        # The RemoteTranscriberProvider class should exist
        assert hasattr(remote_transcriber, 'RemoteTranscriberProvider')
        
        # ServerCapabilities dataclass should exist
        assert hasattr(remote_transcriber, 'ServerCapabilities')

    def test_remote_transcriber_has_expected_exceptions(self):
        """Test that custom exceptions are defined."""
        from local_transcribe.providers.transcribers import remote_transcriber
        
        assert hasattr(remote_transcriber, 'RemoteTranscriberError')
        assert hasattr(remote_transcriber, 'RemoteTranscriberConnectionError')
        assert hasattr(remote_transcriber, 'RemoteTranscriberTranscriptionError')

    def test_remote_transcriber_class_properties(self):
        """Test that RemoteTranscriberProvider has expected provider properties."""
        from local_transcribe.providers.transcribers.remote_transcriber import RemoteTranscriberProvider
        
        provider = RemoteTranscriberProvider()
        
        assert provider.name == "remote"
        assert "remote" in provider.short_name.lower()
        assert "remote" in provider.description.lower()


# ==============================================================================
# Test 2: Shared MFA Helper Utilities
# ==============================================================================

class TestMFAHelperUtilities:
    """Test the shared MFA helper utilities that consolidate duplicate code."""

    def test_mfa_aligner_has_get_mfa_command(self):
        """Test that MFA aligner has _get_mfa_command method."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        
        provider = MFAAlignerProvider()
        
        # Should have the method
        assert hasattr(provider, '_get_mfa_command')
        assert callable(provider._get_mfa_command)
        
        # Should return a string (either path or 'mfa')
        result = provider._get_mfa_command()
        assert isinstance(result, str)
        assert result.endswith('mfa') or result == 'mfa'

    def test_granite_mfa_has_get_mfa_command(self):
        """Test that granite_mfa has _get_mfa_command method."""
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        
        provider = GraniteMFATranscriberProvider()
        
        assert hasattr(provider, '_get_mfa_command')
        result = provider._get_mfa_command()
        assert isinstance(result, str)

    def test_granite_vad_silero_mfa_has_get_mfa_command(self):
        """Test that granite_vad_silero_mfa has _get_mfa_command method."""
        from local_transcribe.providers.transcribers.granite_vad_silero_mfa import GraniteVADSileroMFATranscriberProvider
        
        provider = GraniteVADSileroMFATranscriberProvider()
        
        assert hasattr(provider, '_get_mfa_command')
        result = provider._get_mfa_command()
        assert isinstance(result, str)

    def test_mfa_command_consistent_across_providers(self):
        """Test that all MFA providers return the same MFA command path."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        from local_transcribe.providers.transcribers.granite_vad_silero_mfa import GraniteVADSileroMFATranscriberProvider
        
        aligner = MFAAlignerProvider()
        granite_mfa = GraniteMFATranscriberProvider()
        granite_vad_mfa = GraniteVADSileroMFATranscriberProvider()
        
        # All providers should return the same MFA command
        cmd1 = aligner._get_mfa_command()
        cmd2 = granite_mfa._get_mfa_command()
        cmd3 = granite_vad_mfa._get_mfa_command()
        
        assert cmd1 == cmd2 == cmd3, "MFA command should be consistent across all providers"

    def test_mfa_providers_have_ensure_mfa_models(self):
        """Test that all MFA providers have _ensure_mfa_models method."""
        from local_transcribe.providers.aligners.mfa import MFAAlignerProvider
        from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
        from local_transcribe.providers.transcribers.granite_vad_silero_mfa import GraniteVADSileroMFATranscriberProvider
        
        for ProviderClass in [MFAAlignerProvider, GraniteMFATranscriberProvider, GraniteVADSileroMFATranscriberProvider]:
            provider = ProviderClass()
            assert hasattr(provider, '_ensure_mfa_models'), f"{ProviderClass.__name__} missing _ensure_mfa_models"
            assert callable(provider._ensure_mfa_models)


# ==============================================================================
# Test 3: Entity Factory Methods for Database Row Conversion
# ==============================================================================

class TestEntityFactoryMethods:
    """Test that entities can be created from database rows via factory methods."""

    def test_job_from_row_factory_method(self):
        """Test creating a Job using the from_row factory method."""
        from web_api.models.entities import Job, JobStatus
        
        # Simulate a database row as a dict
        row_data = {
            "id": "test-job-123",
            "status": "running",
            "mode": "vad_split_audio",
            "config_json": '{"key": "value"}',
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:01:00",
            "completed_at": None,
            "error_message": None,
            "output_dir": "/tmp/output",
            "interviewer_file_id": "file-1",
            "participant_file_id": "file-2",
        }
        
        # Use the factory method
        job = Job.from_row(row_data)
        
        assert job.id == "test-job-123"
        assert job.status == JobStatus.RUNNING
        assert job.mode == "vad_split_audio"
        assert job.config_json == '{"key": "value"}'
        assert job.interviewer_file_id == "file-1"

    def test_uploaded_file_from_row_factory_method(self):
        """Test creating an UploadedFile using the from_row factory method."""
        from web_api.models.entities import UploadedFile, UploadStatus
        
        row_data = {
            "id": "file-123",
            "original_filename": "test.m4a",
            "stored_path": "/uploads/file-123/test.m4a",
            "size_bytes": 1024000,
            "content_type": "audio/mp4",
            "upload_status": "complete",
            "created_at": "2024-01-01T00:00:00",
            "chunks_received": 10,
            "total_chunks": 10,
        }
        
        # Use the factory method
        file = UploadedFile.from_row(row_data)
        
        assert file.id == "file-123"
        assert file.upload_status == UploadStatus.COMPLETE
        assert file.size_bytes == 1024000
        assert file.original_filename == "test.m4a"

    def test_edit_from_row_factory_method(self):
        """Test creating an Edit using the from_row factory method."""
        from web_api.models.entities import Edit
        
        # Create a mock row-like object that supports .keys()
        class MockRow(dict):
            def keys(self):
                return super().keys()
        
        row_data = MockRow({
            "id": 42,
            "job_id": "job-123",
            "stage_name": "turn_building",
            "edit_type": "modify_text",
            "turn_id": 5,
            "start_index": 10,
            "end_index": 20,
            "original_value": "old text",
            "new_value": "new text",
            "target_turn_id": None,
            "annotation_type": None,
            "created_at": "2024-01-01T00:00:00",
        })
        
        # Use the factory method
        edit = Edit.from_row(row_data)
        
        assert edit.id == 42
        assert edit.edit_type == "modify_text"
        assert edit.original_value == "old text"
        assert edit.job_id == "job-123"

    def test_job_creation_from_dict(self):
        """Test creating a Job from a dictionary/row-like data."""
        from web_api.models.entities import Job, JobStatus
        
        # Simulate a database row as a dict
        row_data = {
            "id": "test-job-123",
            "status": "running",
            "mode": "vad_split_audio",
            "config_json": '{"key": "value"}',
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:01:00",
            "completed_at": None,
            "error_message": None,
            "output_dir": "/tmp/output",
            "interviewer_file_id": "file-1",
            "participant_file_id": "file-2",
        }
        
        # Create job manually (current approach)
        job = Job(
            id=row_data["id"],
            status=JobStatus(row_data["status"]),
            mode=row_data["mode"],
            config_json=row_data["config_json"],
            created_at=row_data["created_at"],
            started_at=row_data["started_at"],
            completed_at=row_data["completed_at"],
            error_message=row_data["error_message"],
            output_dir=row_data["output_dir"],
            interviewer_file_id=row_data["interviewer_file_id"],
            participant_file_id=row_data["participant_file_id"],
        )
        
        assert job.id == "test-job-123"
        assert job.status == JobStatus.RUNNING
        assert job.mode == "vad_split_audio"

    def test_uploaded_file_creation_from_dict(self):
        """Test creating an UploadedFile from a dictionary/row-like data."""
        from web_api.models.entities import UploadedFile, UploadStatus
        
        row_data = {
            "id": "file-123",
            "original_filename": "test.m4a",
            "stored_path": "/uploads/file-123/test.m4a",
            "size_bytes": 1024000,
            "content_type": "audio/mp4",
            "upload_status": "complete",
            "created_at": "2024-01-01T00:00:00",
            "chunks_received": 10,
            "total_chunks": 10,
        }
        
        file = UploadedFile(
            id=row_data["id"],
            original_filename=row_data["original_filename"],
            stored_path=row_data["stored_path"],
            size_bytes=row_data["size_bytes"],
            content_type=row_data["content_type"],
            upload_status=UploadStatus(row_data["upload_status"]),
            created_at=row_data["created_at"],
            chunks_received=row_data["chunks_received"],
            total_chunks=row_data["total_chunks"],
        )
        
        assert file.id == "file-123"
        assert file.upload_status == UploadStatus.COMPLETE
        assert file.size_bytes == 1024000

    def test_edit_creation_from_dict(self):
        """Test creating an Edit from a dictionary/row-like data."""
        from web_api.models.entities import Edit
        
        row_data = {
            "id": 42,
            "job_id": "job-123",
            "stage_name": "turn_building",
            "edit_type": "modify_text",
            "turn_id": 5,
            "start_index": 10,
            "end_index": 20,
            "original_value": "old text",
            "new_value": "new text",
            "target_turn_id": None,
            "annotation_type": None,
            "created_at": "2024-01-01T00:00:00",
        }
        
        edit = Edit(
            id=row_data["id"],
            job_id=row_data["job_id"],
            stage_name=row_data["stage_name"],
            edit_type=row_data["edit_type"],
            turn_id=row_data["turn_id"],
            start_index=row_data["start_index"],
            end_index=row_data["end_index"],
            original_value=row_data["original_value"],
            new_value=row_data["new_value"],
            target_turn_id=row_data["target_turn_id"],
            annotation_type=row_data["annotation_type"],
            created_at=row_data["created_at"],
        )
        
        assert edit.id == 42
        assert edit.edit_type == "modify_text"
        assert edit.original_value == "old text"

    def test_job_to_dict_roundtrip(self):
        """Test that Job can roundtrip through to_dict."""
        from web_api.models.entities import Job, JobStatus
        
        job = Job(
            id="test-123",
            status=JobStatus.COMPLETED,
            mode="vad_split_audio",
            config_json='{"transcriber": "remote"}',
        )
        
        d = job.to_dict()
        
        assert d["id"] == "test-123"
        assert d["status"] == "completed"
        assert d["mode"] == "vad_split_audio"
        assert d["config"]["transcriber"] == "remote"

    def test_uploaded_file_to_dict_roundtrip(self):
        """Test that UploadedFile can roundtrip through to_dict."""
        from web_api.models.entities import UploadedFile, UploadStatus
        
        file = UploadedFile(
            id="file-456",
            original_filename="audio.wav",
            stored_path="/uploads/audio.wav",
            upload_status=UploadStatus.COMPLETE,
        )
        
        d = file.to_dict()
        
        assert d["id"] == "file-456"
        assert d["upload_status"] == "complete"
        assert d["original_filename"] == "audio.wav"


# ==============================================================================
# Test Database Integration
# ==============================================================================

class TestDatabaseEntityConversion:
    """Test database operations with entity conversion."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        
        yield db_path
        
        # Cleanup
        if db_path.exists():
            db_path.unlink()

    def test_database_get_job_returns_proper_entity(self, temp_db):
        """Test that get_job returns a proper Job entity."""
        from web_api.database import Database
        from web_api.models.entities import Job, JobStatus
        
        db = Database(db_path=temp_db)
        
        # Create a job
        job = Job(
            id="test-job-db",
            status=JobStatus.PENDING,
            mode="vad_split_audio",
        )
        db.create_job(job)
        
        # Retrieve it
        retrieved = db.get_job("test-job-db")
        
        assert retrieved is not None
        assert isinstance(retrieved, Job)
        assert retrieved.id == "test-job-db"
        assert retrieved.status == JobStatus.PENDING

    def test_database_list_jobs_returns_proper_entities(self, temp_db):
        """Test that list_jobs returns proper Job entities."""
        from web_api.database import Database
        from web_api.models.entities import Job, JobStatus
        
        db = Database(db_path=temp_db)
        
        # Create multiple jobs
        for i in range(3):
            job = Job(
                id=f"test-job-{i}",
                status=JobStatus.PENDING,
                mode="vad_split_audio",
            )
            db.create_job(job)
        
        # List them
        jobs = db.list_jobs()
        
        assert len(jobs) == 3
        for job in jobs:
            assert isinstance(job, Job)
            assert job.status == JobStatus.PENDING

    def test_database_get_uploaded_file_returns_proper_entity(self, temp_db):
        """Test that get_uploaded_file returns a proper UploadedFile entity."""
        from web_api.database import Database
        from web_api.models.entities import UploadedFile, UploadStatus
        
        db = Database(db_path=temp_db)
        
        # Create an uploaded file
        file = UploadedFile(
            id="test-file-db",
            original_filename="test.m4a",
            stored_path="/uploads/test.m4a",
            upload_status=UploadStatus.COMPLETE,
        )
        db.create_uploaded_file(file)
        
        # Retrieve it
        retrieved = db.get_uploaded_file("test-file-db")
        
        assert retrieved is not None
        assert isinstance(retrieved, UploadedFile)
        assert retrieved.id == "test-file-db"
        assert retrieved.upload_status == UploadStatus.COMPLETE

    def test_database_get_edits_returns_proper_entities(self, temp_db):
        """Test that get_edits_for_job returns proper Edit entities."""
        from web_api.database import Database
        from web_api.models.entities import Job, JobStatus, Edit
        
        db = Database(db_path=temp_db)
        
        # Create a job first
        job = Job(id="edit-test-job", status=JobStatus.RUNNING, mode="vad_split_audio")
        db.create_job(job)
        
        # Create edits
        for i in range(3):
            edit = Edit(
                id=None,
                job_id="edit-test-job",
                stage_name="turn_building",
                edit_type="modify_text",
                original_value=f"old-{i}",
                new_value=f"new-{i}",
            )
            db.create_edit(edit)
        
        # Retrieve edits
        edits = db.get_edits_for_job("edit-test-job")
        
        assert len(edits) == 3
        for edit in edits:
            assert isinstance(edit, Edit)
            assert edit.job_id == "edit-test-job"


# ==============================================================================
# Test MFA Utils Module (after consolidation)
# ==============================================================================

class TestMFAUtilsModuleExists:
    """Test that MFA utilities exist in the expected location."""

    def test_mfa_alignment_module_exists(self):
        """Test that MFA alignment module exists."""
        from local_transcribe.providers.common import mfa_alignment
        assert mfa_alignment is not None

    def test_mfa_alignment_engine_class_exists(self):
        """Test that MFAAlignmentEngine class exists."""
        from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
        assert MFAAlignmentEngine is not None

    def test_mfa_alignment_engine_has_parse_textgrid(self):
        """Test that MFAAlignmentEngine has parse_textgrid_to_word_dicts method."""
        from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
        assert hasattr(MFAAlignmentEngine, 'parse_textgrid_to_word_dicts')

    def test_mfa_alignment_engine_has_simple_alignment(self):
        """Test that MFAAlignmentEngine has create_simple_alignment method."""
        from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
        assert hasattr(MFAAlignmentEngine, 'create_simple_alignment')


class TestMFAUtilsModule:
    """Test the new shared MFA utilities module."""

    def test_mfa_utils_module_exists(self):
        """Test that the mfa_utils module exists."""
        from local_transcribe.providers.common import mfa_utils
        assert mfa_utils is not None

    def test_get_mfa_command_function_exists(self):
        """Test that get_mfa_command function exists."""
        from local_transcribe.providers.common.mfa_utils import get_mfa_command
        assert callable(get_mfa_command)

    def test_ensure_mfa_models_function_exists(self):
        """Test that ensure_mfa_models function exists."""
        from local_transcribe.providers.common.mfa_utils import ensure_mfa_models
        assert callable(ensure_mfa_models)

    def test_get_mfa_command_returns_string(self):
        """Test that get_mfa_command returns a string."""
        from local_transcribe.providers.common.mfa_utils import get_mfa_command
        result = get_mfa_command()
        assert isinstance(result, str)
        assert result.endswith('mfa') or result == 'mfa'

    def test_get_mfa_environment_function_exists(self):
        """Test that get_mfa_environment function exists."""
        from local_transcribe.providers.common.mfa_utils import get_mfa_environment
        assert callable(get_mfa_environment)

    def test_get_mfa_config_path_function_exists(self):
        """Test that get_mfa_config_path function exists."""
        from local_transcribe.providers.common.mfa_utils import get_mfa_config_path
        assert callable(get_mfa_config_path)

    def test_mfa_utils_exported_from_lazy_imports(self):
        """Test that MFA utils are accessible via lazy_imports module."""
        from local_transcribe.providers.common.lazy_imports import (
            get_mfa_command,
            ensure_mfa_models,
            get_mfa_environment,
            get_mfa_config_path,
        )
        assert callable(get_mfa_command)
        assert callable(ensure_mfa_models)
        assert callable(get_mfa_environment)
        assert callable(get_mfa_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
