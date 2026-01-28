"""
Integration tests for the transcripts API endpoints.

Tests:
- Transcript retrieval
- Stage-specific transcripts
- Available stages listing
- Edit operations (create, list, delete)
- Stale stage detection
"""

import pytest
import json
from pathlib import Path


class TestTranscriptRetrieval:
    """Tests for GET /api/jobs/{job_id}/transcript endpoint."""
    
    def test_get_transcript_job_not_found(self, client):
        """Test getting transcript for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/transcript")
        
        assert response.status_code == 404
    
    def test_get_transcript_job_not_complete(self, client, uploaded_test_files):
        """Test getting transcript for incomplete job returns error."""
        # Create a job but don't wait for completion
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try to get transcript immediately
        response = client.get(f"/api/jobs/{job_id}/transcript")
        
        # Job likely pending or running
        assert response.status_code == 400
        assert "not complete" in response.json()["detail"].lower()


class TestAvailableStages:
    """Tests for GET /api/jobs/{job_id}/transcript/stages endpoint."""
    
    def test_get_stages_job_not_found(self, client):
        """Test getting stages for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/transcript/stages")
        
        assert response.status_code == 404
    
    def test_get_stages_job_not_complete(self, client, uploaded_test_files):
        """Test getting stages for incomplete job returns error."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try to get stages immediately
        response = client.get(f"/api/jobs/{job_id}/transcript/stages")
        
        assert response.status_code == 400


class TestEditOperations:
    """Tests for edit endpoints on transcripts."""
    
    def test_create_edit_job_not_found(self, client):
        """Test creating edit for nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/edits",
            json={
                "edit_type": "word_change",
                "turn_id": 1,
                "start_index": 0,
                "new_value": "Test"
            }
        )
        
        assert response.status_code == 404
    
    def test_list_edits_job_not_found(self, client):
        """Test listing edits for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/edits")
        
        assert response.status_code == 404
    
    def test_create_edit_validates_type(self, client, uploaded_test_files):
        """Test that invalid edit type is rejected."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try invalid edit type
        response = client.post(
            f"/api/jobs/{job_id}/edits",
            json={
                "edit_type": "invalid_type",
                "turn_id": 1,
                "start_index": 0,
                "new_value": "Test"
            }
        )
        
        assert response.status_code == 422  # Validation error


class TestStaleStages:
    """Tests for stale stage detection."""
    
    def test_get_stale_stages_job_not_found(self, client):
        """Test getting stale stages for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/stale-stages")
        
        assert response.status_code == 404
    
    def test_get_stale_stages_empty_initially(self, client, uploaded_test_files):
        """Test that new job has no stale stages."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Check stale stages
        response = client.get(f"/api/jobs/{job_id}/stale-stages")
        
        assert response.status_code == 200
        data = response.json()
        assert "stale_stages" in data
        assert len(data["stale_stages"]) == 0


class TestOutputFiles:
    """Tests for GET /api/jobs/{job_id}/outputs/{filename} endpoint."""
    
    def test_get_output_job_not_found(self, client):
        """Test getting output for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/outputs/turns.json")
        
        assert response.status_code == 404
    
    def test_get_output_file_not_found(self, client, uploaded_test_files):
        """Test getting nonexistent output file returns 404."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try to get nonexistent file
        response = client.get(f"/api/jobs/{job_id}/outputs/nonexistent-file.json")
        
        assert response.status_code == 404
