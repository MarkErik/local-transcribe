"""
Integration tests for the jobs API endpoints.

Tests:
- Job creation
- Job listing with filters
- Job status retrieval
- SSE progress streaming
- Job re-run functionality
"""

import pytest
import json


class TestJobCreate:
    """Tests for POST /api/jobs endpoint."""
    
    def test_create_job_returns_job_id(self, client, uploaded_test_files):
        """Test that job creation returns a valid job_id."""
        response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {
                    "enable_de_identification": False,
                    "enable_cleanup": False,
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert len(data["job_id"]) > 0
        assert data["status"] == "pending"
    
    def test_create_job_with_invalid_file_id(self, client, uploaded_test_files):
        """Test that invalid file IDs return 400 error."""
        response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": "nonexistent-file-id",
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()
    
    def test_create_job_validates_options(self, client, uploaded_test_files):
        """Test that job options are properly validated."""
        response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {
                    "enable_de_identification": True,
                    "enable_cleanup": True,
                    "output_formats": ["timestamped-txt", "turns-json"],
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestJobList:
    """Tests for GET /api/jobs endpoint."""
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["total"] == 0
    
    def test_list_jobs_with_results(self, client, uploaded_test_files):
        """Test listing jobs after creating some."""
        # Create a job
        client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        
        response = client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1
        assert len(data["jobs"]) >= 1
    
    def test_list_jobs_with_status_filter(self, client, uploaded_test_files):
        """Test filtering jobs by status."""
        # Create a job
        client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {}
            }
        )
        
        # Filter by pending status
        response = client.get("/api/jobs?status=pending")
        
        assert response.status_code == 200
        data = response.json()
        for job in data["jobs"]:
            assert job["status"] in ["pending", "running"]  # May have started
    
    def test_list_jobs_with_pagination(self, client, uploaded_test_files):
        """Test job listing pagination."""
        # Create multiple jobs
        for _ in range(3):
            client.post(
                "/api/jobs",
                json={
                    "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                    "participant_file_id": uploaded_test_files["participant"]["file_id"],
                    "mode": "vad_split_audio",
                    "options": {}
                }
            )
        
        # Get first page
        response = client.get("/api/jobs?limit=2&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["jobs"]) <= 2


class TestJobGet:
    """Tests for GET /api/jobs/{job_id} endpoint."""
    
    def test_get_job_by_id(self, client, uploaded_test_files):
        """Test retrieving a specific job."""
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
        
        # Get the job
        response = client.get(f"/api/jobs/{job_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert data["mode"] == "vad_split_audio"
    
    def test_get_job_invalid_id(self, client):
        """Test getting a job with invalid ID returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id")
        
        assert response.status_code == 404


class TestJobProgress:
    """Tests for GET /api/jobs/{job_id}/progress SSE endpoint."""
    
    def test_progress_endpoint_returns_sse(self, client, uploaded_test_files):
        """Test that progress endpoint returns SSE stream."""
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
        
        # Connect to progress stream
        with client.stream("GET", f"/api/jobs/{job_id}/progress") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            # Just verify it connects, don't wait for events
    
    def test_progress_invalid_job_id(self, client):
        """Test progress for invalid job ID returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/progress")
        
        assert response.status_code == 404


class TestJobRerun:
    """Tests for POST /api/jobs/{job_id}/rerun endpoint."""
    
    def test_rerun_invalid_job_id(self, client):
        """Test re-running with invalid job ID returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/rerun",
            json={"start_stage": "de_identification"}
        )
        
        assert response.status_code == 404
    
    def test_rerun_validates_start_stage(self, client, uploaded_test_files):
        """Test that invalid start_stage is rejected."""
        # Create a job first
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
        
        # Try invalid start stage
        response = client.post(
            f"/api/jobs/{job_id}/rerun",
            json={"start_stage": "invalid_stage"}
        )
        
        # Should fail - either 400 for invalid stage or 400 for incomplete job
        assert response.status_code in [400, 422]
