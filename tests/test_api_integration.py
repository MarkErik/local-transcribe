"""
Integration tests for core API endpoints.

These tests verify basic API functionality without requiring 
background task execution (which complicates test isolation).

For tests requiring full pipeline execution, use manual testing
or an integration test environment with proper setup/teardown.
"""

import pytest


class TestHealthEndpoint:
    """Tests for /api/health endpoint."""
    
    def test_health_check_returns_ok(self, client):
        """Test that health check returns 200 with status ok."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_health_check_includes_version(self, client):
        """Test that health check includes the API version."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
    
    def test_health_check_includes_database_status(self, client):
        """Test that health check reports database connection status."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert data["database"] == "connected"


class TestUploadInit:
    """Tests for POST /api/files/upload/init endpoint."""
    
    def test_init_upload_returns_upload_id(self, client):
        """Test that upload init returns a valid upload_id."""
        response = client.post(
            "/api/files/upload/init",
            json={
                "filename": "test_audio.m4a",
                "size_bytes": 10000,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "upload_id" in data
        assert len(data["upload_id"]) > 0
    
    def test_init_upload_returns_chunk_info(self, client):
        """Test that upload init returns chunk size and count."""
        file_size = 15 * 1024 * 1024  # 15MB
        
        response = client.post(
            "/api/files/upload/init",
            json={
                "filename": "large_audio.m4a",
                "size_bytes": file_size,
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "chunk_size" in data
        assert "total_chunks" in data
        assert data["chunk_size"] > 0
        assert data["total_chunks"] > 1
    
    def test_init_upload_rejects_oversized_files(self, client):
        """Test that files exceeding max size are rejected."""
        huge_size = 600 * 1024 * 1024  # 600MB
        
        response = client.post(
            "/api/files/upload/init",
            json={
                "filename": "huge_audio.m4a",
                "size_bytes": huge_size,
            }
        )
        
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()


class TestJobList:
    """Tests for GET /api/jobs endpoint (without job creation)."""
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/jobs")
        
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert data["total"] == 0


class TestJobNotFound:
    """Tests for endpoints with nonexistent job ID."""
    
    def test_get_job_invalid_id(self, client):
        """Test getting a job with invalid ID returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id")
        
        assert response.status_code == 404

    def test_get_transcript_job_not_found(self, client):
        """Test getting transcript for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/transcript")
        
        assert response.status_code == 404
    
    def test_get_stages_job_not_found(self, client):
        """Test getting stages for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/transcript/stages")
        
        assert response.status_code == 404

    def test_list_edits_job_not_found(self, client):
        """Test listing edits for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/edits")
        
        assert response.status_code == 404
    
    def test_get_output_job_not_found(self, client):
        """Test getting output for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/outputs/turns.json")
        
        assert response.status_code == 404

    def test_rerun_invalid_job_id(self, client):
        """Test re-running with invalid job ID returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/rerun",
            json={"start_stage": "de_identification"}
        )
        
        assert response.status_code == 404


class TestUploadNotFound:
    """Tests for upload endpoints with invalid IDs."""
    
    def test_upload_chunk_invalid_upload_id(self, client, sample_m4a_file):
        """Test uploading to invalid upload ID returns 404."""
        with open(sample_m4a_file, 'rb') as f:
            response = client.post(
                "/api/files/upload/nonexistent-id/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        assert response.status_code == 404

    def test_complete_upload_invalid_id(self, client):
        """Test completing with invalid upload ID returns 404."""
        response = client.post("/api/files/upload/nonexistent-id/complete")
        
        assert response.status_code == 404

    def test_get_upload_status_invalid_id(self, client):
        """Test getting status for invalid upload ID returns 404."""
        response = client.get("/api/files/upload/nonexistent-id/status")
        
        assert response.status_code == 404

    def test_download_file_invalid_id(self, client):
        """Test downloading with invalid file ID returns 404."""
        response = client.get("/api/files/nonexistent-id/audio")
        
        assert response.status_code == 404


class TestDeIDNotFound:
    """Tests for de-identification endpoints with nonexistent job."""
    
    def test_first_pass_job_not_found(self, client):
        """Test first pass for nonexistent job returns 404."""
        response = client.post("/api/jobs/nonexistent-job-id/de-identify/first-pass")
        
        assert response.status_code == 404

    def test_get_names_job_not_found(self, client):
        """Test getting names for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/de-identify/names")
        
        assert response.status_code == 404

    def test_update_names_job_not_found(self, client):
        """Test updating names for nonexistent job returns 404."""
        response = client.put(
            "/api/jobs/nonexistent-job-id/de-identify/names",
            json={"names": []}
        )
        
        assert response.status_code == 404

    def test_second_pass_job_not_found(self, client):
        """Test second pass for nonexistent job returns 404."""
        response = client.post("/api/jobs/nonexistent-job-id/de-identify/second-pass")
        
        assert response.status_code == 404

    def test_get_replacements_job_not_found(self, client):
        """Test getting replacements for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/de-identify/replacements")
        
        assert response.status_code == 404

    def test_get_status_job_not_found(self, client):
        """Test getting de-ID status for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/de-identify/status")
        
        assert response.status_code == 404


class TestExportNotFound:
    """Tests for export endpoints with nonexistent job."""
    
    def test_list_export_formats_job_not_found(self, client):
        """Test listing formats for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/export/formats")
        
        assert response.status_code == 404

    def test_export_job_not_found(self, client):
        """Test exporting from nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/export",
            json={"format": "timestamped-txt"}
        )
        
        assert response.status_code == 404

    def test_get_export_job_not_found(self, client):
        """Test getting export from nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/export/timestamped-txt")
        
        assert response.status_code == 404


class TestFileUploadFlow:
    """Tests for complete file upload workflow."""
    
    def test_upload_single_chunk(self, client, sample_m4a_file):
        """Test uploading a single chunk."""
        # Initialize upload
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Upload chunk 0
        with open(sample_m4a_file, 'rb') as f:
            response = client.post(
                f"/api/files/upload/{upload_id}/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_received"] == 1
        assert data["bytes_received"] > 0

    def test_complete_upload_success(self, client, sample_m4a_file):
        """Test completing an upload successfully."""
        # Initialize
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Upload chunk
        with open(sample_m4a_file, 'rb') as f:
            client.post(
                f"/api/files/upload/{upload_id}/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        # Complete
        response = client.post(f"/api/files/upload/{upload_id}/complete")
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "stored_path" in data
        assert data["filename"] == "test.m4a"

    def test_upload_chunk_out_of_order(self, client, sample_m4a_file):
        """Test uploading chunks out of order is rejected."""
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Try to upload chunk 1 before chunk 0
        with open(sample_m4a_file, 'rb') as f:
            response = client.post(
                f"/api/files/upload/{upload_id}/chunk/1",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        assert response.status_code == 400
        assert "expected chunk" in response.json()["detail"].lower()

    def test_complete_upload_missing_chunks(self, client):
        """Test completing upload with missing chunks fails."""
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": 10 * 1024 * 1024}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Don't upload any chunks
        response = client.post(f"/api/files/upload/{upload_id}/complete")
        
        assert response.status_code == 400
        assert "chunks" in response.json()["detail"].lower()
