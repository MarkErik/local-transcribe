"""
Integration tests for the export API endpoints.

Tests:
- Export format listing
- Single format export
- Bulk export
- Comparison export
- Error handling for incomplete jobs
"""

import pytest


class TestExportFormats:
    """Tests for GET /api/jobs/{job_id}/export/formats endpoint."""
    
    def test_list_export_formats_job_not_found(self, client):
        """Test listing formats for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/export/formats")
        
        assert response.status_code == 404
    
    def test_list_export_formats(self, client, uploaded_test_files):
        """Test listing available export formats."""
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
        
        # Get formats
        response = client.get(f"/api/jobs/{job_id}/export/formats")
        
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert len(data["formats"]) > 0
        
        # Check expected format fields
        for fmt in data["formats"]:
            assert "id" in fmt
            assert "name" in fmt
            assert "extension" in fmt


class TestSingleExport:
    """Tests for POST /api/jobs/{job_id}/export endpoint."""
    
    def test_export_job_not_found(self, client):
        """Test exporting from nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/export",
            json={"format": "timestamped-txt"}
        )
        
        assert response.status_code == 404
    
    def test_export_job_not_complete(self, client, uploaded_test_files):
        """Test exporting from incomplete job returns error."""
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
        
        # Try to export immediately
        response = client.post(
            f"/api/jobs/{job_id}/export",
            json={"format": "timestamped-txt"}
        )
        
        assert response.status_code == 400
        assert "not complete" in response.json()["detail"].lower()
    
    def test_export_invalid_format(self, client, uploaded_test_files):
        """Test exporting with invalid format returns error."""
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
        
        # Try invalid format (422 if validation catches, 400 if runtime check)
        response = client.post(
            f"/api/jobs/{job_id}/export",
            json={"format": "invalid-format"}
        )
        
        # Will be 400 (job not complete) or could be format validation error
        assert response.status_code in [400, 422]


class TestBulkExport:
    """Tests for POST /api/jobs/{job_id}/export/bulk endpoint."""
    
    def test_bulk_export_job_not_found(self, client):
        """Test bulk export from nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/export/bulk",
            json={"formats": ["timestamped-txt", "turns-json"]}
        )
        
        assert response.status_code == 404
    
    def test_bulk_export_validates_formats(self, client, uploaded_test_files):
        """Test bulk export validates format list."""
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
        
        # Empty formats list
        response = client.post(
            f"/api/jobs/{job_id}/export/bulk",
            json={"formats": []}
        )
        
        assert response.status_code in [400, 422]


class TestComparisonExport:
    """Tests for GET /api/jobs/{job_id}/export/comparison endpoint."""
    
    def test_comparison_job_not_found(self, client):
        """Test comparison export from nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/export/comparison")
        
        assert response.status_code == 404
    
    def test_comparison_job_not_complete(self, client, uploaded_test_files):
        """Test comparison export from incomplete job returns error."""
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
        
        # Try comparison export immediately
        response = client.get(f"/api/jobs/{job_id}/export/comparison")
        
        assert response.status_code == 400
        assert "not complete" in response.json()["detail"].lower()
