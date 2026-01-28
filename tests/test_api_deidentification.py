"""
Integration tests for the de-identification API endpoints.

Tests:
- First pass de-identification
- Name list review and update
- Second pass de-identification
- PII replacements audit trail
- Manual redaction operations
"""

import pytest


class TestFirstPass:
    """Tests for POST /api/jobs/{job_id}/de-identify/first-pass endpoint."""
    
    def test_first_pass_job_not_found(self, client):
        """Test first pass for nonexistent job returns 404."""
        response = client.post("/api/jobs/nonexistent-job-id/de-identify/first-pass")
        
        assert response.status_code == 404
    
    def test_first_pass_job_not_complete(self, client, uploaded_test_files):
        """Test first pass for incomplete job returns error."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try first pass immediately (job not complete)
        response = client.post(f"/api/jobs/{job_id}/de-identify/first-pass")
        
        assert response.status_code == 400


class TestNameList:
    """Tests for name list review endpoints."""
    
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
    
    def test_update_names_validates_format(self, client, uploaded_test_files):
        """Test that name list update validates format."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try invalid name format
        response = client.put(
            f"/api/jobs/{job_id}/de-identify/names",
            json={"names": "invalid"}  # Should be a list
        )
        
        assert response.status_code == 422


class TestSecondPass:
    """Tests for POST /api/jobs/{job_id}/de-identify/second-pass endpoint."""
    
    def test_second_pass_job_not_found(self, client):
        """Test second pass for nonexistent job returns 404."""
        response = client.post("/api/jobs/nonexistent-job-id/de-identify/second-pass")
        
        assert response.status_code == 404
    
    def test_second_pass_without_first_pass(self, client, uploaded_test_files):
        """Test second pass without first pass returns error."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Try second pass without first pass
        response = client.post(f"/api/jobs/{job_id}/de-identify/second-pass")
        
        # Will fail - either job not complete or first pass not done
        assert response.status_code == 400


class TestPIIReplacements:
    """Tests for PII replacements audit trail endpoint."""
    
    def test_get_replacements_job_not_found(self, client):
        """Test getting replacements for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/de-identify/replacements")
        
        assert response.status_code == 404
    
    def test_get_replacements_empty_initially(self, client, uploaded_test_files):
        """Test that new job has no PII replacements."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Get replacements (should be empty)
        response = client.get(f"/api/jobs/{job_id}/de-identify/replacements")
        
        assert response.status_code == 200
        data = response.json()
        assert "replacements" in data
        assert len(data["replacements"]) == 0


class TestManualRedaction:
    """Tests for manual redaction endpoints."""
    
    def test_manual_redact_job_not_found(self, client):
        """Test manual redaction for nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/de-identify/manual-redact",
            json={
                "turn_id": 1,
                "start_index": 0,
                "end_index": 1,
                "replacement_text": "[NAME]"
            }
        )
        
        assert response.status_code == 404
    
    def test_manual_redact_validates_request(self, client, uploaded_test_files):
        """Test that manual redaction validates request body."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Missing required fields
        response = client.post(
            f"/api/jobs/{job_id}/de-identify/manual-redact",
            json={"turn_id": 1}  # Missing start_index, end_index
        )
        
        assert response.status_code == 422


class TestOverrideRedaction:
    """Tests for override redaction (restore) endpoints."""
    
    def test_override_job_not_found(self, client):
        """Test override for nonexistent job returns 404."""
        response = client.post(
            "/api/jobs/nonexistent-job-id/de-identify/override",
            json={
                "turn_id": 1,
                "word_index": 0,
                "original_text": "John"
            }
        )
        
        assert response.status_code == 404
    
    def test_override_validates_request(self, client, uploaded_test_files):
        """Test that override validates request body."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Missing required fields
        response = client.post(
            f"/api/jobs/{job_id}/de-identify/override",
            json={"turn_id": 1}  # Missing word_index, original_text
        )
        
        assert response.status_code == 422


class TestDeIDState:
    """Tests for de-identification state endpoint."""
    
    def test_get_state_job_not_found(self, client):
        """Test getting de-ID state for nonexistent job returns 404."""
        response = client.get("/api/jobs/nonexistent-job-id/de-identify/state")
        
        assert response.status_code == 404
    
    def test_get_state_initial(self, client, uploaded_test_files):
        """Test getting de-ID state for new job."""
        # Create a job
        create_response = client.post(
            "/api/jobs",
            json={
                "interviewer_file_id": uploaded_test_files["interviewer"]["file_id"],
                "participant_file_id": uploaded_test_files["participant"]["file_id"],
                "mode": "vad_split_audio",
                "options": {"enable_de_identification": True}
            }
        )
        job_id = create_response.json()["job_id"]
        
        # Get state
        response = client.get(f"/api/jobs/{job_id}/de-identify/state")
        
        assert response.status_code == 200
        data = response.json()
        assert data["first_pass_complete"] == False
        assert data["second_pass_complete"] == False
