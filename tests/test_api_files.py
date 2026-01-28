"""
Integration tests for the file upload/download API endpoints.

Tests:
- Chunked upload initialization
- Chunk upload and assembly
- Upload completion and validation
- File download with range requests
- Error handling for invalid uploads
"""

import pytest
from pathlib import Path


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
        assert data["total_chunks"] > 1  # Should be multiple chunks for 15MB
    
    def test_init_upload_rejects_oversized_files(self, client):
        """Test that files exceeding max size are rejected."""
        huge_size = 600 * 1024 * 1024  # 600MB (over 500MB limit)
        
        response = client.post(
            "/api/files/upload/init",
            json={
                "filename": "huge_audio.m4a",
                "size_bytes": huge_size,
            }
        )
        
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()
    
    def test_init_upload_validates_request_body(self, client):
        """Test that missing fields return validation error."""
        response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a"}  # Missing size_bytes
        )
        
        assert response.status_code == 422  # Validation error


class TestChunkUpload:
    """Tests for POST /api/files/upload/{upload_id}/chunk/{chunk_num} endpoint."""
    
    def test_upload_single_chunk(self, client, sample_m4a_file):
        """Test uploading a single chunk."""
        # Initialize upload
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Upload chunk 0 - use 'file' parameter
        with open(sample_m4a_file, 'rb') as f:
            response = client.post(
                f"/api/files/upload/{upload_id}/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_received"] == 1
        assert data["bytes_received"] > 0
    
    def test_upload_chunk_invalid_upload_id(self, client, sample_m4a_file):
        """Test uploading to invalid upload ID returns 404."""
        with open(sample_m4a_file, 'rb') as f:
            response = client.post(
                "/api/files/upload/nonexistent-id/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        assert response.status_code == 404
    
    def test_upload_chunk_out_of_order(self, client, sample_m4a_file):
        """Test uploading chunks out of order is rejected."""
        # Initialize upload
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


class TestUploadComplete:
    """Tests for POST /api/files/upload/{upload_id}/complete endpoint."""
    
    def test_complete_upload_success(self, client, sample_m4a_file):
        """Test completing an upload successfully."""
        # Initialize upload
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Upload chunk - use 'file' parameter
        with open(sample_m4a_file, 'rb') as f:
            client.post(
                f"/api/files/upload/{upload_id}/chunk/0",
                files={"file": ("test.m4a", f.read(), "application/octet-stream")}
            )
        
        # Complete upload
        response = client.post(f"/api/files/upload/{upload_id}/complete")
        
        assert response.status_code == 200
        data = response.json()
        assert "file_id" in data
        assert "stored_path" in data
        assert data["filename"] == "test.m4a"
    
    def test_complete_upload_invalid_id(self, client):
        """Test completing with invalid upload ID returns 404."""
        response = client.post("/api/files/upload/nonexistent-id/complete")
        
        assert response.status_code == 404
    
    def test_complete_upload_missing_chunks(self, client):
        """Test completing upload with missing chunks fails."""
        # Initialize upload for large file (multiple chunks needed)
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": 10 * 1024 * 1024}  # 10MB
        )
        upload_id = init_response.json()["upload_id"]
        
        # Don't upload any chunks, try to complete
        response = client.post(f"/api/files/upload/{upload_id}/complete")
        
        assert response.status_code == 400
        assert "chunks" in response.json()["detail"].lower()


class TestUploadStatus:
    """Tests for GET /api/files/upload/{upload_id}/status endpoint."""
    
    def test_get_upload_status(self, client, sample_m4a_file):
        """Test getting upload status."""
        # Initialize upload
        file_size = sample_m4a_file.stat().st_size
        init_response = client.post(
            "/api/files/upload/init",
            json={"filename": "test.m4a", "size_bytes": file_size}
        )
        upload_id = init_response.json()["upload_id"]
        
        # Check status
        response = client.get(f"/api/files/upload/{upload_id}/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["upload_id"] == upload_id
        assert "status" in data
        assert "chunks_received" in data
    
    def test_get_upload_status_invalid_id(self, client):
        """Test getting status for invalid upload ID returns 404."""
        response = client.get("/api/files/upload/nonexistent-id/status")
        
        assert response.status_code == 404


class TestFileDownload:
    """Tests for GET /api/files/{file_id}/audio endpoint."""
    
    def test_download_file(self, client, uploaded_test_files):
        """Test downloading a completed file."""
        file_id = uploaded_test_files["interviewer"]["file_id"]
        
        response = client.get(f"/api/files/{file_id}/audio")
        
        assert response.status_code == 200
        assert len(response.content) > 0
    
    def test_download_file_with_range_request(self, client, uploaded_test_files):
        """Test downloading a file with HTTP Range header."""
        file_id = uploaded_test_files["interviewer"]["file_id"]
        
        response = client.get(
            f"/api/files/{file_id}/audio",
            headers={"Range": "bytes=0-99"}
        )
        
        assert response.status_code == 206  # Partial Content
        assert len(response.content) == 100
        assert "Content-Range" in response.headers
    
    def test_download_file_invalid_id(self, client):
        """Test downloading with invalid file ID returns 404."""
        response = client.get("/api/files/nonexistent-id/audio")
        
        assert response.status_code == 404
    
    def test_head_request_for_file_size(self, client, uploaded_test_files):
        """Test HEAD request returns file size in Content-Length."""
        file_id = uploaded_test_files["interviewer"]["file_id"]
        
        response = client.head(f"/api/files/{file_id}/audio")
        
        assert response.status_code == 200
        assert "Content-Length" in response.headers
        assert int(response.headers["Content-Length"]) > 0
