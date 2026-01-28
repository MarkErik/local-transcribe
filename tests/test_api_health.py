"""
Integration tests for the health check API endpoint.

Tests:
- Basic health check returns 200
- Health check includes version and database status
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
    
    def test_health_check_response_schema(self, client):
        """Test that health check response matches expected schema."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        required_fields = ["status", "version", "database"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
