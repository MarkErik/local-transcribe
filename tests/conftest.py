"""
Pytest configuration and fixtures for API integration tests.

This module provides shared fixtures for testing the web API,
including a test client, test database, and sample data.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import FastAPI test client
from fastapi.testclient import TestClient

# Import the app and database components
from web_api.main import app
from web_api.database import init_database, get_database, Database
from web_api.config import get_config, ServerConfig


@pytest.fixture(scope="session")
def test_config() -> ServerConfig:
    """Create a test configuration with temporary directories."""
    config = get_config()
    return config


@pytest.fixture(scope="function")
def temp_dirs():
    """Create temporary directories for testing."""
    temp_base = Path(tempfile.mkdtemp(prefix="local_transcribe_test_"))
    
    dirs = {
        "base": temp_base,
        "uploads": temp_base / "uploads",
        "output": temp_base / "output",
        "database": temp_base / "test.db",
    }
    
    # Create directories
    dirs["uploads"].mkdir(parents=True, exist_ok=True)
    dirs["output"].mkdir(parents=True, exist_ok=True)
    
    yield dirs
    
    # Cleanup
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture(scope="function")
def test_db(temp_dirs) -> Generator[Database, None, None]:
    """Create a test database."""
    db_path = temp_dirs["database"]
    init_database(db_path)
    db = get_database()
    
    yield db
    
    # Database is cleaned up with temp_dirs


@pytest.fixture(scope="function")
def client(temp_dirs, monkeypatch) -> Generator[TestClient, None, None]:
    """Create a test client with isolated database and directories."""
    # Patch config to use temp directories
    monkeypatch.setenv("TRANSCRIBE_UPLOAD_DIR", str(temp_dirs["uploads"]))
    monkeypatch.setenv("TRANSCRIBE_OUTPUT_DIR", str(temp_dirs["output"]))
    monkeypatch.setenv("TRANSCRIBE_DATABASE_PATH", str(temp_dirs["database"]))
    
    # Re-initialize config with new env vars
    from web_api import config as config_module
    from web_api import database as db_module
    
    config_module._config = None  # Reset cached config
    db_module._db = None  # Reset cached database
    
    # Initialize database
    init_database(temp_dirs["database"])
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up cached instances after test
    config_module._config = None
    db_module._db = None


@pytest.fixture
def sample_audio_file(temp_dirs) -> Path:
    """Create a minimal valid audio file for testing.
    
    Creates a small MP3 file with valid magic bytes.
    """
    audio_path = temp_dirs["base"] / "test_audio.mp3"
    
    # Minimal MP3 frame header (not actually playable, but valid magic bytes)
    # ID3v2 header followed by MP3 frame sync
    mp3_header = b'ID3\x04\x00\x00\x00\x00\x00\x00'
    mp3_frame = b'\xff\xfb\x90\x00' + b'\x00' * 417  # MP3 frame
    
    with open(audio_path, 'wb') as f:
        f.write(mp3_header)
        f.write(mp3_frame * 10)  # Write a few frames
    
    return audio_path


@pytest.fixture
def sample_m4a_file(temp_dirs) -> Path:
    """Create a minimal valid M4A file for testing."""
    audio_path = temp_dirs["base"] / "test_audio.m4a"
    
    # M4A files have ftyp box at bytes 4-7
    # Format: [size:4][ftyp:4][brand:4][version:4]...
    ftyp_box = (
        b'\x00\x00\x00\x18'  # box size (24 bytes)
        b'ftyp'              # box type
        b'M4A '              # brand
        b'\x00\x00\x00\x00'  # version
        b'isom'              # compatible brand 1
        b'M4A '              # compatible brand 2
    )
    
    # Add some padding to make it a reasonable size
    padding = b'\x00' * 1000
    
    with open(audio_path, 'wb') as f:
        f.write(ftyp_box)
        f.write(padding)
    
    return audio_path


@pytest.fixture
def sample_transcript() -> dict:
    """Create a sample transcript for testing."""
    return {
        'turns': [
            {
                'turn_id': 1,
                'primary_speaker': 'Interviewer',
                'speaker': 'Interviewer',
                'text': 'Hello how are you today',
                'start_time': 0.0,
                'end_time': 2.0,
                'words': [
                    {'word': 'Hello', 'start_time': 0.0, 'end_time': 0.5, 'confidence': 0.95},
                    {'word': 'how', 'start_time': 0.5, 'end_time': 0.8, 'confidence': 0.90},
                    {'word': 'are', 'start_time': 0.8, 'end_time': 1.2, 'confidence': 0.92},
                    {'word': 'you', 'start_time': 1.2, 'end_time': 1.6, 'confidence': 0.98},
                    {'word': 'today', 'start_time': 1.6, 'end_time': 2.0, 'confidence': 0.97},
                ],
                'interjections': []
            },
            {
                'turn_id': 2,
                'primary_speaker': 'Participant',
                'speaker': 'Participant',
                'text': 'I am doing great thanks for asking',
                'start_time': 2.5,
                'end_time': 5.0,
                'words': [
                    {'word': 'I', 'start_time': 2.5, 'end_time': 2.6, 'confidence': 0.99},
                    {'word': 'am', 'start_time': 2.6, 'end_time': 2.8, 'confidence': 0.97},
                    {'word': 'doing', 'start_time': 2.8, 'end_time': 3.2, 'confidence': 0.94},
                    {'word': 'great', 'start_time': 3.2, 'end_time': 3.6, 'confidence': 0.96},
                    {'word': 'thanks', 'start_time': 3.6, 'end_time': 4.0, 'confidence': 0.93},
                    {'word': 'for', 'start_time': 4.0, 'end_time': 4.3, 'confidence': 0.91},
                    {'word': 'asking', 'start_time': 4.3, 'end_time': 5.0, 'confidence': 0.95},
                ],
                'interjections': []
            },
        ],
        'metadata': {
            'pipeline_mode': 'vad_split_audio',
            'created_at': '2026-01-28T12:00:00Z',
        }
    }


@pytest.fixture
def uploaded_test_files(client, sample_m4a_file) -> dict:
    """Upload two test files and return their IDs."""
    files = {}
    
    for role in ['interviewer', 'participant']:
        # Initialize upload
        init_response = client.post(
            "/api/files/upload/init",
            json={
                "filename": f"test_{role}.m4a",
                "size_bytes": sample_m4a_file.stat().st_size,
            }
        )
        assert init_response.status_code == 200
        init_data = init_response.json()
        upload_id = init_data["upload_id"]
        
        # Upload chunk - use 'file' parameter to match endpoint
        with open(sample_m4a_file, 'rb') as f:
            chunk_response = client.post(
                f"/api/files/upload/{upload_id}/chunk/0",
                files={"file": (f"test_{role}.m4a", f.read(), "application/octet-stream")}
            )
        assert chunk_response.status_code == 200
        
        # Complete upload
        complete_response = client.post(f"/api/files/upload/{upload_id}/complete")
        assert complete_response.status_code == 200
        complete_data = complete_response.json()
        
        files[role] = {
            "upload_id": upload_id,
            "file_id": complete_data["file_id"],
            "stored_path": complete_data["stored_path"],
        }
    
    return files
