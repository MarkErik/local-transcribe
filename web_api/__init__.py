"""
Web API for local-transcribe.

Provides a FastAPI-based REST API and SSE endpoints for:
- Job submission and monitoring
- File upload/download
- Transcript retrieval and editing
- Pipeline re-execution
"""

try:
    from local_transcribe._version import version as __version__
except ImportError:
    # Fallback for development when setuptools-scm hasn't generated _version.py
    __version__ = "0.1.0-dev"
