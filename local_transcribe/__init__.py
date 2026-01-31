"""Local Transcribe - Audio transcription pipeline."""

try:
    from local_transcribe._version import version as __version__
except ImportError:
    # Fallback for development when setuptools-scm hasn't generated _version.py
    __version__ = "0.1.0-dev"

__all__ = ["__version__"]

