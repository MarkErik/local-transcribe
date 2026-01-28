"""
Configuration for the web API.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ServerConfig:
    """Server configuration settings."""
    
    # Server binding
    host: str = "0.0.0.0"
    port: int = 8099
    
    # CORS settings
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    
    # File storage paths
    upload_dir: Path = field(default_factory=lambda: Path("./uploads"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    
    # Upload settings
    max_file_size_mb: int = 500
    chunk_size_bytes: int = 5 * 1024 * 1024  # 5MB chunks
    upload_timeout_hours: int = 24  # Cleanup incomplete uploads after this
    
    # Database
    database_path: Path = field(default_factory=lambda: Path("./data/transcribe.db"))
    
    # Pipeline settings
    default_mode: str = "vad_split_audio"
    
    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("TRANSCRIBE_HOST", "0.0.0.0"),
            port=int(os.getenv("TRANSCRIBE_PORT", "8099")),
            cors_origins=os.getenv("TRANSCRIBE_CORS_ORIGINS", "*").split(","),
            upload_dir=Path(os.getenv("TRANSCRIBE_UPLOAD_DIR", "./uploads")),
            output_dir=Path(os.getenv("TRANSCRIBE_OUTPUT_DIR", "./output")),
            data_dir=Path(os.getenv("TRANSCRIBE_DATA_DIR", "./data")),
            max_file_size_mb=int(os.getenv("TRANSCRIBE_MAX_FILE_SIZE_MB", "500")),
            chunk_size_bytes=int(os.getenv("TRANSCRIBE_CHUNK_SIZE_BYTES", str(5 * 1024 * 1024))),
            database_path=Path(os.getenv("TRANSCRIBE_DATABASE_PATH", "./data/transcribe.db")),
            default_mode=os.getenv("TRANSCRIBE_DEFAULT_MODE", "vad_split_audio"),
        )
    
    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global config instance - initialized on startup
_config: Optional[ServerConfig] = None


def get_config() -> ServerConfig:
    """Get the current server configuration."""
    global _config
    if _config is None:
        _config = ServerConfig.from_env()
        _config.ensure_directories()
    return _config


def set_config(config: ServerConfig) -> None:
    """Set the server configuration (for testing)."""
    global _config
    _config = config
