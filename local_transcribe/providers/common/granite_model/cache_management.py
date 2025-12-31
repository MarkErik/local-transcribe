# Granite Model Cache Management
# Handles cache directory resolution and management for Granite models

import os
import pathlib
from typing import Any


class CacheManager:
    """Manages cache directory resolution for Granite models."""
    
    def __init__(self, logger: Any):
        """
        Initialize the cache manager.
        
        Args:
            logger: Logger instance for logging messages
        """
        self.logger = logger
    
    def resolve_cache_directory(self) -> pathlib.Path:
        """
        Resolve the cache directory for storing models.
        
        Returns:
            Path to the resolved cache directory
        """
        # Check for XDG_CACHE_HOME environment variable first
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            models_root = pathlib.Path(xdg_cache_home)
        else:
            # Fallback to standard HuggingFace cache location
            models_root = pathlib.Path.home() / ".cache" / "huggingface"
        
        # The models are stored in the standard HuggingFace hub structure
        cache_dir = models_root / "huggingface" / "hub"
        
        try:
            # Create directory structure if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Log the resolved cache directory for debugging
            self.logger.debug(f"Resolved cache directory: {cache_dir}")
            
            return cache_dir
            
        except OSError as e:
            error_msg = f"Failed to create cache directory {cache_dir}: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg)
