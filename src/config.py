# src/config.py
from __future__ import annotations
from typing import Any, Dict
import threading

class GlobalConfig:
    """
    Global configuration object to store application-wide settings.
    Thread-safe implementation for use across all modules.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Default configuration values
        self._config: Dict[str, Any] = {
            "debug_enabled": False,
            "info_enabled": False,
        }
        
        self._lock = threading.Lock()
        self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        with self._lock:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        with self._lock:
            self._config[key] = value
    
    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments."""
        with self._lock:
            self._config["debug_enabled"] = getattr(args, 'debug', False)
            self._config["info_enabled"] = getattr(args, 'info', False)
    
    @property
    def debug_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.get("debug_enabled", False)
    
    @property
    def info_enabled(self) -> bool:
        """Check if info logging is enabled."""
        return self.get("info_enabled", False)


# Global configuration instance
_global_config: GlobalConfig = GlobalConfig()


def get_global_config() -> GlobalConfig:
    """Get the global configuration instance."""
    return _global_config


def configure_from_args(args) -> None:
    """Configure global settings from command line arguments."""
    _global_config.update_from_args(args)


def is_debug_enabled() -> bool:
    """Check if debug logging is enabled."""
    return _global_config.debug_enabled


def is_info_enabled() -> bool:
    """Check if info logging is enabled."""
    return _global_config.info_enabled