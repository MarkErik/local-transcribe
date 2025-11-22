#!/usr/bin/env python3
# lib/system_capability_utils.py - System capability and device utilities for the application

# Global configuration variables
_system_capability = "cpu"  # Default to CPU

def set_system_capability(capability: str) -> None:
    """Set the global system capability."""
    global _system_capability
    _system_capability = capability

def get_system_capability() -> str:
    """Get the current system capability."""
    return _system_capability

def clear_device_cache() -> None:
    """Clear the device cache based on the current system capability."""
    try:
        import torch
        device = get_system_capability()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass  # torch not available