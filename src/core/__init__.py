#!/usr/bin/env python3
"""
Core plugin system initialization and utilities.
"""

from .plugins import (
    PluginRegistry,
    ASRProvider,
    DiarizationProvider,
    OutputWriter,
    WordSegment,
    Turn,
    registry
)
from .builtin_plugins import register_builtin_plugins

__all__ = [
    'PluginRegistry',
    'ASRProvider',
    'DiarizationProvider',
    'OutputWriter',
    'WordSegment',
    'Turn',
    'registry',
    'register_builtin_plugins'
]