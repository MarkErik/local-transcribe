#!/usr/bin/env python3
"""
Core plugin system initialization and utilities.
"""

from .plugin_interfaces import (
    PluginRegistry,
    TranscriberProvider,
    AlignerProvider,
    DiarizationProvider,
    UnifiedProvider,
    WordWriter,
    WordSegment,
    Turn,
    registry
)

__all__ = [
    'PluginRegistry',
    'TranscriberProvider',
    'AlignerProvider',
    'DiarizationProvider',
    'UnifiedProvider',
    'WordWriter',
    'WordSegment',
    'Turn',
    'registry'
]