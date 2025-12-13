#!/usr/bin/env python3
"""
Core plugin system initialization and utilities.
"""

from local_transcribe.framework.plugin_interfaces import (
    PluginRegistry,
    TranscriberProvider,
    AlignerProvider,
    DiarizationProvider,
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
    'WordWriter',
    'WordSegment',
    'Turn',
    'registry'
]