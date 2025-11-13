#!/usr/bin/env python3
"""
Core plugin system initialization and utilities.
"""

from .plugins import (
    PluginRegistry,
    TranscriberProvider,
    AlignerProvider,
    DiarizationProvider,
    UnifiedProvider,
    TurnBuilderProvider,
    OutputWriter,
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
    'TurnBuilderProvider',
    'OutputWriter',
    'WordSegment',
    'Turn',
    'registry'
]