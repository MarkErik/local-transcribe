#!/usr/bin/env python3
"""
Markdown Writer - Now delegates to Annotated Markdown Writer.

This module is kept for backward compatibility. The actual implementation
has moved to annotated_markdown_writer.py which provides rich hierarchical
output with inline interjections.
"""

from __future__ import annotations
from typing import List, Optional, Any
from pathlib import Path

from local_transcribe.framework.plugin_interfaces import OutputWriter, registry, WordSegment

# Import the new annotated markdown writer
from local_transcribe.providers.file_writers.annotated_markdown_writer import write_annotated_markdown


class MarkdownWriter(OutputWriter):
    """
    Markdown writer that delegates to the annotated markdown writer.
    
    This provides backward compatibility for code that references 'markdown'
    as an output format while using the new hierarchical output.
    """
    
    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return "Annotated Markdown format with hierarchical turn structure"

    @property
    def supported_formats(self) -> List[str]:
        return [".md"]

    def write(self, turns: Any, output_path: str, word_segments: Optional[List[WordSegment]] = None, **kwargs) -> None:
        """
        Write transcript to Markdown format.
        
        Delegates to the annotated markdown writer for hierarchical output.
        """
        write_annotated_markdown(turns, output_path)


# Register the writer
registry.register_output_writer(MarkdownWriter())
