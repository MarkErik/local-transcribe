#!/usr/bin/env python3
# framework/output_manager.py - Output writing and formatting logic

from typing import List, Dict, Any, Optional
import pathlib


class OutputManager:
    """Handles writing transcript outputs in various formats."""
    
    _instance = None
    _registry = None
    
    def __new__(cls, registry=None):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = registry
        return cls._instance
    
    def __init__(self, registry=None):
        """Initialize with registry (only once due to singleton pattern)."""
        # Only set registry if this is the first initialization
        if self._registry is None and registry is not None:
            self._registry = registry
    
    def write_selected_outputs(self, transcript: List[Any], paths: Dict[str, pathlib.Path],
                             selected_formats: List[str], audio_path: Optional[pathlib.Path] = None) -> None:
        """
        Write selected outputs for merged transcript.
        
        Args:
            transcript: List of conversation turns to write
            paths: Dictionary containing output paths
            selected_formats: List of output formats to generate
            audio_path: Optional audio path for video rendering
        """
        print(f"[*] Writing output files...")
        
        # Write text-based outputs
        self._write_text_outputs(transcript, paths["merged"], selected_formats)
        
        # Write subtitle outputs
        self._write_subtitle_outputs(transcript, paths["merged"], audio_path, selected_formats)
        
        # Print completion message
        print("[✓] Output files written successfully.")
    
    def _write_text_outputs(self, transcript: List[Any], merged_dir: pathlib.Path, selected_formats: List[str]) -> None:
        """Write text-based output formats."""
        if 'timestamped-txt' in selected_formats:
            timestamped_writer = self._registry.get_output_writer("timestamped-txt")
            timestamped_writer.write(transcript, merged_dir / "transcript.timestamped.txt")

        if 'plain-txt' in selected_formats:
            plain_writer = self._registry.get_output_writer("plain-txt")
            plain_writer.write(transcript, merged_dir / "transcript.txt")

        if 'csv' in selected_formats:
            csv_writer = self._registry.get_output_writer("csv")
            csv_writer.write(transcript, merged_dir / "transcript.csv")

        if 'markdown' in selected_formats:
            markdown_writer = self._registry.get_output_writer("markdown")
            markdown_writer.write(transcript, merged_dir / "transcript.md")

        if 'turns-json' in selected_formats:
            json_turns_writer = self._registry.get_output_writer("turns-json")
            json_turns_writer.write(transcript, merged_dir / "transcript.turns.json")
            print(f"[i] Final transcript JSON saved to: transcript.turns.json")
    
    def _write_subtitle_outputs(self, transcript: List[Any], merged_dir: pathlib.Path,
                              audio_path: Optional[pathlib.Path], selected_formats: List[str]) -> None:
        """Write subtitle outputs and optionally render video."""
        srt_path = None
        
        if 'srt' in selected_formats:
            srt_writer = self._registry.get_output_writer("srt")
            srt_path = merged_dir / "subtitles.srt"
            srt_writer.write(transcript, srt_path)

        if 'vtt' in selected_formats:
            vtt_writer = self._registry.get_output_writer("vtt")
            vtt_writer.write(transcript, merged_dir / "subtitles.vtt")

        # Render video with subtitles if SRT is available and audio path is provided
        if srt_path and audio_path is not None:
            self._render_video_with_subtitles(srt_path, merged_dir / "video_with_subtitles.mp4", audio_path)
    
    def _render_video_with_subtitles(self, srt_path: pathlib.Path, output_path: pathlib.Path, 
                                   audio_path: pathlib.Path) -> None:
        """Render video with subtitles using SRT file."""
        print(f"[*] Rendering video with subtitles...")
        try:
            from local_transcribe.providers.writers.render_video import render_video
            render_video(srt_path, output_path, audio_path=audio_path)
            print(f"[✓] Video rendered successfully: {output_path.name}")
        except Exception as e:
            print(f"[!] Warning: Video rendering failed: {e}")
    
    @classmethod
    def get_instance(cls, registry=None):
        """Get the singleton instance of OutputManager."""
        if cls._instance is None:
            cls(registry)
        return cls._instance
    
    def reset_singleton(self):
        """Reset the singleton instance (useful for testing)."""
        self.__class__._instance = None
        self.__class__._registry = None