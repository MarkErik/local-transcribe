#!/usr/bin/env python3
# framework/output_manager.py - Output writing and formatting logic

from typing import List, Dict, Any, Optional
import pathlib


class OutputManager:
    """Handles writing transcript outputs in various formats."""
    
    def __init__(self, registry):
        self.registry = registry
    
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
        self._write_text_outputs(transcript, paths["merged"])
        
        # Write subtitle outputs
        self._write_subtitle_outputs(transcript, paths["merged"], audio_path)
        
        # Print completion message
        print("[✓] Output files written successfully.")
    
    def _write_text_outputs(self, transcript: List[Any], merged_dir: pathlib.Path) -> None:
        """Write text-based output formats."""
        if 'timestamped-txt' in self._get_selected_formats():
            timestamped_writer = self.registry.get_output_writer("timestamped-txt")
            timestamped_writer.write(transcript, merged_dir / "transcript.timestamped.txt")

        if 'plain-txt' in self._get_selected_formats():
            plain_writer = self.registry.get_output_writer("plain-txt")
            plain_writer.write(transcript, merged_dir / "transcript.txt")

        if 'csv' in self._get_selected_formats():
            csv_writer = self.registry.get_output_writer("csv")
            csv_writer.write(transcript, merged_dir / "transcript.csv")

        if 'markdown' in self._get_selected_formats():
            markdown_writer = self.registry.get_output_writer("markdown")
            markdown_writer.write(transcript, merged_dir / "transcript.md")

        if 'turns-json' in self._get_selected_formats():
            json_turns_writer = self.registry.get_output_writer("turns-json")
            json_turns_writer.write(transcript, merged_dir / "transcript.turns.json")
            print(f"[i] Final transcript JSON saved to: transcript.turns.json")
    
    def _write_subtitle_outputs(self, transcript: List[Any], merged_dir: pathlib.Path, 
                              audio_path: Optional[pathlib.Path]) -> None:
        """Write subtitle outputs and optionally render video."""
        srt_path = None
        
        if 'srt' in self._get_selected_formats():
            srt_writer = self.registry.get_output_writer("srt")
            srt_path = merged_dir / "subtitles.srt"
            srt_writer.write(transcript, srt_path)

        if 'vtt' in self._get_selected_formats():
            vtt_writer = self.registry.get_output_writer("vtt")
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
    
    def _get_selected_formats(self) -> List[str]:
        """Get the list of selected formats (placeholder - will be passed from caller)."""
        # This method is a placeholder since the actual selected formats are passed to write_selected_outputs
        # In practice, this information comes from the args.selected_formats parameter
        return []