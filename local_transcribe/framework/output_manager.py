#!/usr/bin/env python3
# framework/output_manager.py - Output writing and formatting logic

from typing import List, Dict, Any, Optional, Union
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
                             selected_formats: List[str], audio_config: Optional[Union[pathlib.Path, Dict[str, str], list[pathlib.Path]]] = None,
                             generate_video: bool = True, word_segments: Optional[List] = None) -> None:
        """
        Write selected outputs for merged transcript.
        
        Args:
            transcript: List of conversation turns to write
            paths: Dictionary containing output paths
            selected_formats: List of output formats to generate
            audio_config: Optional audio configuration for video rendering:
                - Single path for combined_audio mode
                - Dict mapping speaker names to audio paths for split_audio mode
                - List of audio paths (legacy format)
            generate_video: Whether to generate video output (default: True)
            word_segments: Optional list of word segments for detailed timing in outputs
        """
        print(f"[*] Writing output files...")
        print(f"[i] Selected formats: {selected_formats}")
        print(f"[i] Generate video: {generate_video}")
        
        # Write text-based outputs
        self._write_text_outputs(transcript, paths["merged"], selected_formats, word_segments)
        
        # Write video output (includes SRT generation internally)
        if 'video' in selected_formats and generate_video:
            print(f"[i] Video format detected, attempting to generate...")
            self._write_video_output(transcript, paths["merged"], audio_config, word_segments)
        else:
            print(f"[i] Video skipped - format not in {selected_formats} or generate_video={generate_video}")
        
        # Print completion message
        print("[✓] Output files written successfully.")
    
    def _write_text_outputs(self, transcript: List[Any], merged_dir: pathlib.Path, selected_formats: List[str], word_segments: Optional[List] = None) -> None:
        """Write text-based output formats."""
        if 'timestamped-txt' in selected_formats:
            timestamped_writer = self._registry.get_output_writer("timestamped-txt")
            timestamped_writer.write(transcript, merged_dir / "transcript.timestamped.txt", word_segments=word_segments)

        if 'plain-txt' in selected_formats:
            plain_writer = self._registry.get_output_writer("plain-txt")
            plain_writer.write(transcript, merged_dir / "transcript.txt", word_segments=word_segments)

        # Annotated Markdown format with hierarchical turn structure
        if 'markdown' in selected_formats:
            annotated_md_writer = self._registry.get_output_writer("markdown")
            annotated_md_writer.write(transcript, merged_dir / "transcript.md", word_segments=word_segments)

        # Dialogue Script format
        if 'dialogue-script' in selected_formats:
            script_writer = self._registry.get_output_writer("dialogue-script")
            script_writer.write(transcript, merged_dir / "transcript.script.txt", word_segments=word_segments)

        # Interactive HTML Timeline
        if 'html-timeline' in selected_formats:
            html_writer = self._registry.get_output_writer("html-timeline")
            html_writer.write(transcript, merged_dir / "transcript_timeline.html", word_segments=word_segments)

        if 'turns-json' in selected_formats:
            json_turns_writer = self._registry.get_output_writer("turns-json")
            json_turns_writer.write(transcript, merged_dir / "transcript.turns.json", word_segments=word_segments)
            print(f"[i] Final transcript JSON saved to: transcript.turns.json")
    
    def _write_subtitle_outputs(self, transcript: List[Any], merged_dir: pathlib.Path,
                              audio_path: Optional[pathlib.Path], selected_formats: List[str]) -> None:
        """Write subtitle outputs (deprecated - kept for backward compatibility)."""
        # This method is deprecated and kept only for backward compatibility
        # Video output now handles SRT generation internally
        pass
    
    def _write_video_output(self, transcript: List[Any], merged_dir: pathlib.Path,
                           audio_config: Optional[Union[pathlib.Path, Dict[str, str], list[pathlib.Path]]] = None,
                           word_segments: Optional[List] = None) -> None:
        """Write video output with subtitles."""
        print(f"[*] Rendering video with subtitles...")
        try:
            # Check if video writer is available
            video_writer = self._registry.get_output_writer("video")
            if video_writer is None:
                print("[!] Warning: Video writer not found in registry")
                return
                
            print(f"[i] Found video writer: {video_writer.name}")
            video_path = merged_dir / "video_with_subtitles.mp4"
            
            # Check audio config
            if audio_config is None:
                print("[!] Warning: No audio configuration provided for video generation")
                return
                
            print(f"[i] Audio config type: {type(audio_config)}")
            if isinstance(audio_config, dict):
                print(f"[i] Number of audio tracks: {len(audio_config)}")
            
            # Pass audio configuration as kwargs to the video writer
            video_writer.write(transcript, str(video_path), word_segments=word_segments, audio_config=audio_config)
            print(f"[✓] Video rendered successfully: {video_path.name}")
        except Exception as e:
            print(f"[!] Warning: Video rendering failed: {e}")
            import traceback
            traceback.print_exc()
    
    # _render_video_with_subtitles is deprecated and replaced by _write_video_output
    
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