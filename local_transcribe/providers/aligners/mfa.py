#!/usr/bin/env python3
"""
Aligner plugin using Montreal Forced Aligner (MFA).

Uses MFAAlignmentEngine for TextGrid parsing and word alignment utilities.
"""

from typing import List, Optional
import os
import pathlib
import tempfile
import subprocess
from local_transcribe.framework.plugin_interfaces import AlignerProvider, WordSegment, registry
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug

# Lazy import to avoid loading torch at module import time
_mfa_alignment_engine_class = None

def _get_mfa_alignment_engine_class():
    """Lazily import MFAAlignmentEngine."""
    global _mfa_alignment_engine_class
    if _mfa_alignment_engine_class is None:
        from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine
        _mfa_alignment_engine_class = MFAAlignmentEngine
    return _mfa_alignment_engine_class


class MFAAlignerProvider(AlignerProvider):
    """Aligner provider using Montreal Forced Aligner for word-level timestamps."""

    def __init__(self):
        # MFA setup
        self.mfa_models_dir = None
        self.logger = get_logger()
        self._alignment_engine = None
    
    @property
    def word_alignment_engine(self):
        """Lazily initialize the alignment engine."""
        if self._alignment_engine is None:
            MFAAlignmentEngine = _get_mfa_alignment_engine_class()
            self._alignment_engine = MFAAlignmentEngine(self.logger)
        return self._alignment_engine

    @property
    def name(self) -> str:
        return "mfa"

    @property
    def short_name(self) -> str:
        return "MFA"

    @property
    def description(self) -> str:
        return "Montreal Forced Aligner for precise word-level timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        # MFA doesn't use Hugging Face models, but we return empty list for compatibility
        return []

    def get_available_models(self) -> List[str]:
        # MFA uses pre-trained acoustic models and dictionaries
        return ["english_us_arpa"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload MFA models to cache."""
        # MFA models are downloaded on-demand, so this is a no-op
        pass

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which MFA models are available offline."""
        # For simplicity, assume MFA models need to be downloaded
        return models

    def _get_mfa_command(self):
        """Get the MFA command, checking local environment first."""
        # Check if MFA is available in project-local environment
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        local_mfa_env = project_root / ".mfa_env" / "bin" / "mfa"
        
        if local_mfa_env.exists():
            self.logger.info(f"[MFA] Using local MFA: {local_mfa_env}")
            return str(local_mfa_env)
        
        # Fall back to system MFA
        self.logger.info(f"[MFA] Using system MFA: mfa")
        return "mfa"

    def _ensure_mfa_models(self):
        """Ensure MFA acoustic model and dictionary are downloaded to project directory."""
        self.logger.info(f"[MFA] Checking MFA models in {self.mfa_models_dir}")
        # Set MFA_ROOT_DIR environment variable to use project models directory
        env = os.environ.copy()
        env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)

        mfa_cmd = self._get_mfa_command()
        self.logger.info(f"[MFA] Using MFA command: {mfa_cmd}")
        
        try:
            # Check if models are already downloaded
            result = subprocess.run(
                [mfa_cmd, "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            log_debug(f"[MFA] Available acoustic models: {result.stdout.strip()}")

            if "english_us_arpa" not in result.stdout:
                log_progress(f"[MFA] Downloading MFA English acoustic model to {self.mfa_models_dir}...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "acoustic", "english_us_arpa"],
                    check=True,
                    env=env
                )
                log_completion(f"[MFA] Acoustic model downloaded successfully")
            else:
                self.logger.info(f"[MFA] Acoustic model english_us_arpa already available")

            result = subprocess.run(
                [mfa_cmd, "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            log_debug(f"[MFA] Available dictionaries: {result.stdout.strip()}")

            if "english_us_arpa" not in result.stdout:
                log_progress(f"[MFA] Downloading MFA English dictionary to {self.mfa_models_dir}...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "dictionary", "english_us_arpa"],
                    check=True,
                    env=env
                )
                log_completion(f"[MFA] Dictionary downloaded successfully")
            else:
                self.logger.info(f"[MFA] Dictionary english_us_arpa already available")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"[MFA] ERROR: Failed to check/download MFA models: {e}")
            self.logger.error(f"[MFA] stdout: {e.stdout}")
            self.logger.error(f"[MFA] stderr: {e.stderr}")
            raise

    def _parse_textgrid(self, textgrid_path: pathlib.Path, original_transcript: str, speaker: Optional[str] = None) -> List[WordSegment]:
        """Parse MFA TextGrid output to extract word timestamps.
        
        Delegates to MFAAlignmentEngine for consistent TextGrid parsing
        across all MFA-based providers.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript with punctuation/capitalization
            speaker: Speaker identifier
            
        Returns:
            List of WordSegment objects with timestamps
        """
        self.logger.info(f"[MFA] Parsing TextGrid: {textgrid_path}")
        
        # Use the alignment engine to parse the TextGrid
        # segment_start_time=0.0 and segment_end_time=0.0 because MFA aligns from audio start
        word_dicts = self.word_alignment_engine.parse_textgrid_to_word_dicts(
            textgrid_path, original_transcript,
            segment_start_time=0.0, segment_end_time=0.0,
            speaker=speaker
        )
        
        # Convert word dicts to WordSegment objects
        segments = [
            WordSegment(
                text=wd["text"],
                start=wd["start"],
                end=wd["end"],
                speaker=wd.get("speaker")
            )
            for wd in word_dicts
        ]
        
        self.logger.info(f"[MFA] Parsed {len(segments)} word segments from TextGrid")
        return segments

    def _simple_alignment(self, audio_path: str, transcript: str, speaker: Optional[str] = None) -> List[WordSegment]:
        """Fallback to simple even-distribution alignment.
        
        Delegates to the alignment engine for consistent behavior.
        """
        import librosa

        # Get audio duration
        duration = librosa.get_duration(filename=audio_path)
        
        self.logger.info(f"[MFA] Simple alignment: Audio duration={duration:.2f}s")

        # Use alignment engine for consistent simple alignment
        word_dicts = self.word_alignment_engine.create_simple_alignment(
            transcript, segment_start_time=0.0, segment_duration=duration, speaker=speaker
        )
        
        # Convert to WordSegment objects
        segments = [
            WordSegment(
                text=wd["text"],
                start=wd["start"],
                end=wd["end"],
                speaker=wd.get("speaker")
            )
            for wd in word_dicts
        ]
        
        self.logger.info(f"[MFA] Simple alignment: created {len(segments)} segments")
        return segments

    def align_transcript(
        self,
        audio_path: str,
        transcript: str,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """Align transcript to audio using Montreal Forced Aligner.
        
        Note: MFA is a command-line tool and doesn't use GPU acceleration,
        so the device parameter is ignored.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript text
            device: Device to use (ignored for MFA)
            **kwargs: Additional options including 'role' or 'speaker'
        """
        self.logger.info(f"[MFA] Starting alignment for audio: {audio_path}")
        log_debug(f"[MFA] Transcript: {transcript[:200]}..." if len(transcript) > 200 else f"[MFA] Transcript: {transcript}")
        
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        log_debug(f"[MFA] Speaker: {speaker}")
        
        # Ensure MFA models directory exists
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / ".models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[MFA] Using MFA models directory: {self.mfa_models_dir}")

        # Download MFA models if needed
        self._ensure_mfa_models()
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Create a temporary directory for MFA processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            # Prepare input files for MFA
            # MFA expects: audio files in one directory and matching .lab (transcript) files
            audio_dir = temp_path / "audio"
            audio_dir.mkdir()

            # Copy audio to temp directory with a simple name
            audio_name = "audio.wav"
            audio_file = audio_dir / audio_name

            # MFA requires 16kHz mono WAV - audio_path should already be standardized
            # but let's copy it to ensure proper format
            import shutil
            shutil.copy(audio_path, audio_file)
            log_debug(f"[MFA] Audio file copied to: {audio_file} (size: {audio_file.stat().st_size} bytes)")
            
            # Get audio duration for debugging
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_file))
            log_debug(f"[MFA] Audio duration: {audio_duration:.2f} seconds")
            log_debug(f"[MFA] Audio file copied to: {audio_file} (size: {audio_file.stat().st_size} bytes)")
            
            # Get audio duration for debugging
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_file))
            log_debug(f"[MFA] Audio duration: {audio_duration:.2f} seconds")

            # Create matching transcript file (.lab extension)
            # MFA needs text without punctuation, so normalize it
            normalized_transcript = ' '.join(
                ''.join(c for c in word if c.isalnum() or c == "'")
                for word in transcript.split()
            )
            
            log_debug(f"[MFA] Normalized transcript: {normalized_transcript[:200]}..." if len(normalized_transcript) > 200 else f"[MFA] Normalized transcript: {normalized_transcript}")
            log_debug(f"[MFA] Transcript word count: {len(normalized_transcript.split())}")
            log_debug(f"[MFA] Temp directory: {temp_path}")
            
            transcript_file = audio_dir / f"{audio_name.rsplit('.', 1)[0]}.lab"
            transcript_file.write_text(normalized_transcript, encoding='utf-8')
            log_debug(f"[MFA] Transcript file written: {transcript_file} (size: {transcript_file.stat().st_size} bytes)")

            # Setup output directory for alignments
            output_dir = temp_path / "output"
            output_dir.mkdir()

            # Run MFA alignment
            try:
                # MFA command: mfa align_one <audio_file> <text_file> <dictionary> <acoustic_model> <output_path>
                # Using English dictionary and acoustic model
                # Set MFA_ROOT_DIR to use project models directory
                env = os.environ.copy()
                env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)
                env["MFA_NO_HISTORY"] = "1"  # Disable command history to prevent atexit issues
                
                # Add MFA environment bin directory to PATH so MFA can find OpenFST binaries
                mfa_env_bin = pathlib.Path(self._get_mfa_command()).parent
                env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")
                
                log_debug(f"[MFA] MFA environment bin: {mfa_env_bin}")
                log_debug(f"[MFA] Updated PATH: {env['PATH']}")
                log_debug(f"[MFA] Checking if fstcompile exists: {(mfa_env_bin / 'fstcompile').exists()}")

                # Output TextGrid file
                textgrid_file = output_dir / f"{audio_name.rsplit('.', 1)[0]}.TextGrid"

                project_root = pathlib.Path(__file__).parent.parent.parent.parent
                mfa_cmd = self._get_mfa_command()
                config_path = project_root / "mfa_config.yaml"
                cmd = [
                    mfa_cmd, "align_one",
                    str(audio_file),  # Audio file path
                    str(transcript_file),  # Text file path
                    "english_us_arpa",  # Dictionary
                    "english_us_arpa",  # Acoustic model
                    str(textgrid_file),  # Output TextGrid path
                    "--config_path", str(config_path),
                    "--single_speaker",  # Single speaker mode
                    "--clean",
                    "--final_clean",
                    "--verbose",
                    "--debug",
                ]

                log_debug(f"[MFA] Running command: {' '.join(cmd)}")
                log_debug(f"[MFA] Audio file: {audio_file} (exists: {audio_file.exists()})")
                log_debug(f"[MFA] Transcript file: {transcript_file} (content: {normalized_transcript[:100]}...)")
                log_debug(f"[MFA] Expected output: {textgrid_file}")
                log_debug(f"[MFA] Starting MFA subprocess now...")

                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Changed to False to see live output
                    text=True,
                    check=True,
                    env=env,
                    timeout=3600  # 1 hour timeout
                )

                self.logger.info(f"[MFA] MFA subprocess completed!")
                self.logger.info(f"[MFA] Command completed successfully. Exit code: {result.returncode}")
                # Note: stdout/stderr not captured when capture_output=False

                # Parse TextGrid to extract word timestamps
                if textgrid_file.exists():
                    self.logger.info(f"[MFA] TextGrid file exists at {textgrid_file}, parsing...")
                    log_debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
                    segments = self._parse_textgrid(textgrid_file, transcript, speaker=speaker)
                    self.logger.info(f"[MFA] Successfully parsed {len(segments)} word segments from TextGrid")
                    return segments
                else:
                    self.logger.error(f"[MFA] ERROR: TextGrid file not found at {textgrid_file}")
                    self.logger.info(f"[MFA] Falling back to simple alignment")
                    return self._simple_alignment(audio_path, transcript, speaker=speaker)

            except subprocess.TimeoutExpired:
                self.logger.error(f"[MFA] ERROR: MFA alignment timed out after 3600 seconds")
                self.logger.error(f"[MFA] Command: {' '.join(cmd)}")
                log_debug(f"[MFA] Checking if TextGrid was created despite timeout: {textgrid_file.exists()}")
                if textgrid_file.exists():
                    log_debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
                self.logger.info(f"[MFA] Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript, speaker=speaker)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"[MFA] ERROR: MFA alignment failed with exit code {e.returncode}")
                self.logger.error(f"[MFA] Command: {' '.join(cmd)}")
                # Note: stdout/stderr not captured when capture_output=False
                log_debug(f"[MFA] Checking if TextGrid was created despite error: {textgrid_file.exists()}")
                if textgrid_file.exists():
                    log_debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
                self.logger.info(f"[MFA] Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript, speaker=speaker)
            except FileNotFoundError:
                self.logger.error(f"[MFA] ERROR: MFA command not found: {mfa_cmd}")
                self.logger.error(f"[MFA] Run: bash setup_mfa.sh")
                self.logger.info(f"[MFA] Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript, speaker=speaker)

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure MFA models are available."""
        # MFA models are handled in _ensure_mfa_models()
        pass


def register_aligner_plugins():
    """Register aligner plugins."""
    registry.register_aligner_provider(MFAAlignerProvider())


# Auto-register on import
register_aligner_plugins()