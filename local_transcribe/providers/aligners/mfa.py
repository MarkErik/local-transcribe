#!/usr/bin/env python3
"""
Aligner plugin using Montreal Forced Aligner (MFA).
"""

from typing import List, Optional
import os
import pathlib
import tempfile
import subprocess
from local_transcribe.framework.plugin_interfaces import AlignerProvider, WordSegment, registry


class MFAAlignerProvider(AlignerProvider):
    """Aligner provider using Montreal Forced Aligner for word-level timestamps."""

    def __init__(self):
        # MFA setup
        self.mfa_models_dir = None

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
            return str(local_mfa_env)
        
        # Fall back to system MFA
        return "mfa"

    def _ensure_mfa_models(self):
        """Ensure MFA acoustic model and dictionary are downloaded to project directory."""
        # Set MFA_ROOT_DIR environment variable to use project models directory
        env = os.environ.copy()
        env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)

        mfa_cmd = self._get_mfa_command()
        
        try:
            # Check if models are already downloaded
            result = subprocess.run(
                [mfa_cmd, "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            if "english_us_arpa" not in result.stdout:
                print(f"[*] Downloading MFA English acoustic model to {self.mfa_models_dir}...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "acoustic", "english_us_arpa"],
                    check=True,
                    env=env
                )

            result = subprocess.run(
                [mfa_cmd, "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            if "english_us_arpa" not in result.stdout:
                print(f"[*] Downloading MFA English dictionary to {self.mfa_models_dir}...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "dictionary", "english_us_arpa"],
                    check=True,
                    env=env
                )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to check/download MFA models: {e}")
            raise

    def _parse_textgrid(self, textgrid_path: pathlib.Path, original_transcript: str, speaker: Optional[str] = None) -> List[WordSegment]:
        """Parse MFA TextGrid output to extract word timestamps.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript with punctuation/capitalization
            speaker: Speaker identifier
        """
        segments = []

        try:
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Build a mapping of normalized words to original words
            original_words = original_transcript.split()
            normalized_to_original = {}
            word_usage_count = {}  # Track how many times we've used each normalized form
            
            for orig_word in original_words:
                # Normalize: lowercase and strip punctuation
                normalized = ''.join(c.lower() for c in orig_word if c.isalnum())
                if normalized:
                    if normalized not in normalized_to_original:
                        normalized_to_original[normalized] = []
                        word_usage_count[normalized] = 0
                    normalized_to_original[normalized].append(orig_word)

            # Find the word tier
            word_tier_start = None
            word_tier_end = None
            
            for i, line in enumerate(lines):
                if 'name = "words"' in line:
                    word_tier_start = i
                # Find the next tier (phones) to know where words tier ends
                elif word_tier_start is not None and 'name = "phones"' in line:
                    word_tier_end = i
                    break

            if word_tier_start is None:
                raise ValueError("Could not find word tier in TextGrid")
            
            # If we didn't find the phones tier, parse until end of file
            if word_tier_end is None:
                word_tier_end = len(lines)

            # Parse intervals only within the words tier
            i = word_tier_start
            while i < word_tier_end:
                line = lines[i].strip()
                if line.startswith('intervals ['):
                    # Parse interval
                    # Skip to xmin, xmax, text lines
                    i += 1  # Move to xmin line
                    if i >= word_tier_end:
                        break
                    xmin_line = lines[i].strip()
                    i += 1  # Move to xmax line
                    if i >= word_tier_end:
                        break
                    xmax_line = lines[i].strip()
                    i += 1  # Move to text line
                    if i >= word_tier_end:
                        break
                    text_line = lines[i].strip()

                    # Extract values
                    try:
                        start = float(xmin_line.split('=')[1].strip())
                        end = float(xmax_line.split('=')[1].strip())
                        mfa_text = text_line.split('=')[1].strip().strip('"')

                        # Skip empty intervals, silence markers, and special tokens
                        if mfa_text and mfa_text not in ["", "<eps>", "sil", "sp", "spn"]:
                            # Map MFA's normalized text back to original formatting
                            normalized_key = mfa_text.lower()
                            
                            if normalized_key in normalized_to_original:
                                # Get the next occurrence of this word from the original transcript
                                word_list = normalized_to_original[normalized_key]
                                usage_idx = word_usage_count[normalized_key] % len(word_list)
                                original_text = word_list[usage_idx]
                                word_usage_count[normalized_key] += 1
                            else:
                                # Fallback: use MFA's text if we can't find a mapping
                                original_text = mfa_text
                            
                            segments.append(WordSegment(
                                text=original_text,
                                start=start,
                                end=end,
                                speaker=speaker
                            ))
                    except (ValueError, IndexError):
                        pass

                i += 1

        except Exception as e:
            print(f"Warning: Failed to parse TextGrid: {e}")
            raise

        return segments

    def _simple_alignment(self, audio_path: str, transcript: str, speaker: Optional[str] = None) -> List[WordSegment]:
        """Fallback to simple even-distribution alignment."""
        import librosa

        # Get audio duration
        duration = librosa.get_duration(filename=audio_path)

        # Split transcript into words
        words = transcript.split()

        if not words:
            return []

        # Simple even distribution
        word_duration = duration / len(words)

        segments = []
        current_time = 0.0

        for word in words:
            segments.append(WordSegment(
                text=word,
                start=current_time,
                end=current_time + word_duration,
                speaker=speaker
            ))
            current_time += word_duration

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
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        
        # Ensure MFA models directory exists
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)

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

            # Create matching transcript file (.lab extension)
            # MFA needs text without punctuation, so normalize it
            normalized_transcript = ' '.join(
                ''.join(c for c in word if c.isalnum() or c == "'")
                for word in transcript.split()
            )
            
            transcript_file = audio_dir / f"{audio_name.rsplit('.', 1)[0]}.lab"
            transcript_file.write_text(normalized_transcript, encoding='utf-8')

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
                
                # Add MFA environment bin directory to PATH so MFA can find OpenFST binaries
                mfa_env_bin = pathlib.Path(self._get_mfa_command()).parent
                env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")

                # Output TextGrid file
                textgrid_file = output_dir / f"{audio_name.rsplit('.', 1)[0]}.TextGrid"

                mfa_cmd = self._get_mfa_command()
                cmd = [
                    mfa_cmd, "align_one",
                    str(audio_file),  # Audio file path
                    str(transcript_file),  # Text file path
                    "english_us_arpa",  # Dictionary
                    "english_us_arpa",  # Acoustic model
                    str(textgrid_file),  # Output TextGrid path
                    "--single_speaker",  # Single speaker mode
                    "--beam", "15",  # Increase beam size for better alignment (higher = more tolerant)
                    "--retry_beam", "60",
                    "--lattice_beam","15",
                    "--quiet",  # Suppress verbose output
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env
                )

                # Parse TextGrid to extract word timestamps
                if textgrid_file.exists():
                    segments = self._parse_textgrid(textgrid_file, transcript, speaker=speaker)
                    return segments
                else:
                    print(f"Warning: MFA did not produce TextGrid output at {textgrid_file}")
                    return self._simple_alignment(audio_path, transcript, speaker=speaker)

            except subprocess.CalledProcessError as e:
                print(f"Warning: MFA alignment failed: {e.stderr}")
                print(f"Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript, speaker=speaker)
            except FileNotFoundError:
                print(f"Warning: MFA not installed or not in PATH")
                print(f"Run: bash setup_mfa.sh")
                print(f"Falling back to simple alignment")
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