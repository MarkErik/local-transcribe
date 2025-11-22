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
from local_transcribe.lib.system_output import get_logger, log_progress, log_completion


class MFAAlignerProvider(AlignerProvider):
    """Aligner provider using Montreal Forced Aligner for word-level timestamps."""

    def __init__(self):
        # MFA setup
        self.mfa_models_dir = None
        self.logger = get_logger()

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
            self.logger.debug(f"[MFA] Available acoustic models: {result.stdout.strip()}")

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
            self.logger.debug(f"[MFA] Available dictionaries: {result.stdout.strip()}")

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
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript with punctuation/capitalization
            speaker: Speaker identifier
        """
        self.logger.info(f"[MFA] Parsing TextGrid: {textgrid_path}")
        segments = []

        try:
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.logger.debug(f"[MFA] TextGrid has {len(lines)} lines")
            if len(lines) < 10:
                self.logger.debug(f"[MFA] TextGrid content (first 10 lines): {lines}")
            
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
                    self.logger.debug(f"[MFA] Found word tier at line {i}")
                # Find the next tier (phones) to know where words tier ends
                elif word_tier_start is not None and 'name = "phones"' in line:
                    word_tier_end = i
                    self.logger.debug(f"[MFA] Word tier ends at line {i}")
                    break

            if word_tier_start is None:
                self.logger.error(f"[MFA] ERROR: Could not find word tier in TextGrid")
                raise ValueError("Could not find word tier in TextGrid")
            
            # If we didn't find the phones tier, parse until end of file
            if word_tier_end is None:
                word_tier_end = len(lines)
                self.logger.debug(f"[MFA] No phones tier found, parsing until end of file")

            # Parse intervals only within the words tier
            i = word_tier_start
            interval_count = 0
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
                            interval_count += 1
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
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"[MFA] Warning: Failed to parse interval at line {i}: {e}")
                        pass

                i += 1

            self.logger.info(f"[MFA] Successfully parsed {interval_count} intervals, created {len(segments)} segments")

        except Exception as e:
            self.logger.error(f"[MFA] ERROR: Failed to parse TextGrid: {e}")
            raise

        # Replace <unk> with original words using context
        self._replace_unk_with_original(segments, original_transcript)

        return segments

    def _replace_unk_with_original(self, segments: List[WordSegment], original_transcript: str) -> None:
        """Replace <unk> tokens in aligned segments with words from the original transcript using two-pointer alignment."""
        self.logger.debug(f"[MFA UNK REPLACE] Starting <unk> replacement")
        
        aligned_texts = [seg.text for seg in segments]
        original_words = original_transcript.split()
        
        self.logger.debug(f"[MFA UNK REPLACE] Original transcript word count: {len(original_words)}")
        self.logger.debug(f"[MFA UNK REPLACE] MFA aligned word count before replacement: {len(aligned_texts)}")
        self.logger.debug(f"[MFA UNK REPLACE] Aligned texts ({len(aligned_texts)} words): {' '.join(aligned_texts[:10])} ... {' '.join(aligned_texts[-10:])}" if len(aligned_texts) > 20 else f"[MFA UNK REPLACE] Aligned texts: {' '.join(aligned_texts)}")
        self.logger.debug(f"[MFA UNK REPLACE] Original words ({len(original_words)} words): {' '.join(original_words[:10])} ... {' '.join(original_words[-10:])}" if len(original_words) > 20 else f"[MFA UNK REPLACE] Original words: {' '.join(original_words)}")
        
        ptr = 0
        for i, seg in enumerate(segments):
            if seg.text == "<unk>":
                # Debug: show context
                start_idx = max(0, i - 5)
                end_idx = min(len(aligned_texts), i + 6)
                aligned_context = aligned_texts[start_idx:end_idx]
                
                orig_start = max(0, ptr - 5)
                orig_end = min(len(original_words), ptr + 6)
                original_context = original_words[orig_start:orig_end]
                
                self.logger.debug(f"[MFA UNK REPLACE] Replacing <unk> at position {i}: Aligned context: {' '.join(aligned_context)} | Original context around ptr {ptr}: {' '.join(original_context)}")
                
                if ptr < len(original_words):
                    replacement = original_words[ptr]
                    seg.text = replacement
                    self.logger.debug(f"[MFA UNK REPLACE] Replaced with: '{replacement}'")
                    ptr += 1
                else:
                    self.logger.debug(f"[MFA UNK REPLACE] No more original words available, leaving as <unk>")
            else:
                if ptr < len(original_words) and seg.text.lower() == original_words[ptr].lower():
                    self.logger.debug(f"[MFA UNK REPLACE] Matched '{seg.text}' with original '{original_words[ptr]}', advancing ptr to {ptr+1}")
                    ptr += 1
                else:
                    self.logger.debug(f"[MFA UNK REPLACE] No match for '{seg.text}' at ptr {ptr}, not advancing ptr")
        
        final_texts = [seg.text for seg in segments]
        self.logger.debug(f"[MFA UNK REPLACE] MFA aligned word count after replacement: {len(final_texts)}")
        self.logger.debug(f"[MFA UNK REPLACE] Final aligned texts: {' '.join(final_texts[:10])} ... {' '.join(final_texts[-10:])}" if len(final_texts) > 20 else f"[MFA UNK REPLACE] Final aligned texts: {' '.join(final_texts)}")

    def _simple_alignment(self, audio_path: str, transcript: str, speaker: Optional[str] = None) -> List[WordSegment]:
        """Fallback to simple even-distribution alignment."""
        import librosa

        # Get audio duration
        duration = librosa.get_duration(filename=audio_path)

        # Split transcript into words
        words = transcript.split()

        if not words:
            self.logger.info(f"[MFA] Simple alignment: No words in transcript")
            return []

        # Simple even distribution
        word_duration = duration / len(words)

        self.logger.info(f"[MFA] Simple alignment: Audio duration={duration:.2f}s, {len(words)} words, word_duration={word_duration:.3f}s")
        self.logger.debug(f"[MFA] First 5 words: {words[:5]}")
        self.logger.debug(f"[MFA] Last 5 words: {words[-5:]}")

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
        self.logger.info(f"[MFA] Starting alignment for audio: {audio_path}")
        self.logger.debug(f"[MFA] Transcript: {transcript[:200]}..." if len(transcript) > 200 else f"[MFA] Transcript: {transcript}")
        
        # Extract speaker from kwargs (passed from split_audio mode)
        speaker = kwargs.get('role') or kwargs.get('speaker')
        self.logger.debug(f"[MFA] Speaker: {speaker}")
        
        # Ensure MFA models directory exists
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models")))
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
            self.logger.debug(f"[MFA] Audio file copied to: {audio_file} (size: {audio_file.stat().st_size} bytes)")
            
            # Get audio duration for debugging
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_file))
            self.logger.debug(f"[MFA] Audio duration: {audio_duration:.2f} seconds")
            self.logger.debug(f"[MFA] Audio file copied to: {audio_file} (size: {audio_file.stat().st_size} bytes)")
            
            # Get audio duration for debugging
            import librosa
            audio_duration = librosa.get_duration(filename=str(audio_file))
            self.logger.debug(f"[MFA] Audio duration: {audio_duration:.2f} seconds")

            # Create matching transcript file (.lab extension)
            # MFA needs text without punctuation, so normalize it
            normalized_transcript = ' '.join(
                ''.join(c for c in word if c.isalnum() or c == "'")
                for word in transcript.split()
            )
            
            self.logger.debug(f"[MFA] Normalized transcript: {normalized_transcript[:200]}..." if len(normalized_transcript) > 200 else f"[MFA] Normalized transcript: {normalized_transcript}")
            self.logger.debug(f"[MFA] Transcript word count: {len(normalized_transcript.split())}")
            self.logger.debug(f"[MFA] Temp directory: {temp_path}")
            
            transcript_file = audio_dir / f"{audio_name.rsplit('.', 1)[0]}.lab"
            transcript_file.write_text(normalized_transcript, encoding='utf-8')
            self.logger.debug(f"[MFA] Transcript file written: {transcript_file} (size: {transcript_file.stat().st_size} bytes)")

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
                
                self.logger.debug(f"[MFA] MFA environment bin: {mfa_env_bin}")
                self.logger.debug(f"[MFA] Updated PATH: {env['PATH']}")
                self.logger.debug(f"[MFA] Checking if fstcompile exists: {(mfa_env_bin / 'fstcompile').exists()}")

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

                self.logger.debug(f"[MFA] Running command: {' '.join(cmd)}")
                self.logger.debug(f"[MFA] Audio file: {audio_file} (exists: {audio_file.exists()})")
                self.logger.debug(f"[MFA] Transcript file: {transcript_file} (content: {normalized_transcript[:100]}...)")
                self.logger.debug(f"[MFA] Expected output: {textgrid_file}")
                self.logger.debug(f"[MFA] Starting MFA subprocess now...")

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
                    self.logger.debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
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
                self.logger.debug(f"[MFA] Checking if TextGrid was created despite timeout: {textgrid_file.exists()}")
                if textgrid_file.exists():
                    self.logger.debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
                self.logger.info(f"[MFA] Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript, speaker=speaker)
            except subprocess.CalledProcessError as e:
                self.logger.error(f"[MFA] ERROR: MFA alignment failed with exit code {e.returncode}")
                self.logger.error(f"[MFA] Command: {' '.join(cmd)}")
                # Note: stdout/stderr not captured when capture_output=False
                self.logger.debug(f"[MFA] Checking if TextGrid was created despite error: {textgrid_file.exists()}")
                if textgrid_file.exists():
                    self.logger.debug(f"[MFA] TextGrid file size: {textgrid_file.stat().st_size} bytes")
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