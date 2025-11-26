#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with MFA alignment.

This plugin combines Granite's transcription capabilities with Montreal Forced Aligner
to produce chunked transcripts where each word has timestamps. Unlike the separate
granite + mfa pipeline, this processes each chunk with alignment before moving to the next,
resulting in chunks that contain timestamped words ready for intelligent stitching.
"""

from typing import List, Optional, Dict, Any
import os
import pathlib
import re
import torch
import math
import librosa
import tempfile
import subprocess
import json
from datetime import datetime
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug


class GraniteMFATranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + MFA for chunked transcription with timestamps."""

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite MFA Transcriber Provider")
        # Granite configuration
        self.model_mapping = {
            "granite-2b": "ibm-granite/granite-speech-3.3-2b",
            "granite-8b": "ibm-granite/granite-speech-3.3-8b"
        }
        self.selected_model = None
        self.processor = None
        self.model = None
        self.chunk_length_seconds = 60.0
        self.overlap_seconds = 4.0
        self.min_chunk_seconds = 7.0
        
        # MFA configuration
        self.mfa_models_dir = None

    @property
    def device(self):
        return get_system_capability()

    @property
    def name(self) -> str:
        return "granite_mfa"

    @property
    def short_name(self) -> str:
        return "Granite + MFA"

    @property
    def description(self) -> str:
        return "IBM Granite transcription with MFA alignment (produces chunked timestamped output)"

    @property
    def has_builtin_alignment(self) -> bool:
        """This provider combines transcription and alignment."""
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return required Granite models (MFA models handled separately)."""
        if selected_model and selected_model in self.model_mapping:
            return [self.model_mapping[selected_model]]
        return [self.model_mapping["granite-8b"]]

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        self.logger.info("Starting model preload for Granite models")
        import sys

        cache_dir = models_dir / "transcribers" / "granite"
        cache_dir.mkdir(parents=True, exist_ok=True)

        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        os.environ["HF_HUB_OFFLINE"] = "0"

        # Reload huggingface_hub modules
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]

        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]

        from huggingface_hub import snapshot_download

        try:
            for model in models:
                if model in self.model_mapping.values():
                    os.environ["HF_HOME"] = str(cache_dir)
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, token=token)
                    log_completion(f"{model} downloaded successfully.")
        except Exception as e:
            raise Exception(f"Failed to download {model}: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline."""
        missing_models = []
        for model in models:
            if model in self.model_mapping.values():
                xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
                if xdg_cache_home:
                    models_root = pathlib.Path(xdg_cache_home)
                else:
                    models_root = pathlib.Path.home() / ".cache" / "huggingface"
                
                hub_dir = models_root / "huggingface" / "hub"
                hf_model_name = model.replace("/", "--")
                model_dir = hub_dir / f"models--{hf_model_name}"
                
                if not model_dir.exists() or not any(model_dir.rglob("*.bin")) and not any(model_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_granite_model(self):
        """Load the Granite model if not already loaded."""
        log_progress("Loading Granite model...")
        if self.model is None:
            model_name = self.model_mapping.get(self.selected_model, self.model_mapping["granite-8b"])
            
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                models_root = pathlib.Path.home() / ".cache" / "huggingface"
            
            cache_dir = models_root / "huggingface" / "hub"

            try:
                token = os.getenv("HF_TOKEN")
                self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True, token=token)
                self.tokenizer = self.processor.tokenizer

                log_progress(f"Loading Granite model on device: {self.device}")
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_name, 
                    local_files_only=True, 
                    token=token
                ).to(self.device)
                log_completion("Granite model loaded successfully")
            except Exception as e:
                log_debug(f"Failed to load model {model_name}")
                log_debug(f"Cache directory exists: {cache_dir.exists()}")
                if cache_dir.exists():
                    log_debug(f"Cache directory contents: {list(cache_dir.iterdir())}")
                raise e

    def _get_mfa_command(self):
        """Get the MFA command, checking local environment first."""
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        local_mfa_env = project_root / ".mfa_env" / "bin" / "mfa"
        
        if local_mfa_env.exists():
            return str(local_mfa_env)
        
        return "mfa"

    def _ensure_mfa_models(self):
        """Ensure MFA acoustic model and dictionary are downloaded."""
        log_progress("Ensuring MFA models are available...")
        log_progress(f"Checking MFA models in {self.mfa_models_dir}")
        env = os.environ.copy()
        env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)

        mfa_cmd = self._get_mfa_command()
        
        try:
            result = subprocess.run(
                [mfa_cmd, "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            if "english_us_arpa" not in result.stdout:
                log_progress("Downloading MFA English acoustic model...")
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
                log_progress("Downloading MFA English dictionary...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "dictionary", "english_us_arpa"],
                    check=True,
                    env=env
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to check/download MFA models: {e}")
            raise

    def _transcribe_single_chunk(self, wav, **kwargs) -> str:
        """Transcribe a single audio chunk using Granite."""
        log_progress("Transcribing audio chunk with Granite")
        try:
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)

            chat = [
                {
                    "role": "system",
                    "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
                },
                {
                    "role": "user",
                    "content": "<|audio|>can you transcribe the speech into a written format?",
                }
            ]

            text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.processor(
                text,
                wav_tensor,
                device=self.device,
                return_tensors="pt",
            ).to(self.device)

            # The recommended repetition penalty is 3 as long as input IDs are excluded.
            # Otherwise, you should use a repetition penalty of 1 to keep results stable.
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=3.0,
                prompt_ignore_length=model_inputs["input_ids"].shape[-1],
            )

            with torch.no_grad():
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    early_stopping = True,
                    logits_processor=[repetition_penalty_processor],
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )

            cleaned_text = self._clean_transcription_output(output_text[0].strip(), verbose=kwargs.get('verbose', False))

            return cleaned_text
            
        finally:
            if 'wav_tensor' in locals():
                del wav_tensor
            if 'model_inputs' in locals():
                for key in list(model_inputs.keys()):
                    del model_inputs[key]
                del model_inputs
            if 'model_outputs' in locals():
                del model_outputs
            if 'new_tokens' in locals():
                del new_tokens
            
            import gc
            gc.collect()
            clear_device_cache()

    def _clean_transcription_output(self, text: str, verbose: bool = False) -> str:
        """Clean the transcription output by removing dialogue markers and quotation marks."""
        if verbose:
            user_count = len(re.findall(r'\bUser:\s*', text, flags=re.IGNORECASE))
            assistant_count = len(re.findall(r'\bAI Assistant:\s*', text, flags=re.IGNORECASE))
            assistant_short_count = len(re.findall(r'\bAssistant:\s*', text, flags=re.IGNORECASE))
            total_removed = user_count + assistant_count + assistant_short_count
        
        text = re.sub(r'\bUser:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAI Assistant:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAssistant:\s*', '', text, flags=re.IGNORECASE)
        
        text = text.replace('"', '')
        text = text.replace('\u201C', '')
        text = text.replace('\u201D', '')
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if verbose and 'total_removed' in locals() and total_removed > 0:
            self.logger.info(f"Removed {total_removed} labels from chunk transcript.")
        
        return text

    def _align_chunk_with_mfa(self, chunk_wav, chunk_transcript: str, chunk_start_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Align a single chunk using MFA and return timestamped words.
        
        Args:
            chunk_wav: Audio data for this chunk
            chunk_transcript: Transcript text for this chunk
            chunk_start_time: Absolute start time of this chunk in the full audio (seconds)
            speaker: Speaker identifier
        """
        log_progress(f"Aligning transcript with MFA (chunk starts at {chunk_start_time:.2f}s)")
        
        # Calculate chunk end time for potential timestamp interpolation
        chunk_duration = len(chunk_wav) / 16000.0
        chunk_end_time = chunk_start_time + chunk_duration
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            audio_dir = temp_path / "audio"
            audio_dir.mkdir()

            # Save chunk audio as WAV
            audio_file = audio_dir / "chunk.wav"
            import soundfile as sf
            sf.write(str(audio_file), chunk_wav, 16000)

            # Normalize transcript for MFA
            normalized_transcript = ' '.join(
                ''.join(c for c in word if c.isalnum() or c == "'")
                for word in chunk_transcript.split()
            )
            
            transcript_file = audio_dir / "chunk.lab"
            transcript_file.write_text(normalized_transcript, encoding='utf-8')

            output_dir = temp_path / "output"
            output_dir.mkdir()

            env = os.environ.copy()
            env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)
            env["MFA_NO_HISTORY"] = "1"
            
            mfa_env_bin = pathlib.Path(self._get_mfa_command()).parent
            env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")

            textgrid_file = output_dir / "chunk.TextGrid"

            project_root = pathlib.Path(__file__).parent.parent.parent.parent
            mfa_cmd = self._get_mfa_command()
            config_path = project_root / "mfa_config.yaml"
            
            cmd = [
                mfa_cmd, "align_one",
                str(audio_file),
                str(transcript_file),
                "english_us_arpa",
                "english_us_arpa",
                str(textgrid_file),
                "--config_path", str(config_path),
                "--single_speaker",
                "--clean",
                "--final_clean",
            ]

            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    env=env,
                    timeout=600
                )

                if textgrid_file.exists():
                    return self._parse_textgrid_to_word_dicts(textgrid_file, chunk_transcript, chunk_start_time, chunk_end_time, speaker)
                else:
                    # Fallback to simple distribution
                    return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback to simple distribution
                return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

    def _parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str, chunk_start_time: float = 0.0, chunk_end_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse MFA TextGrid and return list of word dicts with timestamps.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript with punctuation/capitalization
            chunk_start_time: Absolute start time of this chunk in the full audio (seconds)
            chunk_end_time: Absolute end time of this chunk in the full audio (seconds)
            speaker: Speaker identifier
        """
        word_dicts = []

        try:
            with open(textgrid_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Build mapping of normalized to original words
            original_words = original_transcript.split()
            normalized_to_original = {}
            word_usage_count = {}
            
            for orig_word in original_words:
                normalized = ''.join(c.lower() for c in orig_word if c.isalnum())
                if normalized:
                    if normalized not in normalized_to_original:
                        normalized_to_original[normalized] = []
                        word_usage_count[normalized] = 0
                    normalized_to_original[normalized].append(orig_word)

            # Find word tier
            word_tier_start = None
            word_tier_end = None
            
            for i, line in enumerate(lines):
                if 'name = "words"' in line:
                    word_tier_start = i
                elif word_tier_start is not None and 'name = "phones"' in line:
                    word_tier_end = i
                    break

            if word_tier_start is None:
                raise ValueError("Could not find word tier in TextGrid")
            
            if word_tier_end is None:
                word_tier_end = len(lines)

            # Parse intervals
            i = word_tier_start
            while i < word_tier_end:
                line = lines[i].strip()
                if line.startswith('intervals ['):
                    i += 1
                    if i >= word_tier_end:
                        break
                    xmin_line = lines[i].strip()
                    i += 1
                    if i >= word_tier_end:
                        break
                    xmax_line = lines[i].strip()
                    i += 1
                    if i >= word_tier_end:
                        break
                    text_line = lines[i].strip()

                    try:
                        start = float(xmin_line.split('=')[1].strip())
                        end = float(xmax_line.split('=')[1].strip())
                        mfa_text = text_line.split('=')[1].strip().strip('"')

                        if mfa_text and mfa_text not in ["", "<eps>", "sil", "sp", "spn"]:
                            normalized_key = mfa_text.lower()
                            
                            if normalized_key in normalized_to_original:
                                word_list = normalized_to_original[normalized_key]
                                usage_idx = word_usage_count[normalized_key] % len(word_list)
                                original_text = word_list[usage_idx]
                                word_usage_count[normalized_key] += 1
                            else:
                                original_text = mfa_text
                            
                            # Adjust timestamps to absolute time in full audio
                            word_dicts.append({
                                "text": original_text,
                                "start": start + chunk_start_time,
                                "end": end + chunk_start_time,
                                "speaker": speaker
                            })
                    except (ValueError, IndexError):
                        pass

                i += 1

        except Exception as e:
            self.logger.warning(f"Failed to parse TextGrid: {e}")
            return self._simple_alignment_to_word_dicts(None, original_transcript, chunk_start_time, speaker)

        # Replace MFA word text with Granite's original words
        return self._replace_words_with_granite_text(word_dicts, original_transcript, chunk_start_time, chunk_end_time, speaker)

    def _normalize_word_for_matching(self, word: str) -> str:
        """Normalize a word for comparison during alignment.
        
        Args:
            word: The word to normalize
            
        Returns:
            Lowercase word with only alphanumeric characters
        """
        return ''.join(c.lower() for c in word if c.isalnum())

    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words for alignment purposes.
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        norm1 = self._normalize_word_for_matching(word1)
        norm2 = self._normalize_word_for_matching(word2)
        
        # Handle empty strings
        if not norm1 or not norm2:
            return 0.0
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Prefix match (handles "we" matching "we'll" or "well")
        if norm1.startswith(norm2) or norm2.startswith(norm1):
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return 0.7 + (0.2 * shorter / longer)
        
        # Character-level Levenshtein ratio
        # Simple implementation for small words
        len1, len2 = len(norm1), len(norm2)
        if abs(len1 - len2) > max(len1, len2) // 2:
            return 0.0
        
        # Compute Levenshtein distance
        if len1 < len2:
            norm1, norm2 = norm2, norm1
            len1, len2 = len2, len1
        
        prev_row = list(range(len2 + 1))
        for i, c1 in enumerate(norm1):
            curr_row = [i + 1]
            for j, c2 in enumerate(norm2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (0 if c1 == c2 else 1)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        
        distance = prev_row[-1]
        max_len = max(len1, len2)
        ratio = 1.0 - (distance / max_len)
        
        # Only return meaningful similarity for close matches
        return ratio if ratio >= 0.6 else 0.0

    def _align_word_sequences(self, granite_words: List[str], mfa_words: List[Dict[str, Any]]) -> List[tuple]:
        """Align Granite words with MFA words using dynamic programming.
        
        Uses a similarity-aware alignment algorithm to find the best mapping
        between the two sequences, handling insertions, deletions, and substitutions.
        
        Args:
            granite_words: List of words from Granite transcript
            mfa_words: List of word dicts from MFA alignment
            
        Returns:
            List of tuples (granite_idx, mfa_idx) representing aligned pairs.
            granite_idx is None if MFA has an extra word (to be merged).
            mfa_idx is None if Granite has an extra word (needs interpolation).
        """
        n = len(granite_words)
        m = len(mfa_words)
        
        mfa_texts = [wd["text"] for wd in mfa_words]
        
        # DP table: dp[i][j] = (score, backpointer)
        # Score represents alignment quality (higher is better)
        INF = float('-inf')
        
        # Initialize DP table
        dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
        backptr = [[None for _ in range(m + 1)] for _ in range(n + 1)]
        
        # Gap penalties
        GAP_PENALTY = -0.5
        
        # Initialize first row and column
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] + GAP_PENALTY
            backptr[i][0] = (i-1, 0, 'granite_gap')
        
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] + GAP_PENALTY
            backptr[0][j] = (0, j-1, 'mfa_gap')
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                granite_word = granite_words[i-1]
                mfa_word = mfa_texts[j-1]
                
                sim = self._word_similarity(granite_word, mfa_word)
                
                # Option 1: Match/substitute
                match_score = dp[i-1][j-1] + sim
                
                # Option 2: Gap in MFA (Granite has extra word)
                granite_gap_score = dp[i-1][j] + GAP_PENALTY
                
                # Option 3: Gap in Granite (MFA has extra word - likely a split)
                mfa_gap_score = dp[i][j-1] + GAP_PENALTY
                
                # Choose best option
                if match_score >= granite_gap_score and match_score >= mfa_gap_score:
                    dp[i][j] = match_score
                    backptr[i][j] = (i-1, j-1, 'match')
                elif granite_gap_score >= mfa_gap_score:
                    dp[i][j] = granite_gap_score
                    backptr[i][j] = (i-1, j, 'granite_gap')
                else:
                    dp[i][j] = mfa_gap_score
                    backptr[i][j] = (i, j-1, 'mfa_gap')
        
        # Traceback to find alignment
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i == 0:
                # Remaining MFA words have no Granite match
                alignment.append((None, j-1))
                j -= 1
            elif j == 0:
                # Remaining Granite words have no MFA match
                alignment.append((i-1, None))
                i -= 1
            else:
                _, _, move = backptr[i][j]
                if move == 'match':
                    alignment.append((i-1, j-1))
                    i -= 1
                    j -= 1
                elif move == 'granite_gap':
                    alignment.append((i-1, None))
                    i -= 1
                else:  # mfa_gap
                    alignment.append((None, j-1))
                    j -= 1
        
        alignment.reverse()
        return alignment

    def _interpolate_timestamps(self, prev_end: float, next_start: float, num_words: int, 
                                 words: List[str], speaker: Optional[str]) -> List[Dict[str, Any]]:
        """Create word dicts with interpolated timestamps for words that MFA dropped.
        
        Args:
            prev_end: End time of the previous aligned word
            next_start: Start time of the next aligned word
            num_words: Number of words to interpolate
            words: The actual word texts
            speaker: Speaker identifier
            
        Returns:
            List of word dicts with interpolated timestamps
        """
        if num_words == 0:
            return []
        
        duration = next_start - prev_end
        word_duration = duration / num_words
        
        result = []
        current_time = prev_end
        
        for word in words:
            result.append({
                "text": word,
                "start": current_time,
                "end": current_time + word_duration,
                "speaker": speaker
            })
            current_time += word_duration
        
        return result

    def _merge_mfa_timestamps(self, mfa_words: List[Dict[str, Any]], start_idx: int, end_idx: int) -> tuple:
        """Get merged start/end times from a range of MFA words.
        
        Args:
            mfa_words: List of MFA word dicts
            start_idx: First index to include
            end_idx: Last index to include (exclusive)
            
        Returns:
            Tuple of (start_time, end_time)
        """
        if start_idx >= end_idx or start_idx >= len(mfa_words):
            return (0.0, 0.0)
        
        start_time = mfa_words[start_idx]["start"]
        end_time = mfa_words[min(end_idx - 1, len(mfa_words) - 1)]["end"]
        
        return (start_time, end_time)

    def _replace_words_with_granite_text(self, word_dicts: List[Dict[str, Any]], original_transcript: str, 
                                          chunk_start_time: float, chunk_end_time: float, 
                                          speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Replace MFA word text with Granite's original words, using MFA timestamps.
        
        This function handles three cases:
        1. Word counts match: Direct positional replacement
        2. MFA has fewer words: Use alignment + interpolation for gaps
        3. MFA has more words: Use alignment + timestamp merging for splits
        
        Args:
            word_dicts: List of word dictionaries from MFA alignment
            original_transcript: Original transcript from Granite
            chunk_start_time: Start time of the chunk (for gap interpolation at start)
            chunk_end_time: End time of the chunk (for gap interpolation at end)
            speaker: Speaker identifier
            
        Returns:
            New list of word dicts with Granite text and MFA/interpolated timestamps
        """
        log_progress("Replacing MFA words with Granite text")
        
        granite_words = original_transcript.split()
        mfa_count = len(word_dicts)
        granite_count = len(granite_words)
        
        log_debug(f"Granite word count: {granite_count}, MFA word count: {mfa_count}")
        
        # Fast path: word counts match - direct positional replacement
        if mfa_count == granite_count:
            log_debug("Word counts match - using direct positional replacement")
            result = []
            for i, word_dict in enumerate(word_dicts):
                result.append({
                    "text": granite_words[i],
                    "start": word_dict["start"],
                    "end": word_dict["end"],
                    "speaker": word_dict.get("speaker", speaker)
                })
            return result
        
        # Counts differ - need alignment
        log_debug(f"Word counts differ ({granite_count} vs {mfa_count}) - using sequence alignment")
        alignment = self._align_word_sequences(granite_words, word_dicts)
        
        log_debug(f"Alignment computed with {len(alignment)} entries")
        
        # Process alignment to build result
        result = []
        granite_idx = 0
        pending_granite_words = []  # Words waiting for timestamps
        last_end_time = chunk_start_time
        
        # Group alignment by Granite words
        i = 0
        while i < len(alignment):
            g_idx, m_idx = alignment[i]
            
            if g_idx is not None and m_idx is not None:
                # Matched pair - first handle any pending words
                if pending_granite_words:
                    current_start = word_dicts[m_idx]["start"]
                    interpolated = self._interpolate_timestamps(
                        last_end_time, current_start, 
                        len(pending_granite_words), pending_granite_words, speaker
                    )
                    result.extend(interpolated)
                    pending_granite_words = []
                
                # Check if next alignments are MFA gaps that should merge into this Granite word
                mfa_start_idx = m_idx
                mfa_end_idx = m_idx + 1
                
                j = i + 1
                while j < len(alignment):
                    next_g, next_m = alignment[j]
                    if next_g is None and next_m is not None:
                        # MFA extra word - merge it
                        mfa_end_idx = next_m + 1
                        j += 1
                    else:
                        break
                
                # Get merged timestamps
                start_time, end_time = self._merge_mfa_timestamps(word_dicts, mfa_start_idx, mfa_end_idx)
                
                result.append({
                    "text": granite_words[g_idx],
                    "start": start_time,
                    "end": end_time,
                    "speaker": word_dicts[m_idx].get("speaker", speaker)
                })
                
                last_end_time = end_time
                i = j  # Skip merged MFA words
                
            elif g_idx is not None and m_idx is None:
                # Granite word with no MFA match - queue for interpolation
                pending_granite_words.append(granite_words[g_idx])
                i += 1
                
            elif g_idx is None and m_idx is not None:
                # MFA extra word with no Granite match at this position
                # This shouldn't happen if alignment is correct, but handle gracefully
                log_debug(f"Unexpected MFA extra word at position {m_idx}: '{word_dicts[m_idx]['text']}'")
                i += 1
                
            else:
                # Both None - shouldn't happen
                i += 1
        
        # Handle any remaining pending words at the end
        if pending_granite_words:
            interpolated = self._interpolate_timestamps(
                last_end_time, chunk_end_time,
                len(pending_granite_words), pending_granite_words, speaker
            )
            result.extend(interpolated)
        
        # Validation
        if len(result) != granite_count:
            self.logger.warning(
                f"Word count mismatch after alignment: expected {granite_count}, got {len(result)}. "
                "Falling back to simple distribution."
            )
            # Ultimate fallback - simple distribution with Granite words
            return self._simple_alignment_to_word_dicts(None, original_transcript, chunk_start_time, speaker)
        
        # Verify no <unk> tokens remain
        unk_count = sum(1 for wd in result if wd["text"] == "<unk>")
        if unk_count > 0:
            self.logger.warning(f"Found {unk_count} remaining <unk> tokens after replacement")
        
        # Verify timestamps are monotonic
        for i in range(1, len(result)):
            if result[i]["start"] < result[i-1]["end"]:
                log_debug(f"Non-monotonic timestamps at position {i}: {result[i-1]['end']:.3f} > {result[i]['start']:.3f}")
        
        log_debug(f"Replacement complete: {len(result)} words")
        return result

    def _simple_alignment_to_word_dicts(self, chunk_wav, transcript: str, chunk_start_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback: simple even distribution of timestamps.
        
        Args:
            chunk_wav: Audio data for this chunk (or None)
            transcript: Transcript text
            chunk_start_time: Absolute start time of this chunk in the full audio (seconds)
            speaker: Speaker identifier
        """
        words = transcript.split()
        if not words:
            return []

        # Calculate duration
        if chunk_wav is not None:
            duration = len(chunk_wav) / 16000.0
        else:
            duration = len(words) * 0.5  # Estimate

        word_duration = duration / len(words)
        
        word_dicts = []
        current_time = chunk_start_time

        for word in words:
            word_dicts.append({
                "text": word,
                "start": current_time,
                "end": current_time + word_duration,
                "speaker": speaker
            })
            current_time += word_duration

        return word_dicts

    def _save_debug_chunk(self, debug_dir: pathlib.Path, chunk_num: int, stage: str, data: Dict[str, Any]) -> None:
        """Save debug files for a processing stage.
        
        Args:
            debug_dir: Directory to save debug files
            chunk_num: Chunk number (1-indexed)
            stage: Processing stage name (granite_output, mfa_input, mfa_output)
            data: Data to save
        """
        chunk_num_str = f"{chunk_num:03d}"
        
        # Save JSON file
        json_path = debug_dir / f"chunk_{chunk_num_str}_{stage}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Save human-readable text file
        txt_path = debug_dir / f"chunk_{chunk_num_str}_{stage}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"CHUNK {chunk_num} - {stage.upper().replace('_', ' ')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Chunk start time: {data.get('chunk_start_time', 0):.2f}s\n")
            
            if 'text' in data:
                f.write(f"Word count: {data.get('word_count', len(data['text'].split()))}\n")
                f.write("-" * 60 + "\n\n")
                f.write(data['text'])
                f.write("\n")
            elif 'normalized_text' in data:
                f.write(f"Original word count: {data.get('original_word_count', 0)}\n")
                f.write(f"Normalized word count: {data.get('normalized_word_count', 0)}\n")
                f.write("-" * 60 + "\n")
                f.write("ORIGINAL TEXT:\n")
                f.write(data['original_text'])
                f.write("\n\n")
                f.write("NORMALIZED TEXT (sent to MFA):\n")
                f.write(data['normalized_text'])
                f.write("\n")
            elif 'words' in data:
                f.write(f"Word count: {data.get('word_count', len(data['words']))}\n")
                if data['words']:
                    first_word = data['words'][0]
                    last_word = data['words'][-1]
                    f.write(f"Time range: {first_word.get('start', 0):.2f}s - {last_word.get('end', 0):.2f}s\n")
                f.write("-" * 60 + "\n\n")
                # Write words with timestamps
                for word in data['words']:
                    f.write(f"[{word.get('start', 0):.2f}-{word.get('end', 0):.2f}] {word.get('text', '')}\n")

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs):
        """Not implemented - this provider requires alignment. Use transcribe_with_alignment()."""
        raise NotImplementedError(
            "granite_mfa is a combined transcriber+aligner. Use transcribe_with_alignment() instead."
        )

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio with alignment, returning chunked data with timestamped words.
        
        Returns:
            List of chunk dicts, each with:
            {
                "chunk_id": int,
                "words": List[Dict[str, Any]]  # Each word has "text", "start", "end"
            }
        """
        log_progress("Starting transcription with alignment using Granite + MFA")
        
        # Check if DEBUG logging is enabled and setup debug directory
        from local_transcribe.lib.program_logger import get_output_context
        debug_enabled = get_output_context().should_log("DEBUG")
        debug_dir = None
        intermediate_dir = kwargs.get('intermediate_dir')
        if debug_enabled and intermediate_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = pathlib.Path(intermediate_dir) / "transcription_alignment" / "granite_mfa_debug" / timestamp
            debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Debug mode enabled - saving debug files to {debug_dir}")
        else:
            log_debug(f"Debug not enabled: debug_enabled={debug_enabled}, intermediate_dir={intermediate_dir}")
        
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')
        if transcriber_model not in self.model_mapping:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        self._load_granite_model()

        # Setup MFA
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_mfa_models()

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(wav) / sr
        
        if duration < self.chunk_length_seconds:
            raise ValueError(
                f"Audio duration ({duration:.1f}s) is shorter than minimum chunk length ({self.chunk_length_seconds}s)"
            )
        
        verbose = kwargs.get('verbose', False)
        
        # Calculate chunks
        chunk_samples = int(self.chunk_length_seconds * sr)
        overlap_samples = int(self.overlap_seconds * sr)
        min_chunk_samples = int(self.min_chunk_seconds * sr)

        chunks_with_timestamps = []
        total_samples = len(wav)
        effective_chunk_length = self.chunk_length_seconds - self.overlap_seconds
        num_chunks = math.ceil(duration / effective_chunk_length) if effective_chunk_length > 0 else 1
        
        if verbose:
            log_progress(f"Audio duration: {duration:.1f}s - processing in {num_chunks} chunks")
        
        chunk_start = 0
        chunk_num = 0
        prev_chunk_wav = None
        
        while chunk_start < total_samples:
            chunk_num += 1
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            chunk_wav = wav[chunk_start:chunk_end]
            
            # Calculate absolute start time of this chunk in the full audio
            chunk_start_time = chunk_start / sr
            
            log_progress(f"Processing chunk {chunk_num} of {num_chunks} (starts at {chunk_start_time:.2f}s)")
            
            if verbose:
                chunk_duration_sec = len(chunk_wav) / sr
                log_progress(f"Processing chunk {chunk_num} of {num_chunks} ({chunk_duration_sec:.1f}s)...")
            
            if len(chunk_wav) < min_chunk_samples:
                if prev_chunk_wav is not None:
                    # Merge with previous chunk
                    non_overlapping_part = chunk_wav[overlap_samples:]
                    merged_tensor = torch.cat([torch.from_numpy(prev_chunk_wav), torch.from_numpy(non_overlapping_part)])
                    merged_wav = merged_tensor.numpy()
                    
                    # Use the start time from the previous chunk (which is being extended)
                    prev_chunk_start_time = chunks_with_timestamps[-1].get("chunk_start_time", chunk_start_time - (len(prev_chunk_wav) / sr))
                    prev_chunk_id = chunks_with_timestamps[-1]["chunk_id"]
                    
                    # Transcribe merged chunk
                    chunk_text = self._transcribe_single_chunk(merged_wav, **kwargs)
                    
                    # Save Granite output for debug (merged chunk)
                    if debug_dir:
                        self._save_debug_chunk(debug_dir, prev_chunk_id, "granite_output_merged", {
                            "chunk_id": prev_chunk_id,
                            "chunk_start_time": prev_chunk_start_time,
                            "text": chunk_text,
                            "word_count": len(chunk_text.split()),
                            "note": f"Merged with short final chunk {chunk_num}"
                        })
                    
                    # Save normalized transcript for MFA input (merged chunk)
                    normalized_for_mfa = ' '.join(
                        ''.join(c for c in word if c.isalnum() or c == "'")
                        for word in chunk_text.split()
                    )
                    if debug_dir:
                        self._save_debug_chunk(debug_dir, prev_chunk_id, "mfa_input_merged", {
                            "chunk_id": prev_chunk_id,
                            "chunk_start_time": prev_chunk_start_time,
                            "original_text": chunk_text,
                            "normalized_text": normalized_for_mfa,
                            "original_word_count": len(chunk_text.split()),
                            "normalized_word_count": len(normalized_for_mfa.split()),
                            "note": f"Merged with short final chunk {chunk_num}"
                        })
                    
                    # Align merged chunk with absolute timestamps
                    timestamped_words = self._align_chunk_with_mfa(merged_wav, chunk_text, prev_chunk_start_time, role)
                    
                    # Save MFA output for debug (merged chunk)
                    if debug_dir:
                        self._save_debug_chunk(debug_dir, prev_chunk_id, "mfa_output_merged", {
                            "chunk_id": prev_chunk_id,
                            "chunk_start_time": prev_chunk_start_time,
                            "word_count": len(timestamped_words),
                            "words": timestamped_words,
                            "note": f"Merged with short final chunk {chunk_num}"
                        })
                    
                    # Update last chunk
                    chunks_with_timestamps[-1] = {
                        "chunk_id": prev_chunk_id,
                        "chunk_start_time": prev_chunk_start_time,
                        "words": timestamped_words
                    }
            else:
                # Normal chunk processing
                chunk_text = self._transcribe_single_chunk(chunk_wav, **kwargs)
                
                # Save Granite output for debug
                if debug_dir:
                    self._save_debug_chunk(debug_dir, chunk_num, "granite_output", {
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split())
                    })
                
                # Save normalized transcript for MFA input
                normalized_for_mfa = ' '.join(
                    ''.join(c for c in word if c.isalnum() or c == "'")
                    for word in chunk_text.split()
                )
                if debug_dir:
                    self._save_debug_chunk(debug_dir, chunk_num, "mfa_input", {
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "original_text": chunk_text,
                        "normalized_text": normalized_for_mfa,
                        "original_word_count": len(chunk_text.split()),
                        "normalized_word_count": len(normalized_for_mfa.split())
                    })
                
                # Align this chunk with absolute timestamps
                timestamped_words = self._align_chunk_with_mfa(chunk_wav, chunk_text, chunk_start_time, role)
                
                # Save MFA output for debug
                if debug_dir:
                    self._save_debug_chunk(debug_dir, chunk_num, "mfa_output", {
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "word_count": len(timestamped_words),
                        "words": timestamped_words
                    })
                
                chunks_with_timestamps.append({
                    "chunk_id": chunk_num,
                    "chunk_start_time": chunk_start_time,
                    "words": timestamped_words
                })
            
            prev_chunk_wav = chunk_wav

            if chunk_end == total_samples:
                break
            
            chunk_start = chunk_start + chunk_samples - overlap_samples
        
        if verbose:
            total_words = sum(len(chunk["words"]) for chunk in chunks_with_timestamps)
            log_completion(f"Transcription and alignment complete: {len(chunks_with_timestamps)} chunks, {total_words} words")
        
        return chunks_with_timestamps

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteMFATranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
