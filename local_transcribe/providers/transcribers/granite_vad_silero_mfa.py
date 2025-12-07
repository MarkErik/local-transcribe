#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with Silero VAD-based segmentation and MFA alignment.

This plugin combines:
- Silero VAD for intelligent audio segmentation at speech boundaries
- Granite's transcription capabilities
- Montreal Forced Aligner for precise word-level timestamps

The integrated stitcher produces continuous WordSegments output.
Debug mode saves individual segment transcripts when DEBUG logging is enabled.
"""

from typing import List, Optional, Dict, Any
import os
import pathlib
import re
import torch
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
from local_transcribe.lib.vad_silero_segmenter import SileroVADSegmenter


class GraniteVADSileroMFATranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + Silero VAD segmentation + MFA alignment."""

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite VAD (Silero) MFA Transcriber Provider")
        
        # Granite configuration
        self.model_mapping = {
            "granite-2b": "ibm-granite/granite-speech-3.3-2b",
            "granite-8b": "ibm-granite/granite-speech-3.3-8b"
        }
        self.selected_model = None
        self.processor = None
        self.model = None
        self.tokenizer = None
        
        # Segmenter instance
        self.vad_segmenter = None
        self.models_dir = None
        
        # MFA configuration
        self.mfa_models_dir = None

    @property
    def device(self):
        return get_system_capability()

    @property
    def name(self) -> str:
        return "granite_vad_silero_mfa"

    @property
    def short_name(self) -> str:
        return "Granite + VAD (Silero) + MFA"

    @property
    def description(self) -> str:
        return "IBM Granite transcription with Silero VAD-based segmentation and MFA alignment (produces continuous timestamped output)"

    @property
    def has_builtin_alignment(self) -> bool:
        """This provider combines transcription and alignment."""
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return required Granite models (MFA and Silero VAD models handled separately)."""
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
        """Check which models are available offline."""
        missing_models = []
        
        # Check Granite models
        granite_missing = self._check_granite_models_offline(models, models_dir)
        missing_models.extend(granite_missing)
        
        # Note: Silero VAD doesn't require pre-download check as it's auto-downloaded via torch.hub
        # and is very lightweight (~2MB)
        
        return missing_models
    
    def _check_granite_models_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
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

    def _init_vad_segmenter(self):
        """Initialize the Silero VAD segmenter if not already done."""
        if self.vad_segmenter is None:
            self.vad_segmenter = SileroVADSegmenter(
                device=self.device if self.device != "mps" else "cpu",  # Silero works best on CPU for MPS
                models_dir=self.models_dir
            )

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

    def _transcribe_single_segment(self, wav, sample_rate=16000, **kwargs) -> str:
        """Transcribe a single audio segment using Granite."""
        log_progress("Transcribing audio segment with Granite")
        try:
            wav_tensor = torch.from_numpy(wav).unsqueeze(0)
            
            # Calculate segment duration from wav array using provided sample rate
            segment_duration = len(wav) / sample_rate

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

            # Adjust parameters based on segment duration
            if segment_duration < 8.0:
                # For segments less than 8 seconds: reduce max_new_tokens and exclude logits_processor
                max_new_tokens = 192
                logits_processor = None
            else:
                # For segments 8 seconds or longer: use current settings
                max_new_tokens = 256
                repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                    penalty=3.0,
                    prompt_ignore_length=model_inputs["input_ids"].shape[-1],
                )
                logits_processor = [repetition_penalty_processor]

            with torch.no_grad():
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    early_stopping=True,
                    logits_processor=logits_processor,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )

            cleaned_text = self._clean_transcription_output(output_text[0].strip())

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

    def _clean_transcription_output(self, text: str) -> str:
        """Clean the transcription output by removing dialogue markers and quotation marks."""
        # Count labels before removal for debug logging
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
        
        # Log count if any labels were removed
        if total_removed > 0:
            log_debug(f"Removed {total_removed} labels from segment transcript.")
        
        return text

    def _align_segment_with_mfa(self, segment_wav, segment_transcript: str, segment_start_time: float = 0.0, speaker: Optional[str] = None, debug_dir: Optional[pathlib.Path] = None, segment_num: Optional[int] = None) -> List[Dict[str, Any]]:
        """Align a single segment using MFA and return timestamped words.
        
        Args:
            segment_wav: Audio data for this segment
            segment_transcript: Transcript text for this segment
            segment_start_time: Absolute start time of this segment in the full audio (seconds)
            speaker: Speaker identifier
        """
        log_progress(f"Aligning transcript with MFA (segment starts at {segment_start_time:.2f}s)")
        
        segment_duration = len(segment_wav) / 16000.0
        segment_end_time = segment_start_time + segment_duration
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            audio_dir = temp_path / "audio"
            audio_dir.mkdir()

            # Save segment audio as WAV
            audio_file = audio_dir / "segment.wav"
            import soundfile as sf
            sf.write(str(audio_file), segment_wav, 16000)

            # Normalize transcript for MFA
            normalized_transcript = ' '.join(
                ''.join(c for c in word if c.isalnum() or c == "'")
                for word in segment_transcript.split()
            )
            
            transcript_file = audio_dir / "segment.lab"
            transcript_file.write_text(normalized_transcript, encoding='utf-8')

            output_dir = temp_path / "output"
            output_dir.mkdir()

            env = os.environ.copy()
            env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)
            env["MFA_NO_HISTORY"] = "1"
            
            mfa_env_bin = pathlib.Path(self._get_mfa_command()).parent
            env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")

            textgrid_file = output_dir / "segment.TextGrid"

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
                    word_dicts = self._parse_textgrid_to_word_dicts(textgrid_file, segment_transcript, segment_start_time, segment_end_time, speaker)
                    
                    # Save TextGrid content for debugging if debug mode is enabled
                    if debug_dir and segment_num is not None:
                        textgrid_content = textgrid_file.read_text(encoding='utf-8')
                        textgrid_debug_path = debug_dir / f"segment_{segment_num:03d}_textgrid.TextGrid"
                        textgrid_debug_path.write_text(textgrid_content, encoding='utf-8')
                        log_debug(f"Saved TextGrid for segment {segment_num} to {textgrid_debug_path}")
                    
                    return word_dicts
                else:
                    self.logger.warning("MFA alignment completed but no TextGrid file was produced. Falling back to simple alignment.")
                    return self._simple_alignment_to_word_dicts(segment_wav, segment_transcript, segment_start_time, speaker)

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.logger.warning(f"MFA alignment failed: {e}")
                return self._simple_alignment_to_word_dicts(segment_wav, segment_transcript, segment_start_time, speaker)

    def _parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str, segment_start_time: float = 0.0, segment_end_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse MFA TextGrid and return list of word dicts with timestamps."""
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
                            
                            word_dicts.append({
                                "text": original_text,
                                "start": start + segment_start_time,
                                "end": end + segment_start_time,
                                "speaker": speaker
                            })
                    except (ValueError, IndexError):
                        pass

                i += 1

        except Exception as e:
            self.logger.warning(f"Failed to parse TextGrid: {e}")
            return self._simple_alignment_to_word_dicts(None, original_transcript, segment_start_time, speaker)

        return self._replace_words_with_granite_text(word_dicts, original_transcript, segment_start_time, segment_end_time, speaker)

    def _normalize_word_for_matching(self, word: str) -> str:
        """Normalize a word for comparison during alignment."""
        return ''.join(c.lower() for c in word if c.isalnum())

    def _word_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words for alignment purposes."""
        norm1 = self._normalize_word_for_matching(word1)
        norm2 = self._normalize_word_for_matching(word2)
        
        if not norm1 or not norm2:
            return 0.0
        
        if norm1 == norm2:
            return 1.0
        
        if norm1.startswith(norm2) or norm2.startswith(norm1):
            shorter = min(len(norm1), len(norm2))
            longer = max(len(norm1), len(norm2))
            return 0.7 + (0.2 * shorter / longer)
        
        len1, len2 = len(norm1), len(norm2)
        if abs(len1 - len2) > max(len1, len2) // 2:
            return 0.0
        
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
        
        return ratio if ratio >= 0.6 else 0.0

    def _align_word_sequences(self, granite_words: List[str], mfa_words: List[Dict[str, Any]]) -> List[tuple]:
        """Align Granite words with MFA words using dynamic programming."""
        n = len(granite_words)
        m = len(mfa_words)
        
        mfa_texts = [wd["text"] for wd in mfa_words]
        
        dp = [[0.0 for _ in range(m + 1)] for _ in range(n + 1)]
        backptr = [[None for _ in range(m + 1)] for _ in range(n + 1)]
        
        GAP_PENALTY = -0.5
        
        for i in range(1, n + 1):
            dp[i][0] = dp[i-1][0] + GAP_PENALTY
            backptr[i][0] = (i-1, 0, 'granite_gap')
        
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j-1] + GAP_PENALTY
            backptr[0][j] = (0, j-1, 'mfa_gap')
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                granite_word = granite_words[i-1]
                mfa_word = mfa_texts[j-1]
                
                sim = self._word_similarity(granite_word, mfa_word)
                
                match_score = dp[i-1][j-1] + sim
                granite_gap_score = dp[i-1][j] + GAP_PENALTY
                mfa_gap_score = dp[i][j-1] + GAP_PENALTY
                
                if match_score >= granite_gap_score and match_score >= mfa_gap_score:
                    dp[i][j] = match_score
                    backptr[i][j] = (i-1, j-1, 'match')
                elif granite_gap_score >= mfa_gap_score:
                    dp[i][j] = granite_gap_score
                    backptr[i][j] = (i-1, j, 'granite_gap')
                else:
                    dp[i][j] = mfa_gap_score
                    backptr[i][j] = (i, j-1, 'mfa_gap')
        
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i == 0:
                alignment.append((None, j-1))
                j -= 1
            elif j == 0:
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
                else:
                    alignment.append((None, j-1))
                    j -= 1
        
        alignment.reverse()
        return alignment

    def _interpolate_timestamps(self, prev_end: float, next_start: float, num_words: int, 
                                 words: List[str], speaker: Optional[str]) -> List[Dict[str, Any]]:
        """Create word dicts with interpolated timestamps for words that MFA dropped."""
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
        """Get merged start/end times from a range of MFA words."""
        if start_idx >= end_idx or start_idx >= len(mfa_words):
            return (0.0, 0.0)
        
        start_time = mfa_words[start_idx]["start"]
        end_time = mfa_words[min(end_idx - 1, len(mfa_words) - 1)]["end"]
        
        return (start_time, end_time)

    def _replace_words_with_granite_text(self, word_dicts: List[Dict[str, Any]], original_transcript: str, 
                                          segment_start_time: float, segment_end_time: float, 
                                          speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Replace MFA word text with Granite's original words, using MFA timestamps."""
        log_progress("Replacing MFA words with Granite text")
        
        granite_words = original_transcript.split()
        mfa_count = len(word_dicts)
        granite_count = len(granite_words)
        
        log_debug(f"Granite word count: {granite_count}, MFA word count: {mfa_count}")
        
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
        
        log_debug(f"Word counts differ ({granite_count} vs {mfa_count}) - using sequence alignment")
        alignment = self._align_word_sequences(granite_words, word_dicts)
        
        log_debug(f"Alignment computed with {len(alignment)} entries")
        
        result = []
        pending_granite_words = []
        last_end_time = segment_start_time
        
        i = 0
        while i < len(alignment):
            g_idx, m_idx = alignment[i]
            
            if g_idx is not None and m_idx is not None:
                if pending_granite_words:
                    self.logger.warning(f"MFA dropped {len(pending_granite_words)} word(s): {pending_granite_words}. Using interpolated timestamps.")
                    current_start = word_dicts[m_idx]["start"]
                    interpolated = self._interpolate_timestamps(
                        last_end_time, current_start, 
                        len(pending_granite_words), pending_granite_words, speaker
                    )
                    result.extend(interpolated)
                    pending_granite_words = []
                
                mfa_start_idx = m_idx
                mfa_end_idx = m_idx + 1
                
                j = i + 1
                while j < len(alignment):
                    next_g, next_m = alignment[j]
                    if next_g is None and next_m is not None:
                        mfa_end_idx = next_m + 1
                        j += 1
                    else:
                        break
                
                start_time, end_time = self._merge_mfa_timestamps(word_dicts, mfa_start_idx, mfa_end_idx)
                
                result.append({
                    "text": granite_words[g_idx],
                    "start": start_time,
                    "end": end_time,
                    "speaker": word_dicts[m_idx].get("speaker", speaker)
                })
                
                last_end_time = end_time
                i = j
                
            elif g_idx is not None and m_idx is None:
                pending_granite_words.append(granite_words[g_idx])
                i += 1
                
            elif g_idx is None and m_idx is not None:
                log_debug(f"Unexpected MFA extra word at position {m_idx}: '{word_dicts[m_idx]['text']}'")
                i += 1
                
            else:
                i += 1
        
        if pending_granite_words:
            self.logger.warning(f"MFA dropped {len(pending_granite_words)} word(s) at end of segment: {pending_granite_words}. Using interpolated timestamps.")
            interpolated = self._interpolate_timestamps(
                last_end_time, segment_end_time,
                len(pending_granite_words), pending_granite_words, speaker
            )
            result.extend(interpolated)
        
        if len(result) != granite_count:
            self.logger.warning(
                f"Word count mismatch after alignment: expected {granite_count}, got {len(result)}. "
                "Falling back to simple distribution."
            )
            return self._simple_alignment_to_word_dicts(None, original_transcript, segment_start_time, speaker)
        
        log_debug(f"Replacement complete: {len(result)} words")
        return result

    def _simple_alignment_to_word_dicts(self, segment_wav, transcript: str, segment_start_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback: simple even distribution of timestamps."""
        words = transcript.split()
        if not words:
            return []

        if segment_wav is not None:
            duration = len(segment_wav) / 16000.0
        else:
            duration = len(words) * 0.5

        word_duration = duration / len(words)
        
        word_dicts = []
        current_time = segment_start_time

        for word in words:
            word_dicts.append({
                "text": word,
                "start": current_time,
                "end": current_time + word_duration,
                "speaker": speaker
            })
            current_time += word_duration

        return word_dicts

    # =========================================================================
    # VAD-specific stitching (simple concatenation with gap handling)
    # =========================================================================
    
    def _stitch_vad_segments(self, all_segment_words: List[List[Dict[str, Any]]]) -> List[WordSegment]:
        """
        Stitch VAD segment words into continuous output.
        
        Since VAD segments are non-overlapping and separated by silence,
        this is primarily simple concatenation with timestamp validation.
        
        Args:
            all_segment_words: List of word dict lists, one per VAD segment
            
        Returns:
            List of WordSegment objects in chronological order
        """
        log_progress("Stitching VAD segments into continuous output...")
        
        result: List[WordSegment] = []
        last_end_time = 0.0
        
        for segment_idx, segment_words in enumerate(all_segment_words):
            if not segment_words:
                continue
            
            for word_dict in segment_words:
                word_start = word_dict["start"]
                word_end = word_dict["end"]
                
                # Validate timestamp monotonicity
                if word_start < last_end_time:
                    # Small overlap possible at segment boundaries due to VAD imprecision
                    # Adjust start time to maintain monotonicity
                    gap = last_end_time - word_start
                    if gap < 0.1:  # Small gap, just adjust
                        word_start = last_end_time
                        if word_end < word_start:
                            word_end = word_start + 0.01
                    else:
                        log_debug(f"Larger timestamp overlap detected: {gap:.3f}s at segment {segment_idx}")
                        word_start = last_end_time
                        if word_end < word_start:
                            word_end = word_start + 0.01
                
                result.append(WordSegment(
                    text=word_dict["text"],
                    start=word_start,
                    end=word_end,
                    speaker=word_dict.get("speaker")
                ))
                
                last_end_time = word_end
        
        log_completion(f"Stitching complete: {len(result)} words")
        return result

    # =========================================================================
    # Debug helpers
    # =========================================================================
    
    def _save_debug_segment(self, debug_dir: pathlib.Path, segment_num: int, stage: str, data: Dict[str, Any]) -> None:
        """Save debug files for a processing stage."""
        segment_num_str = f"{segment_num:03d}"
        
        json_path = debug_dir / f"segment_{segment_num_str}_{stage}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        txt_path = debug_dir / f"segment_{segment_num_str}_{stage}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"SEGMENT {segment_num} - {stage.upper().replace('_', ' ')}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Segment start time: {data.get('segment_start_time', 0):.2f}s\n")
            f.write(f"Segment end time: {data.get('segment_end_time', 0):.2f}s\n")
            
            if 'text' in data:
                f.write(f"Word count: {data.get('word_count', len(data['text'].split()))}\n")
                f.write("-" * 60 + "\n\n")
                f.write(data['text'])
                f.write("\n")
            elif 'words' in data:
                f.write(f"Word count: {data.get('word_count', len(data['words']))}\n")
                if data['words']:
                    first_word = data['words'][0]
                    last_word = data['words'][-1]
                    f.write(f"Time range: {first_word.get('start', 0):.2f}s - {last_word.get('end', 0):.2f}s\n")
                f.write("-" * 60 + "\n\n")
                for word in data['words']:
                    f.write(f"[{word.get('start', 0):.2f}-{word.get('end', 0):.2f}] {word.get('text', '')}\n")

    # =========================================================================
    # Main entry points
    # =========================================================================
    
    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs):
        """Not implemented - this provider requires alignment. Use transcribe_with_alignment()."""
        raise NotImplementedError(
            "granite_vad_silero_mfa is a combined transcriber+aligner. Use transcribe_with_alignment() instead."
        )

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio with Silero VAD-based segmentation and MFA alignment.
        
        Returns:
            List[WordSegment] - Continuous list of timestamped words
        """
        log_progress("Starting transcription with Silero VAD segmentation + Granite + MFA")
        
        # Extract models_dir from kwargs if provided
        models_dir = kwargs.get('models_dir')
        if models_dir:
            self.models_dir = pathlib.Path(models_dir)
        
        # Check if DEBUG logging is enabled and setup debug directory
        from local_transcribe.lib.program_logger import get_output_context
        debug_enabled = get_output_context().should_log("DEBUG")
        debug_dir = None
        intermediate_dir = kwargs.get('intermediate_dir')
        if debug_enabled and intermediate_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = pathlib.Path(intermediate_dir) / "transcription_alignment" / "granite_vad_silero_mfa_debug" / timestamp
            debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Debug mode enabled - saving debug files to {debug_dir}")
        
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')
        if transcriber_model not in self.model_mapping:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        self._load_granite_model()

        # Initialize Silero VAD segmenter
        self._init_vad_segmenter()

        # Setup MFA
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / ".models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_mfa_models()

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(wav) / sr
        
        log_progress(f"Audio duration: {duration:.1f}s")
        
        # Step 1: Silero VAD segmentation
        log_progress("Running Silero VAD segmentation...")
        debug_file_path = debug_dir / "vad_segmentation_debug.txt" if debug_dir else None
        vad_segments = self.vad_segmenter.segment_audio(wav, sr, debug_file_path=str(debug_file_path) if debug_file_path else None)
        
        log_progress(f"Silero VAD produced {len(vad_segments)} segments")
        
        # Save VAD segments info for debug
        if debug_dir:
            vad_info = {
                "vad_engine": "silero",
                "vad_params": {
                    "threshold": self.vad_segmenter.threshold,
                    "min_speech_duration_ms": self.vad_segmenter.min_speech_duration_ms,
                    "min_silence_duration_ms": self.vad_segmenter.min_silence_duration_ms,
                    "speech_pad_ms": self.vad_segmenter.speech_pad_ms,
                    "max_segment_duration": self.vad_segmenter.max_segment_duration,
                },
                "total_segments": len(vad_segments),
                "audio_duration": duration,
                "segments": [
                    {"start": start, "end": end, "duration": end - start}
                    for start, end in vad_segments
                ]
            }
            with open(debug_dir / "vad_segments.json", 'w') as f:
                json.dump(vad_info, f, indent=2)
        
        # Step 2: Process each VAD segment
        all_segment_words: List[List[Dict[str, Any]]] = []
        
        for seg_idx, (seg_start, seg_end) in enumerate(vad_segments):
            segment_num = seg_idx + 1
            
            log_progress(f"Processing segment {segment_num}/{len(vad_segments)} ({seg_start:.2f}s - {seg_end:.2f}s)")
            
            # Extract segment audio
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            segment_wav = wav[start_sample:end_sample]
            
            # Transcribe segment with Granite
            segment_text = self._transcribe_single_segment(segment_wav, sample_rate=sr, **kwargs)
            
            if debug_dir:
                self._save_debug_segment(debug_dir, segment_num, "granite_output", {
                    "segment_id": segment_num,
                    "segment_start_time": seg_start,
                    "segment_end_time": seg_end,
                    "text": segment_text,
                    "word_count": len(segment_text.split())
                })
            
            if not segment_text.strip():
                log_debug(f"Segment {segment_num} produced empty transcript, skipping")
                all_segment_words.append([])
                continue
            
            # Align segment with MFA
            timestamped_words = self._align_segment_with_mfa(segment_wav, segment_text, seg_start, role, debug_dir, segment_num)
            
            if debug_dir:
                self._save_debug_segment(debug_dir, segment_num, "mfa_output", {
                    "segment_id": segment_num,
                    "segment_start_time": seg_start,
                    "segment_end_time": seg_end,
                    "word_count": len(timestamped_words),
                    "words": timestamped_words
                })
            
            all_segment_words.append(timestamped_words)
        
        # Step 3: Stitch all segments into continuous output
        result = self._stitch_vad_segments(all_segment_words)
        
        if debug_enabled:
            log_debug(f"Transcription complete: {len(result)} words from {len(vad_segments)} Silero VAD segments")
        
        return result

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.models_dir = models_dir
        
        # Preload Granite models
        granite_models = [m for m in models if m in self.model_mapping.values()]
        if granite_models:
            self.preload_models(granite_models, models_dir)
        
        # Preload Silero VAD model
        log_progress("Preloading Silero VAD segmentation model...")
        try:
            self._init_vad_segmenter()
            self.vad_segmenter.preload_models()
            log_completion("Silero VAD model preloaded successfully")
        except Exception as e:
            log_progress(f"Failed to preload Silero VAD model: {e}")
            raise


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteVADSileroMFATranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
