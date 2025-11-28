#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with Wav2Vec2 alignment.

This plugin combines Granite's transcription capabilities with Wav2Vec2 forced alignment
to produce chunked transcripts where each word has timestamps. Unlike the separate
granite + wav2vec2 pipeline, this processes each chunk with alignment before moving to the next,
resulting in chunks that contain timestamped words ready for intelligent stitching.

If Wav2Vec2 alignment fails, it falls back to MFA alignment if available.
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
import warnings
import numpy as np
import torchaudio
from datetime import datetime
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug


class GraniteWav2Vec2TranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + Wav2Vec2 for chunked transcription with timestamps."""

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite Wav2Vec2 Transcriber Provider")
        
        # Granite configuration
        self.model_mapping = {
            "granite-2b": "ibm-granite/granite-speech-3.3-2b",
            "granite-8b": "ibm-granite/granite-speech-3.3-8b"
        }
        self.selected_model = None
        self.processor = None
        self.model = None
        self.tokenizer = None
        
        # Wav2Vec2 configuration
        self.wav2vec2_model_name = "facebook/wav2vec2-large-960h"
        self.wav2vec2_processor = None
        self.wav2vec2_model = None
        
        # Chunking configuration
        self.chunk_length_seconds = 60.0
        self.overlap_seconds = 4.0
        self.min_chunk_seconds = 7.0
        
        # MFA fallback configuration
        self.mfa_models_dir = None
        self.mfa_available = None  # Lazy check

    @property
    def device(self):
        return get_system_capability()

    @property
    def name(self) -> str:
        return "granite_wav2vec2"

    @property
    def short_name(self) -> str:
        return "Granite + Wav2Vec2"

    @property
    def description(self) -> str:
        return "IBM Granite transcription with Wav2Vec2 alignment (produces chunked timestamped output)"

    @property
    def has_builtin_alignment(self) -> bool:
        """This provider combines transcription and alignment."""
        return True

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return required Granite and Wav2Vec2 models."""
        models = []
        
        # Granite model
        if selected_model and selected_model in self.model_mapping:
            models.append(self.model_mapping[selected_model])
        else:
            models.append(self.model_mapping["granite-8b"])
        
        # Wav2Vec2 model
        models.append(self.wav2vec2_model_name)
        
        return models

    def get_available_models(self) -> List[str]:
        return list(self.model_mapping.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite and Wav2Vec2 models to cache."""
        self.logger.info("Starting model preload for Granite and Wav2Vec2 models")
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
                    # Granite model - use HF_HOME
                    os.environ["HF_HOME"] = str(cache_dir)
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, token=token)
                    log_completion(f"{model} downloaded successfully.")
                elif model == self.wav2vec2_model_name:
                    # Wav2Vec2 model - use standard HF cache
                    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
                    if xdg_cache_home:
                        models_root = pathlib.Path(xdg_cache_home)
                    else:
                        models_root = pathlib.Path.home() / ".cache" / "huggingface"
                    
                    hub_cache_dir = models_root / "huggingface" / "hub"
                    hub_cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    snapshot_download(model, cache_dir=hub_cache_dir, token=os.getenv("HF_TOKEN"))
                    log_completion(f"{model} downloaded successfully.")
        except Exception as e:
            raise Exception(f"Failed to download model: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite and Wav2Vec2 models are available offline."""
        missing_models = []
        
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            models_root = pathlib.Path(xdg_cache_home)
        else:
            models_root = pathlib.Path.home() / ".cache" / "huggingface"
        
        hub_dir = models_root / "huggingface" / "hub"
        
        for model in models:
            if "/" in model:
                hf_model_name = model.replace("/", "--")
                model_dir = hub_dir / f"models--{hf_model_name}"
                
                has_model_files = (
                    model_dir.exists() and (
                        any(model_dir.rglob("*.bin")) or
                        any(model_dir.rglob("*.safetensors")) or
                        any(model_dir.rglob("*.pt")) or
                        any(model_dir.rglob("*.pth"))
                    )
                )
                
                if not has_model_files:
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

    def _load_wav2vec2_model(self):
        """Load the Wav2Vec2 model for alignment if not already loaded."""
        if self.wav2vec2_model is None:
            log_progress("Loading Wav2Vec2 model for alignment...")
            
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                models_root = pathlib.Path.home() / ".cache" / "huggingface"

            try:
                token = os.getenv("HF_TOKEN")
                
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
                    warnings.filterwarnings("ignore", message=".*masked_spec_embed.*")
                    self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                        self.wav2vec2_model_name, local_files_only=True, token=token
                    )
                    self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                        self.wav2vec2_model_name, local_files_only=True, token=token
                    ).to(self.device)
                
                log_completion("Wav2Vec2 model loaded successfully")
            except Exception as e:
                log_debug(f"Failed to load Wav2Vec2 model {self.wav2vec2_model_name}")
                log_debug(f"Cache directory exists: {(models_root / 'huggingface' / 'hub').exists()}")
                raise e

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
                    early_stopping=True,
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

    def _get_token_timestamps(self, emissions: torch.Tensor, transcript: str) -> List[tuple]:
        """Extract token timestamps using CTC alignment with frame-level paths."""
        vocab = self.wav2vec2_processor.tokenizer.get_vocab()
        dictionary = {c: i for i, c in enumerate(vocab.keys())}
        
        transcript_normalized = transcript.upper()
        tokens = []
        for char in transcript_normalized:
            if char == ' ':
                tokens.append('|')
            elif char in dictionary:
                tokens.append(char)
        
        if not tokens:
            return []
        
        token_ids = [dictionary.get(t, dictionary.get('[UNK]', 0)) for t in tokens]
        
        try:
            log_probs = emissions.log_softmax(dim=-1).cpu()
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*forced_align.*deprecated.*")
                aligned_labels, scores = torchaudio.functional.forced_align(
                    log_probs,
                    torch.tensor([token_ids]),
                    blank=self.wav2vec2_processor.tokenizer.pad_token_id
                )
            
            token_timestamps = self._extract_token_boundaries(
                aligned_labels[0],
                tokens,
                token_ids,
                self.wav2vec2_processor.tokenizer.pad_token_id
            )
            
            return token_timestamps
            
        except Exception as e:
            self.logger.warning(f"Forced alignment failed ({e}), using fallback method")
            return self._fallback_token_alignment(emissions, tokens)

    def _extract_token_boundaries(self, aligned_labels: torch.Tensor, tokens: List[str],
                                    token_ids: List[int], blank_id: int) -> List[tuple]:
        """Extract token boundaries from frame-level aligned labels."""
        token_timestamps = []
        aligned_labels_list = aligned_labels.tolist()
        
        current_token_idx = 0
        frame_start = None
        
        for frame_idx, label_id in enumerate(aligned_labels_list):
            if label_id == blank_id:
                if frame_start is not None and current_token_idx < len(token_ids):
                    start_time = frame_start * 0.02 * 1000
                    end_time = frame_idx * 0.02 * 1000
                    token_timestamps.append((tokens[current_token_idx], start_time, end_time))
                    
                    current_token_idx += 1
                    frame_start = None
                continue
            
            if current_token_idx < len(token_ids) and label_id == token_ids[current_token_idx]:
                if frame_start is None:
                    frame_start = frame_idx
            else:
                # Handle case where we're in the middle of a token but encounter a different label
                if frame_start is not None and current_token_idx < len(token_ids):
                    start_time = frame_start * 0.02 * 1000
                    end_time = frame_idx * 0.02 * 1000
                    token_timestamps.append((tokens[current_token_idx], start_time, end_time))
                    
                    current_token_idx += 1
                    frame_start = None
                    
                    # Check if the new label matches the next expected token
                    if current_token_idx < len(token_ids) and label_id == token_ids[current_token_idx]:
                        frame_start = frame_idx
        
        # Handle final token if still tracking
        if frame_start is not None and current_token_idx < len(token_ids):
            start_time = frame_start * 0.02 * 1000
            end_time = len(aligned_labels_list) * 0.02 * 1000
            token_timestamps.append((tokens[current_token_idx], start_time, end_time))
        
        return token_timestamps

    def _fallback_token_alignment(self, emissions: torch.Tensor, tokens: List[str]) -> List[tuple]:
        """Fallback token alignment using simple peak detection."""
        token_timestamps = []
        emissions_np = emissions[0].cpu().numpy()
        
        dictionary = {c: i for i, c in enumerate(self.wav2vec2_processor.tokenizer.get_vocab().keys())}
        
        time_per_token = emissions_np.shape[0] / max(len(tokens), 1)
        
        for i, token in enumerate(tokens):
            token_id = dictionary.get(token, dictionary.get('[UNK]', 0))
            
            expected_time = int(i * time_per_token)
            window_start = max(0, expected_time - int(time_per_token))
            window_end = min(emissions_np.shape[0], expected_time + int(time_per_token * 2))
            
            window_probs = emissions_np[window_start:window_end, token_id]
            if len(window_probs) > 0:
                peak_pos = window_start + np.argmax(window_probs)
                start_time = peak_pos * 0.02 * 1000
                end_time = (peak_pos + 1) * 0.02 * 1000
                token_timestamps.append((token, start_time, end_time))
        
        return token_timestamps

    def _chars_to_word_dicts(self, transcript: str, token_timestamps: List[tuple], 
                             chunk_start_time: float, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert character-level token timestamps to word dicts with absolute timestamps."""
        words = transcript.split()
        if not words:
            return []
        
        if not token_timestamps:
            return self._simple_alignment_to_word_dicts(None, transcript, chunk_start_time, speaker)
        
        word_dicts = []
        token_idx = 0
        
        for word in words:
            word_chars = [c for c in word.upper() if c.isalnum()]
            word_token_times = []
            
            chars_matched = 0
            while token_idx < len(token_timestamps) and chars_matched < len(word_chars):
                token_char, start_ms, end_ms = token_timestamps[token_idx]
                
                if token_char == '|':
                    token_idx += 1
                    continue
                
                if chars_matched < len(word_chars) and token_char.upper() == word_chars[chars_matched].upper():
                    word_token_times.append((start_ms, end_ms))
                    chars_matched += 1
                    token_idx += 1
                else:
                    found = False
                    for skip in range(1, min(10, len(token_timestamps) - token_idx)):
                        if token_timestamps[token_idx + skip][0] == '|':
                            continue
                        if token_timestamps[token_idx + skip][0].upper() == word_chars[chars_matched].upper():
                            token_idx += skip
                            found = True
                            break
                    
                    if not found:
                        chars_matched += 1
                        token_idx += 1
            
            if word_token_times:
                start_time = min(t[0] for t in word_token_times)
                end_time = max(t[1] for t in word_token_times)
            else:
                if word_dicts:
                    start_time = word_dicts[-1]["end"] * 1000 - chunk_start_time * 1000
                else:
                    start_time = 0
                end_time = start_time + max(len(word) * 150, 200)
            
            # Convert to seconds and add chunk_start_time for absolute timestamps
            word_dicts.append({
                "text": word,
                "start": start_time / 1000 + chunk_start_time,
                "end": end_time / 1000 + chunk_start_time,
                "speaker": speaker
            })
        
        return word_dicts

    def _align_chunk_with_wav2vec2(self, chunk_wav, chunk_transcript: str, 
                                    chunk_start_time: float = 0.0, 
                                    speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Align a single chunk using Wav2Vec2 and return timestamped words.
        
        If Wav2Vec2 fails, falls back to MFA alignment.
        """
        log_progress(f"Aligning transcript with Wav2Vec2 (chunk starts at {chunk_start_time:.2f}s)")
        
        try:
            # Process audio for Wav2Vec2
            inputs = self.wav2vec2_processor(
                chunk_wav, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits

            # Get token timestamps using CTC forced alignment
            token_timestamps = self._get_token_timestamps(logits, chunk_transcript)

            # Convert to word dicts with absolute timestamps
            if token_timestamps:
                word_dicts = self._chars_to_word_dicts(
                    chunk_transcript, token_timestamps, chunk_start_time, speaker
                )
                
                # Validate we got words
                if word_dicts and len(word_dicts) > 0:
                    log_debug(f"Wav2Vec2 alignment produced {len(word_dicts)} words")
                    return word_dicts
            
            # If we got here, Wav2Vec2 didn't produce good results
            self.logger.warning("Wav2Vec2 alignment produced no words, trying MFA fallback")
            return self._align_chunk_with_mfa_fallback(
                chunk_wav, chunk_transcript, chunk_start_time, speaker
            )
            
        except Exception as e:
            self.logger.warning(f"Wav2Vec2 alignment failed: {e}, trying MFA fallback")
            return self._align_chunk_with_mfa_fallback(
                chunk_wav, chunk_transcript, chunk_start_time, speaker
            )
        finally:
            # Clean up
            if 'inputs' in locals():
                for key in list(inputs.keys()):
                    del inputs[key]
                del inputs
            if 'logits' in locals():
                del logits
            
            import gc
            gc.collect()
            clear_device_cache()

    def _check_mfa_available(self) -> bool:
        """Check if MFA is available for fallback alignment."""
        if self.mfa_available is not None:
            return self.mfa_available
        
        # Check if MFA is available in project-local environment
        project_root = pathlib.Path(__file__).parent.parent.parent.parent
        local_mfa_env = project_root / ".mfa_env" / "bin" / "mfa"
        
        if local_mfa_env.exists():
            self.mfa_available = True
            return True
        
        # Check system MFA
        try:
            result = subprocess.run(
                ["mfa", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.mfa_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.mfa_available = False
        
        return self.mfa_available

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

    def _align_chunk_with_mfa_fallback(self, chunk_wav, chunk_transcript: str, 
                                        chunk_start_time: float = 0.0, 
                                        speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback: Align a single chunk using MFA."""
        if not self._check_mfa_available():
            self.logger.warning("MFA not available, using simple timestamp distribution")
            return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)
        
        log_progress(f"Aligning transcript with MFA fallback (chunk starts at {chunk_start_time:.2f}s)")
        
        # Setup MFA models directory if needed
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / "models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._ensure_mfa_models()
        except Exception as e:
            self.logger.warning(f"MFA model setup failed: {e}, using simple distribution")
            return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)
        
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
                    return self._parse_textgrid_to_word_dicts(
                        textgrid_file, chunk_transcript, chunk_start_time, chunk_end_time, speaker
                    )
                else:
                    return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                self.logger.warning(f"MFA alignment failed: {e}")
                return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

    def _parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str, 
                                       chunk_start_time: float = 0.0, chunk_end_time: float = 0.0, 
                                       speaker: Optional[str] = None) -> List[Dict[str, Any]]:
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

        # Replace MFA words with Granite's original words if counts match
        granite_words = original_transcript.split()
        if len(word_dicts) == len(granite_words):
            for i, word_dict in enumerate(word_dicts):
                word_dict["text"] = granite_words[i]
        
        return word_dicts

    def _simple_alignment_to_word_dicts(self, chunk_wav, transcript: str, 
                                         chunk_start_time: float = 0.0, 
                                         speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback: simple even distribution of timestamps."""
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
        """Save debug files for a processing stage."""
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
                f.write("NORMALIZED TEXT (sent to aligner):\n")
                f.write(data['normalized_text'])
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

    def transcribe(self, audio_path: str, device: Optional[str] = None, **kwargs):
        """Not implemented - this provider requires alignment. Use transcribe_with_alignment()."""
        raise NotImplementedError(
            "granite_wav2vec2 is a combined transcriber+aligner. Use transcribe_with_alignment() instead."
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
        log_progress("Starting transcription with alignment using Granite + Wav2Vec2")
        
        # Check if DEBUG logging is enabled and setup debug directory
        from local_transcribe.lib.program_logger import get_output_context
        debug_enabled = get_output_context().should_log("DEBUG")
        debug_dir = None
        intermediate_dir = kwargs.get('intermediate_dir')
        if debug_enabled and intermediate_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = pathlib.Path(intermediate_dir) / "transcription_alignment" / "granite_wav2vec2_debug" / timestamp
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
        self._load_wav2vec2_model()

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
                    
                    # Save normalized transcript for aligner input (merged chunk)
                    normalized_for_aligner = ' '.join(
                        ''.join(c for c in word if c.isalnum() or c == "'")
                        for word in chunk_text.split()
                    )
                    if debug_dir:
                        self._save_debug_chunk(debug_dir, prev_chunk_id, "wav2vec2_input_merged", {
                            "chunk_id": prev_chunk_id,
                            "chunk_start_time": prev_chunk_start_time,
                            "original_text": chunk_text,
                            "normalized_text": normalized_for_aligner,
                            "original_word_count": len(chunk_text.split()),
                            "normalized_word_count": len(normalized_for_aligner.split()),
                            "note": f"Merged with short final chunk {chunk_num}"
                        })
                    
                    # Align merged chunk with absolute timestamps
                    timestamped_words = self._align_chunk_with_wav2vec2(merged_wav, chunk_text, prev_chunk_start_time, role)
                    
                    # Save aligner output for debug (merged chunk)
                    if debug_dir:
                        self._save_debug_chunk(debug_dir, prev_chunk_id, "wav2vec2_output_merged", {
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
                
                # Save normalized transcript for aligner input
                normalized_for_aligner = ' '.join(
                    ''.join(c for c in word if c.isalnum() or c == "'")
                    for word in chunk_text.split()
                )
                if debug_dir:
                    self._save_debug_chunk(debug_dir, chunk_num, "wav2vec2_input", {
                        "chunk_id": chunk_num,
                        "chunk_start_time": chunk_start_time,
                        "original_text": chunk_text,
                        "normalized_text": normalized_for_aligner,
                        "original_word_count": len(chunk_text.split()),
                        "normalized_word_count": len(normalized_for_aligner.split())
                    })
                
                # Align this chunk with absolute timestamps
                timestamped_words = self._align_chunk_with_wav2vec2(chunk_wav, chunk_text, chunk_start_time, role)
                
                # Save aligner output for debug
                if debug_dir:
                    self._save_debug_chunk(debug_dir, chunk_num, "wav2vec2_output", {
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
    registry.register_transcriber_provider(GraniteWav2Vec2TranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
