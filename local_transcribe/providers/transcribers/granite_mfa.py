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
        self.overlap_seconds = 3.0
        self.min_chunk_seconds = 6.0
        
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

            with torch.no_grad():
                model_outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens=400,
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    repetition_penalty=3.0,
                    length_penalty=1.0,
                    temperature=1.0,
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
                    return self._parse_textgrid_to_word_dicts(textgrid_file, chunk_transcript, chunk_start_time, speaker)
                else:
                    # Fallback to simple distribution
                    return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback to simple distribution
                return self._simple_alignment_to_word_dicts(chunk_wav, chunk_transcript, chunk_start_time, speaker)

    def _parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str, chunk_start_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse MFA TextGrid and return list of word dicts with timestamps.
        
        Args:
            textgrid_path: Path to the TextGrid file
            original_transcript: Original transcript with punctuation/capitalization
            chunk_start_time: Absolute start time of this chunk in the full audio (seconds)
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

        # Replace <unk> tokens with words from original transcript
        self._replace_unk_in_word_dicts(word_dicts, original_transcript)

        return word_dicts

    def _replace_unk_in_word_dicts(self, word_dicts: List[Dict[str, Any]], original_transcript: str) -> None:
        """Replace <unk> tokens in aligned word dicts with words from the original transcript using two-pointer alignment.
        
        Args:
            word_dicts: List of word dictionaries from MFA alignment (modified in place)
            original_transcript: Original transcript from Granite with all words
        """
        log_progress("Starting <unk> replacement")
        
        aligned_texts = [wd["text"] for wd in word_dicts]
        original_words = original_transcript.split()
        
        log_debug(f"Original transcript word count: {len(original_words)}")
        log_debug(f"MFA aligned word count before replacement: {len(aligned_texts)}")
        
        ptr = 0
        for i, word_dict in enumerate(word_dicts):
            if word_dict["text"] == "<unk>":
                # Debug: show context
                start_idx = max(0, i - 5)
                end_idx = min(len(aligned_texts), i + 6)
                aligned_context = aligned_texts[start_idx:end_idx]
                
                orig_start = max(0, ptr - 5)
                orig_end = min(len(original_words), ptr + 6)
                original_context = original_words[orig_start:orig_end]
                
                log_debug(f"Replacing <unk> at position {i}: Aligned context: {' '.join(aligned_context)} | Original context around ptr {ptr}: {' '.join(original_context)}")
                
                if ptr < len(original_words):
                    replacement = original_words[ptr]
                    word_dict["text"] = replacement
                    aligned_texts[i] = replacement  # Update local copy for debugging
                    log_debug(f"Replaced with: '{replacement}'")
                    ptr += 1
                else:
                    log_debug("No more original words available, leaving as <unk>")
            else:
                if ptr < len(original_words) and word_dict["text"].lower() == original_words[ptr].lower():
                    log_debug(f"Matched '{word_dict['text']}' with original '{original_words[ptr]}', advancing ptr to {ptr+1}")
                    ptr += 1
                else:
                    log_debug(f"No match for '{word_dict['text']}' at ptr {ptr}, not advancing ptr")
        
        final_texts = [wd["text"] for wd in word_dicts]
        log_debug(f"MFA aligned word count after replacement: {len(final_texts)}")

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
