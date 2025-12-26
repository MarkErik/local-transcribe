#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with MFA alignment.

This plugin combines Granite's transcription capabilities with Montreal Forced Aligner
to produce chunked transcripts where each word has timestamps. 
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
from local_transcribe.providers.common.granite_model_manager import GraniteModelManager
from local_transcribe.providers.common.mfa_word_alignment_engine import MFAWordAlignmentEngine


class GraniteMFATranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + MFA for chunked transcription with timestamps."""

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite MFA Transcriber Provider")
        
        # Replace duplicated model management with GraniteModelManager
        self.model_manager = GraniteModelManager(self.logger)
        
        # Initialize MFA Word Alignment Engine
        self.word_alignment_engine = MFAWordAlignmentEngine(self.logger)
        
        # Keep MFA-specific configuration
        self.chunk_length_seconds = 30.0
        self.overlap_seconds = 4.0
        self.min_chunk_seconds = 7.0
        
        # MFA configuration
        self.mfa_models_dir = None
        
        # Remote transcription settings
        self.use_remote_granite: bool = False
        self.remote_granite_url: Optional[str] = None

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
        return self.model_manager.get_required_models(selected_model)

    def get_available_models(self) -> List[str]:
        return list(self.model_manager.MODEL_MAPPING.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        self.model_manager.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline."""
        return self.model_manager.check_models_available_offline(models, models_dir)

    def _load_granite_model(self):
        """Load the Granite model if not already loaded."""
        if self.model_manager.model is None:
            # Set the selected model in the model manager
            if self.selected_model:
                self.model_manager.selected_model = self.selected_model
            
            # Load the model using the model manager
            model_name = self.model_manager.get_required_models()[0]
            self.model_manager._load_model(model_name)
            
            # Copy the loaded model components to the instance for backward compatibility
            self.model = self.model_manager.model
            self.processor = self.model_manager.processor
            self.tokenizer = self.model_manager.tokenizer

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

    def _transcribe_single_chunk(self, wav, sample_rate: int = 16000, **kwargs) -> str:
        """Transcribe a single audio chunk using Granite (local or remote)."""
        segment_duration: float = len(wav) / sample_rate
        
        # Check if we should use remote transcription
        if self.use_remote_granite and self.model_manager.should_use_remote():
            log_progress(f"Transcribing {segment_duration:.1f}s chunk with remote Granite server")
            try:
                text = self.model_manager.transcribe_remote(
                    audio=wav,
                    sample_rate=sample_rate,
                    segment_duration=segment_duration,
                    include_disfluencies=True
                )
                # Apply local cleaning
                cleaned_text = self._clean_transcription_output(text, verbose=kwargs.get('verbose', False))
                return cleaned_text
            except Exception as e:
                self.logger.warning(f"Remote transcription failed, falling back to local: {e}")
                # Fall through to local transcription
        
        # Local transcription
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
                    return self.word_alignment_engine.create_simple_alignment(
                        chunk_transcript, chunk_start_time, len(chunk_wav) / 16000.0 if chunk_wav is not None else None, speaker, chunk_wav
                    )

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Fallback to simple distribution
                return self.word_alignment_engine.create_simple_alignment(
                    chunk_transcript, chunk_start_time, len(chunk_wav) / 16000.0 if chunk_wav is not None else None, speaker, chunk_wav
                )

    def _parse_textgrid_to_word_dicts(self, textgrid_path: pathlib.Path, original_transcript: str, chunk_start_time: float = 0.0, chunk_end_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Parse MFA TextGrid and return list of word dicts with timestamps.
        
        Uses the shared MFAWordAlignmentEngine for parsing.
        """
        return self.word_alignment_engine.parse_textgrid_to_word_dicts(
            textgrid_path, original_transcript, chunk_start_time, chunk_end_time, speaker
        )



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

        Supports both local and remote Granite transcription. Configure remote
        transcription by passing:
            use_remote_granite=True
            remote_granite_url="http://server:7070"
        
        Args:
            audio_path: Path to the audio file
            role: Optional speaker role
            device: Device for processing
            **kwargs: Additional options including:
                - transcriber_model: Model to use (granite-2b or granite-8b)
                - intermediate_dir: Directory for intermediate files
                - use_remote_granite: Use remote Granite server (default: False)
                - remote_granite_url: URL of remote Granite server
        
        Returns:
            List of dictionaries with chunk data and timestamped words
        """
        log_progress("Starting transcription with alignment using Granite + MFA")
        
        # Configure remote Granite if requested
        self.use_remote_granite = kwargs.get('use_remote_granite', False)
        self.remote_granite_url = kwargs.get('remote_granite_url')
        
        if self.use_remote_granite:
            log_progress(f"Remote Granite transcription enabled: {self.remote_granite_url or 'default URL'}")
            self.model_manager.configure_remote(
                use_remote=True,
                remote_url=self.remote_granite_url
            )
            
            # Check if remote is available
            if not self.model_manager.is_remote_available():
                self.logger.warning("Remote Granite server not available, falling back to local")
                self.use_remote_granite = False
                self.model_manager.configure_remote(use_remote=False)
        else:
            self.model_manager.configure_remote(use_remote=False)
        
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
        if transcriber_model not in self.model_manager.MODEL_MAPPING:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        
        # Only load local model if not using remote (or as fallback)
        if not self.use_remote_granite:
            self._load_granite_model()
        else:
            log_progress("Using remote Granite server - skipping local model load")

        # Setup MFA
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / ".models")))
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
        self.model_manager.ensure_models_available(models, models_dir)


def register_transcriber_plugins():
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteMFATranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
