#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with Wav2Vec2 alignment.

This plugin combines Granite's transcription capabilities with Wav2Vec2 forced alignment
to produce chunked transcripts where each word has timestamps. Unlike the separate
granite + wav2vec2 pipeline, this processes each chunk with alignment before moving to the next,
resulting in chunks that contain timestamped words ready for stitching.

If Wav2Vec2 alignment fails, it falls back to simple timestamp distribution.

Uses GraniteModelManager for consolidated model management and transcription.

Note: Heavy imports (torch, librosa, transformers, torchaudio) are lazily loaded when needed
to avoid slow startup times when this provider is not used.
"""

from typing import List, Optional, Dict, Any, TYPE_CHECKING, Sequence, Union
import os
import pathlib
import math
import json
import warnings
from datetime import datetime
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug

# Type hints for lazy-loaded modules
if TYPE_CHECKING:
    import torch
    import numpy as np
    import torchaudio
    import librosa
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Lazy import for GraniteModelManager to avoid torch import at module load
_granite_model_manager_class = None

def _get_granite_model_manager_class():
    """Lazily import GraniteModelManager to defer torch import."""
    global _granite_model_manager_class
    if _granite_model_manager_class is None:
        from local_transcribe.providers.common.granite_model import GraniteModelManager
        _granite_model_manager_class = GraniteModelManager
    return _granite_model_manager_class


class GraniteWav2Vec2TranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + Wav2Vec2 for chunked transcription with timestamps.
    
    Uses GraniteModelManager for consolidated model management and transcription.
    """

    def __init__(self):
        self.logger = get_logger()
        self.logger.info("Initializing Granite Wav2Vec2 Transcriber Provider")
        
        # Model manager will be lazily initialized
        self._model_manager = None
        
        # Track selected model
        self.selected_model: Optional[str] = None
        
        # Wav2Vec2 configuration
        self.wav2vec2_model_name = "facebook/wav2vec2-large-960h"
        self.wav2vec2_processor: Optional[Any] = None
        self.wav2vec2_model: Optional[Any] = None
        
        # Chunking configuration
        self.chunk_length_seconds = 60.0
        self.overlap_seconds = 4.0
        self.min_chunk_seconds = 7.0

    @property
    def model_manager(self):
        """Lazily initialize the model manager to defer torch import."""
        if self._model_manager is None:
            GraniteModelManager = _get_granite_model_manager_class()
            self._model_manager = GraniteModelManager(self.logger)
        return self._model_manager

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
        
        # Granite model - delegate to GraniteModelManager
        granite_models = self.model_manager.get_required_models(selected_model)
        models.extend(granite_models)
        
        # Wav2Vec2 model
        models.append(self.wav2vec2_model_name)
        
        return models

    def get_available_models(self) -> List[str]:
        return list(self.model_manager.MODEL_MAPPING.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite and Wav2Vec2 models to cache."""
        self.logger.info("Starting model preload for Granite and Wav2Vec2 models")
        
        # Separate Granite models from Wav2Vec2 models
        granite_models = []
        wav2vec2_models = []
        
        for model in models:
            if model == self.wav2vec2_model_name:
                wav2vec2_models.append(model)
            else:
                # Assume it's a Granite model
                granite_models.append(model)
        
        # Preload Granite models using GraniteModelManager
        if granite_models:
            self.model_manager.preload_models(granite_models, models_dir)
        
        # Preload Wav2Vec2 models separately
        if wav2vec2_models:
            for model in wav2vec2_models:
                if model == self.wav2vec2_model_name:
                    # Wav2Vec2 model - use standard HF cache
                    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
                    if xdg_cache_home:
                        models_root = pathlib.Path(xdg_cache_home)
                    else:
                        models_root = pathlib.Path.home() / ".cache" / "huggingface"
                    
                    hub_cache_dir = models_root / "huggingface" / "hub"
                    hub_cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    from huggingface_hub import snapshot_download
                    snapshot_download(model, cache_dir=hub_cache_dir, token=os.getenv("HF_TOKEN"))
                    log_completion(f"{model} downloaded successfully.")

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite and Wav2Vec2 models are available offline."""
        missing_models = []
        
        # Separate Granite models from Wav2Vec2 models
        granite_models = []
        wav2vec2_models = []
        
        for model in models:
            if model == self.wav2vec2_model_name:
                wav2vec2_models.append(model)
            else:
                # Assume it's a Granite model
                granite_models.append(model)
        
        # Check Granite models using GraniteModelManager
        if granite_models:
            granite_missing = self.model_manager.check_models_available_offline(granite_models, models_dir)
            missing_models.extend(granite_missing)
        
        # Check Wav2Vec2 models separately
        if wav2vec2_models:
            xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
            if xdg_cache_home:
                models_root = pathlib.Path(xdg_cache_home)
            else:
                models_root = pathlib.Path.home() / ".cache" / "huggingface"
            
            hub_dir = models_root / "huggingface" / "hub"
            
            for model in wav2vec2_models:
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

    def _load_granite_model(self) -> None:
        """Load the Granite model if not already loaded."""
        log_progress("Loading Granite model...")
        
        if self.model_manager.model is None:
            # Set selected model in the manager
            if self.selected_model:
                self.model_manager.selected_model = self.selected_model
            
            # Load the model
            model_name = self.model_manager.get_required_models()[0]
            self.model_manager._load_model(model_name)
            
            log_completion("Granite model loaded successfully")

    def _load_wav2vec2_model(self):
        """Load the Wav2Vec2 model for alignment if not already loaded."""
        if self.wav2vec2_model is None:
            log_progress("Loading Wav2Vec2 model for alignment...")
            
            # Lazy import of transformers
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
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

    def _transcribe_single_chunk(self, wav, sample_rate: int = 16000, **kwargs) -> str:
        """Transcribe a single audio chunk using consolidated GraniteModelManager."""
        return self.model_manager.transcribe_segment(wav, sample_rate)

    def _get_token_timestamps(self, emissions: "torch.Tensor", transcript: str) -> List[tuple]:
        """Extract token timestamps using CTC alignment with frame-level paths."""
        # Lazy imports
        import torch
        import torchaudio
        
        if self.wav2vec2_processor is None:
            raise RuntimeError("Wav2Vec2 processor not loaded")
        
        # Access tokenizer from processor
        tokenizer = getattr(self.wav2vec2_processor, 'tokenizer', None)
        if tokenizer is None:
            raise RuntimeError("Wav2Vec2 processor has no tokenizer attribute")
        
        vocab = tokenizer.get_vocab()
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
            
            pad_token_id = getattr(tokenizer, 'pad_token_id', 0)
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*forced_align.*deprecated.*")
                aligned_labels, scores = torchaudio.functional.forced_align(
                    log_probs,
                    torch.tensor([token_ids]),
                    blank=pad_token_id
                )
            
            token_timestamps = self._extract_token_boundaries(
                aligned_labels[0],
                tokens,
                token_ids,
                pad_token_id
            )
            
            return token_timestamps
            
        except Exception as e:
            self.logger.warning(f"Forced alignment failed ({e}), using fallback method")
            return self._fallback_token_alignment(emissions, tokens)

    def _extract_token_boundaries(self, aligned_labels: "torch.Tensor", tokens: List[str],
                                    token_ids: List[int], blank_id: int) -> List[tuple]:
        """Extract token boundaries from frame-level aligned labels.
        
        Args:
            aligned_labels: Frame-level alignment output from CTC forced alignment
            tokens: List of token characters to align
            token_ids: Corresponding token IDs for each token
            blank_id: The CTC blank token ID
            
        Returns:
            List of tuples: (token_char, start_time_ms, end_time_ms)
        """
        token_timestamps = []
        aligned_labels_list = aligned_labels.tolist()
        total_frames = len(aligned_labels_list)
        total_duration_ms = total_frames * 0.02 * 1000  # 20ms per frame
        
        current_token_idx = 0
        frame_start = None
        
        for frame_idx, label_id in enumerate(aligned_labels_list):
            if label_id == blank_id:
                if frame_start is not None and current_token_idx < len(token_ids):
                    start_time = frame_start * 0.02 * 1000
                    end_time = frame_idx * 0.02 * 1000
                    
                    # Bounds check: ensure end_time doesn't exceed audio duration
                    end_time = min(end_time, total_duration_ms)
                    
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
                    
                    # Bounds check
                    end_time = min(end_time, total_duration_ms)
                    
                    token_timestamps.append((tokens[current_token_idx], start_time, end_time))
                    
                    current_token_idx += 1
                    frame_start = None
                    
                    # Check if the new label matches the next expected token
                    if current_token_idx < len(token_ids) and label_id == token_ids[current_token_idx]:
                        frame_start = frame_idx
        
        # Handle final token if still tracking
        if frame_start is not None and current_token_idx < len(token_ids):
            start_time = frame_start * 0.02 * 1000
            end_time = total_duration_ms  # Use total duration for final token
            token_timestamps.append((tokens[current_token_idx], start_time, end_time))
            current_token_idx += 1
        
        # Validation: check if we missed any tokens
        tokens_found = len(token_timestamps)
        tokens_expected = len(tokens)
        if tokens_found < tokens_expected:
            missing_count = tokens_expected - tokens_found
            self.logger.warning(
                f"Token alignment incomplete: found {tokens_found}/{tokens_expected} tokens "
                f"({missing_count} missing). Alignment quality may be degraded."
            )
            
            # Fill in missing tokens with estimated timestamps at the end
            if token_timestamps:
                last_end_time = token_timestamps[-1][2]
            else:
                last_end_time = 0
            
            remaining_duration = total_duration_ms - last_end_time
            remaining_tokens = tokens[current_token_idx:]
            
            if remaining_tokens and remaining_duration > 0:
                time_per_missing = remaining_duration / len(remaining_tokens)
                for i, token in enumerate(remaining_tokens):
                    start_time = last_end_time + (i * time_per_missing)
                    end_time = start_time + time_per_missing
                    token_timestamps.append((token, start_time, end_time))
        
        return token_timestamps

    def _fallback_token_alignment(self, emissions: "torch.Tensor", tokens: List[str]) -> List[tuple]:
        """Fallback token alignment using simple peak detection."""
        # Lazy import of numpy
        import numpy as np
        
        token_timestamps = []
        emissions_np = emissions[0].cpu().numpy()
        
        if self.wav2vec2_processor is None:
            raise RuntimeError("Wav2Vec2 processor not loaded")
        tokenizer = getattr(self.wav2vec2_processor, 'tokenizer', None)
        if tokenizer is None:
            raise RuntimeError("Wav2Vec2 processor has no tokenizer attribute")
        
        dictionary = {c: i for i, c in enumerate(tokenizer.get_vocab().keys())}
        
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
                             chunk_start_time: float, speaker: Optional[str] = None,
                             chunk_duration_ms: Optional[float] = None) -> List[Dict[str, Any]]:
        """Convert character-level token timestamps to word dicts with absolute timestamps.
        
        Args:
            transcript: The transcribed text
            token_timestamps: List of (char, start_ms, end_ms) tuples from alignment
            chunk_start_time: Absolute start time of this chunk in seconds
            speaker: Optional speaker label
            chunk_duration_ms: Optional chunk duration in ms for bounds checking
            
        Returns:
            List of word dicts with "text", "start", "end", "speaker" keys
        """
        words = transcript.split()
        if not words:
            return []
        
        if not token_timestamps:
            return self._simple_alignment_to_word_dicts(None, transcript, chunk_start_time, speaker)
        
        # Calculate chunk duration from tokens if not provided
        if chunk_duration_ms is None and token_timestamps:
            chunk_duration_ms = max(t[2] for t in token_timestamps)  # Max end time
        
        word_dicts = []
        token_idx = 0
        fallback_count = 0
        
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
                # Fallback: estimate timestamps based on previous word
                fallback_count += 1
                if word_dicts:
                    # Previous word's end is in absolute seconds, convert back to chunk-relative ms
                    prev_end_absolute = word_dicts[-1]["end"]  # in seconds
                    start_time = (prev_end_absolute - chunk_start_time) * 1000  # chunk-relative ms
                else:
                    start_time = 0
                
                # Estimate duration based on word length (roughly 150ms per character, min 200ms)
                estimated_duration = max(len(word) * 150, 200)
                end_time = start_time + estimated_duration
                
                # Bounds check: ensure we don't exceed chunk duration
                if chunk_duration_ms is not None and end_time > chunk_duration_ms:
                    end_time = chunk_duration_ms
                    # Adjust start if needed to maintain some duration
                    if end_time - start_time < 50:  # Minimum 50ms
                        start_time = max(0, end_time - 50)
            
            # Convert to seconds and add chunk_start_time for absolute timestamps
            word_dicts.append({
                "text": word,
                "start": start_time / 1000 + chunk_start_time,
                "end": end_time / 1000 + chunk_start_time,
                "speaker": speaker
            })
        
        # Log warning if many words used fallback
        if fallback_count > 0:
            fallback_pct = (fallback_count / len(words)) * 100
            if fallback_pct > 20:
                self.logger.warning(
                    f"Word alignment quality degraded: {fallback_count}/{len(words)} words "
                    f"({fallback_pct:.1f}%) used fallback timestamps"
                )
        
        return word_dicts

    def _align_chunk_with_wav2vec2(self, chunk_wav, chunk_transcript: str, 
                                    chunk_start_time: float = 0.0, 
                                    speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Align a single chunk using Wav2Vec2 and return timestamped words.
        
        If Wav2Vec2 fails, falls back to simple timestamp distribution.
        """
        log_progress(f"Aligning transcript with Wav2Vec2 (chunk starts at {chunk_start_time:.2f}s)")
        
        if self.wav2vec2_processor is None or self.wav2vec2_model is None:
            raise RuntimeError("Wav2Vec2 models not loaded. Call _load_wav2vec2_model() first.")
        
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
            self.logger.warning("Wav2Vec2 alignment produced no words, using simple timestamp distribution")
            return self._simple_alignment_to_word_dicts(
                chunk_wav, chunk_transcript, chunk_start_time, speaker
            )
            
        except Exception as e:
            self.logger.warning(f"Wav2Vec2 alignment failed: {e}, using simple timestamp distribution")
            return self._simple_alignment_to_word_dicts(
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

    def _enforce_timestamp_monotonicity(self, chunks: List[Dict[str, Any]], 
                                         verbose: bool = False,
                                         intermediate_dir: Optional[pathlib.Path] = None,
                                         audio_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Post-process chunks to ensure timestamp monotonicity across all words.
        
        This fixes overlapping timestamps that can occur at chunk boundaries or due to
        alignment errors. For each overlap, it adjusts timestamps by splitting the
        difference between overlapping words.
        
        Args:
            chunks: List of chunk dicts, each with "words" containing timestamped word dicts
            verbose: Whether to log adjustment details
            intermediate_dir: Optional directory to save overlap fix details to a text file
            audio_path: Optional path to the audio file, used for naming the overlap report
            
        Returns:
            The same chunks list with adjusted timestamps (modified in place)
        """
        # Collect all words across chunks for sequential processing
        all_words = []
        for chunk in chunks:
            all_words.extend(chunk.get("words", []))
        
        if not all_words:
            return chunks
        
        overlap_count = 0
        total_adjustment = 0.0
        overlap_details = []
        
        for i in range(1, len(all_words)):
            prev_word = all_words[i - 1]
            curr_word = all_words[i]
            
            prev_end = prev_word.get("end", 0)
            curr_start = curr_word.get("start", 0)
            
            if curr_start < prev_end:
                # Overlap detected
                overlap_count += 1
                overlap_duration = prev_end - curr_start
                total_adjustment += overlap_duration
                
                # Strategy: Split the difference - move both words' boundary to midpoint
                midpoint = (prev_end + curr_start) / 2
                
                # Ensure midpoint is within reasonable bounds
                # Don't let previous word's end go before its start
                if midpoint < prev_word.get("start", 0):
                    midpoint = prev_word.get("start", 0) + 0.01
                
                # Adjust timestamps
                prev_word["end"] = midpoint
                curr_word["start"] = midpoint
                
                # Also ensure current word's end is after its start
                if curr_word.get("end", 0) <= curr_word["start"]:
                    curr_word["end"] = curr_word["start"] + 0.02  # Minimum 20ms duration
                
                # Record details for auditing
                overlap_details.append({
                    "word_index": i,
                    "prev_word": prev_word.get("text", ""),
                    "curr_word": curr_word.get("text", ""),
                    "original_prev_end": prev_end,
                    "original_curr_start": curr_start,
                    "overlap_duration": overlap_duration,
                    "new_midpoint": midpoint,
                    "adjustment": overlap_duration
                })
                
                if verbose:
                    log_debug(
                        f"Fixed overlap between '{prev_word.get('text', '')}' and "
                        f"'{curr_word.get('text', '')}': adjusted by {overlap_duration:.3f}s"
                    )
        
        if overlap_count > 0:
            self.logger.info(
                f"Timestamp monotonicity: fixed {overlap_count} overlaps, "
                f"total adjustment: {total_adjustment:.3f}s"
            )
            
            # Save detailed overlap fixes to file for auditing
            if intermediate_dir:
                if audio_path:
                    audio_basename = pathlib.Path(audio_path).stem
                    overlap_file = intermediate_dir / f"overlap_fixes_{audio_basename}.txt"
                else:
                    overlap_file = intermediate_dir / "overlap_fixes.txt"
                try:
                    with open(overlap_file, 'w', encoding='utf-8') as f:
                        f.write("Overlap Fixes Report\n")
                        f.write("=" * 50 + "\n")
                        f.write(f"Total overlaps fixed: {overlap_count}\n")
                        f.write(f"Total adjustment time: {total_adjustment:.3f}s\n")
                        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        f.write("Detailed Fixes:\n")
                        f.write("-" * 50 + "\n")
                        for detail in overlap_details:
                            f.write(f"Word {detail['word_index']}: '{detail['prev_word']}' -> '{detail['curr_word']}'\n")
                            f.write(f"  Original: prev_end={detail['original_prev_end']:.3f}s, curr_start={detail['original_curr_start']:.3f}s\n")
                            f.write(f"  Overlap: {detail['overlap_duration']:.3f}s\n")
                            f.write(f"  New midpoint: {detail['new_midpoint']:.3f}s\n")
                            f.write(f"  Adjustment: {detail['adjustment']:.3f}s\n\n")
                        
                    self.logger.info(f"Overlap fix details saved to: {overlap_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to save overlap fix details: {e}")
        
        return chunks

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
        if not self.model_manager.validate_model_selection(transcriber_model):
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.model_manager.selected_model = transcriber_model
        
        # Load the Granite model
        self._load_granite_model()
        
        self._load_wav2vec2_model()

        # Lazy import of librosa
        import librosa

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
                    # Handle case where short chunk might be smaller than overlap
                    if len(chunk_wav) > overlap_samples:
                        non_overlapping_part = chunk_wav[overlap_samples:]
                    else:
                        # Chunk is very short, use all of it but log warning
                        non_overlapping_part = chunk_wav
                        self.logger.warning(
                            f"Short final chunk ({len(chunk_wav)} samples) smaller than overlap "
                            f"({overlap_samples} samples), using entire chunk"
                        )
                    # Lazy import of torch
                    import torch
                    merged_tensor = torch.cat([torch.from_numpy(prev_chunk_wav), torch.from_numpy(non_overlapping_part)])
                    merged_wav = merged_tensor.numpy()
                    
                    # Get previous chunk's start time - it should always be present
                    if "chunk_start_time" not in chunks_with_timestamps[-1]:
                        self.logger.warning("Previous chunk missing chunk_start_time, calculating from current position")
                    prev_chunk_start_time = chunks_with_timestamps[-1].get(
                        "chunk_start_time", 
                        max(0, chunk_start_time - (len(prev_chunk_wav) / sr))  # Ensure non-negative
                    )
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
        
        # Post-process to ensure timestamp monotonicity across all chunks
        chunks_with_timestamps = self._enforce_timestamp_monotonicity(chunks_with_timestamps, verbose=verbose, intermediate_dir=intermediate_dir, audio_path=audio_path)
        
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
