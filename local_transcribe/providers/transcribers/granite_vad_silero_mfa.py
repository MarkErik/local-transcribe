#!/usr/bin/env python3
"""
Combined Transcriber+Aligner plugin using IBM Granite with Silero VAD-based segmentation and MFA alignment.

The integrated stitcher produces continuous WordSegments output.
Debug mode saves individual segment transcripts when DEBUG logging is enabled.

Uses GraniteModelManager for consolidated model management and transcription.

"""

from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import os
import pathlib
from numpy.typing import NDArray
import tempfile
import subprocess
import json
from datetime import datetime
from local_transcribe.framework.plugin_interfaces import TranscriberProvider, WordSegment, registry
from local_transcribe.lib.system_capability_utils import get_system_capability
from local_transcribe.lib.program_logger import get_logger, log_progress, log_completion, log_debug
from local_transcribe.processing.chunk_stitching import ChunkStitcher

# Use shared lazy imports to avoid code duplication
from local_transcribe.providers.common.lazy_imports import (
    get_granite_model_manager_class,
    get_mfa_alignment_engine_class,
    get_silero_vad_provider_class,
    get_vad_segmenter_func,
)

# Keep module-level references for backward compatibility with tests
_get_granite_model_manager_class = get_granite_model_manager_class
_get_mfa_alignment_engine_class = get_mfa_alignment_engine_class
_get_silero_vad_provider_class = get_silero_vad_provider_class
_get_vad_segmenter_func = get_vad_segmenter_func

# Type hints for lazy-loaded modules
if TYPE_CHECKING:
    import torch
    import librosa


class GraniteVADSileroMFATranscriberProvider(TranscriberProvider):
    """Combined transcriber+aligner using IBM Granite + Silero VAD segmentation + MFA alignment.
    
    Uses GraniteModelManager for consolidated model management and transcription.
    """

    def __init__(self) -> None:
        self.logger = get_logger()
        self.logger.info("Initializing Granite VAD (Silero) MFA Transcriber Provider")
        
        # Model managers will be lazily initialized
        self._model_manager = None
        self._alignment_engine = None
        
        # Track selected model
        self.selected_model: Optional[str] = None
        
        # VAD provider instance (lazily initialized)
        self._vad_provider = None
        self.models_dir: Optional[pathlib.Path] = None
        
        # MFA configuration
        self.mfa_models_dir: Optional[pathlib.Path] = None

    @property
    def model_manager(self):
        """Lazily initialize the model manager to defer torch import."""
        if self._model_manager is None:
            GraniteModelManager = get_granite_model_manager_class()
            self._model_manager = GraniteModelManager(self.logger)
        return self._model_manager
    
    @property
    def word_alignment_engine(self):
        """Lazily initialize the alignment engine."""
        if self._alignment_engine is None:
            MFAAlignmentEngine = get_mfa_alignment_engine_class()
            self._alignment_engine = MFAAlignmentEngine(self.logger)
        return self._alignment_engine
    
    @property
    def vad_provider(self):
        """Lazily initialize the VAD provider."""
        if self._vad_provider is None:
            SileroVADProvider = get_silero_vad_provider_class()
            self._vad_provider = SileroVADProvider(models_dir=self.models_dir)
        return self._vad_provider

    @property
    def device(self) -> str:
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

    @property
    def max_audio_chunk_duration_s(self) -> float:
        """Maximum audio chunk duration this transcriber can handle."""
        return 30.0

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """Return required Granite models (MFA and Silero VAD models handled separately)."""
        return self.model_manager.get_required_models(selected_model)

    def get_available_models(self) -> List[str]:
        return list(self.model_manager.MODEL_MAPPING.keys())

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        self.model_manager.preload_models(models, models_dir)

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which models are available offline."""
        # Note: Silero VAD doesn't require pre-download check as it's auto-downloaded via torch.hub
        # and is very lightweight (~2MB)
        return self.model_manager.check_models_available_offline(models, models_dir)

    def _load_granite_model(self) -> None:
        """Load the Granite model if not already loaded."""
        if self.model_manager.model is None:
            # Set the selected model in the model manager
            self.model_manager.selected_model = self.selected_model
            
            # Load the model using the model manager
            self.model_manager._load_model(self.model_manager.get_required_models()[0])

    def _init_vad_provider(self) -> None:
        """Initialize the Silero VAD provider if not already done."""
        if self._vad_provider is None:
            SileroVADProvider = _get_silero_vad_provider_class()
            self._vad_provider = SileroVADProvider(models_dir=self.models_dir)

    def _get_mfa_command(self) -> str:
        """Get the MFA command, checking local environment first."""
        project_root: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.parent
        local_mfa_env: pathlib.Path = project_root / ".mfa_env" / "bin" / "mfa"
        
        if local_mfa_env.exists():
            return str(local_mfa_env)
        
        return "mfa"

    def _ensure_mfa_models(self) -> None:
        """Ensure MFA acoustic model and dictionary are downloaded."""
        log_progress("Ensuring MFA models are available...")
        log_progress(f"Checking MFA models in {self.mfa_models_dir}")
        env: Dict[str, str] = os.environ.copy()
        env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)

        mfa_cmd: str = self._get_mfa_command()
        
        try:
            result_acoustic: subprocess.CompletedProcess[str] = subprocess.run(
                [mfa_cmd, "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            if "english_us_arpa" not in result_acoustic.stdout:
                log_progress("Downloading MFA English acoustic model...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "acoustic", "english_us_arpa"],
                    check=True,
                    env=env
                )

            result_dictionary: subprocess.CompletedProcess[str] = subprocess.run(
                [mfa_cmd, "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            if "english_us_arpa" not in result_dictionary.stdout:
                log_progress("Downloading MFA English dictionary...")
                subprocess.run(
                    [mfa_cmd, "model", "download", "dictionary", "english_us_arpa"],
                    check=True,
                    env=env
                )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to check/download MFA models: {e}")
            raise

    def _transcribe_single_segment(self, wav: NDArray[Any], sample_rate: int = 16000, **kwargs) -> str:
        """Transcribe a single audio segment using consolidated GraniteModelManager."""
        return self.model_manager.transcribe_segment(wav, sample_rate)

    def _align_segment_with_mfa(self, segment_wav: NDArray[Any], segment_transcript: str, segment_start_time: float = 0.0, speaker: Optional[str] = None, debug_dir: Optional[pathlib.Path] = None, segment_num: Optional[int] = None) -> List[Dict[str, Any]]:
        """Align a single segment using MFA and return timestamped words."""
        log_progress(f"Aligning transcript with MFA (segment starts at {segment_start_time:.2f}s)")
        
        segment_duration: float = len(segment_wav) / 16000.0
        segment_end_time: float = segment_start_time + segment_duration
        
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

            output_dir: pathlib.Path = temp_path / "output"
            output_dir.mkdir()

            env: Dict[str, str] = os.environ.copy()
            env["MFA_ROOT_DIR"] = str(self.mfa_models_dir)
            env["MFA_NO_HISTORY"] = "1"
            
            mfa_env_bin: pathlib.Path = pathlib.Path(self._get_mfa_command()).parent
            env["PATH"] = str(mfa_env_bin) + os.pathsep + env.get("PATH", "")

            textgrid_file: pathlib.Path = output_dir / "segment.TextGrid"

            project_root: pathlib.Path = pathlib.Path(__file__).parent.parent.parent.parent
            mfa_cmd: str = self._get_mfa_command()
            config_path: pathlib.Path = project_root / "mfa_config.yaml"
            
            cmd: List[str] = [
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
                    word_dicts: List[Dict[str, Any]] = self._parse_textgrid_to_word_dicts(textgrid_file, segment_transcript, segment_start_time, segment_end_time, speaker)
                    
                    # Save TextGrid content for debugging if debug mode is enabled
                    if debug_dir and segment_num is not None:
                        textgrid_content: str = textgrid_file.read_text(encoding='utf-8')
                        textgrid_debug_path: pathlib.Path = debug_dir / f"segment_{segment_num:03d}_textgrid.TextGrid"
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
        return self.word_alignment_engine.parse_textgrid_to_word_dicts(
            textgrid_path, original_transcript, segment_start_time, segment_end_time, speaker
        )


    def _replace_words_with_granite_text(self, word_dicts: List[Dict[str, Any]], original_transcript: str,
                                          segment_start_time: float, segment_end_time: float,
                                          speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Replace MFA word text with Granite's original words, using MFA timestamps."""
        return self.word_alignment_engine.replace_words_with_original_text(
            word_dicts, original_transcript, segment_start_time, segment_end_time, speaker
        )

    def _simple_alignment_to_word_dicts(self, segment_wav: Optional[NDArray[Any]], transcript: str, segment_start_time: float = 0.0, speaker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback: simple even distribution of timestamps."""
        if segment_wav is not None:
            duration: float = len(segment_wav) / 16000.0
        else:
            duration: float = len(transcript.split()) * 0.5

        return self.word_alignment_engine.create_simple_alignment(
            transcript, segment_start_time, duration, speaker, segment_wav
        )

    # =========================================================================
    # Chunking strategy for long segments
    # =========================================================================
    
    def _should_chunk_segment(self, segment_duration: float) -> bool:
        """Returns True if segment_duration > max_audio_chunk_duration_s.
        
        Note: Segments should already be ≤30s.
        This is a fallback for edge cases where a segment still exceeds the limit.
        """
        return segment_duration > self.max_audio_chunk_duration_s
    
    def _chunk_segment(self, segment_wav: NDArray, segment_start: float, segment_end: float, sr: int) -> List[Tuple[NDArray, float, float]]:
        """
        Split a long segment into chunks with 25-second length and 4-second overlaps.
        
        Args:
            segment_wav: Full segment audio data
            segment_start: Absolute start time of the segment
            segment_end: Absolute end time of the segment
            sr: Sample rate
            
        Returns:
            List of (chunk_audio, chunk_start_time, chunk_end_time)
        """
        segment_duration: float = segment_end - segment_start
        if segment_duration <= 50.0:
            # No chunking needed for segments <= 50 seconds
            return [(segment_wav, segment_start, segment_end)]
        
        log_debug(f"Chunking segment of {segment_duration:.1f}s into smaller chunks")
        
        # Chunking parameters
        CHUNK_LENGTH = 25.0  # seconds
        OVERLAP = 4.0  # seconds
        MIN_CHUNK_SIZE = 7.0  # seconds
        
        chunks = []
        
        # Calculate number of full chunks and remaining duration
        num_full_chunks = int(segment_duration // CHUNK_LENGTH)
        remaining = segment_duration - (num_full_chunks * CHUNK_LENGTH)
        
        # Process full chunks
        for i in range(num_full_chunks):
            # Calculate chunk boundaries with overlap (relative to segment start)
            if i == 0:
                # First chunk starts at 0 (relative)
                chunk_start_relative = 0.0
            else:
                # Subsequent chunks start with overlap
                chunk_start_relative = (i * CHUNK_LENGTH) - OVERLAP
            
            chunk_end_relative = (i + 1) * CHUNK_LENGTH
            
            # Convert relative times to sample indices (within segment_wav)
            start_sample = int(chunk_start_relative * sr)
            end_sample = int(chunk_end_relative * sr)
            
            # Ensure we don't go beyond the segment boundaries
            start_sample = max(0, start_sample)
            end_sample = min(len(segment_wav), end_sample)
            
            # Extract chunk audio
            chunk_audio = segment_wav[start_sample:end_sample]
            
            # Convert relative times to absolute times for return
            chunk_start_time = segment_start + chunk_start_relative
            chunk_end_time = segment_start + chunk_end_relative
            
            chunks.append((chunk_audio, chunk_start_time, chunk_end_time))
        
        # Handle remaining duration
        if remaining > 0:
            if remaining < MIN_CHUNK_SIZE:
                # Merge remaining with last chunk
                log_debug(f"Remaining {remaining:.1f}s < {MIN_CHUNK_SIZE}s, merging with last chunk")
                if chunks:
                    # Extend the last chunk to include remaining audio
                    last_chunk_audio, last_start, last_end = chunks[-1]
                    new_end_time = segment_end
                    
                    # Calculate relative position of last chunk start
                    last_start_relative = last_start - segment_start
                    
                    # Convert to sample indices using relative times
                    start_sample = int(last_start_relative * sr)
                    end_sample = len(segment_wav)  # Go to end of segment
                    
                    # Extract the extended audio
                    extended_audio = segment_wav[start_sample:end_sample]
                    chunks[-1] = (extended_audio, last_start, new_end_time)
            else:
                # Add final chunk for remaining duration
                final_start_relative = segment_duration - remaining
                final_start_time = segment_start + final_start_relative
                
                # Convert to sample indices using relative times
                start_sample = int(final_start_relative * sr)
                end_sample = len(segment_wav)  # Go to end of segment
                
                final_chunk_audio = segment_wav[start_sample:end_sample]
                chunks.append((final_chunk_audio, final_start_time, segment_end))
        
        log_debug(f"Created {len(chunks)} chunks from {segment_duration:.1f}s segment")
        for i, (chunk_audio, chunk_start, chunk_end) in enumerate(chunks):
            log_debug(f"  Chunk {i+1}: {chunk_start:.2f}s - {chunk_end:.2f}s ({chunk_end - chunk_start:.1f}s)")
        
        return chunks
    
    def _process_chunked_segment(self, segment_wav: NDArray, segment_start: float, segment_end: float, role: Optional[str], debug_dir: Optional[pathlib.Path], segment_num: int) -> List[Dict[str, Any]]:
        """
        Process a segment that requires chunking.
        
        Args:
            segment_wav: Full segment audio data
            segment_start: Absolute start time of the segment
            segment_end: Absolute end time of the segment
            role: Speaker role information
            debug_dir: Debug directory path
            segment_num: Segment number for debug files
            
        Returns:
            List of word dictionaries (same format as current _align_segment_with_mfa)
        """
        sr = 16000  # Sample rate used throughout the system
        
        log_debug(f"Processing chunked segment {segment_num} ({segment_start:.2f}s - {segment_end:.2f}s)")
        
        # Step 1: Split phase - Calculate chunk boundaries and extract audio
        chunks = self._chunk_segment(segment_wav, segment_start, segment_end, sr)
        
        # Prepare chunks for processing
        chunk_data = []
        for i, (chunk_audio, chunk_start_time, chunk_end_time) in enumerate(chunks):
            chunk_id = f"{segment_num}-{i+1}"
            chunk_data.append({
                'chunk_id': chunk_id,
                'audio': chunk_audio,
                'start_time': chunk_start_time,
                'end_time': chunk_end_time,
                'duration': chunk_end_time - chunk_start_time
            })
        
        # Step 2: Transcription phase - Transcribe each chunk
        log_progress(f"Transcribing {len(chunk_data)} chunks for segment {segment_num}")
        chunk_transcripts = []
        
        for i, chunk in enumerate(chunk_data):
            chunk_id = chunk['chunk_id']
            chunk_audio_data: NDArray = chunk['audio']
            chunk_start_time = chunk['start_time']
            
            log_debug(f"Transcribing chunk {chunk_id} ({chunk_start_time:.2f}s)")
            
            # Transcribe the chunk
            chunk_text: str = self._transcribe_single_segment(chunk_audio_data, sample_rate=sr)
            
            if not chunk_text.strip():
                log_debug(f"Chunk {chunk_id} produced empty transcript, skipping")
                continue
                
            chunk_transcripts.append({
                'chunk_id': chunk_id,
                'text': chunk_text,
                'start_time': chunk_start_time,
                'end_time': chunk['end_time']
            })
            
            if debug_dir:
                chunk_debug_num = int(f"{segment_num}{i+1:02d}")
                self._save_debug_segment(debug_dir, chunk_debug_num, "chunk_granite_output", {
                    "chunk_id": chunk_id,
                    "segment_start_time": chunk_start_time,
                    "segment_end_time": chunk['end_time'],
                    "text": chunk_text,
                    "word_count": len(chunk_text.split())
                })
        
        if not chunk_transcripts:
            log_debug(f"All chunks for segment {segment_num} produced empty transcripts")
            return []
        
        # Step 3: Alignment phase - Align each chunk with MFA
        log_progress(f"Aligning {len(chunk_transcripts)} chunks for segment {segment_num}")
        chunk_words = []
        
        for chunk_transcript in chunk_transcripts:
            chunk_id = chunk_transcript['chunk_id']
            chunk_audio_for_alignment: Optional[NDArray] = None  # We'll need to get this from chunk_data
            chunk_start_time = chunk_transcript['start_time']
            
            # Find the corresponding audio data
            for chunk in chunk_data:
                if chunk['chunk_id'] == chunk_id:
                    chunk_audio_for_alignment = chunk['audio']
                    break
            
            if chunk_audio_for_alignment is None:
                log_debug(f"Could not find audio data for chunk {chunk_id}, skipping")
                continue
            
            # Align the chunk with MFA
            chunk_num_suffix = int(chunk_id.split('-')[1])
            timestamped_words: List[Dict[str, Any]] = self._align_segment_with_mfa(
                chunk_audio_for_alignment,
                chunk_transcript['text'],
                chunk_start_time,
                role,
                debug_dir,
                chunk_num_suffix
            )
            
            chunk_words.append({
                'chunk_id': chunk_id,
                'words': timestamped_words,
                'start_time': chunk_start_time,
                'end_time': chunk_transcript['end_time']
            })
            
            if debug_dir:
                self._save_debug_segment(debug_dir, chunk_num_suffix, "chunk_mfa_output", {
                    "chunk_id": chunk_id,
                    "segment_start_time": chunk_start_time,
                    "segment_end_time": chunk_transcript['end_time'],
                    "word_count": len(timestamped_words),
                    "words": timestamped_words
                })
        
        if not chunk_words:
            log_debug(f"All chunks for segment {segment_num} failed alignment")
            return []
        
        # Step 4: Merging phase - Use ChunkStitcher to merge overlapping chunks
        log_progress(f"Merging {len(chunk_words)} chunks for segment {segment_num}")
        
        # Prepare chunks for stitching
        chunks_for_stitching = []
        for chunk_word in chunk_words:
            chunks_for_stitching.append({
                'chunk_id': chunk_word['chunk_id'],
                'words': chunk_word['words']
            })
        
        # Use ChunkStitcher to merge chunks
        stitcher = ChunkStitcher()
        merged_words = stitcher.stitch_chunks(chunks_for_stitching)
        
        # Convert to list of dictionaries if needed
        if isinstance(merged_words, list) and merged_words and hasattr(merged_words[0], 'text'):
            # Convert WordSegment objects to dictionaries
            result_words = []
            for word in merged_words:
                result_words.append({
                    'text': word.text,
                    'start': word.start,
                    'end': word.end,
                    'speaker': word.speaker
                })
            merged_words = result_words
        elif isinstance(merged_words, str):
            # If stitcher returns a string (shouldn't happen with timestamped words),
            # create simple alignment
            log_debug("ChunkStitcher returned string, creating simple alignment")
            segment_duration = segment_end - segment_start
            merged_words = self._simple_alignment_to_word_dicts(
                segment_wav, merged_words, segment_start, role
            )
        elif isinstance(merged_words, list) and merged_words and isinstance(merged_words[0], WordSegment):
            # Convert WordSegment objects to dictionaries
            result_words = []
            for word in merged_words:
                result_words.append({
                    'text': word.text,
                    'start': word.start,
                    'end': word.end,
                    'speaker': word.speaker
                })
            merged_words = result_words
        
        log_debug(f"Chunk merging complete for segment {segment_num}: {len(merged_words)} words")
        
        # Step 5: Integration - Return the merged words
        # Type cast to satisfy the return type annotation
        return merged_words  # type: ignore

    # =========================================================================
    # VAD-specific stitching (simple concatenation with gap handling)
    # =========================================================================
    
    def _stitch_vad_segments(self, all_segment_words: List[List[Dict[str, Any]]]) -> List[WordSegment]:
        """
        Stitch VAD segment words into continuous output.

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
        segment_num_str: str = f"{segment_num:03d}"
        
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
        
        Args:
            audio_path: Path to the audio file
            role: Optional speaker role
            device: Device for processing
            **kwargs: Additional options including:
                - models_dir: Directory for models
                - transcriber_model: Model to use (granite-2b or granite-8b)
                - intermediate_dir: Directory for intermediate files
        
        Returns:
            List of WordSegment objects with timestamps
        """
        log_progress("Starting transcription with Silero VAD segmentation + Granite + MFA")
        
        # Extract models_dir from kwargs if provided
        models_dir = kwargs.get('models_dir')
        if models_dir:
            self.models_dir = pathlib.Path(models_dir)
        
        # Check if DEBUG logging is enabled and setup debug directory
        from local_transcribe.lib.program_logger import get_output_context
        debug_enabled = get_output_context().should_log("DEBUG")
        debug_dir: Optional[pathlib.Path] = None
        intermediate_dir = kwargs.get('intermediate_dir')
        if debug_enabled and intermediate_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = pathlib.Path(intermediate_dir) / "transcription_alignment" / "granite_vad_silero_mfa_debug" / timestamp
            debug_dir.mkdir(parents=True, exist_ok=True)
            log_debug(f"Debug mode enabled - saving debug files to {debug_dir}")
        
        transcriber_model = kwargs.get('transcriber_model', 'granite-8b')
        if transcriber_model not in self.model_manager.MODEL_MAPPING:
            self.logger.warning(f"Unknown model {transcriber_model}, defaulting to granite-8b")
            transcriber_model = 'granite-8b'

        self.selected_model = transcriber_model
        
        # Load the Granite model
        self._load_granite_model()

        # Initialize Silero VAD provider
        self._init_vad_provider()

        # Setup MFA
        if self.mfa_models_dir is None:
            models_root = pathlib.Path(os.environ.get("HF_HOME", str(pathlib.Path.cwd() / ".models")))
            self.mfa_models_dir = models_root / "aligners" / "mfa"
            self.mfa_models_dir.mkdir(parents=True, exist_ok=True)

        self._ensure_mfa_models()

        # Lazy import of librosa
        import librosa

        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration: float = len(wav) / sr
        
        log_progress(f"Audio duration: {duration:.1f}s")
        
        # Step 1: Silero VAD detection using new provider
        log_progress("Running Silero VAD detection...")
        raw_vad_segments = self.vad_provider.detect_speech_from_array(wav, int(sr), speaker_id=role or "Speaker")
        
        # Step 2: Use segmenter for ASR-ready chunks
        segment_for_asr = _get_vad_segmenter_func()
        combined_segments = segment_for_asr(
            raw_vad_segments, 
            max_segment_duration=self.max_audio_chunk_duration_s
        )
        
        # Convert CombinedSegment to (start, end) tuples for processing
        vad_segments: List[Tuple[float, float]] = [
            (seg.start_s, seg.end_s) for seg in combined_segments
        ]
        
        log_progress(f"Silero VAD produced {len(raw_vad_segments)} raw segments, combined into {len(vad_segments)} ASR chunks")
        
        # Save VAD segments info for debug
        if debug_dir:
            vad_info = {
                "vad_engine": "silero",
                "vad_params": self.vad_provider.config,
                "raw_segments": len(raw_vad_segments),
                "combined_segments": len(vad_segments),
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
            segment_num: int = seg_idx + 1
            segment_duration = seg_end - seg_start
            
            log_progress(f"Processing segment {segment_num}/{len(vad_segments)} ({seg_start:.2f}s - {seg_end:.2f}s, duration: {segment_duration:.1f}s)")
            
            # Extract segment audio
            start_sample = int(seg_start * sr)
            end_sample = int(seg_end * sr)
            segment_wav: NDArray[Any] = wav[start_sample:end_sample]
            
            # NEW: Check if segment needs chunking
            if self._should_chunk_segment(segment_duration):
                log_debug(f"Segment {segment_num} is {segment_duration:.1f}s > 50s, applying chunking strategy")
                processed_words = self._process_chunked_segment(
                    segment_wav, seg_start, seg_end, role, debug_dir, segment_num
                )
                all_segment_words.append(processed_words)
            else:
                # Existing processing for short segments
                log_debug(f"Segment {segment_num} is {segment_duration:.1f}s <= 50s, processing as single segment")
                # Transcribe segment (GraniteModelManager handles cleaning and prompt stripping)
                segment_text: str = self._transcribe_single_segment(segment_wav, sample_rate=int(sr), **kwargs)
                
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
                timestamped_words: List[Dict[str, Any]] = self._align_segment_with_mfa(segment_wav, segment_text, seg_start, role, debug_dir, segment_num)
                
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
        result: List[WordSegment] = self._stitch_vad_segments(all_segment_words)
        
        if debug_enabled:
            log_debug(f"Transcription complete: {len(result)} words from {len(vad_segments)} Silero VAD segments")
        
        return result

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.models_dir = models_dir
        
        # Ensure Granite models are available
        self.model_manager.ensure_models_available(models, models_dir)
        
        # Preload Silero VAD model
        log_progress("Preloading Silero VAD model...")
        try:
            self._init_vad_provider()
            self.vad_provider.preload_model()
            log_completion("Silero VAD model preloaded successfully")
        except Exception as e:
            log_progress(f"Failed to preload Silero VAD model: {e}")
            raise


def register_transcriber_plugins() -> None:
    """Register transcriber plugins."""
    registry.register_transcriber_provider(GraniteVADSileroMFATranscriberProvider())


# Auto-register on import
register_transcriber_plugins()
