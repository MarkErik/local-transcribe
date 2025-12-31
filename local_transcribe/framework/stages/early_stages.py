#!/usr/bin/env python3
"""
Early pipeline stages: audio processing, transcription, de-identification, diarization.

These stages handle the initial processing of audio files before turn building.
"""

from typing import List, Optional, Any, Dict
from pathlib import Path

from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.framework.stages.base import PipelineStage, StageError
from local_transcribe.lib.program_logger import (
    log_status, log_progress, log_intermediate_save, log_completion
)


class AudioStandardizationStage(PipelineStage):
    """Stage for standardizing audio files to a common format."""
    
    @property
    def name(self) -> str:
        return "audio_standardization"
    
    @property
    def description(self) -> str:
        return "Convert audio to standardized format (16kHz mono WAV)"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["speaker_files", "api"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["standardized_audio", "standardized_speaker_files"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.audio_processor import standardize_audio
        
        outdir = context.get_output_dir()
        
        if context.mode == "combined_audio":
            # Single file to standardize
            audio_path = list(context.speaker_files.values())[0]
            log_progress(f"Standardizing audio: {Path(audio_path).name}")
            
            std_audio = standardize_audio(str(audio_path), outdir, context.api)
            context.standardized_audio = std_audio
            context.standardized_speaker_files = {"combined_audio": str(std_audio)}
            
        elif context.mode == "single_speaker_audio":
            # Single speaker file
            audio_path = context.speaker_files["speaker"]
            log_progress(f"Standardizing audio: {Path(audio_path).name}")
            
            std_audio = standardize_audio(str(audio_path), outdir, context.api)
            context.standardized_audio = std_audio
            context.standardized_speaker_files = {"speaker": str(std_audio)}
            
        else:
            # Multiple speaker files (split_audio mode)
            context.standardized_speaker_files = {}
            
            for speaker_name, audio_path in context.speaker_files.items():
                log_progress(f"Standardizing audio for {speaker_name}: {Path(audio_path).name}")
                std_audio = standardize_audio(str(audio_path), outdir, context.api, speaker_name)
                context.standardized_speaker_files[speaker_name] = str(std_audio)
            
            # For split audio, standardized_audio is None (use standardized_speaker_files instead)
            context.standardized_audio = None
        
        log_completion("Audio standardization complete")
        return context


class TranscriptionAlignmentStage(PipelineStage):
    """Stage for transcribing audio and aligning words with timestamps."""
    
    @property
    def name(self) -> str:
        return "transcription_alignment"
    
    @property
    def description(self) -> str:
        return "Transcribe audio and align words with timestamps"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["transcriber_provider"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["word_segments", "speaker_word_segments"]
    
    @property
    def applicable_modes(self) -> List[str]:
        # Applies to all modes except vad_split_audio (which has its own transcription)
        return ["combined_audio", "split_audio", "single_speaker_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.system_capability_utils import get_system_capability
        
        args = context.args
        intermediate_dir = context.get_intermediate_dir()
        registry = context.api.get("registry")
        
        if registry is None:
            raise StageError(self.name, "Registry not found in api")
        
        # Build common kwargs for transcription
        transcribe_kwargs = {
            "registry": registry,
            "transcriber_model": getattr(args, 'transcriber_model', None),
            "output_format": getattr(args, 'output_format', 'stitched'),
        }
        
        # Pass include_disfluencies setting if specified (for remote transcriber)
        include_disfluencies = getattr(args, 'include_disfluencies', None)
        if include_disfluencies is not None:
            transcribe_kwargs["include_disfluencies"] = include_disfluencies
        
        if context.mode == "single_speaker_audio":
            # Transcription only (no alignment) for single speaker
            words = self._transcribe_only(context, transcribe_kwargs)
            context.word_segments = words
            context.speaker_word_segments = {"speaker": words}
            
        elif context.mode == "combined_audio":
            # Single file transcription + alignment
            audio_path = context.standardized_audio
            if audio_path is None:
                raise StageError(self.name, "No standardized audio available")
            
            words = self._transcribe_with_alignment(
                context, str(audio_path), None, "", transcribe_kwargs
            )
            context.word_segments = words
            context.speaker_word_segments = None  # Not applicable for combined audio
            
        else:
            # Split audio - process each speaker file
            if not context.standardized_speaker_files:
                raise StageError(self.name, "No standardized speaker files available for split audio mode")
            
            context.speaker_word_segments = {}
            all_words = []
            
            for speaker_name, audio_path in context.standardized_speaker_files.items():
                log_progress(f"Transcribing {speaker_name}...")
                
                words = self._transcribe_with_alignment(
                    context, audio_path, speaker_name, 
                    f"{speaker_name.lower()}_", transcribe_kwargs
                )
                context.speaker_word_segments[speaker_name] = words
                all_words.extend(words)
            
            context.word_segments = all_words
        
        log_completion(f"Transcription complete: {len(context.word_segments or [])} word segments")
        return context
    
    def _transcribe_only(self, context: PipelineContext, kwargs: Dict[str, Any]) -> Any:
        """Transcribe without alignment (for single_speaker_audio mode)."""
        from local_transcribe.lib.system_capability_utils import get_system_capability
        from local_transcribe.processing.chunk_stitching import stitch_chunks
        import json
        
        transcriber = context.transcriber_provider
        if transcriber is None:
            raise StageError(self.name, "No transcriber provider configured")
        
        audio_path = context.standardized_audio
        intermediate_dir = context.get_intermediate_dir()
        device = get_system_capability()
        
        # Transcribe
        transcript = transcriber.transcribe(str(audio_path), device=device, **kwargs)
        
        # Handle chunked output
        if isinstance(transcript, list):
            log_progress(f"Received chunked output with {len(transcript)} chunks")
            
            # Save chunks
            if intermediate_dir:
                chunk_file = intermediate_dir / "transcription" / "raw_chunks.json"
                chunk_file.parent.mkdir(parents=True, exist_ok=True)
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                log_intermediate_save(str(chunk_file), "Raw chunks saved to")
            
            transcript_text = stitch_chunks(transcript, **kwargs)
        else:
            transcript_text = transcript
        
        # Save transcript
        if intermediate_dir:
            transcript_file = intermediate_dir / "transcription" / "raw_transcript.txt"
            transcript_file.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(str(transcript_text))
            log_intermediate_save(str(transcript_file), "Raw transcript saved to")
        
        return str(transcript_text)
    
    def _transcribe_with_alignment(
        self, 
        context: PipelineContext,
        audio_path: str,
        role: Optional[str],
        base_name: str,
        kwargs: Dict[str, Any]
    ) -> List[Any]:
        """Transcribe and align audio."""
        from local_transcribe.lib.system_capability_utils import get_system_capability
        from local_transcribe.processing.chunk_stitching import stitch_chunks
        import json
        
        transcriber = context.transcriber_provider
        if transcriber is None:
            raise StageError(self.name, "No transcriber provider configured")
        
        aligner = context.aligner_provider
        intermediate_dir = context.get_intermediate_dir()
        device = get_system_capability()
        
        # Add role and intermediate_dir to kwargs
        full_kwargs = {
            **kwargs,
            'role': role,
            'intermediate_dir': intermediate_dir,
        }
        if context.models_dir:
            full_kwargs['models_dir'] = context.models_dir
        
        if transcriber.has_builtin_alignment:
            # Transcriber has built-in alignment
            segments = transcriber.transcribe_with_alignment(
                audio_path, device=device, **full_kwargs
            )
            
            # Handle chunked output
            if isinstance(segments, list) and segments and isinstance(segments[0], dict) and "chunk_id" in segments[0]:
                log_progress(f"Received chunked output with timestamps, {len(segments)} chunks")
                
                if intermediate_dir:
                    chunk_file = Path(intermediate_dir) / "transcription_alignment" / f"{base_name}raw_chunks_timestamped.json"
                    chunk_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        json.dump(segments, f, indent=2, ensure_ascii=False)
                    log_intermediate_save(str(chunk_file), "Raw timestamped chunks saved to")
                
                log_progress("Stitching chunks with timestamps using overlap detection")
                segments = stitch_chunks(segments, **full_kwargs)
            
            # Save word segments
            if intermediate_dir:
                registry = kwargs.get('registry')
                if registry:
                    json_word_writer = registry.get_word_writer("word-segments-json")
                    word_file = Path(intermediate_dir) / "transcription_alignment" / f"{base_name}word_segments.json"
                    word_file.parent.mkdir(parents=True, exist_ok=True)
                    json_word_writer.write(segments, word_file)
                    log_intermediate_save(str(word_file), "Word segments saved to")
        else:
            # Use transcriber + aligner composition
            transcript_result = transcriber.transcribe(audio_path, device=device, **full_kwargs)
            
            # Handle chunked output
            if isinstance(transcript_result, list):
                log_progress(f"Received chunked output with {len(transcript_result)} chunks")
                
                if intermediate_dir:
                    serializable_chunks = []
                    for chunk in transcript_result:
                        words = chunk["words"]
                        if words and isinstance(words[0], dict):
                            serializable_chunks.append(chunk)
                        else:
                            serializable_chunks.append({"chunk_id": chunk["chunk_id"], "words": list(words)})
                    
                    chunk_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_chunks.json"
                    chunk_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(chunk_file, "w", encoding="utf-8") as f:
                        json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
                    log_intermediate_save(str(chunk_file), "Raw chunks saved to")
                
                log_progress("Stitching chunks using overlap detection")
                transcript = stitch_chunks(transcript_result, **full_kwargs)
            else:
                transcript = transcript_result
            
            # Save raw transcript
            if intermediate_dir:
                transcript_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_transcript.txt"
                transcript_file.parent.mkdir(parents=True, exist_ok=True)
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(str(transcript))
                log_intermediate_save(str(transcript_file), "Raw transcript saved to")
            
            # Align
            if aligner is None:
                raise StageError(self.name, "Aligner required but not available")
            
            segments = aligner.align_transcript(audio_path, transcript, device=device, **full_kwargs)
            
            # Save word segments
            if intermediate_dir:
                registry = kwargs.get('registry')
                if registry:
                    json_word_writer = registry.get_word_writer("word-segments-json")
                    word_file = Path(intermediate_dir) / "alignment" / f"{base_name}word_segments.json"
                    word_file.parent.mkdir(parents=True, exist_ok=True)
                    json_word_writer.write(segments, word_file)
                    log_intermediate_save(str(word_file), "Word segments saved to")
        
        return segments if isinstance(segments, list) else [segments]


class DeIdentificationStage(PipelineStage):
    """Stage for de-identifying transcripts by replacing personal names."""
    
    @property
    def name(self) -> str:
        return "de_identification"
    
    @property
    def description(self) -> str:
        return "Remove personal names and identifiers"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["word_segments"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["word_segments", "speaker_word_segments"]  # Modified in place
    
    @property
    def is_optional(self) -> bool:
        return True
    
    def can_execute(self, context: PipelineContext) -> tuple[bool, str]:
        # Check base requirements first
        can_run, reason = super().can_execute(context)
        if not can_run:
            return can_run, reason
        
        # Only run if de_identify flag is set
        if not getattr(context.args, 'de_identify', False):
            return False, "De-identification not enabled (use --de-identify flag)"
        
        return True, ""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.processing.de_identification import DeIdentificationOrchestrator
        
        args = context.args
        intermediate_dir = context.get_intermediate_dir()
        registry = context.api.get("registry")
        
        orchestrator = DeIdentificationOrchestrator(
            llm_url=getattr(args, 'llm_de_identifier_url', 'http://0.0.0.0:8080'),
            intermediate_dir=intermediate_dir
        )
        
        if context.mode == "single_speaker_audio":
            # Text-based de-identification for single speaker
            log_progress("De-identifying transcript (text mode)")
            # For single speaker mode, word_segments holds text as a string
            transcript_text = str(context.word_segments) if context.word_segments else ""
            deidentified_text = orchestrator.de_identify_text(transcript_text)
            # Store de-identified text - this mode doesn't use word segments
            context.deidentified_text = deidentified_text
            log_progress("De-identification complete")
            
        elif context.mode == "combined_audio":
            # Word segment de-identification
            log_progress("De-identifying word segments")
            
            if context.word_segments is None:
                raise StageError(self.name, "No word segments available for de-identification")
            
            result = orchestrator.de_identify(context.word_segments, speaker_name=None)
            context.word_segments = list(result.segments)
            
            # Save de-identified segments
            if intermediate_dir and registry:
                json_word_writer = registry.get_word_writer("word-segments-json")
                deidentified_file = intermediate_dir / "de_identification" / "word_segments_deidentified.json"
                deidentified_file.parent.mkdir(parents=True, exist_ok=True)
                json_word_writer.write(context.word_segments, deidentified_file)
                log_intermediate_save(str(deidentified_file), "De-identified word segments saved to")
            
            log_progress(
                f"De-identification complete: {result.total_replacements} names replaced "
                f"({len(result.discovered_names)} unique)"
            )
            
        else:
            # Multi-speaker de-identification (split_audio mode)
            log_status("Starting de-identification across all speakers")
            
            if context.speaker_word_segments is None:
                raise StageError(self.name, "No speaker word segments available")
            
            results = orchestrator.de_identify_multi_speaker(context.speaker_word_segments)
            
            # Update segments with de-identified versions
            all_words = []
            
            for speaker_name, result in results.items():
                context.speaker_word_segments[speaker_name] = result.segments
                all_words.extend(result.segments)
                
                # Save de-identified segments
                if intermediate_dir and registry:
                    json_word_writer = registry.get_word_writer("word-segments-json")
                    deidentified_file = intermediate_dir / "de_identification" / f"{speaker_name.lower()}_word_segments_deidentified.json"
                    deidentified_file.parent.mkdir(parents=True, exist_ok=True)
                    json_word_writer.write(result.segments, deidentified_file)
                    log_intermediate_save(str(deidentified_file), f"De-identified segments saved for {speaker_name}")
                
                log_progress(
                    f"De-identification for {speaker_name}: {result.total_replacements} names replaced"
                )
            
            context.word_segments = all_words
            log_status("De-identification complete for all speakers")
        
        return context


class DiarizationStage(PipelineStage):
    """Stage for assigning speaker labels to word segments."""
    
    @property
    def name(self) -> str:
        return "diarization"
    
    @property
    def description(self) -> str:
        return "Assign speaker labels to word segments"
    
    @property
    def required_inputs(self) -> List[str]:
        return ["word_segments", "diarization_provider", "standardized_audio"]
    
    @property
    def produces_outputs(self) -> List[str]:
        return ["diarized_segments"]
    
    @property
    def applicable_modes(self) -> List[str]:
        # Only applies to combined_audio mode - split_audio already has speaker labels
        return ["combined_audio"]
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        from local_transcribe.lib.system_capability_utils import get_system_capability
        
        args = context.args
        diarization_provider = context.diarization_provider
        intermediate_dir = context.get_intermediate_dir()
        registry = context.api.get("registry")
        device = get_system_capability()
        
        if diarization_provider is None:
            raise StageError(self.name, "Diarization provider not available")
        
        if context.standardized_audio is None:
            raise StageError(self.name, "No standardized audio available")
        
        log_progress(f"Diarizing with {args.num_speakers} expected speakers")
        
        diarized_segments = diarization_provider.diarize(
            str(context.standardized_audio),
            context.word_segments,
            args.num_speakers,
            device=device,
            models_dir=context.models_dir
        )
        
        context.diarized_segments = diarized_segments
        
        # Save diarized segments
        if intermediate_dir and registry:
            json_word_writer = registry.get_word_writer("word-segments-json")
            diarization_file = intermediate_dir / "diarization" / "diarized_word_segments.json"
            diarization_file.parent.mkdir(parents=True, exist_ok=True)
            json_word_writer.write(diarized_segments, diarization_file)
            log_intermediate_save(str(diarization_file), "Diarized word segments saved to")
        
        log_completion(f"Diarization complete: {len(diarized_segments)} segments")
        return context
