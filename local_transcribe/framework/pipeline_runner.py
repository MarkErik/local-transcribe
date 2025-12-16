#!/usr/bin/env python3
# framework/pipeline_runner.py - Pipeline execution and processing logic

import os
import sys
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

from local_transcribe.framework.model_downloader import ensure_models_available
from local_transcribe.framework.provider_setup import ProviderSetup
from local_transcribe.framework.output_manager import OutputManager
from local_transcribe.lib.program_logger import log_status, log_progress, log_intermediate_save, log_completion
from local_transcribe.lib.speaker_namer import assign_speaker_names
from local_transcribe.lib.audio_processor import standardize_audio, cleanup_temp_audio
from local_transcribe.processing.pre_LLM_transcript_preparation import prepare_transcript_for_llm
from local_transcribe.processing.llm_de_identifier import de_identify_word_segments, de_identify_text, DeIdentificationResult
from local_transcribe.processing.turn_building import TranscriptFlow
from local_transcribe.processing.llm_second_pass_de_identifier import (
    second_pass_de_identify,
    build_global_name_list,
    DiscoveredName
)
from local_transcribe.processing.turn_building import build_turns

def transcribe_with_alignment(transcriber_provider, aligner_provider, audio_path: str, role: Optional[str], intermediate_dir: Optional[Union[str, os.PathLike]] = None, base_name: str = "", models_dir: Optional[Union[str, os.PathLike]] = None, **kwargs) -> List[Any]:
    """Transcribe audio and return word segments with timestamps."""
    from local_transcribe.lib.system_capability_utils import get_system_capability
    
    # Add role to kwargs so it's passed to providers
    kwargs['role'] = role
    kwargs['intermediate_dir'] = intermediate_dir
    if models_dir:
        kwargs['models_dir'] = models_dir
    
    # Get device from global config to pass explicitly
    device = get_system_capability()
    
    if transcriber_provider.has_builtin_alignment:
        # Transcriber has built-in alignment
        segments = transcriber_provider.transcribe_with_alignment(
            audio_path,
            device=device,
            **kwargs
        )
        
        # Check if segments are chunked (for granite_mfa and similar plugins)
        if isinstance(segments, list) and segments and isinstance(segments[0], dict) and "chunk_id" in segments[0]:
            # Chunked output with timestamps - need to stitch
            log_progress(f"Received chunked output with timestamps from {transcriber_provider.name}, {len(segments)} chunks")
            
            # Save chunked data
            if intermediate_dir:
                import json
                # Chunks already have serializable format (dicts with text/start/end)
                chunk_file = Path(intermediate_dir) / "transcription_alignment" / f"{base_name}raw_chunks_timestamped.json"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2, ensure_ascii=False)
                log_intermediate_save(str(chunk_file), "Raw timestamped chunks saved to")
            
            # Use local_chunk_stitcher (which now handles timestamped words)
            from local_transcribe.processing.local_chunk_stitcher import stitch_chunks
            log_progress("Stitching chunks with timestamps using intelligent local overlap detection")
            segments = stitch_chunks(segments, **kwargs)
            # Now segments is List[WordSegment]
        
        # Save word segments
        if intermediate_dir:
            registry = kwargs.get('registry')
            if registry is None:
                raise ValueError("Registry not found in kwargs")
            json_word_writer = registry.get_word_writer("word-segments-json")
            word_file = Path(intermediate_dir) / "transcription_alignment" / f"{base_name}word_segments.json"
            json_word_writer.write(segments, word_file)
            log_intermediate_save(str(word_file), "Word segments saved to")
    else:
        # Use transcriber + aligner composition
        transcript_result = transcriber_provider.transcribe(audio_path, device=device, **kwargs)
        
        # Handle chunked output (list of dicts) vs simple string output
        if isinstance(transcript_result, list):
            # Chunked output - need to stitch
            log_progress(f"Received chunked output with {len(transcript_result)} chunks")
            
            # Save chunked data
            if intermediate_dir:
                import json
                # Handle both string words and dict words (timestamped)
                serializable_chunks = []
                for chunk in transcript_result:
                    words = chunk["words"]
                    if words and isinstance(words[0], dict):
                        # Already serializable (timestamps included)
                        serializable_chunks.append(chunk)
                    else:
                        # String words (convert to list for JSON)
                        serializable_chunks.append({"chunk_id": chunk["chunk_id"], "words": list(words)})
                
                chunk_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_chunks.json"
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
                log_intermediate_save(str(chunk_file), "Raw chunks saved to")
            
            # Use intelligent local chunk stitching
            from local_transcribe.processing.local_chunk_stitcher import stitch_chunks
            log_progress("Stitching chunks using intelligent local overlap detection")
            transcript = stitch_chunks(transcript_result, **kwargs)
        else:
            # Simple string output - no stitching needed
            transcript = transcript_result
        
        # Save raw transcript
        if intermediate_dir:
            transcript_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(str(transcript))
            log_intermediate_save(str(transcript_file), "Raw transcript saved to")
        
        # Pass role to aligner via kwargs (already added above)
        segments = aligner_provider.align_transcript(audio_path, transcript, device=device, **kwargs)
        
        # Save word segments (speaker should already be assigned by aligner)
        if intermediate_dir:
            registry = kwargs.get('registry')
            if registry is None:
                raise ValueError("Registry not found in kwargs")
            json_word_writer = registry.get_word_writer("word-segments-json")
            word_file = Path(intermediate_dir) / "alignment" / f"{base_name}word_segments.json"
            json_word_writer.write(segments, word_file)
            log_intermediate_save(str(word_file), "Word segments saved to")
    
    return segments if isinstance(segments, list) else [segments]

def only_transcribe(transcriber_provider, audio_path: str, role: Optional[str], intermediate_dir: Optional[Union[str, os.PathLike]] = None, base_name: str = "", **kwargs) -> str:
    """Transcribe audio and return transcript text only (no alignment)."""
    from local_transcribe.lib.system_capability_utils import get_system_capability
    
    # Remove role from kwargs to avoid duplicates (it's an explicit param)
    kwargs.pop('role', None)
    
    # Get device from global config to pass explicitly
    device = get_system_capability()
    
    # Transcribe without alignment
    transcript = transcriber_provider.transcribe(audio_path, device=device, **kwargs)
    
    # Handle chunked output (list of dicts) vs simple string output
    if isinstance(transcript, list):
        # Chunked output - need to stitch
        log_progress(f"Received chunked output with {len(transcript)} chunks")
        
        # Use intelligent local chunk stitching
        from local_transcribe.processing.local_chunk_stitcher import stitch_chunks
        log_progress("Stitching chunks using intelligent local overlap detection")
        transcript_text = stitch_chunks(transcript, **kwargs)
        
        # Save chunked data
        if intermediate_dir:
            import json
            # Handle both string words and dict words
            serializable_chunks = []
            for chunk in transcript:
                words = chunk["words"]
                if words and isinstance(words[0], dict):
                    # Already serializable (timestamps included)
                    serializable_chunks.append(chunk)
                else:
                    # String words
                    serializable_chunks.append({"chunk_id": chunk["chunk_id"], "words": list(words)})
            
            chunk_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_chunks.json"
            with open(chunk_file, "w", encoding="utf-8") as f:
                json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)
            log_intermediate_save(str(chunk_file), "Raw chunks saved to")
    else:
        # Simple string output
        transcript_text = transcript
    
    # Save final stitched transcript
    if intermediate_dir:
        transcript_file = Path(intermediate_dir) / "transcription" / f"{base_name}raw_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(str(transcript_text))
        log_intermediate_save(str(transcript_file), "Raw transcript saved to")
    
    return str(transcript_text)

def run_pipeline(args, api: Dict[str, Any], root: Union[str, os.PathLike]) -> int:
    """Main pipeline execution function.
    
    Args:
        args: Command line arguments
        api: API dictionary with registry and other services
        root: Root directory path
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Early return if no audio files provided
    if not hasattr(args, 'audio_files') or not args.audio_files:
        print("ERROR: No audio files provided.")
        return 1
    
    # Initialize return value
    return_code = 0
    
    from local_transcribe.lib.environment import ensure_file, ensure_outdir

    models_dir = Path(root) / ".models"

    # Determine mode and speaker mapping
    if hasattr(args, 'single_speaker_audio') and args.single_speaker_audio:
        if len(args.audio_files) != 1:
            print("ERROR: Single speaker audio mode requires exactly one audio file.")
            return 1
        mode = "single_speaker_audio"
        speaker_files = {"speaker": str(root / args.audio_files[0])}
    else:
        num_files = len(args.audio_files)
        if num_files == 1:
            mode = "combined_audio"
            speaker_files = {"combined_audio": str(root / args.audio_files[0])}  # Single file with multiple speakers
        else:
            mode = "split_audio"
            speaker_files = {}
            
            if num_files == 2:
                # Auto-assign: first file = interviewer, second = participant
                speaker_files["Interviewer"] = str(root / args.audio_files[0])
                speaker_files["Participant"] = str(root / args.audio_files[1])
            else:
                # 3+ files: prompt for speaker names
                print(f"You provided {num_files} audio files. Please assign a speaker name to each:")
                for i, audio_file in enumerate(args.audio_files):
                    while True:
                        speaker_name = input(f"Speaker name for '{audio_file}': ").strip()
                        if speaker_name:
                            speaker_files[speaker_name] = str(root / audio_file)
                            break
                        else:
                            print("Speaker name cannot be empty.")

    # Set default num_speakers if not provided
    if not hasattr(args, 'num_speakers') or args.num_speakers is None:
        if mode == "combined_audio":
            args.num_speakers = 2  # Default for single file mode
        else:
            args.num_speakers = len(speaker_files)  # One speaker per file

    # Set default outputs for non-interactive
    if not hasattr(args, 'selected_outputs') or not args.selected_outputs:
        if mode == "single_speaker_audio":
            # Single speaker audio mode uses custom CSV output
            args.selected_outputs = ['csv']
        elif args.only_final_transcript:
            args.selected_outputs = ['timestamped-txt']
        else:
            # Include JSON outputs for debugging alignment
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            all_writers = list(registry.list_output_writers().keys())
            print(f"[i] Available output writers: {all_writers}")
            args.selected_outputs = all_writers

    # Validate audio files exist
    for speaker, audio_file in speaker_files.items():
        try:
            ensure_file(audio_file, speaker)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    # Early validation for single_speaker_audio mode
    if mode == "single_speaker_audio" and hasattr(args, 'transcriber_provider') and args.transcriber_provider:
        try:
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            temp_provider = registry.get_transcriber_provider(args.transcriber_provider)
            if temp_provider.has_builtin_alignment:
                print(f"ERROR: Provider '{args.transcriber_provider}' has built-in alignment and is not allowed in single-speaker-audio mode.")
                print("       Use granite or openai_whisper for this mode.")
                print("Use --list-plugins to see available options.")
                return 1
        except ValueError:
            pass  # Will be caught in provider setup below

    # Setup providers using ProviderSetup class
    try:
        registry = api.get("registry")
        if registry is None:
            raise ValueError("Registry not found in api")
        provider_setup = ProviderSetup(registry, args)
        providers = provider_setup.setup_providers(mode)
        
        # Extract individual providers for easier access
        transcriber_provider = providers.get('transcriber')
        aligner_provider = providers.get('aligner')
        diarization_provider = providers.get('diarization')
        transcript_cleanup_provider = providers.get('transcript_cleanup')
        
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1

    # Download required models for selected providers
    model_download_providers = provider_setup.get_model_download_providers()

    download_result = ensure_models_available(model_download_providers, models_dir, args)
    if download_result != 0:
        return download_result

    # Configure logging based on log level
    api["configure_global_logging"](log_level=args.log_level)

    # Compute capabilities for directory creation
    capabilities = {
        "mode": mode,
        "has_builtin_alignment": transcriber_provider.has_builtin_alignment if transcriber_provider else False,
        "aligner": aligner_provider is not None,
        "diarization": diarization_provider is not None
    }

    # Modify output directory name if DEBUG flag is set
    if args.log_level == "DEBUG":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        transcriber_name = transcriber_provider.name if transcriber_provider else "unknown"
        args.outdir = f"{args.outdir}_{transcriber_name}_{timestamp}"

    # Ensure outdir & subdirs
    outdir = ensure_outdir(args.outdir)
    ensure_session_dirs = api.get("ensure_session_dirs")
    if ensure_session_dirs is None:
        raise ValueError("ensure_session_dirs not found in api")
    paths = ensure_session_dirs(outdir, mode, speaker_files, capabilities)

    # Write settings to file if DEBUG log level is set
    if args.log_level == "DEBUG":
        from local_transcribe.lib.system_capability_utils import get_system_capability
        settings_path = os.path.join(outdir, "settings.txt")
        with open(settings_path, 'w') as f:
            f.write("Local-Transcribe Settings\n")
            f.write("=" * 30 + "\n\n")
            
            # Write all command line arguments
            f.write("Command Line Arguments:\n")
            f.write("-" * 25 + "\n")
            for key, value in vars(args).items():
                if key not in ['audio_files', 'outdir']:  # Skip these as they can be long
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Write provider information
            f.write("Selected Providers:\n")
            f.write("-" * 20 + "\n")
            if transcriber_provider:
                f.write(f"Transcriber: {transcriber_provider.name}\n")
                if hasattr(transcriber_provider, 'model') and transcriber_provider.model:
                    f.write(f"Transcriber Model: {transcriber_provider.model}\n")
                elif hasattr(args, 'transcriber_model') and args.transcriber_model:
                    f.write(f"Transcriber Model: {args.transcriber_model}\n")
            if aligner_provider:
                f.write(f"Aligner: {aligner_provider.name}\n")
            if diarization_provider:
                f.write(f"Diarization: {diarization_provider.name}\n")
            if transcript_cleanup_provider:
                f.write(f"Transcript Cleanup: {transcript_cleanup_provider.name}\n")
            f.write("\n")
            
            # Write processing mode
            f.write("Processing Mode:\n")
            f.write("-" * 17 + "\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"System Capability: {get_system_capability()}\n")
            f.write(f"Number of Speakers: {args.num_speakers}\n")
            f.write(f"Selected Outputs: {', '.join(args.selected_outputs)}\n")
            f.write("\n")
            
            # Write audio files info
            f.write("Audio Files:\n")
            f.write("-" * 13 + "\n")
            for speaker, path in speaker_files.items():
                f.write(f"{speaker}: {os.path.basename(path)}\n")
        
        print(f"[DEBUG] Settings written to {settings_path}")

    if mode == "single_speaker_audio":
        log_status(f"Mode: {mode} | System: {args.system.upper()} | Transcriber: {args.transcriber_provider} | Outputs: CSV")
    else:
        provider_info = []
        if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
            provider_info.append(f"Transcriber: {args.transcriber_provider}")
        if hasattr(args, 'aligner_provider') and args.aligner_provider:
            provider_info.append(f"Aligner: {args.aligner_provider}")
        if hasattr(args, 'diarization_provider') and args.diarization_provider:
            provider_info.append(f"Diarization: {args.diarization_provider}")
        provider_str = " | ".join(provider_info) if provider_info else "Default providers"
        from local_transcribe.lib.system_capability_utils import get_system_capability
        log_status(f"Mode: {mode} | System: {get_system_capability().upper()} | {provider_str} | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if mode == "single_speaker_audio":
            speaker_path = ensure_file(speaker_files["speaker"], "Single Speaker Audio")

            # 1) Standardize
            std_audio = standardize_audio(str(speaker_path), outdir, api)

            # 2) Transcribe only
            kwargs = vars(args).copy()
            # Remove parameters that are already explicit arguments
            kwargs.pop('transcriber_provider', None)
            kwargs.pop('audio_files', None)
            kwargs.pop('outdir', None)
            kwargs.pop('interactive', None)
            kwargs.pop('list_plugins', None)
            kwargs.pop('show_defaults', None)
            kwargs.pop('system', None)
            kwargs.pop('single_speaker_audio', None)
            
            transcript = only_transcribe(
                transcriber_provider,
                str(std_audio),
                "speaker",
                paths["intermediate"],
                "",
                **kwargs
            )

            # 2.5) De-identification (if enabled)
            if args.de_identify:
                log_progress("De-identifying transcript (text mode)")
                transcript = de_identify_text(
                    transcript,
                    intermediate_dir=paths["intermediate"],
                    llm_url=args.llm_de_identifier_url
                )
                log_progress("De-identification complete")

            # 3) Process transcript into words and save as CSV
            import csv
            words = transcript.split()
            csv_path = outdir / "transcript.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Line', 'Word'])
                for i, word in enumerate(words, 1):
                    writer.writerow([i, word])
            
            print(f"[✓] Transcript saved to {csv_path}")
            print(f"[i] Artifacts written to: {paths['root']}")
            
            # Clean up temporary audio files
            cleanup_temp_audio(outdir)
            print("[✓] Temporary audio files cleaned up.")
            
            return 0

        elif mode == "combined_audio":
            mixed_path = ensure_file(speaker_files["combined_audio"], "Combined Audio")

            # 1) Standardize
            std_audio = standardize_audio(str(mixed_path), outdir, api)

            # 2) Transcription + alignment
            words = transcribe_with_alignment(
                transcriber_provider,
                aligner_provider,
                str(std_audio),
                None,
                intermediate_dir=paths.get("intermediate"),
                base_name="",
                models_dir=models_dir,
                registry=api["registry"],
                transcriber_model=args.transcriber_model,
                output_format=getattr(args, 'output_format', 'stitched')
            )

            # 2.5) De-identification (if enabled) - BEFORE diarization
            if args.de_identify:
                log_progress("De-identifying word segments")
                
                # For combined_audio, second pass is less useful since there's only one transcript
                # But we still support it for consistency
                use_second_pass = getattr(args, 'de_identify_second_pass', False)
                
                result = de_identify_word_segments(
                    words,
                    intermediate_dir=paths["intermediate"],
                    llm_url=args.llm_de_identifier_url,
                    return_result_object=True,
                    skip_audit_log=use_second_pass
                )
                if isinstance(result, DeIdentificationResult):
                    words = result.segments
                    first_pass_replacements = result.replacements
                    log_progress(f"First pass complete: {len(result.discovered_names)} unique names found")
                
                # Save first-pass de-identified word segments
                registry = api.get("registry")
                if registry is None:
                    raise ValueError("Registry not found in api")
                json_word_writer = registry.get_word_writer("word-segments-json")
                if use_second_pass:
                    deidentified_file = paths["intermediate"] / "de_identification" / "word_segments_first_pass.json"
                else:
                    deidentified_file = paths["intermediate"] / "de_identification" / "word_segments_deidentified.json"
                json_word_writer.write(words, deidentified_file)
                log_intermediate_save(str(deidentified_file), "De-identified word segments saved to")
                
                # Second pass (if enabled)
                if use_second_pass and isinstance(result, DeIdentificationResult) and result.discovered_names:
                    log_progress("Running second pass on combined audio (single transcript mode)")
                    
                    # Build name list from first pass
                    global_names = build_global_name_list({"combined": first_pass_replacements})
                    
                    if global_names:
                        second_pass_result = second_pass_de_identify(
                            list(words),  # type: ignore
                            global_names,
                            intermediate_dir=paths["intermediate"],
                            llm_url=args.llm_de_identifier_url,
                            speaker_name=None,
                            first_pass_replacements=first_pass_replacements
                        )
                        
                        words = list(second_pass_result.segments)  # type: ignore
                        
                        # Save final de-identified segments
                        final_file = paths["intermediate"] / "de_identification" / "word_segments_deidentified.json"
                        json_word_writer.write(words, final_file)
                        log_intermediate_save(str(final_file), "Final de-identified segments saved")
                        
                        if second_pass_result.additional_replacements:
                            log_progress(f"Second pass found {len(second_pass_result.additional_replacements)} additional names")
                elif use_second_pass:
                    log_progress("No names discovered in first pass, skipping second pass")
                
                log_progress("De-identification complete")

            # 3) Diarize (assign speakers to words)
            from local_transcribe.lib.system_capability_utils import get_system_capability
            device = get_system_capability()
            
            if diarization_provider is None:
                raise ValueError("Diarization provider is not available")
            words_with_speakers = diarization_provider.diarize(
                str(std_audio), 
                words, 
                args.num_speakers,
                device=device,
                models_dir=models_dir
            )
            # Save diarized segments
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            json_word_writer = registry.get_word_writer("word-segments-json")
            diarization_file = paths["intermediate"] / "diarization" / "diarized_word_segments.json"
            json_word_writer.write(words_with_speakers, diarization_file)
            log_intermediate_save(str(diarization_file), "Diarized word segments saved to")

            # 4) Build turns
            turn_kwargs = {'intermediate_dir': paths["intermediate"]}
            transcript = build_turns(words_with_speakers, mode=mode, **turn_kwargs)  # type: ignore
            # Save raw turns
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            json_turns_writer = registry.get_output_writer("turns-json")
            turns_file = paths["intermediate"] / "turns" / "raw_turns.json"
            json_turns_writer.write(transcript, turns_file)
            log_intermediate_save(str(turns_file), "Raw turns saved to")

            # Assign speaker names if interactive
            transcript = assign_speaker_names(transcript, getattr(args, 'interactive', False), mode)

            # Write raw outputs including video (this is the only place video should be generated)
            raw_dir = paths["root"] / "Transcript_Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            log_status(f"Writing raw outputs to {raw_dir}")
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            output_manager = OutputManager.get_instance(registry)
            # For combined_audio mode, pass the single standardized audio file
            audio_config = std_audio if mode == "combined_audio" else None
            log_progress(f"Writing raw outputs with formats: {args.selected_outputs}")
            if output_manager is None:
                raise ValueError("Output manager is not available")
            output_manager.write_selected_outputs(transcript, {**paths, "merged": raw_dir}, args.selected_outputs, audio_config, generate_video=True, word_segments=words_with_speakers)

            # 5) Prepare transcript for LLM processing
            log_status("Preparing transcript for LLM processing")
            try:
                prep_result = prepare_transcript_for_llm(
                    transcript,
                    max_words_per_segment=getattr(args, 'max_words_per_segment', 500),
                    preparation_mode=getattr(args, 'preparation_mode', 'basic'),
                    standardize_speakers=getattr(args, 'standardize_speakers', True),
                    normalize_whitespace=getattr(args, 'normalize_whitespace', True),
                    handle_special_chars=getattr(args, 'handle_special_chars', True)
                )
                
                # Update transcript with processed turns
                transcript = prep_result['turns']
                
                log_completion(f"Transcript preparation complete: {prep_result['stats']['segments_created']} segments created", {
                    "original_turns": prep_result['stats']['original_turns'],
                    "words_processed": prep_result['stats']['words_processed'],
                    "turns_split": prep_result['stats']['turns_split']
                })
            except Exception as e:
                log_status(f"Warning: Error during transcript preparation: {str(e)}", "WARNING")
                log_progress("Continuing with original transcript")

            # 6) Optional transcript LLM-based cleanup
            if transcript_cleanup_provider:
                log_status(f"Cleaning up transcript with {args.transcript_cleanup_provider}")
                log_progress(f"Processing {len(prep_result['segments'])} segments (max {getattr(args, 'max_words_per_segment', 500)} words each)")
                
                # Process each segment through LLM
                cleaned_segments = []
                for idx, segment in enumerate(prep_result['segments']):
                    log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Processing: {segment[:60]}...")
                    cleaned = transcript_cleanup_provider.transcript_cleanup_segment(segment)
                    cleaned_segments.append(cleaned)
                    log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Cleaned: {cleaned[:60]}...")
                
                # Write cleaned transcript to processed directory
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                # Write as plain text (no timestamps needed for cleaned transcript)
                cleaned_text_file = processed_dir / "transcript_cleaned.txt"
                cleaned_text_file.write_text('\n\n'.join(cleaned_segments) + '\n', encoding='utf-8')
                
                log_completion(f"Transcript cleanup complete: {cleaned_text_file}")
                log_progress("Raw transcript with timestamps available in Transcript_Raw/")
            else:
                log_progress("No transcript cleanup selected, raw outputs already written.")

            print(f"[i] Artifacts written to: {paths['root']}")

            print("[✓] Single file processing complete.")

        else:
            # Process separate audio files
            all_words = []
            
            # For two-pass de-identification, we need to collect data per speaker
            speaker_segments = {}  # speaker_name -> word segments
            speaker_replacements = {}  # speaker_name -> first-pass replacements
            
            for speaker_name, audio_file in speaker_files.items():
                print(f"[*] Processing {speaker_name}...")
                
                audio_path = ensure_file(audio_file, speaker_name)
                
                # 1) Standardize
                std_audio = standardize_audio(str(audio_path), outdir, api, speaker_name)
                
                # 2) ASR + alignment
                print(f"[*] Performing transcription and alignment for {speaker_name}...")
                words = transcribe_with_alignment(
                    transcriber_provider,
                    aligner_provider,
                    str(std_audio),
                    speaker_name,
                    intermediate_dir=paths.get("intermediate"),
                    base_name=f"{speaker_name.lower()}_",
                    registry=api["registry"],
                    transcriber_model=args.transcriber_model,
                    output_format=getattr(args, 'output_format', 'stitched')
                )
                
                # 2.5) De-identification FIRST PASS per speaker (if enabled)
                if args.de_identify:
                    log_progress(f"De-identifying word segments for {speaker_name} (first pass)")
                    
                    # Use return_result_object to get replacements for second pass
                    # Skip individual audit log if second pass is enabled (combined log will be created)
                    skip_audit = getattr(args, 'de_identify_second_pass', False)
                    
                    result = de_identify_word_segments(
                        words,
                        intermediate_dir=paths["intermediate"],
                        llm_url=args.llm_de_identifier_url,
                        speaker_name=speaker_name,
                        return_result_object=True,
                        skip_audit_log=skip_audit
                    )
                    
                    if isinstance(result, DeIdentificationResult):
                        words = result.segments
                        speaker_replacements[speaker_name] = result.replacements
                        log_progress(f"First pass complete for {speaker_name}: {len(result.discovered_names)} unique names found")
                    
                    # Save first-pass de-identified word segments per speaker
                    registry = api.get("registry")
                    if registry is None:
                        raise ValueError("Registry not found in api")
                    json_word_writer = registry.get_word_writer("word-segments-json")
                    deidentified_file = paths["intermediate"] / "de_identification" / f"{speaker_name.lower()}_word_segments_first_pass.json"
                    json_word_writer.write(words, deidentified_file)
                    log_intermediate_save(str(deidentified_file), f"First-pass de-identified segments saved for {speaker_name}")
                
                # Store segments for potential second pass
                speaker_segments[speaker_name] = words
                
                # Add words to the combined list
                all_words.extend(words)
            
            # 2.6) De-identification SECOND PASS (if enabled)
            if args.de_identify and getattr(args, 'de_identify_second_pass', False):
                log_status("Starting second-pass de-identification across all speakers")
                
                # Build global name list from all speakers' first-pass results
                global_names = build_global_name_list(speaker_replacements)
                
                if global_names:
                    log_progress(f"Global name list: {', '.join(n.name for n in global_names)}")
                    
                    # Clear all_words to rebuild with second-pass results
                    all_words = []
                    
                    for speaker_name in speaker_files.keys():
                        log_progress(f"Running second pass for {speaker_name}")
                        
                        first_pass_segments = speaker_segments[speaker_name]
                        first_pass_reps = speaker_replacements.get(speaker_name, [])
                        
                        second_pass_result = second_pass_de_identify(
                            list(first_pass_segments),  # type: ignore
                            global_names,
                            intermediate_dir=paths["intermediate"],
                            llm_url=args.llm_de_identifier_url,
                            speaker_name=speaker_name,
                            first_pass_replacements=first_pass_reps
                        )
                        
                        # Update speaker segments with final de-identified version
                        speaker_segments[speaker_name] = second_pass_result.segments
                        
                        # Save final de-identified segments
                        registry = api.get("registry")
                        if registry is None:
                            raise ValueError("Registry not found in api")
                        json_word_writer = registry.get_word_writer("word-segments-json")
                        final_file = paths["intermediate"] / "de_identification" / f"{speaker_name.lower()}_word_segments_deidentified.json"
                        json_word_writer.write(second_pass_result.segments, final_file)
                        log_intermediate_save(str(final_file), f"Final de-identified segments saved for {speaker_name}")
                        
                        # Add to combined list
                        all_words.extend(second_pass_result.segments)
                        
                        if second_pass_result.additional_replacements:
                            log_progress(f"Second pass found {len(second_pass_result.additional_replacements)} additional names for {speaker_name}")
                    
                    log_status("Second-pass de-identification complete")
                else:
                    log_progress("No names discovered in first pass, skipping second pass")
            
            # 3) Build and merge turns using the turn building processor
            turn_kwargs = {'intermediate_dir': paths["intermediate"]}
            transcript = build_turns(all_words, mode=mode, **turn_kwargs)  # type: ignore
            
            # Save merged turns
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            json_turns_writer = registry.get_output_writer("turns-json")
            merged_file = paths["intermediate"] / "turns" / "merged_turns.json"
            json_turns_writer.write(transcript, merged_file)
            log_intermediate_save(str(merged_file), "Merged turns saved to")

            # Write raw outputs including video (this is the only place video should be generated)
            raw_dir = paths["root"] / "Transcript_Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            log_status(f"Writing raw outputs to {raw_dir}")
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            output_manager = OutputManager.get_instance(registry)
            
            # For split audio mode, pass the speaker_files dictionary for video generation
            audio_config = speaker_files if mode == "split_audio" else None
            
            print(f"[i] Writing raw outputs with formats: {args.selected_outputs}")
            if output_manager is None:
                raise ValueError("Output manager is not available")
            # Cast transcript to List[Any] to satisfy type checker
            # Use type: ignore to suppress the type checker error for TranscriptFlow
            transcript_list = transcript  # type: ignore
            output_manager.write_selected_outputs(
                transcript_list,  # type: ignore
                {**paths, "merged": raw_dir},
                args.selected_outputs,
                audio_config,
                generate_video=True,
                word_segments=all_words
            )

            # 5) Prepare transcript for LLM processing
            log_status("Preparing transcript for LLM processing")
            try:
                prep_result = prepare_transcript_for_llm(
                    transcript,
                    max_words_per_segment=getattr(args, 'max_words_per_segment', 500),
                    preparation_mode=getattr(args, 'preparation_mode', 'basic'),
                    standardize_speakers=getattr(args, 'standardize_speakers', True),
                    normalize_whitespace=getattr(args, 'normalize_whitespace', True),
                    handle_special_chars=getattr(args, 'handle_special_chars', True)
                )
                
                # Update transcript with processed turns
                transcript = prep_result['turns']
                
                log_completion(f"Transcript preparation complete: {prep_result['stats']['segments_created']} segments created", {
                    "original_turns": prep_result['stats']['original_turns'],
                    "words_processed": prep_result['stats']['words_processed'],
                    "turns_split": prep_result['stats']['turns_split']
                })
            except Exception as e:
                log_status(f"Warning: Error during transcript preparation: {str(e)}", "WARNING")
                log_progress("Continuing with original transcript")

            # 6) Optional LLM-based transcript cleanup
            if transcript_cleanup_provider:
                log_status(f"Cleaning up transcript with {args.transcript_cleanup_provider}")
                log_progress(f"Processing {len(prep_result['segments'])} segments (max {getattr(args, 'max_words_per_segment', 500)} words each)")
                
                # Process each segment through LLM
                cleaned_segments = []
                for idx, segment in enumerate(prep_result['segments']):
                    log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Processing: {segment[:60]}...")
                    cleaned = transcript_cleanup_provider.transcript_cleanup_segment(segment)
                    cleaned_segments.append(cleaned)
                    log_progress(f"[{idx+1}/{len(prep_result['segments'])}] Cleaned: {cleaned[:60]}...")
                
                # Write cleaned transcript to processed directory
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                # Write as plain text (no timestamps needed for cleaned transcript)
                cleaned_text_file = processed_dir / "transcript_cleaned.txt"
                cleaned_text_file.write_text('\n\n'.join(cleaned_segments) + '\n', encoding='utf-8')
                
                log_completion(f"Transcript cleanup complete: {cleaned_text_file}")
                log_progress("Raw transcript with timestamps available in Transcript_Raw/")
            else:
                log_progress("No transcript cleanup selected, raw outputs already written.")

            log_progress(f"Artifacts written to: {paths['root']}")
            
            log_completion("Separate audio processing complete.")

        return return_code

    # This should never be reached, but we need to satisfy the type checker
    return return_code
