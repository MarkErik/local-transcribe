#!/usr/bin/env python3
# framework/pipeline_runner.py - Pipeline execution and processing logic

import os
import sys
from typing import Optional

from local_transcribe.framework.model_downloader import ensure_models_available
from local_transcribe.framework.provider_setup import ProviderSetup
from local_transcribe.framework.output_manager import OutputManager
from local_transcribe.lib.speaker_namer import assign_speaker_names
from local_transcribe.lib.audio_processor import standardize_audio, cleanup_temp_audio
from local_transcribe.processing.pre_LLM_transcript_preparation import prepare_transcript_for_llm

def transcribe_with_alignment(transcriber_provider, aligner_provider, audio_path, role, intermediate_dir=None, verbose=False, base_name="", **kwargs):
    """Transcribe audio and return word segments with timestamps."""
    from local_transcribe.lib.config import get_system_capability
    
    # Add verbose and role to kwargs so they're passed to providers
    kwargs['verbose'] = verbose
    kwargs['role'] = role
    
    # Get device from global config to pass explicitly
    device = get_system_capability()
    
    if transcriber_provider.has_builtin_alignment:
        # Transcriber has built-in alignment
        segments = transcriber_provider.transcribe_with_alignment(
            audio_path,
            device=device,
            **kwargs
        )
        # Verbose: Save word segments
        if verbose and intermediate_dir:
            json_word_writer = kwargs.get('registry').get_word_writer("word-segments-json")
            json_word_writer.write(segments, intermediate_dir / "transcription_alignment" / f"{base_name}word_segments.json")
            print(f"[i] Verbose: Word segments saved to Intermediate_Outputs/transcription_alignment/{base_name}word_segments.json")
    else:
        # Use transcriber + aligner composition
        output_mode = kwargs.get('output_mode', 'stitched')
        transcript_result = transcriber_provider.transcribe(audio_path, device=device, **kwargs)
        
        if output_mode == 'chunked':
            # Chunked output: stitch using LLM
            from local_transcribe.processing.llm_stitcher import stitch_chunks
            transcript = stitch_chunks(transcript_result, **kwargs)
        else:
            transcript = transcript_result
        
        # Verbose: Save raw transcript
        if verbose and intermediate_dir:
            with open(intermediate_dir / "transcription" / f"{base_name}raw_transcript.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
            print(f"[i] Verbose: Raw transcript saved to Intermediate_Outputs/transcription/{base_name}raw_transcript.txt")
        # Pass role to aligner via kwargs (already added above)
        segments = aligner_provider.align_transcript(audio_path, transcript, device=device, **kwargs)
        # Verbose: Save word segments (speaker should already be assigned by aligner)
        if verbose and intermediate_dir:
            json_word_writer = kwargs.get('registry').get_word_writer("word-segments-json")
            json_word_writer.write(segments, intermediate_dir / "alignment" / f"{base_name}word_segments.json")
            print(f"[i] Verbose: Word segments saved to Intermediate_Outputs/alignment/{base_name}word_segments.json")
    return segments

def run_pipeline(args, api, root):
    from local_transcribe.lib.environment import ensure_file, ensure_outdir

    models_dir = root / "models"

    # Determine mode and speaker mapping
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
        if args.only_final_transcript:
            args.selected_outputs = ['timestamped-txt']
        else:
            # Include JSON outputs for debugging alignment
            all_writers = list(api["registry"].list_output_writers().keys())
            print(f"[i] Available output writers: {all_writers}")
            args.selected_outputs = all_writers

    # Validate audio files exist
    for speaker, audio_file in speaker_files.items():
        try:
            ensure_file(audio_file, speaker)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    # Setup providers using ProviderSetup class
    try:
        provider_setup = ProviderSetup(api["registry"], args)
        providers = provider_setup.setup_providers(mode)
        
        # Extract individual providers for easier access
        unified_provider = providers.get('unified')
        transcriber_provider = providers.get('transcriber')
        aligner_provider = providers.get('aligner')
        diarization_provider = providers.get('diarization')
        turn_builder_provider = providers.get('turn_builder')
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

    # Configure logging based on verbose flag
    api["configure_global_logging"](log_level="INFO" if args.verbose else "WARNING")

    # Compute capabilities for directory creation if the verbose flag is set
    capabilities = {
        "mode": mode,
        "unified": unified_provider is not None,
        "has_builtin_alignment": transcriber_provider.has_builtin_alignment if transcriber_provider else False,
        "aligner": aligner_provider is not None,
        "diarization": diarization_provider is not None
    }

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir, mode, speaker_files, args.verbose, capabilities)

        if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
            print(f"[*] Mode: {mode} (combined_audio) | System: {args.system.upper()} | Provider: {args.unified_provider} | Outputs: {', '.join(args.selected_outputs)}")
        else:
            provider_info = []
            if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
                provider_info.append(f"Transcriber: {args.transcriber_provider}")
            if hasattr(args, 'aligner_provider') and args.aligner_provider:
                provider_info.append(f"Aligner: {args.aligner_provider}")
            if hasattr(args, 'diarization_provider') and args.diarization_provider:
                provider_info.append(f"Diarization: {args.diarization_provider}")
            provider_str = " | ".join(provider_info) if provider_info else "Default providers"
            print(f"[*] Mode: {mode} | System: {args.system.upper()} | {provider_str} | Turn Builder: {turn_builder_provider.name} | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if mode == "combined_audio":
            mixed_path = ensure_file(speaker_files["combined_audio"], "Combined Audio")

            # 1) Standardize
            std_audio = standardize_audio(mixed_path, outdir, tracker, api)

            if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
                # Use unified provider
                from local_transcribe.lib.config import get_system_capability
                device = get_system_capability()
                
                transcript = unified_provider.transcribe_and_diarize(
                    str(std_audio),
                    args.num_speakers,
                    device=device,
                    model=args.unified_model
                )
                # Verbose: Save unified turns
                if args.verbose:
                    json_turns_writer = api["registry"].get_output_writer("turns-json")
                    json_turns_writer.write(transcript, paths["intermediate"] / "turns" / "unified_turns.json")
                    print("[i] Verbose: Unified turns saved to Intermediate_Outputs/turns/unified_turns.json")
            else:
                # 2) Transcription + alignment
                words = transcribe_with_alignment(
                    transcriber_provider,
                    aligner_provider,
                    str(std_audio),
                    None,
                    intermediate_dir=paths.get("intermediate"),
                    verbose=args.verbose,
                    base_name="",
                    registry=api["registry"],
                    transcriber_model=args.transcriber_model,
                    disable_chunking=getattr(args, 'disable_chunking', False),
                    output_mode=getattr(args, 'output_mode', 'stitched'),
                    llm_stitcher_url=getattr(args, 'llm_stitcher_url', 'http://0.0.0.0:8080')
                )

                # 3) Diarize (assign speakers to words)
                from local_transcribe.lib.config import get_system_capability
                device = get_system_capability()
                
                words_with_speakers = diarization_provider.diarize(
                    str(std_audio), 
                    words, 
                    args.num_speakers,
                    device=device,
                    models_dir=models_dir
                )
                # Verbose: Save diarized segments
                if args.verbose:
                    json_word_writer = api["registry"].get_word_writer("word-segments-json")
                    json_word_writer.write(words_with_speakers, paths["intermediate"] / "diarization" / "diarized_word_segments.json")
                    print("[i] Verbose: Diarized word segments saved to Intermediate_Outputs/diarization/diarized_word_segments.json")

                # 4) Build turns
                turn_kwargs = {}
                if hasattr(args, 'llm_turn_builder_url') and args.llm_turn_builder_url:
                    turn_kwargs['llm_url'] = args.llm_turn_builder_url
                transcript = turn_builder_provider.build_turns(words_with_speakers, **turn_kwargs)
                # Verbose: Save raw turns
                if args.verbose:
                    json_turns_writer = api["registry"].get_output_writer("turns-json")
                    json_turns_writer.write(transcript, paths["intermediate"] / "turns" / "raw_turns.json")
                    print("[i] Verbose: Raw turns saved to Intermediate_Outputs/turns/raw_turns.json")
    
                # Assign speaker names if interactive
                transcript = assign_speaker_names(transcript, getattr(args, 'interactive', False), mode)
    
                # Write raw outputs including video (this is the only place video should be generated)
                raw_dir = paths["root"] / "Transcript_Raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                print(f"[*] Writing raw outputs to {raw_dir}...")
                output_manager = OutputManager.get_instance(api["registry"])
                # For combined_audio mode, pass the single standardized audio file
                audio_config = std_audio if mode == "combined_audio" else None
                print(f"[i] Writing raw outputs with formats: {args.selected_outputs}")
                output_manager.write_selected_outputs(transcript, {**paths, "merged": raw_dir}, args.selected_outputs, audio_config, generate_video=True, word_segments=words_with_speakers)

            # 5) Prepare transcript for LLM processing
            print("[*] Preparing transcript for LLM processing...")
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
                
                print(f"[✓] Transcript preparation complete: {prep_result['stats']['segments_created']} segments created")
                if args.verbose:
                    print(f"    - Original turns: {prep_result['stats']['original_turns']}")
                    print(f"    - Words processed: {prep_result['stats']['words_processed']}")
                    print(f"    - Turns split: {prep_result['stats']['turns_split']}")
            except Exception as e:
                print(f"[!] Warning: Error during transcript preparation: {str(e)}")
                print("[i] Continuing with original transcript...")

            # 6) Optional transcript LLM-based cleanup
            if transcript_cleanup_provider:
                print(f"[*] Cleaning up transcript with {args.transcript_cleanup_provider}...")
                print(f"    Processing {len(prep_result['segments'])} segments (max {getattr(args, 'max_words_per_segment', 500)} words each)")
                
                # Process each segment through LLM
                cleaned_segments = []
                for idx, segment in enumerate(prep_result['segments']):
                    if args.verbose:
                        print(f"    [{idx+1}/{len(prep_result['segments'])}] Processing: {segment[:60]}...")
                    cleaned = transcript_cleanup_provider.transcript_cleanup_segment(segment)
                    cleaned_segments.append(cleaned)
                    if args.verbose:
                        print(f"    [{idx+1}/{len(prep_result['segments'])}] Cleaned: {cleaned[:60]}...")
                
                # Write cleaned transcript to processed directory
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                # Write as plain text (no timestamps needed for cleaned transcript)
                cleaned_text_file = processed_dir / "transcript_cleaned.txt"
                cleaned_text_file.write_text('\n\n'.join(cleaned_segments) + '\n', encoding='utf-8')
                
                print(f"[✓] Transcript cleanup complete: {cleaned_text_file}")
                print("[i] Raw transcript with timestamps available in Transcript_Raw/")
            else:
                print("[i] No transcript cleanup selected, raw outputs already written.")

            print(f"[i] Artifacts written to: {paths['root']}")

            print("[✓] Single file processing complete.")

        else:
            # Process separate audio files
            all_words = []
            
            for speaker_name, audio_file in speaker_files.items():
                print(f"[*] Processing {speaker_name}...")
                
                audio_path = ensure_file(audio_file, speaker_name)
                
                # 1) Standardize
                std_audio = standardize_audio(audio_path, outdir, tracker, api, speaker_name)
                
                # 2) ASR + alignment
                print(f"[*] Performing transcription and alignment for {speaker_name}...")
                words = transcribe_with_alignment(
                    transcriber_provider,
                    aligner_provider,
                    str(std_audio),
                    speaker_name,
                    intermediate_dir=paths.get("intermediate"),
                    verbose=args.verbose,
                    base_name=f"{speaker_name.lower()}_",
                    registry=api["registry"],
                    transcriber_model=args.transcriber_model,
                    disable_chunking=getattr(args, 'disable_chunking', False),
                    output_mode=getattr(args, 'output_mode', 'stitched'),
                    llm_stitcher_url=getattr(args, 'llm_stitcher_url', 'http://0.0.0.0:8080')
                )
                
                # Add words to the combined list
                all_words.extend(words)            # 3) Build and merge turns using the split_audio_turn_builder
            turns_task = tracker.add_task("Building and merging conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=50, description="Building optimal turns from all speakers")
            
            turn_kwargs = {}
            if hasattr(args, 'llm_turn_builder_url') and args.llm_turn_builder_url:
                turn_kwargs['llm_url'] = args.llm_turn_builder_url
            transcript = turn_builder_provider.build_turns(all_words, **turn_kwargs)
            
            tracker.update(turns_task, advance=50, description="Conversation turns built and merged")
            tracker.complete_task(turns_task, stage="turns")
            
            # Verbose: Save merged turns
            if args.verbose:
                json_turns_writer = api["registry"].get_output_writer("turns-json")
                json_turns_writer.write(transcript, paths["intermediate"] / "turns" / "merged_turns.json")
                print("[i] Verbose: Merged turns saved to Intermediate_Outputs/turns/merged_turns.json")

            # Write raw outputs including video (this is the only place video should be generated)
            raw_dir = paths["root"] / "Transcript_Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Writing raw outputs to {raw_dir}...")
            output_manager = OutputManager.get_instance(api["registry"])
            
            # For split audio mode, pass the speaker_files dictionary for video generation
            audio_config = speaker_files if mode == "split_audio" else None
            
            print(f"[i] Writing raw outputs with formats: {args.selected_outputs}")
            output_manager.write_selected_outputs(
                transcript,
                {**paths, "merged": raw_dir},
                args.selected_outputs,
                audio_config,
                generate_video=True,
                word_segments=all_words
            )

            # 5) Prepare transcript for LLM processing
            print("[*] Preparing transcript for LLM processing...")
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
                
                print(f"[✓] Transcript preparation complete: {prep_result['stats']['segments_created']} segments created")
                if args.verbose:
                    print(f"    - Original turns: {prep_result['stats']['original_turns']}")
                    print(f"    - Words processed: {prep_result['stats']['words_processed']}")
                    print(f"    - Turns split: {prep_result['stats']['turns_split']}")
            except Exception as e:
                print(f"[!] Warning: Error during transcript preparation: {str(e)}")
                print("[i] Continuing with original transcript...")

            # 6) Optional LLM-based transcript cleanup
            if transcript_cleanup_provider:
                print(f"[*] Cleaning up transcript with {args.transcript_cleanup_provider}...")
                print(f"    Processing {len(prep_result['segments'])} segments (max {getattr(args, 'max_words_per_segment', 500)} words each)")
                
                # Process each segment through LLM
                cleaned_segments = []
                for idx, segment in enumerate(prep_result['segments']):
                    if args.verbose:
                        print(f"    [{idx+1}/{len(prep_result['segments'])}] Processing: {segment[:60]}...")
                    cleaned = transcript_cleanup_provider.transcript_cleanup_segment(segment)
                    cleaned_segments.append(cleaned)
                    if args.verbose:
                        print(f"    [{idx+1}/{len(prep_result['segments'])}] Cleaned: {cleaned[:60]}...")
                
                # Write cleaned transcript to processed directory
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                
                # Write as plain text (no timestamps needed for cleaned transcript)
                cleaned_text_file = processed_dir / "transcript_cleaned.txt"
                cleaned_text_file.write_text('\n\n'.join(cleaned_segments) + '\n', encoding='utf-8')
                
                print(f"[✓] Transcript cleanup complete: {cleaned_text_file}")
                print("[i] Raw transcript with timestamps available in Transcript_Raw/")
            else:
                print("[i] No transcript cleanup selected, raw outputs already written.")

            print(f"[i] Artifacts written to: {paths['root']}")
            
            print("[✓] Separate audio processing complete.")

        # Summary
        print(f"[i] Artifacts written to: {paths['root']}")
        
        # Print performance summary
        tracker.print_summary()
        
        # Clean up temporary audio files
        cleanup_temp_audio(outdir)
        print("[✓] Temporary audio files cleaned up.")
        
        return 0
        
    finally:
        # Always stop progress tracking
        tracker.stop()
