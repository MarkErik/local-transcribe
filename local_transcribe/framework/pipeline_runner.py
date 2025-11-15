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

def transcribe_with_alignment(transcriber_provider, aligner_provider, audio_path, role, **kwargs):
    """Transcribe audio and return word segments with timestamps."""
    if transcriber_provider.has_builtin_alignment:
        # Transcriber has built-in alignment
        return transcriber_provider.transcribe_with_alignment(
            audio_path,
            role=role,
            **kwargs
        )
    else:
        # Use transcriber + aligner composition
        transcript = transcriber_provider.transcribe(audio_path, **kwargs)
        segments = aligner_provider.align_transcript(audio_path, transcript, **kwargs)
        # Add speaker role if provided
        if role:
            for segment in segments:
                segment.speaker = role
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
            args.selected_outputs = list(api["registry"].list_output_writers().keys())

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

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir, mode, speaker_files)

        if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
            print(f"[*] Mode: {mode} (combined_audio) | Provider: {args.unified_provider} | Outputs: {', '.join(args.selected_outputs)}")
        else:
            provider_info = []
            if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
                provider_info.append(f"Transcriber: {args.transcriber_provider}")
            if hasattr(args, 'aligner_provider') and args.aligner_provider:
                provider_info.append(f"Aligner: {args.aligner_provider}")
            if hasattr(args, 'diarization_provider') and args.diarization_provider:
                provider_info.append(f"Diarization: {args.diarization_provider}")
            provider_str = " | ".join(provider_info) if provider_info else "Default providers"
            print(f"[*] Mode: {mode} | {provider_str} | Turn Builder: {turn_builder_provider.name} | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if mode == "combined_audio":
            mixed_path = ensure_file(speaker_files["combined_audio"], "Combined Audio")

            # 1) Standardize
            std_audio = standardize_audio(mixed_path, outdir, tracker, api)

            if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
                # Use unified provider
                transcript = unified_provider.transcribe_and_diarize(
                    str(std_audio),
                    args.num_speakers,
                    model=args.unified_model
                )
            else:
                # 2) Transcription + alignment
                words = transcribe_with_alignment(
                    transcriber_provider,
                    aligner_provider,
                    str(std_audio),
                    role=None,
                    transcriber_model=args.transcriber_model
                )

                # 3) Diarize (assign speakers to words)
                words_with_speakers = diarization_provider.diarize(str(std_audio), words, args.num_speakers, models_dir)

                # 4) Build turns
                transcript = turn_builder_provider.build_turns(words_with_speakers)

            # Assign speaker names if interactive
            transcript = assign_speaker_names(transcript, getattr(args, 'interactive', False), mode)

            # Write raw outputs before cleanup
            raw_dir = paths["root"] / "Transcript_Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Writing raw outputs to {raw_dir}...")
            output_manager = OutputManager.get_instance(api["registry"])
            output_manager.write_selected_outputs(transcript, {**paths, "merged": raw_dir}, args.selected_outputs, std_audio)

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
            transcript_cleanup_done = False
            if transcript_cleanup_provider:
                print(f"[*] Cleaning up transcript with {args.transcript_cleanup_provider}...")
                for turn in transcript:
                    original_text = turn.text
                    turn.text = transcript_cleanup_provider.transcript_cleanup_segment(original_text)
                    if turn.text != original_text:
                        print(f"  Cleaned: '{original_text[:50]}...' -> '{turn.text[:50]}...'")
                print("[✓] Transcript cleanup complete.")
                transcript_cleanup_done = True

            # Write processed outputs if transcript cleanup was done
            if transcript_cleanup_done:
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                print(f"[*] Writing processed outputs to {processed_dir}...")
                output_manager = OutputManager.get_instance(api["registry"])
                output_manager.write_selected_outputs(transcript, {**paths, "merged": processed_dir}, args.selected_outputs, std_audio)
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
                    role=speaker_name,
                    transcriber_model=args.transcriber_model
                )
                
                # Debug: Save alignment results as JSON for inspection
                if args.verbose:
                    json_word_writer = api["registry"].get_word_writer("word-segments-json")
                    json_word_writer.write(words, paths[f"speaker_{speaker_name.lower()}"] / f"{speaker_name.lower()}_alignment.json")
                    print(f"[i] Alignment debug data saved to: speaker_{speaker_name.lower()}_alignment.json")
                
                # Save individual transcription results
                word_writer = api["registry"].get_word_writer("word-segments")
                word_writer.write(words, paths[f"speaker_{speaker_name.lower()}"] / f"{speaker_name.lower()}.txt")
                
                # Add words to the combined list
                all_words.extend(words)
            
            # 3) Build and merge turns using the split_audio_turn_builder
            turns_task = tracker.add_task("Building and merging conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=50, description="Building optimal turns from all speakers")
            
            transcript = turn_builder_provider.build_turns(all_words)
            
            tracker.update(turns_task, advance=50, description="Conversation turns built and merged")
            tracker.complete_task(turns_task, stage="turns")

            # Write raw outputs before cleanup
            raw_dir = paths["root"] / "Transcript_Raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            print(f"[*] Writing raw outputs to {raw_dir}...")
            output_manager = OutputManager.get_instance(api["registry"])
            output_manager.write_selected_outputs(transcript, {**paths, "merged": raw_dir}, args.selected_outputs, None)

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
            transcript_cleanup_done = False
            if transcript_cleanup_provider:
                print(f"[*] Cleaning up transcript with {args.transcript_cleanup_provider}...")
                for turn in transcript:
                    original_text = turn.text
                    turn.text = transcript_cleanup_provider.transcript_cleanup_segment(original_text)
                    if turn.text != original_text:
                        print(f"  Cleaned: '{original_text[:50]}...' -> '{turn.text[:50]}...'")
                print("[✓] Transcript cleanup complete.")
                transcript_cleanup_done = True

            # Write processed outputs if transcript cleanup was done
            if transcript_cleanup_done:
                processed_dir = paths["root"] / "Transcript_Processed"
                processed_dir.mkdir(parents=True, exist_ok=True)
                print(f"[*] Writing processed outputs to {processed_dir}...")
                output_manager = OutputManager.get_instance(api["registry"])
                output_manager.write_selected_outputs(transcript, {**paths, "merged": processed_dir}, args.selected_outputs, None)
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
