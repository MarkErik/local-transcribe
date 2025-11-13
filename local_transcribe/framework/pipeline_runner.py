#!/usr/bin/env python3
# framework/pipeline_runner.py - Pipeline execution and processing logic

import os
import sys
from typing import Optional

from local_transcribe.framework.model_downloader import ensure_models_available

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
        speaker_files = {"combined_audio": args.audio_files[0]}  # Single file with multiple speakers
    else:
        mode = "split_audio"
        speaker_files = {}
        
        if num_files == 2:
            # Auto-assign: first file = interviewer, second = participant
            speaker_files["Interviewer"] = args.audio_files[0]
            speaker_files["Participant"] = args.audio_files[1]
        else:
            # 3+ files: prompt for speaker names
            print(f"You provided {num_files} audio files. Please assign a speaker name to each:")
            for i, audio_file in enumerate(args.audio_files):
                while True:
                    speaker_name = input(f"Speaker name for '{audio_file}': ").strip()
                    if speaker_name:
                        speaker_files[speaker_name] = audio_file
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
            args.selected_outputs = list(api["registry"].list_output_writers().keys())

    # Validate audio files exist
    for speaker, audio_file in speaker_files.items():
        try:
            ensure_file(audio_file, speaker)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

    # Check required providers
    try:
        if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
            unified_provider = api["registry"].get_unified_provider(args.unified_provider)
        else:
            transcriber_provider = api["registry"].get_transcriber_provider(args.transcriber_provider)
            # Check if transcriber has built-in alignment
            if transcriber_provider.has_builtin_alignment:
                aligner_provider = None
            else:
                # Pure transcribers require an aligner
                aligner_provider = api["registry"].get_aligner_provider(args.aligner_provider)
            
            # Diarization is only needed for combined audio in separate processing mode
            if mode == "combined_audio":
                diarization_provider = api["registry"].get_diarization_provider(args.diarization_provider)
            else:
                diarization_provider = None
            
            # For separate processing, always need a turn builder
            turn_builder_provider = api["registry"].get_turn_builder_provider("general")  # Default to general
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1

    # Set default model if not specified
    if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
        if not hasattr(args, 'unified_model') or args.unified_model is None:
            available_models = unified_provider.get_available_models()
            args.unified_model = available_models[0] if available_models else None
    else:
        if not hasattr(args, 'transcriber_model') or args.transcriber_model is None:
            available_models = transcriber_provider.get_available_models()
            args.transcriber_model = available_models[0] if available_models else None

    # Download required models for selected providers
    providers = {}
    if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
        providers['unified'] = unified_provider
    else:
        providers['transcriber'] = transcriber_provider
        if aligner_provider:
            providers['aligner'] = aligner_provider
        if diarization_provider:
            providers['diarization'] = diarization_provider

    download_result = ensure_models_available(providers, models_dir, args)
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
            print(f"[*] Mode: {mode} | {provider_str} | Turn Builder: general | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if mode == "combined_audio":
            mixed_path = ensure_file(speaker_files["combined_audio"], "Combined Audio")

            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)

            # Standardize combined audio
            tracker.update(std_task, advance=50, description="Standardizing combined audio")
            std_audio = api["standardize_and_get_path"](mixed_path, tmpdir=temp_audio_dir)

            # Complete the standardization task
            tracker.update(std_task, advance=50, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")

            if hasattr(args, 'processing_mode') and args.processing_mode == "unified":
                # Use unified provider
                turns = unified_provider.transcribe_and_diarize(
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

                # Save transcription results as plain text before diarization
                word_writer = api["registry"].get_word_writer("word-segments")
                word_writer.write(words, paths["merged"] / "transcription.txt")

                # 3) Diarize (assign speakers to words)
                words_with_speakers = diarization_provider.diarize(str(std_audio), words, args.num_speakers)

                # 4) Build turns
                turns = turn_builder_provider.build_turns(words_with_speakers)

            # 4) Outputs
            write_selected_outputs(turns, paths, args.selected_outputs, tracker, api["registry"], std_audio)

            print("[✓] Single file processing complete.")

        else:
            # Process separate audio files
            speaker_turns = {}
            
            for speaker_name, audio_file in speaker_files.items():
                print(f"[*] Processing {speaker_name}...")
                
                audio_path = ensure_file(audio_file, speaker_name)
                
                # 1) Standardize
                std_task = tracker.add_task(f"Audio standardization for {speaker_name}", total=100, stage="standardization")
                temp_audio_dir = outdir / "temp_audio"
                temp_audio_dir.mkdir(exist_ok=True)
                
                tracker.update(std_task, advance=50, description=f"Standardizing {speaker_name} audio")
                std_audio = api["standardize_and_get_path"](audio_path, tmpdir=temp_audio_dir)
                
                tracker.update(std_task, advance=50, description=f"{speaker_name} audio standardization complete")
                tracker.complete_task(std_task, stage="standardization")
                
                # 2) ASR + alignment
                words = transcribe_with_alignment(
                    transcriber_provider,
                    aligner_provider,
                    str(std_audio),
                    role=speaker_name,
                    transcriber_model=args.transcriber_model
                )
                
                # Save individual transcription results
                word_writer = api["registry"].get_word_writer("word-segments")
                word_writer.write(words, paths[f"speaker_{speaker_name.lower()}"] / f"{speaker_name.lower()}.txt")
                
                # 3) Build turns (no diarization needed for separate files)
                turns = turn_builder_provider.build_turns(words)
                speaker_turns[speaker_name] = turns
                
                # Write individual speaker outputs
                timestamped_writer = api["registry"].get_output_writer("timestamped-txt")
                timestamped_writer.write(turns, paths[f"speaker_{speaker_name.lower()}"] / f"{speaker_name.lower()}.timestamped.txt")
            
            # 3) Merge turns
            turns_task = tracker.add_task("Merging conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=50, description="Merging conversation turns")
            
            from local_transcribe.processing.merge import merge_turns
            all_turns = list(speaker_turns.values())
            turns = merge_turns(*all_turns)
            
            tracker.update(turns_task, advance=50, description="Conversation turns merged")
            tracker.complete_task(turns_task, stage="turns")
            
            # 4) Outputs
            write_selected_outputs(turns, paths, args.selected_outputs, tracker, api["registry"], None)
            
            print("[✓] Separate audio processing complete.")

        # Summary
        print(f"[i] Artifacts written to: {paths['root']}")
        
        # Print performance summary
        tracker.print_summary()
        
        return 0
        
    finally:
        # Always stop progress tracking
        tracker.stop()

def write_selected_outputs(turns, paths, selected, tracker, registry, audio_path=None):
    """Write selected outputs for merged turns."""
    output_task = tracker.add_task("Writing output files", total=100, stage="output")
    tracker.update(output_task, advance=20, description="Writing transcript files")

    srt_path = None

    if 'timestamped-txt' in selected:
        timestamped_writer = registry.get_output_writer("timestamped-txt")
        timestamped_writer.write(turns, paths["merged"] / "transcript.timestamped.txt")

    if 'plain-txt' in selected:
        plain_writer = registry.get_output_writer("plain-txt")
        plain_writer.write(turns, paths["merged"] / "transcript.txt")

    if 'csv' in selected:
        csv_writer = registry.get_output_writer("csv")
        csv_writer.write(turns, paths["merged"] / "transcript.csv")

    if 'markdown' in selected:
        markdown_writer = registry.get_output_writer("markdown")
        markdown_writer.write(turns, paths["merged"] / "transcript.md")

    tracker.update(output_task, advance=20, description="Writing subtitle files")

    if 'srt' in selected:
        srt_writer = registry.get_output_writer("srt")
        srt_path = paths["merged"] / "subtitles.srt"
        srt_writer.write(turns, srt_path)

    if 'vtt' in selected:
        vtt_writer = registry.get_output_writer("vtt")
        vtt_writer.write(turns, paths["merged"] / "subtitles.vtt")

    if srt_path and audio_path is not None:
        tracker.update(output_task, advance=30, description="Rendering video with subtitles")
        from local_transcribe.providers.writers.render_video import render_video
        render_video(srt_path, paths["merged"] / "video_with_subtitles.mp4", audio_path=audio_path)
    else:
        tracker.update(output_task, advance=30, description="Skipping video rendering")

    tracker.update(output_task, advance=30, description="Finalizing outputs")
    tracker.complete_task(output_task, stage="output")