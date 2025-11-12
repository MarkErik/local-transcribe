#!/usr/bin/env python3
# framework/pipeline_runner.py - Pipeline execution and processing logic

import os
import sys
from typing import Optional

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

def run_pipeline(args, api, root):
    from local_transcribe.lib.environment import ensure_file, ensure_outdir

    models_dir = root / "models"

    # Set default num_speakers if not provided
    if not hasattr(args, 'num_speakers') or args.num_speakers is None:
        if single_file_mode:
            args.num_speakers = 2  # Default for single file mode
        else:
            args.num_speakers = 1  # Not used in separate audio

    # Set default outputs for non-interactive
    if not hasattr(args, 'selected_outputs') or not args.selected_outputs:
        if args.only_final_transcript:
            args.selected_outputs = ['timestamped-txt']
        else:
            args.selected_outputs = list(api["registry"].list_output_writers().keys())

    # Validate separate vs single file
    separate_audio_mode = args.interviewer is not None
    single_file_mode = args.combined is not None
    if separate_audio_mode:
        if not args.participant:
            print("ERROR: Separate audio mode requires both -i/--interviewer and -p/--participant.")
            return 1
        if single_file_mode:
            print("ERROR: Provide either -c/--combined OR -i/--p, not both.")
            return 1
    elif not single_file_mode:
        print("ERROR: Must provide either -c/--combined or -i/--interviewer")
        return 1
    mode = "single_file" if single_file_mode else "separate_audio"

    # Check required providers
    try:
        if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
            combined_provider = api["registry"].get_combined_provider(args.combined_provider)
        else:
            asr_provider = api["registry"].get_asr_provider(args.asr_provider)
            diarization_provider = api["registry"].get_diarization_provider(args.diarization_provider)
            # For separate processing, always need a turn builder
            turn_builder_provider = api["registry"].get_turn_builder_provider("general")  # Default to general
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1

    # Set default model if not specified
    if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
        if not hasattr(args, 'combined_model') or args.combined_model is None:
            available_models = combined_provider.get_available_models()
            args.combined_model = available_models[0] if available_models else None
    else:
        if not hasattr(args, 'asr_model') or args.asr_model is None:
            available_models = asr_provider.get_available_models()
            args.asr_model = available_models[0] if available_models else None

    # Download required models for selected providers
    # Phase 2: Model Availability Check (Offline)
    if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
        required_combined_models = combined_provider.get_required_models(args.combined_model)
        print(f"[*] Checking model availability offline...")
        missing_combined_models = combined_provider.check_models_available_offline(required_combined_models, models_dir)
        all_missing = missing_combined_models
    else:
        required_asr_models = asr_provider.get_required_models(args.asr_model)
        required_diarization_models = diarization_provider.get_required_models()
        
        print(f"[*] Checking model availability offline...")
        missing_asr_models = []
        missing_diarization_models = []
        
        if required_asr_models:
            missing_asr_models = asr_provider.check_models_available_offline(required_asr_models, models_dir)
        
        if required_diarization_models:
            missing_diarization_models = diarization_provider.check_models_available_offline(required_diarization_models, models_dir)
        
        all_missing = missing_asr_models + missing_diarization_models

    # Phase 3: Conditional Download (Online Only If Needed)
    if all_missing:
        print(f"[!] Missing models detected: {', '.join(all_missing)}")
        print("[!] Switching to online mode for download...")
        
        # Check if HF_TOKEN is available
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[!] WARNING: HF_TOKEN not found in environment variables.")
            print("[!] Please ensure your .env file contains a valid HuggingFace token.")
            print("[!] You can get a token from: https://huggingface.co/settings/tokens")
            print("[!] Alternatively, run: huggingface-cli login")
            print("[!] Make sure the token has 'read' permissions for the required models")
        
        # Explicitly set online mode
        print(f"DEBUG: Setting HF_HUB_OFFLINE to 0 (was: {os.environ.get('HF_HUB_OFFLINE')})")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # Force reload of huggingface_hub modules to pick up new environment
        import sys
        print(f"DEBUG: Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        # Verify the change took effect immediately
        print(f"DEBUG: Immediate verification - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
        
        # Show current environment for debugging
        print(f"[*] Environment check:")
        print(f"    HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"    HF_TOKEN: {'***' if hf_token else 'NOT SET'}")
        print(f"    HF_HOME: {os.environ.get('HF_HOME')}")
        
        # Additional debug info
        print(f"DEBUG: All Hugging Face environment variables:")
        for key, value in os.environ.items():
            if key.startswith('HF_'):
                print(f"    {key}: {'***' if 'TOKEN' in key else value}")
        
        # Test huggingface_hub import and check its view of environment
        try:
            from huggingface_hub import HfApi
            print(f"DEBUG: HfApi().whoami() (tests token): {HfApi().whoami()}")
        except Exception as e:
            print(f"DEBUG: HfApi test failed: {e}")
        
        try:
            # Download missing models
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                if missing_combined_models:
                    print(f"[*] Downloading combined models: {', '.join(missing_combined_models)}")
                    combined_provider.ensure_models_available(missing_combined_models, models_dir)
            else:
                if missing_asr_models:
                    print(f"[*] Downloading ASR models: {', '.join(missing_asr_models)}")
                    asr_provider.ensure_models_available(missing_asr_models, models_dir)
                
                if missing_diarization_models:
                    print(f"[*] Downloading diarization models: {', '.join(missing_diarization_models)}")
                    diarization_provider.ensure_models_available(missing_diarization_models, models_dir)
            
            # Verify downloads actually succeeded
            print("[*] Verifying downloaded models...")
            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                verified_combined = combined_provider.check_models_available_offline(required_combined_models, models_dir)
                if verified_combined:
                    print(f"[!] ERROR: Some models failed to download properly: {', '.join(verified_combined)}")
                    print("[!] Please check your internet connection and HuggingFace token.")
                    return 1
            else:
                verified_asr = asr_provider.check_models_available_offline(required_asr_models, models_dir)
                verified_diar = diarization_provider.check_models_available_offline(required_diarization_models, models_dir)
                
                if verified_asr or verified_diar:
                    print(f"[!] ERROR: Some models failed to download properly: {', '.join(verified_asr + verified_diar)}")
                    print("[!] Please check your internet connection and HuggingFace token.")
                    return 1
            
            print("[✓] All models downloaded successfully and verified.")
        except Exception as e:
            print(f"ERROR: Failed to download models: {e}")
            return 1
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        print("[✓] All required models are available locally.")

    # Configure logging based on verbose flag
    api["configure_global_logging"](log_level="INFO" if args.verbose else "WARNING")

    # Initialize progress tracking
    tracker = api["get_progress_tracker"]()
    tracker.start()

    try:
        # Ensure outdir & subdirs
        outdir = ensure_outdir(args.outdir)
        paths = api["ensure_session_dirs"](outdir, mode)

        if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
            print(f"[*] Mode: {mode} (single_file) | Provider: {args.combined_provider} | Outputs: {', '.join(args.selected_outputs)}")
        else:
            print(f"[*] Mode: {mode} | ASR: {args.asr_provider} | Diarization: {args.diarization_provider} | Turn Builder: general | Outputs: {', '.join(args.selected_outputs)}")

        # Run pipeline
        if single_file_mode:
            mixed_path = ensure_file(args.combined, "Combined")

            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)

            # Standardize mixed audio
            tracker.update(std_task, advance=50, description="Standardizing mixed audio")
            std_mix = api["standardize_and_get_path"](mixed_path, tmpdir=temp_audio_dir)

            # Complete the standardization task
            tracker.update(std_task, advance=50, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")

            if hasattr(args, 'processing_mode') and args.processing_mode == "combined":
                # Use combined provider
                turns = combined_provider.transcribe_and_diarize(
                    str(std_mix),
                    args.num_speakers,
                    model=args.combined_model
                )
            else:
                # 2) ASR + alignment
                words = asr_provider.transcribe_with_alignment(
                    str(std_mix),
                    role=None,
                    asr_model=args.asr_model
                )

                # Save ASR results as plain text before diarization
                # TODO: Use plugin for this too
                from local_transcribe.providers.writers.txt_writer import write_asr_words
                write_asr_words(words, paths["merged"] / "asr.txt")

                # 3) Diarize (assign speakers to words)
                words_with_speakers = diarization_provider.diarize(str(std_mix), words, args.num_speakers)

                # 4) Build turns
                turns = turn_builder_provider.build_turns(words_with_speakers)

            # 4) Outputs
            write_selected_outputs(turns, paths, args.selected_outputs, tracker, api["registry"], std_mix)

            print("[✓] Single file processing complete.")

        else:
            # separate audio
            interviewer_path = ensure_file(args.interviewer, "Interviewer")
            participant_path = ensure_file(args.participant, "Participant")

            # 1) Standardize
            std_task = tracker.add_task("Audio standardization", total=100, stage="standardization")
            # Create a temp dir for standardized audio to avoid ffmpeg in-place editing
            temp_audio_dir = outdir / "temp_audio"
            temp_audio_dir.mkdir(exist_ok=True)

            # Standardize interviewer audio
            tracker.update(std_task, advance=33, description="Standardizing interviewer audio")
            std_int = api["standardize_and_get_path"](interviewer_path, tmpdir=temp_audio_dir)

            # Standardize participant audio
            tracker.update(std_task, advance=33, description="Standardizing participant audio")
            std_part = api["standardize_and_get_path"](participant_path, tmpdir=temp_audio_dir)

            # Complete the standardization task
            tracker.update(std_task, advance=34, description="Audio standardization complete")
            tracker.complete_task(std_task, stage="standardization")

            # 2) ASR + alignment per track
            asr_task = tracker.add_task("ASR Transcription", total=100, stage="asr_transcription")
            tracker.update(asr_task, advance=20, description="Transcribing interviewer audio")
            int_words = asr_provider.transcribe_with_alignment(
                str(std_int),
                role="Interviewer",
                asr_model=args.asr_model
            )

            # Save ASR results and build turns for interviewer
            from local_transcribe.providers.writers.txt_writer import write_asr_words
            write_asr_words(int_words, paths["speaker_interviewer"] / "asr.txt")

            # Use dual-track diarization provider for building turns
            dual_track_provider = api["registry"].get_diarization_provider("dual-track")
            int_turns = turn_builder_provider.build_turns(int_words)

            # Save interviewer timestamped transcript
            timestamped_writer = api["registry"].get_output_writer("timestamped-txt")
            timestamped_writer.write(int_turns, paths["speaker_interviewer"] / "interviewer.timestamped.txt")
            tracker.update(asr_task, advance=30, description="Interviewer ASR and timestamped transcript complete")

            tracker.update(asr_task, advance=20, description="Transcribing participant audio")
            part_words = asr_provider.transcribe_with_alignment(
                str(std_part),
                role="Participant",
                asr_model=args.asr_model
            )

            write_asr_words(part_words, paths["speaker_participant"] / "asr.txt")

            # Build turns for participant
            part_turns = turn_builder_provider.build_turns(part_words)

            # Save participant timestamped transcript
            timestamped_writer.write(part_turns, paths["speaker_participant"] / "participant.timestamped.txt")
            tracker.update(asr_task, advance=30, description="Participant ASR and timestamped transcript complete")
            tracker.complete_task(asr_task, stage="asr_transcription")

            # 3) Merge turns
            turns_task = tracker.add_task("Merging conversation turns", total=100, stage="turns")
            tracker.update(turns_task, advance=50, description="Merging conversation turns")
            # TODO: Use plugin for merging
            from local_transcribe.processing.merge import merge_turns
            merged = merge_turns(int_turns, part_turns)
            tracker.update(turns_task, advance=50, description="Turn merging complete")
            tracker.complete_task(turns_task, stage="turns")

            # 4) Write results
            write_selected_outputs(merged, paths, args.selected_outputs, tracker, api["registry"], [std_int, std_part])

            print("[✓] Separate audio processing complete.")

        # Summary
        print(f"[i] Artifacts written to: {paths['root']}")
        
        # Print performance summary
        tracker.print_summary()
        
        return 0
        
    finally:
        # Always stop progress tracking
        tracker.stop()