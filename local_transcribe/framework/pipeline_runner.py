#!/usr/bin/env python3
"""
Pipeline runner - orchestrates the transcription pipeline.

This module provides the main entry point for running the transcription
pipeline. It handles:
- Argument validation and mode detection
- Provider setup and model downloading
- Pipeline context creation
- Stage-based pipeline execution
"""

import os
import datetime
from pathlib import Path
from typing import Dict, Any, Union

from local_transcribe.framework.model_downloader import ensure_models_available
from local_transcribe.framework.provider_setup import ProviderSetup
from local_transcribe.framework.pipeline_context import PipelineContext
from local_transcribe.framework.stages import PipelineExecutor, create_pipeline_for_mode
from local_transcribe.lib.program_logger import log_status, log_progress


def run_pipeline(args, api: Dict[str, Any], root: Union[str, os.PathLike]) -> int:
    """
    Main pipeline execution function.
    
    This function orchestrates the entire transcription pipeline by:
    1. Validating inputs and determining processing mode
    2. Setting up providers and downloading models
    3. Creating pipeline context
    4. Executing appropriate pipeline stages
    
    Args:
        args: Command line arguments
        api: API dictionary with registry and other services
        root: Root directory path
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Validate audio files
    if not hasattr(args, 'audio_files') or not args.audio_files:
        print("ERROR: No audio files provided.")
        return 1
    
    # Determine mode and speaker mapping
    mode, speaker_files = _determine_mode_and_speakers(args, root)
    if mode is None:
        return 1  # Error already printed
    
    # Set default num_speakers
    _set_default_num_speakers(args, mode, speaker_files)
    
    # Set default outputs
    _set_default_outputs(args, mode, api)
    
    # Validate audio files exist
    if not _validate_audio_files(speaker_files):
        return 1
    
    # Early validation for single_speaker_audio mode
    if mode == "single_speaker_audio":
        if not _validate_single_speaker_mode(args, api):
            return 1
    
    # Check for VAD pipeline flag
    if mode == "split_audio" and getattr(args, 'vad_pipeline', False):
        mode = "vad_split_audio"
    
    # Setup providers
    try:
        registry = api.get("registry")
        if registry is None:
            raise ValueError("Registry not found in api")
        
        provider_setup = ProviderSetup(registry, args)
        providers = provider_setup.setup_providers(mode if mode != "vad_split_audio" else "split_audio")
        
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list-plugins to see available options.")
        return 1
    
    # Download required models
    models_dir = Path(root) / ".models"
    model_download_providers = provider_setup.get_model_download_providers()
    
    download_result = ensure_models_available(model_download_providers, models_dir, args)
    if download_result != 0:
        return download_result
    
    # Configure logging
    api["configure_global_logging"](log_level=args.log_level)
    
    # Setup output directories
    outdir, paths = _setup_output_directories(args, mode, speaker_files, providers, api, root)
    
    # Write debug settings if enabled
    if args.log_level == "DEBUG":
        _write_debug_settings(args, outdir, mode, speaker_files, providers)
    
    # Log pipeline start
    _log_pipeline_start(args, mode, providers)
    
    # Create pipeline context
    context = PipelineContext(
        args=args,
        api=api,
        root=Path(root),
        paths=paths,
        mode=mode,
        speaker_files=speaker_files,
        transcriber_provider=providers.get('transcriber'),
        aligner_provider=providers.get('aligner'),
        diarization_provider=providers.get('diarization'),
        transcript_cleanup_provider=providers.get('transcript_cleanup'),
        models_dir=models_dir,
        dry_run=getattr(args, 'dry_run', False),
    )
    
    # Create and execute pipeline
    stages = create_pipeline_for_mode(mode)
    executor = PipelineExecutor(stages)
    
    if context.dry_run:
        result = executor.execute_dry_run(context)
    else:
        result = executor.execute(context)
    
    # Report completion
    if result.success:
        print(f"[i] Artifacts written to: {paths['root']}")
        _log_mode_completion(mode)
    
    return result.exit_code


def _determine_mode_and_speakers(args, root) -> tuple:
    """Determine processing mode and speaker file mapping."""
    root = Path(root)
    
    if hasattr(args, 'single_speaker_audio') and args.single_speaker_audio:
        if len(args.audio_files) != 1:
            print("ERROR: Single speaker audio mode requires exactly one audio file.")
            return None, None
        return "single_speaker_audio", {"speaker": str(root / args.audio_files[0])}
    
    num_files = len(args.audio_files)
    
    if num_files == 1:
        return "combined_audio", {"combined_audio": str(root / args.audio_files[0])}
    
    # Multiple files = split_audio mode
    speaker_files = {}
    
    if num_files == 2:
        # Auto-assign: first file = interviewer, second = participant
        speaker_files["Interviewer"] = str(root / args.audio_files[0])
        speaker_files["Participant"] = str(root / args.audio_files[1])
    else:
        # 3+ files: prompt for speaker names
        print(f"You provided {num_files} audio files. Please assign a speaker name to each:")
        for audio_file in args.audio_files:
            while True:
                speaker_name = input(f"Speaker name for '{audio_file}': ").strip()
                if speaker_name:
                    speaker_files[speaker_name] = str(root / audio_file)
                    break
                print("Speaker name cannot be empty.")
    
    return "split_audio", speaker_files


def _set_default_num_speakers(args, mode: str, speaker_files: Dict[str, str]) -> None:
    """Set default number of speakers based on mode."""
    if not hasattr(args, 'num_speakers') or args.num_speakers is None:
        if mode == "combined_audio":
            args.num_speakers = 2
        else:
            args.num_speakers = len(speaker_files)


def _set_default_outputs(args, mode: str, api: Dict[str, Any]) -> None:
    """Set default output formats if not specified."""
    if hasattr(args, 'selected_outputs') and args.selected_outputs:
        return
    
    if mode == "single_speaker_audio":
        args.selected_outputs = ['csv']
    elif getattr(args, 'only_final_transcript', False):
        args.selected_outputs = ['timestamped-txt']
    else:
        registry = api.get("registry")
        if registry:
            all_writers = list(registry.list_output_writers().keys())
            print(f"[i] Available output writers: {all_writers}")
            args.selected_outputs = all_writers
        else:
            args.selected_outputs = ['timestamped-txt', 'plain-txt']


def _validate_audio_files(speaker_files: Dict[str, str]) -> bool:
    """Validate that all audio files exist."""
    from local_transcribe.lib.environment import ensure_file
    
    for speaker, audio_file in speaker_files.items():
        try:
            ensure_file(audio_file, speaker)
        except Exception as e:
            print(f"ERROR: {e}")
            return False
    return True


def _validate_single_speaker_mode(args, api: Dict[str, Any]) -> bool:
    """Validate provider for single speaker audio mode."""
    if not (hasattr(args, 'transcriber_provider') and args.transcriber_provider):
        return True
    
    try:
        registry = api.get("registry")
        if registry is None:
            return True
        
        temp_provider = registry.get_transcriber_provider(args.transcriber_provider)
        if temp_provider.has_builtin_alignment:
            print(f"ERROR: Provider '{args.transcriber_provider}' has built-in alignment and is not allowed in single-speaker-audio mode.")
            print("       Use granite or openai_whisper for this mode.")
            print("Use --list-plugins to see available options.")
            return False
    except ValueError:
        pass  # Will be caught in provider setup
    
    return True


def _setup_output_directories(args, mode: str, speaker_files: Dict[str, str], 
                              providers: Dict[str, Any], api: Dict[str, Any],
                              root) -> tuple:
    """Setup output directory structure."""
    from local_transcribe.lib.environment import ensure_outdir
    
    # Compute capabilities for directory creation
    transcriber = providers.get('transcriber')
    capabilities = {
        "mode": mode,
        "has_builtin_alignment": transcriber.has_builtin_alignment if transcriber else False,
        "aligner": providers.get('aligner') is not None,
        "diarization": providers.get('diarization') is not None
    }
    
    # Modify output directory name if DEBUG flag is set
    if args.log_level == "DEBUG":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        transcriber_name = transcriber.name if transcriber else "unknown"
        args.outdir = f"{args.outdir}_{transcriber_name}_{timestamp}"
    
    # Ensure outdir & subdirs
    outdir = ensure_outdir(args.outdir)
    ensure_session_dirs = api.get("ensure_session_dirs")
    if ensure_session_dirs is None:
        raise ValueError("ensure_session_dirs not found in api")
    
    paths = ensure_session_dirs(outdir, mode, speaker_files, capabilities)
    
    return outdir, paths


def _write_debug_settings(args, outdir, mode: str, speaker_files: Dict[str, str],
                          providers: Dict[str, Any]) -> None:
    """Write settings to file for debugging."""
    from local_transcribe.lib.system_capability_utils import get_system_capability
    
    settings_path = os.path.join(outdir, "settings.txt")
    
    with open(settings_path, 'w') as f:
        f.write("Local-Transcribe Settings\n")
        f.write("=" * 30 + "\n\n")
        
        # Command line arguments
        f.write("Command Line Arguments:\n")
        f.write("-" * 25 + "\n")
        for key, value in vars(args).items():
            if key not in ['audio_files', 'outdir']:
                f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Provider information
        f.write("Selected Providers:\n")
        f.write("-" * 20 + "\n")
        
        transcriber = providers.get('transcriber')
        if transcriber:
            f.write(f"Transcriber: {transcriber.name}\n")
            if hasattr(transcriber, 'model') and transcriber.model:
                f.write(f"Transcriber Model: {transcriber.model}\n")
            elif hasattr(args, 'transcriber_model') and args.transcriber_model:
                f.write(f"Transcriber Model: {args.transcriber_model}\n")
        
        aligner = providers.get('aligner')
        if aligner:
            f.write(f"Aligner: {aligner.name}\n")
        
        diarization = providers.get('diarization')
        if diarization:
            f.write(f"Diarization: {diarization.name}\n")
        
        cleanup = providers.get('transcript_cleanup')
        if cleanup:
            f.write(f"Transcript Cleanup: {cleanup.name}\n")
        f.write("\n")
        
        # Processing mode
        f.write("Processing Mode:\n")
        f.write("-" * 17 + "\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"System Capability: {get_system_capability()}\n")
        f.write(f"Number of Speakers: {args.num_speakers}\n")
        f.write(f"Selected Outputs: {', '.join(args.selected_outputs)}\n")
        f.write("\n")
        
        # Audio files
        f.write("Audio Files:\n")
        f.write("-" * 13 + "\n")
        for speaker, path in speaker_files.items():
            f.write(f"{speaker}: {os.path.basename(path)}\n")
    
    print(f"[DEBUG] Settings written to {settings_path}")


def _log_pipeline_start(args, mode: str, providers: Dict[str, Any]) -> None:
    """Log pipeline start information."""
    from local_transcribe.lib.system_capability_utils import get_system_capability
    
    if mode == "single_speaker_audio":
        log_status(f"Mode: {mode} | System: {args.system.upper()} | Transcriber: {args.transcriber_provider} | Outputs: CSV")
        return
    
    provider_info = []
    if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
        provider_info.append(f"Transcriber: {args.transcriber_provider}")
    if hasattr(args, 'aligner_provider') and args.aligner_provider:
        provider_info.append(f"Aligner: {args.aligner_provider}")
    if hasattr(args, 'diarization_provider') and args.diarization_provider:
        provider_info.append(f"Diarization: {args.diarization_provider}")
    
    provider_str = " | ".join(provider_info) if provider_info else "Default providers"
    
    log_status(
        f"Mode: {mode} | System: {get_system_capability().upper()} | "
        f"{provider_str} | Outputs: {', '.join(args.selected_outputs)}"
    )


def _log_mode_completion(mode: str) -> None:
    """Log completion message based on mode."""
    messages = {
        "single_speaker_audio": "[✓] Single speaker processing complete.",
        "combined_audio": "[✓] Single file processing complete.",
        "split_audio": "[✓] Separate audio processing complete.",
        "vad_split_audio": "[✓] VAD-first pipeline complete.",
    }
    print(messages.get(mode, "[✓] Processing complete."))
