#!/usr/bin/env python3
"""
Mode-specific interactive flows for the CLI.

This module contains the interactive prompts for each pipeline mode:
- Single speaker audio
- Combined audio (multi-speaker)
- Split audio (standard)
- VAD split audio
"""

import argparse

from local_transcribe.framework.cli_prompts import (
    prompt_selection,
    prompt_yes_no,
    prompt_url,
    print_mode_header,
)
from local_transcribe.framework.cli_providers import (
    configure_transcriber,
    configure_aligner_if_needed,
    configure_diarization,
    configure_num_speakers,
)


# =============================================================================
# Pipeline Mode Constants
# =============================================================================

class PipelineMode:
    """Enumeration of pipeline processing modes."""
    SINGLE_SPEAKER = "single_speaker_audio"
    COMBINED_AUDIO = "combined_audio"
    SPLIT_AUDIO = "split_audio"
    VAD_SPLIT_AUDIO = "vad_split_audio"


# =============================================================================
# System Capability Prompt
# =============================================================================

def prompt_system_capability(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for system capability (MPS/CUDA/CPU) if not already set."""
    from local_transcribe.lib.environment import get_available_system_capabilities
    
    if args.system:
        print(f"  ✓ System: {args.system.upper()} (set via --system)")
        return args
    
    available_capabilities = get_available_system_capabilities()
    
    # Determine preferred default: MPS > CUDA > CPU
    if "mps" in available_capabilities:
        default_capability = "mps"
    elif "cuda" in available_capabilities:
        default_capability = "cuda"
    else:
        default_capability = "cpu"
    
    if len(available_capabilities) == 1:
        args.system = available_capabilities[0]
        print(f"  ✓ System: {args.system.upper()} (only option)")
        return args
    
    default_index = available_capabilities.index(default_capability)
    options = [(cap, cap.upper()) for cap in available_capabilities]
    
    print("\nSystem Capability:")
    selected = prompt_selection(options, "Select system capability (number)", default_index)
    args.system = available_capabilities[selected]
    print(f"  ✓ System: {args.system.upper()}")
    
    return args


# =============================================================================
# De-identification Prompts
# =============================================================================

def _prompt_llm_de_identifier_url(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for the LLM de-identifier URL."""
    default_url = args.llm_de_identifier_url
    print(f"\n  Enter the LLM de-identifier URL, or press Enter for default [{default_url}]:")
    user_input = input("  LLM URL: ").strip()
    
    if user_input:
        args.llm_de_identifier_url = user_input
        print(f"  ✓ LLM de-identifier URL: {args.llm_de_identifier_url}")
    else:
        print(f"  ✓ LLM de-identifier URL: {default_url} (default)")
    
    return args


def prompt_de_identification(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    """Prompt for de-identification settings if not already set.
    """
    # Check if URL was explicitly provided via CLI (not just using default)
    url_was_set_via_cli = hasattr(args, '_llm_de_identifier_url_set') and args._llm_de_identifier_url_set
    
    # Prompt for de-identification if not already set via CLI
    if not args.de_identify:
        args.de_identify = prompt_yes_no(
            "\nEnable de-identification (replace names with [REDACTED])?",
            default=True
        )
        if args.de_identify:
            # Prompt for LLM URL if not already set via CLI
            if not url_was_set_via_cli:
                args = _prompt_llm_de_identifier_url(args)
            else:
                print(f"  ✓ LLM de-identifier URL: {args.llm_de_identifier_url} (set via --llm-de-identifier-url)")
            
            if mode == PipelineMode.SINGLE_SPEAKER:
                print("  ✓ De-identification enabled")
            else:
                print("  ✓ De-identification enabled (includes two-pass processing)")
        else:
            print("  ✓ De-identification disabled")
    else:
        # De-identification was set via CLI, prompt for URL if not also set
        if not url_was_set_via_cli:
            args = _prompt_llm_de_identifier_url(args)
        else:
            print(f"  ✓ LLM de-identifier URL: {args.llm_de_identifier_url} (set via --llm-de-identifier-url)")
        
        if mode == PipelineMode.SINGLE_SPEAKER:
            print("  ✓ De-identification enabled (set via --de-identify)")
        else:
            print("  ✓ De-identification enabled (set via --de-identify, includes two-pass processing)")
    
    return args


# =============================================================================
# Output Format Prompts
# =============================================================================

def get_available_writers(mode: str, registry, exclude_internal: bool = True) -> dict:
    """
    Get output writers available for a specific pipeline mode.
    
    Filters the registry's output writers to only include those that support
    the given mode. This keeps UI code mode-aware without scattering mode
    checks throughout the codebase.
    
    Args:
        mode: Pipeline mode (e.g., 'combined_audio', 'split_audio', 'vad_split_audio')
        registry: Plugin registry instance
        exclude_internal: If True, exclude internal writers like 'srt' (default: True)
        
    Returns:
        Dictionary mapping writer names to their descriptions for compatible writers
    """
    all_writers = registry.list_output_writers_with_metadata()
    filtered = {}
    
    # Internal writers that shouldn't be shown in UI
    internal_writers = {'srt'} if exclude_internal else set()
    
    for name, metadata in all_writers.items():
        # Skip internal writers
        if name in internal_writers:
            continue
            
        # Check if writer supports this mode
        supported_modes = metadata.get('supported_modes', [])
        if mode in supported_modes:
            filtered[name] = metadata['description']
    
    return filtered


def filter_incompatible_writers(selected: list, mode: str, registry) -> tuple:
    """
    Filter out writers that are incompatible with the selected mode.
    
    Validates user selections and removes any writer not compatible with
    the current mode, warning the user about removed selections.
    
    Args:
        selected: List of selected writer names
        mode: Pipeline mode (e.g., 'combined_audio', 'split_audio', 'vad_split_audio')
        registry: Plugin registry instance
        
    Returns:
        Tuple of (compatible_writers, removed_writers) where:
        - compatible_writers: List of writer names that are compatible
        - removed_writers: List of writer names that were removed
    """
    all_writers = registry.list_output_writers_with_metadata()
    compatible = []
    removed = []
    
    for name in selected:
        if name not in all_writers:
            # Unknown writer - skip it
            removed.append(name)
            continue
            
        metadata = all_writers[name]
        supported_modes = metadata.get('supported_modes', [])
        
        if mode in supported_modes:
            compatible.append(name)
        else:
            removed.append(name)
    
    return compatible, removed


def prompt_output_formats(args: argparse.Namespace, registry, mode: str = None) -> argparse.Namespace:
    """Prompt for output format selection if not already set.
    
    Args:
        args: Parsed command line arguments
        registry: Plugin registry instance
        mode: Pipeline mode for filtering compatible writers. If None, uses
              mode from args or defaults to 'combined_audio'
    
    Returns:
        Updated args namespace with selected_outputs populated
    """
    # Determine the mode to use for filtering
    if mode is None:
        mode = getattr(args, 'mode', None) or 'combined_audio'
    
    if hasattr(args, 'selected_outputs') and args.selected_outputs:
        # Validate pre-configured selections against mode
        compatible, removed = filter_incompatible_writers(args.selected_outputs, mode, registry)
        if removed:
            print(f"  ⚠ Removed incompatible writers for mode '{mode}': {', '.join(removed)}")
            args.selected_outputs = compatible
        print(f"  ✓ Output formats: {', '.join(args.selected_outputs)} (pre-configured)")
        return args
    
    # Get writers compatible with current mode (excludes internal writers like 'srt')
    filtered_writers = get_available_writers(mode, registry, exclude_internal=True)
    
    print("\nAvailable Output Formats:")
    for i, (name, desc) in enumerate(filtered_writers.items(), 1):
        print(f"  {i}. {name}: {desc}")
    
    print("\n  Enter numbers separated by commas (e.g., 1,3,5), or press Enter for all formats")
    choice = input("  Select output formats [Default: all]: ").strip()
    
    if not choice:
        args.selected_outputs = list(filtered_writers.keys())
        print("  ✓ Selected: All output formats")
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',') if x.strip()]
            valid_indices = [i for i in indices if 0 <= i < len(filtered_writers)]
            args.selected_outputs = [list(filtered_writers.keys())[i] for i in valid_indices]
            if not args.selected_outputs:
                print("  Error: No valid choices, selecting all.")
                args.selected_outputs = list(filtered_writers.keys())
            print(f"  ✓ Selected: {', '.join(args.selected_outputs)}")
        except ValueError:
            print("  Error: Invalid input, selecting all.")
            args.selected_outputs = list(filtered_writers.keys())
            print(f"  ✓ Selected: {', '.join(args.selected_outputs)}")
    
    return args


# =============================================================================
# Transcript Cleanup Prompts
# =============================================================================

def prompt_transcript_cleanup(args: argparse.Namespace, registry) -> argparse.Namespace:
    """Prompt for optional transcript cleanup provider."""
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider is not None:
        if args.transcript_cleanup_provider:
            # Also set enable_cleanup when provider is set via CLI
            args.enable_cleanup = True
            print(f"  ✓ Transcript cleanup: {args.transcript_cleanup_provider} (set via CLI)")
        else:
            print("  ✓ Transcript cleanup: None (set via CLI)")
        return args
    
    providers = registry.list_transcript_cleanup_providers()
    
    if not providers:
        args.transcript_cleanup_provider = None
        print("  ✓ No transcript cleanup providers available")
        return args
    
    print("\nTranscript Cleanup (optional LLM-based cleaning):")
    
    options = [(name, f"{name}: {desc}") for name, desc in providers.items()]
    
    # Default is None (index -1)
    selected = prompt_selection(
        options, 
        "Select transcript cleanup provider (number)", 
        default_index=-1,
        allow_none=True,
        none_label="None (skip cleanup)"
    )
    
    if selected == -1:
        args.transcript_cleanup_provider = None
        args.enable_cleanup = False
        print("  ✓ Transcript cleanup: None")
    else:
        args.transcript_cleanup_provider = options[selected][0]
        args.enable_cleanup = True  # Enable cleanup when a provider is selected
        
        # If remote provider, ask for URL
        if args.transcript_cleanup_provider == "llm_transcript_cleanup":
            default_url = getattr(args, 'llm_transcript_cleanup_url', 'http://0.0.0.0:8080')
            args.llm_transcript_cleanup_url = prompt_url(
                "Enter LLM server URL",
                default_url
            )
        
        print(f"  ✓ Transcript cleanup: {args.transcript_cleanup_provider}")
    
    return args


# =============================================================================
# Mode-Specific Interactive Flows
# =============================================================================

def interactive_single_speaker(args: argparse.Namespace, api) -> argparse.Namespace:
    """Interactive prompts for single speaker audio mode."""
    registry = api["registry"]
    
    print_mode_header("Single Speaker Audio", "Transcription only, output as CSV")
    
    args = prompt_system_capability(args)
    args = configure_transcriber(args, registry, require_pure=True, default_provider="granite")
    args = prompt_de_identification(args, PipelineMode.SINGLE_SPEAKER)
    
    # Output is fixed to CSV for single speaker
    args.selected_outputs = ['csv']
    print("  ✓ Output format: CSV (fixed for single speaker mode)")
    
    return args


def interactive_vad_split_audio(args: argparse.Namespace, api) -> argparse.Namespace:
    """Interactive prompts for VAD-first pipeline with split audio."""
    registry = api["registry"]
    
    print_mode_header("VAD Pipeline (Split Audio)", "Using Voice Activity Detection for turn segmentation")
    
    args = prompt_system_capability(args)
    args = configure_transcriber(args, registry, require_pure=True, default_provider="granite", warn_builtin_alignment=True)
    
    # De-identification note
    args = prompt_de_identification(args, PipelineMode.VAD_SPLIT_AUDIO)
    
    # Output formats (filtered for VAD mode)
    args = prompt_output_formats(args, registry, mode=PipelineMode.VAD_SPLIT_AUDIO)
    
    # Transcript cleanup
    args = prompt_transcript_cleanup(args, registry)
    
    return args


def interactive_combined_audio(args: argparse.Namespace, api) -> argparse.Namespace:
    """Interactive prompts for combined audio (single file, multiple speakers)."""
    registry = api["registry"]
    
    print_mode_header("Combined Audio", "Single audio file with multiple speakers")
    
    args = prompt_system_capability(args)
    args = configure_transcriber(args, registry, default_provider="granite_mfa")
    args = configure_aligner_if_needed(args, registry)
    args = configure_diarization(args, registry)
    args = configure_num_speakers(args)
    args = prompt_de_identification(args, PipelineMode.COMBINED_AUDIO)
    args = prompt_output_formats(args, registry, mode=PipelineMode.COMBINED_AUDIO)
    args = prompt_transcript_cleanup(args, registry)
    
    return args


def interactive_split_audio(args: argparse.Namespace, api) -> argparse.Namespace:
    """Interactive prompts for split audio (separate files per speaker)."""
    registry = api["registry"]
    
    print_mode_header("Split Audio", "Separate audio files per speaker")
    
    # Offer VAD pipeline as an option
    print("\n--- Pipeline Selection ---")
    
    use_vad = prompt_yes_no("Use VAD for Turn Building?", default=True)
    
    if use_vad:
        args.vad_pipeline = True
        return interactive_vad_split_audio(args, api)
    
    # Continue with standard split audio flow
    args.vad_pipeline = False
    print("\n  Continuing with standard split audio pipeline...")
    
    args = prompt_system_capability(args)
    args = configure_transcriber(args, registry, default_provider="granite_mfa")
    args = configure_aligner_if_needed(args, registry)
    
    # No diarization needed for split audio
    args.diarization_provider = None
    
    args = prompt_de_identification(args, PipelineMode.SPLIT_AUDIO)
    args = prompt_output_formats(args, registry, mode=PipelineMode.SPLIT_AUDIO)
    args = prompt_transcript_cleanup(args, registry)
    
    return args


# =============================================================================
# Configuration Summary
# =============================================================================

def display_configuration_summary(args: argparse.Namespace, mode: str):
    """Display a summary of the selected configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    mode_names = {
        PipelineMode.SINGLE_SPEAKER: "Single Speaker Audio",
        PipelineMode.COMBINED_AUDIO: "Combined Audio (Multi-Speaker)",
        PipelineMode.SPLIT_AUDIO: "Split Audio (Standard)",
        PipelineMode.VAD_SPLIT_AUDIO: "Split Audio (VAD Turn Building)",
    }
    
    print(f"\n  Mode: {mode_names.get(mode, mode)}")
    print(f"  System: {getattr(args, 'system', 'auto').upper()}")
    
    if hasattr(args, 'transcriber_provider') and args.transcriber_provider:
        print(f"  Transcriber: {args.transcriber_provider}")
        # Show remote server URL if using remote transcriber
        if args.transcriber_provider == "remote":
            print(f"  Remote Server: {getattr(args, 'remote_transcriber_url', 'http://0.0.0.0:7070')}")
    if hasattr(args, 'transcriber_model') and args.transcriber_model:
        print(f"  Model: {args.transcriber_model}")
    if hasattr(args, 'aligner_provider') and args.aligner_provider:
        print(f"  Aligner: {args.aligner_provider}")
    if hasattr(args, 'diarization_provider') and args.diarization_provider:
        print(f"  Diarization: {args.diarization_provider}")
    if hasattr(args, 'num_speakers') and args.num_speakers:
        print(f"  Number of speakers: {args.num_speakers}")
    
    print(f"  De-identification: {'Enabled (two-pass)' if getattr(args, 'de_identify', False) else 'Disabled'}")
    
    if hasattr(args, 'selected_outputs') and args.selected_outputs:
        print(f"  Output formats: {', '.join(args.selected_outputs)}")
    
    if hasattr(args, 'transcript_cleanup_provider') and args.transcript_cleanup_provider:
        print(f"  Transcript cleanup: {args.transcript_cleanup_provider}")
    
    print("\n" + "=" * 60)
