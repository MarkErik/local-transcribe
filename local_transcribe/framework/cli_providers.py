#!/usr/bin/env python3
"""
Provider selection helpers for the CLI.

This module provides functions for selecting transcriber, aligner,
diarization, and other providers during interactive mode.
"""

import argparse
from typing import Optional

from local_transcribe.framework.cli_prompts import (
    prompt_selection,
    prompt_yes_no,
    prompt_url,
)


def select_provider(
    registry,
    provider_type: str,
    display_title: str,
    default_provider: Optional[str] = None,
    filter_func: Optional[callable] = None,
    default_to_first: bool = True
) -> str:
    """
    Generic provider selection function.
    
    Args:
        registry: Plugin registry
        provider_type: Type of provider ('transcriber', 'aligner', 'diarization')
        display_title: Title to display (e.g., 'Transcriber Providers')
        default_provider: Name of default provider (if any)
        filter_func: Optional function(provider) -> bool to filter providers
        default_to_first: If True and no default_provider matches, default to first option
        
    Returns:
        str: Selected provider name
    """
    # Get providers and getter based on type
    list_method = getattr(registry, f'list_{provider_type}_providers')
    get_method = getattr(registry, f'get_{provider_type}_provider')
    
    providers = list_method()
    
    # Apply filter if provided
    if filter_func:
        providers = {
            name: desc for name, desc in providers.items()
            if filter_func(get_method(name))
        }
    
    if not providers:
        raise ValueError(f"No suitable {provider_type} providers available.")
    
    # Build options list with display names
    options = []
    default_index = None
    for i, (name, desc) in enumerate(providers.items()):
        provider = get_method(name)
        display_name = getattr(provider, 'short_name', desc)
        options.append((name, display_name))
        if name == default_provider:
            default_index = i
    
    # Default to first if not specified and allowed
    if default_index is None and default_to_first:
        default_index = 0
    
    print(f"\nAvailable {display_title}:")
    selected = prompt_selection(options, f"Select {provider_type} (number)", default_index)
    
    return options[selected][0]


def select_transcriber_provider(
    registry,
    filter_pure_only: bool = False,
    default_provider: Optional[str] = None
) -> str:
    """Select a transcriber provider with optional filtering for pure transcribers."""
    filter_func = (lambda p: not p.has_builtin_alignment) if filter_pure_only else None
    return select_provider(
        registry, 'transcriber', 'Transcriber Providers',
        default_provider, filter_func, default_to_first=False
    )


def select_transcriber_model(registry, provider_name: str, default_model: Optional[str] = None) -> Optional[str]:
    """Select a model for the given transcriber provider."""
    provider = registry.get_transcriber_provider(provider_name)
    available_models = provider.get_available_models()
    
    if not available_models:
        return None
    
    if len(available_models) == 1:
        print(f"  ✓ Using model: {available_models[0]}")
        return available_models[0]
    
    # For granite, default to 8b
    if provider_name == "granite" and default_model is None:
        default_model = "granite-8b"
    
    default_index = available_models.index(default_model) if default_model in available_models else 0
    options = [(m, m) for m in available_models]
    
    print(f"\nAvailable models for {getattr(provider, 'short_name', provider_name)}:")
    selected = prompt_selection(options, "Select model (number)", default_index)
    
    return available_models[selected]


def select_aligner_provider(registry, default_provider: Optional[str] = None) -> str:
    """Select an aligner provider."""
    return select_provider(registry, 'aligner', 'Aligner Providers', default_provider)


def select_diarization_provider(registry, default_provider: Optional[str] = None) -> str:
    """Select a diarization provider."""
    return select_provider(registry, 'diarization', 'Diarization Providers', default_provider)


def configure_transcriber(
    args: argparse.Namespace,
    registry,
    require_pure: bool = False,
    default_provider: str = "granite_mfa",
    warn_builtin_alignment: bool = False
) -> argparse.Namespace:
    """
    Configure transcriber provider and model with consistent logic.
    
    Args:
        args: Command line arguments
        registry: Plugin registry
        require_pure: If True, only allow pure transcribers (no built-in alignment)
        default_provider: Default provider name for selection
        warn_builtin_alignment: If True, warn but allow providers with built-in alignment
        
    Returns:
        Updated args namespace
    """
    if args.transcriber_provider is None:
        args.transcriber_provider = select_transcriber_provider(
            registry,
            filter_pure_only=require_pure,
            default_provider=default_provider
        )
        print(f"  ✓ Transcriber: {args.transcriber_provider}")
    else:
        # Validate CLI-provided transcriber
        provider = registry.get_transcriber_provider(args.transcriber_provider)
        if require_pure and provider.has_builtin_alignment:
            print(f"  ⚠ Provider '{args.transcriber_provider}' has built-in alignment.")
            print("    This mode requires a pure transcriber (granite, openai_whisper, or remote).")
            args.transcriber_provider = select_transcriber_provider(
                registry,
                filter_pure_only=True,
                default_provider=default_provider
            )
        elif warn_builtin_alignment and provider.has_builtin_alignment:
            print(f"  ⚠ Warning: {args.transcriber_provider} has built-in alignment.")
            print(f"    Consider using: granite, openai_whisper, or remote")
            print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
        else:
            print(f"  ✓ Transcriber: {args.transcriber_provider} (set via CLI)")
    
    # Model selection
    if args.transcriber_model is None:
        args.transcriber_model = select_transcriber_model(registry, args.transcriber_provider)
    else:
        print(f"  ✓ Model: {args.transcriber_model} (set via CLI)")
    
    # Remote transcriber URL prompt (if remote transcriber selected)
    args = prompt_remote_transcriber_url(args)
    
    # Granite-specific settings
    if args.transcriber_provider and 'granite' in args.transcriber_provider:
        args.output_format = "chunked"
        print("  ✓ Using chunk stitching for Granite")
    
    return args


def configure_aligner_if_needed(args: argparse.Namespace, registry) -> argparse.Namespace:
    """Configure aligner provider if transcriber doesn't have built-in alignment."""
    transcriber = registry.get_transcriber_provider(args.transcriber_provider)
    if not transcriber.has_builtin_alignment:
        if args.aligner_provider is None:
            args.aligner_provider = select_aligner_provider(registry)
            print(f"  ✓ Aligner: {args.aligner_provider}")
        else:
            print(f"  ✓ Aligner: {args.aligner_provider} (set via CLI)")
    else:
        print("  ✓ Aligner: Not needed (transcriber has built-in alignment)")
        args.aligner_provider = None
    return args


def configure_diarization(args: argparse.Namespace, registry) -> argparse.Namespace:
    """Configure diarization provider."""
    if args.diarization_provider is None:
        args.diarization_provider = select_diarization_provider(registry)
        print(f"  ✓ Diarization: {args.diarization_provider}")
    else:
        print(f"  ✓ Diarization: {args.diarization_provider} (set via CLI)")
    return args


def configure_num_speakers(args: argparse.Namespace) -> argparse.Namespace:
    """Configure number of speakers for diarization."""
    if not hasattr(args, 'num_speakers') or args.num_speakers is None:
        print("\n--- Speaker Configuration ---")
        while True:
            num_input = input("  Number of speakers expected [Default: 2]: ").strip()
            if not num_input:
                args.num_speakers = 2
                break
            try:
                num = int(num_input)
                if num > 0:
                    args.num_speakers = num
                    break
                print("  Error: Please enter a positive number.")
            except ValueError:
                print("  Error: Please enter a valid number.")
        print(f"  ✓ Number of speakers: {args.num_speakers}")
    else:
        print(f"  ✓ Number of speakers: {args.num_speakers} (set via --num-speakers)")
    return args


def prompt_remote_transcriber_url(args: argparse.Namespace) -> argparse.Namespace:
    """Prompt for remote transcriber server URL and options if remote transcriber is selected."""
    # Only relevant when remote transcriber is selected
    if args.transcriber_provider != "remote":
        return args
    
    print("\n--- Remote Transcription Server ---")
    
    default_url = getattr(args, 'remote_transcriber_url', 'http://0.0.0.0:7070')
    args.remote_transcriber_url = prompt_url("Enter remote transcription server URL", default_url)
    
    # Check server availability
    from local_transcribe.providers.transcribers.remote_transcriber import (
        check_remote_transcriber_available,
        get_remote_server_info
    )
    print(f"  Checking connection to {args.remote_transcriber_url}...")
    
    server_available = check_remote_transcriber_available(args.remote_transcriber_url)
    
    if server_available:
        print(f"  ✓ Remote server is available")
        
        # Get server info for display
        server_info = get_remote_server_info(args.remote_transcriber_url)
        if server_info:
            model_info = server_info.get("model", {})
            model_name = model_info.get("name", "unknown")
            print(f"  ✓ Server model: {model_name}")
    else:
        print(f"  ⚠ Remote server not available at {args.remote_transcriber_url}")
        fallback = prompt_yes_no("Continue anyway (will fail if server unavailable)?", default=False)
        if not fallback:
            # User wants to choose a different transcriber
            print("  ✓ Please select a different transcriber")
            args.transcriber_provider = None  # Reset to force re-selection
    
    return args


# Backward compatibility aliases (prefixed with underscore in original)
_select_provider = select_provider
_configure_transcriber = configure_transcriber
_configure_aligner_if_needed = configure_aligner_if_needed
_configure_diarization = configure_diarization
_configure_num_speakers = configure_num_speakers
