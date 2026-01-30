#!/usr/bin/env python3
"""
Reusable CLI prompt helpers.

This module provides consistent prompt functions for user interaction,
including selection prompts, yes/no prompts, and URL input.
"""

from typing import Optional


def prompt_selection(
    options: list, 
    prompt_text: str, 
    default_index: Optional[int] = None, 
    allow_none: bool = False, 
    none_label: str = "None"
) -> int:
    """
    Generic selection prompt with consistent default handling.
    
    Args:
        options: List of (name, display_name) tuples
        prompt_text: Text to display for the prompt
        default_index: 0-based index of default option (None if no default, -1 for None option)
        allow_none: If True, option 0 is "None/skip"
        none_label: Label for the none option
        
    Returns:
        Selected index (0-based), or -1 if none selected
    """
    if allow_none:
        default_marker = " [Default]" if default_index == -1 else ""
        print(f"  0. {none_label}{default_marker}")
    
    for i, (name, display_name) in enumerate(options):
        is_default = (default_index == i)
        marker = " [Default]" if is_default else ""
        # When allow_none, options start at 1; otherwise at 1 as well
        display_num = i + 1
        print(f"  {display_num}. {display_name}{marker}")
    
    # Build prompt with default info
    if default_index is not None:
        if default_index == -1 and allow_none:
            default_display = "0"
        else:
            default_display = str(default_index + 1)
        full_prompt = f"\n{prompt_text} [Default: {default_display}]: "
    else:
        full_prompt = f"\n{prompt_text}: "
    
    while True:
        choice_input = input(full_prompt).strip()
        
        # Handle default (Enter key)
        if not choice_input:
            if default_index is not None:
                return default_index
            else:
                print("  Error: A selection is required.")
                continue
        
        try:
            choice = int(choice_input)
            
            if allow_none and choice == 0:
                return -1
            
            # Convert 1-based input to 0-based index
            adjusted = choice - 1
            if 0 <= adjusted < len(options):
                return adjusted
            else:
                print("  Error: Please enter a number from the list.")
        except ValueError:
            print("  Error: Please enter a valid number.")


def prompt_yes_no(prompt_text: str, default: bool = True) -> bool:
    """
    Yes/No prompt with consistent default handling.
    
    Args:
        prompt_text: Text to display
        default: Default value (True=Yes, False=No)
        
    Returns:
        bool: User's choice
    """
    default_hint = "Y/n" if default else "y/N"
    full_prompt = f"{prompt_text} [{default_hint}]: "
    
    while True:
        response = input(full_prompt).strip().lower()
        
        if not response:
            return default
        elif response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        else:
            print("  Error: Please enter 'y' or 'n'.")


def prompt_url(prompt_text: str, default_url: str) -> str:
    """
    URL input prompt with validation.
    
    Args:
        prompt_text: Text to display
        default_url: Default URL value
        
    Returns:
        str: Validated URL
    """
    full_prompt = f"{prompt_text} [Default: {default_url}]: "
    url = input(full_prompt).strip()
    
    if not url:
        return default_url
    
    # Add http:// if not present
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    
    return url


def print_mode_header(mode_name: str, description: str) -> None:
    """Print consistent mode header."""
    print("\n" + "-" * 50)
    print(f"MODE: {mode_name}")
    print(description)
    print("-" * 50)


# Backward compatibility aliases (prefixed with underscore in original)
_prompt_selection = prompt_selection
_prompt_yes_no = prompt_yes_no
_prompt_url = prompt_url
_print_mode_header = print_mode_header
