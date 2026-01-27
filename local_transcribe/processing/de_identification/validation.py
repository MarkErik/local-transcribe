#!/usr/bin/env python3
"""
Validation rules for LLM de-identification output.

Contains validators for both first-pass and second-pass de-identification
to check LLM output meets format requirements.
"""

from typing import Dict, Any

from local_transcribe.lib.program_logger import log_progress

from .core import (
    ValidationResult,
    normalize_text_for_comparison,
    words_equivalent,
)


def validate_first_pass_output(
    original: str,
    processed: str,
    max_redaction_rate: float = 0.10
) -> ValidationResult:
    """
    Validate first-pass LLM output.
    
    Checks:
    1. Word count must be exactly the same (name replacement preserves word count)
    2. No changes other than name replacement with [REDACTED]
    3. Non-redacted words must match the original exactly (case-insensitive)
    4. Redaction rate must be reasonable (not everything replaced)
    
    Args:
        original: Original input text
        processed: LLM processed text
        max_redaction_rate: Maximum allowed fraction of words to be redacted (default 10%)
        
    Returns:
        ValidationResult with pass/fail status and details
    """
    orig_words = original.split()
    proc_words = processed.split()
    
    # Count [REDACTED] tokens
    redacted_count = proc_words.count("[REDACTED]")
    
    # For small chunks, allow a minimum number of redactions regardless of rate
    # This prevents overly strict validation when processing small text samples
    # or name-heavy passages (introductions, etc.)
    min_allowed_redactions = max(10, int(len(orig_words) * max_redaction_rate))
    
    # Rule 1: Word count must match exactly
    if len(orig_words) != len(proc_words):
        log_progress(
            f"Validation failed: word count mismatch "
            f"(orig: {len(orig_words)}, proc: {len(proc_words)})"
        )
        return ValidationResult(
            passed=False,
            reason=f'word count mismatch (orig: {len(orig_words)}, proc: {len(proc_words)})',
            details={
                'original_word_count': len(orig_words),
                'processed_word_count': len(proc_words),
                'redacted_count': redacted_count
            }
        )
    
    # Rule 2: Check redaction rate (not everything replaced)
    # Allow at least min_allowed_redactions regardless of rate (for small chunks)
    if len(orig_words) > 0 and redacted_count > min_allowed_redactions:
        log_progress(
            f"Validation failed: too many replacements "
            f"({redacted_count} out of {len(orig_words)} words, max allowed: {min_allowed_redactions})"
        )
        return ValidationResult(
            passed=False,
            reason=f'too many replacements ({redacted_count} out of {len(orig_words)} words)',
            details={
                'original_word_count': len(orig_words),
                'processed_word_count': len(proc_words),
                'redacted_count': redacted_count,
                'redacted_percentage': (redacted_count / len(orig_words)) * 100,
                'max_allowed_redactions': min_allowed_redactions,
                'max_allowed_percentage': max_redaction_rate * 100
            }
        )
    
    # Rule 3: Non-redacted words must match original (Unicode-aware)
    mismatches = []
    for i in range(len(orig_words)):
        if proc_words[i] == "[REDACTED]":
            continue  # Redaction is allowed
        if not words_equivalent(orig_words[i], proc_words[i]):
            mismatches.append({
                'index': i,
                'original': orig_words[i],
                'processed': proc_words[i],
                'normalized_original': normalize_text_for_comparison(orig_words[i]),
                'normalized_processed': normalize_text_for_comparison(proc_words[i])
            })
    
    if mismatches:
        log_progress(f"Validation failed: word mismatches at {len(mismatches)} positions")
        return ValidationResult(
            passed=False,
            reason=f'word mismatches at {len(mismatches)} positions',
            details={
                'original_word_count': len(orig_words),
                'processed_word_count': len(proc_words),
                'redacted_count': redacted_count,
                'mismatches': mismatches[:10]  # Limit for readability
            }
        )
    
    return ValidationResult(
        passed=True,
        reason='validation passed',
        details={
            'original_word_count': len(orig_words),
            'processed_word_count': len(proc_words),
            'redacted_count': redacted_count
        }
    )


def validate_second_pass_output(
    original: str,
    processed: str,
    expected_redacted_min: int = 0,
    max_new_redaction_rate: float = 0.1
) -> ValidationResult:
    """
    Validate second-pass LLM output with stricter rules.
    
    Rules:
    1. Word count must be exactly the same
    2. Must not have fewer [REDACTED] tokens than input (can't remove redactions)
    3. Shouldn't have unreasonably many new redactions
    
    Args:
        original: Original input text (already partially redacted)
        processed: LLM processed text
        expected_redacted_min: Minimum number of [REDACTED] tokens expected
        max_new_redaction_rate: Maximum fraction of non-redacted words that can be newly redacted
        
    Returns:
        ValidationResult with pass/fail status and details
    """
    orig_words = original.split()
    proc_words = processed.split()
    
    # Count [REDACTED] tokens
    orig_redacted = orig_words.count("[REDACTED]")
    proc_redacted = proc_words.count("[REDACTED]")
    new_redactions = proc_redacted - orig_redacted
    
    # Rule 1: Word count must match
    if len(orig_words) != len(proc_words):
        return ValidationResult(
            passed=False,
            reason=f'word count mismatch (orig: {len(orig_words)}, proc: {len(proc_words)})',
            details={
                'original_word_count': len(orig_words),
                'processed_word_count': len(proc_words),
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted
            }
        )
    
    # Rule 2: Must not remove existing [REDACTED] tokens
    if proc_redacted < orig_redacted:
        return ValidationResult(
            passed=False,
            reason=f'removed existing redactions (orig: {orig_redacted}, proc: {proc_redacted})',
            details={
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted,
                'redactions_removed': orig_redacted - proc_redacted
            }
        )
    
    # Rule 3: Sanity check - not too many new redactions
    non_redacted_words = len(orig_words) - orig_redacted
    if non_redacted_words > 0 and new_redactions > non_redacted_words * max_new_redaction_rate:
        max_allowed = int(non_redacted_words * max_new_redaction_rate)
        return ValidationResult(
            passed=False,
            reason=f'too many new redactions ({new_redactions} added, max allowed: {max_allowed})',
            details={
                'original_redacted': orig_redacted,
                'processed_redacted': proc_redacted,
                'new_redactions': new_redactions,
                'max_allowed': max_allowed
            }
        )
    
    return ValidationResult(
        passed=True,
        reason='validation passed',
        details={
            'original_word_count': len(orig_words),
            'processed_word_count': len(proc_words),
            'original_redacted': orig_redacted,
            'processed_redacted': proc_redacted,
            'new_redactions': new_redactions
        }
    )
