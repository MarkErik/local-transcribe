#!/usr/bin/env python3
"""
Audit logging for de-identification.

Creates detailed audit logs documenting all name replacements.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from local_transcribe.lib.program_logger import log_progress

from .core import WordReplacement, DiscoveredName, format_timestamp


class AuditLogger:
    """Creates audit logs for de-identification operations."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the audit logger.
        
        Args:
            output_dir: Directory to save audit logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_first_pass_audit_log(
        self,
        replacements: List[WordReplacement],
        speaker_name: Optional[str] = None,
        total_words: int = 0,
        all_words: Optional[List[str]] = None
    ) -> Path:
        """
        Create audit log for first-pass de-identification.
        
        Args:
            replacements: List of replacements made
            speaker_name: Optional speaker name
            total_words: Total words processed
            all_words: Optional list of all words for context
            
        Returns:
            Path to the created audit log
        """
        if speaker_name:
            filename = f"{speaker_name.lower()}_first_pass_audit.txt"
        else:
            filename = "first_pass_audit.txt"
        
        audit_path = self.output_dir / filename
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            f.write("First-Pass De-Identification Audit Log\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {timestamp}\n")
            if speaker_name:
                f.write(f"Speaker: {speaker_name}\n")
            f.write(f"Total names replaced: {len(replacements)}\n")
            f.write("=" * 50 + "\n\n")
            
            if not replacements:
                f.write("No names were replaced.\n")
            else:
                f.write("Replacements:\n")
                f.write("-" * 50 + "\n")
                
                self._write_replacements(f, replacements, all_words)
            
            self._write_summary(f, replacements, total_words)
        
        log_progress(f"First-pass audit log saved to {audit_path}")
        return audit_path
    
    def create_combined_audit_log(
        self,
        first_pass_replacements: List[WordReplacement],
        second_pass_replacements: List[WordReplacement],
        speaker_name: Optional[str] = None,
        total_words: int = 0,
        global_names: Optional[List[DiscoveredName]] = None,
        all_words: Optional[List[str]] = None
    ) -> Path:
        """
        Create combined audit log for both passes.
        
        Args:
            first_pass_replacements: Replacements from first pass
            second_pass_replacements: Replacements from second pass
            speaker_name: Optional speaker name
            total_words: Total words processed
            global_names: Names searched for in second pass
            all_words: Optional list of all words for context
            
        Returns:
            Path to the created audit log
        """
        if speaker_name:
            filename = f"{speaker_name.lower()}_combined_audit.txt"
        else:
            filename = "combined_audit.txt"
        
        audit_path = self.output_dir / filename
        timestamp = datetime.now().strftime("%H:%M:%S")
        total_replacements = len(first_pass_replacements) + len(second_pass_replacements)
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            f.write("Combined De-Identification Audit Log\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {timestamp}\n")
            if speaker_name:
                f.write(f"Speaker: {speaker_name}\n")
            f.write(f"Total words processed: {total_words:,}\n")
            f.write(f"Total names replaced: {total_replacements}\n")
            f.write(f"  - First pass: {len(first_pass_replacements)}\n")
            f.write(f"  - Second pass: {len(second_pass_replacements)}\n")
            f.write("=" * 60 + "\n\n")
            
            # First pass section
            f.write("FIRST PASS REPLACEMENTS (General De-identification)\n")
            f.write("-" * 60 + "\n")
            if not first_pass_replacements:
                f.write("No names replaced in first pass.\n")
            else:
                self._write_replacements(f, first_pass_replacements, all_words)
            
            f.write("\n")
            
            # Second pass section
            f.write("SECOND PASS REPLACEMENTS (Targeted Name Review)\n")
            f.write("-" * 60 + "\n")
            if global_names:
                f.write(f"Names searched for: {', '.join(n.name for n in global_names)}\n\n")
            
            if not second_pass_replacements:
                f.write("No additional names found in second pass.\n")
            else:
                self._write_replacements(f, second_pass_replacements, all_words, show_matched=True)
            
            # Summary
            f.write("\n" + "=" * 60 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 60 + "\n")
            
            # Unique names
            all_names = set()
            for rep in first_pass_replacements + second_pass_replacements:
                if rep.original:
                    all_names.add(rep.original.lower())
            all_names.discard('')
            
            f.write(f"Unique names redacted: {len(all_names)}\n")
            if all_names:
                f.write("Names: " + ", ".join(sorted(all_names)) + "\n")
            
            if total_words > 0:
                rate = (total_replacements / total_words) * 100
                f.write(f"Replacement rate: {rate:.2f}%\n")
            
            f.write("=" * 60 + "\n")
        
        log_progress(f"Combined audit log saved to {audit_path}")
        return audit_path
    
    def _write_replacements(
        self,
        f,
        replacements: List[WordReplacement],
        all_words: Optional[List[str]] = None,
        show_matched: bool = False
    ) -> None:
        """Write replacement entries to audit file."""
        for rep in replacements:
            if rep.timestamp is not None:
                ts_str = format_timestamp(rep.timestamp)
                speaker_str = f", speaker: {rep.speaker}" if rep.speaker else ""
                
                if show_matched and rep.matched_from_list:
                    f.write(f"[{ts_str}] \"{rep.original}\" → [REDACTED] (matched: {rep.matched_from_list})\n")
                else:
                    f.write(f"[{ts_str}] \"{rep.original}\" → [REDACTED]{speaker_str}\n")
                
                if all_words:
                    context = self._extract_context_words(all_words, rep.word_index)
                    if context:
                        f.write(f"    Context: {context}\n")
            else:
                if show_matched and rep.matched_from_list:
                    f.write(f"[word {rep.word_index}] \"{rep.original}\" → [REDACTED] (matched: {rep.matched_from_list})\n")
                else:
                    f.write(f"[word {rep.word_index}] \"{rep.original}\" → [REDACTED]\n")
                
                if all_words:
                    context = self._extract_context_words(all_words, rep.word_index)
                    if context:
                        f.write(f"    Context: {context}\n")
    
    def _write_summary(
        self,
        f,
        replacements: List[WordReplacement],
        total_words: int
    ) -> None:
        """Write summary section to audit file."""
        f.write("\n" + "=" * 50 + "\n")
        f.write("Summary:\n")
        f.write(f"- Total words processed: {total_words:,}\n")
        f.write(f"- Names replaced: {len(replacements)}\n")
        if total_words > 0:
            rate = (len(replacements) / total_words) * 100
            f.write(f"- Replacement rate: {rate:.2f}%\n")
    
    def _extract_context_words(
        self,
        all_words: List[str],
        word_index: int,
        before: int = 5,
        after: int = 3
    ) -> str:
        """
        Extract context words around a specific word index.
        
        Args:
            all_words: List of all words
            word_index: Index of target word
            before: Words to include before
            after: Words to include after
            
        Returns:
            Context string with target word highlighted
        """
        if not all_words or word_index < 0 or word_index >= len(all_words):
            return ""
        
        start_idx = max(0, word_index - before)
        end_idx = min(len(all_words), word_index + after + 1)
        
        context_words = list(all_words[start_idx:end_idx])
        
        if word_index >= start_idx and word_index < end_idx:
            target_pos = word_index - start_idx
            context_words[target_pos] = f">>>{context_words[target_pos]}<<<"
        
        return " ".join(context_words)
