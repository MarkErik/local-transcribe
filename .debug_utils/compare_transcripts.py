import csv
import re
import argparse
from pathlib import Path

def read_csv_words(csv_path):
    """Read words from CSV file."""
    words = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 1:
                words.append(row[1].lower())
    return words

def read_txt_words(txt_path):
    """Read words from text file."""
    with open(txt_path, 'r') as file:
        content = file.read().strip()
        # Split by whitespace and convert to lowercase
        words = re.findall(r'\b\w+\b', content.lower())
    return words

def read_full_content(file_path):
    """Read full content of file."""
    with open(file_path, 'r') as file:
        return file.read().strip()

def count_punctuation(text):
    """Count punctuation characters in text."""
    return len(re.findall(r'[^\w\s]', text))

def log_output(message, log_file=None):
    """Print to console and optionally to log file."""
    print(message)
    if log_file:
        log_file.write(message + '\n')

def compare_transcripts(csv_path, txt_path, log_file_path=None):
    """Compare two transcript files."""
    log_file = None
    if log_file_path:
        log_file = open(log_file_path, 'w', encoding='utf-8')
    
    try:
        csv_words = read_csv_words(csv_path)
        txt_words = read_txt_words(txt_path)
        
        csv_content = read_full_content(csv_path)
        txt_content = read_full_content(txt_path)
        
        log_output(f"CSV file ({csv_path}) contains {len(csv_words)} words", log_file)
        log_output(f"TXT file ({txt_path}) contains {len(txt_words)} words", log_file)
        log_output("", log_file)
        
        # Punctuation comparison
        csv_punct = count_punctuation(csv_content)
        txt_punct = count_punctuation(txt_content)
        punct_diff = abs(csv_punct - txt_punct)
        log_output(f"Punctuation count - CSV: {csv_punct}, TXT: {txt_punct} (difference: {punct_diff})", log_file)
        if punct_diff > 10:  # Arbitrary threshold for "significant"
            log_output("Warning: Significant punctuation difference detected!", log_file)
        log_output("", log_file)
        
        # Find common words
        common_words = set(csv_words) & set(txt_words)
        log_output(f"Number of unique common words: {len(common_words)}", log_file)
        log_output("", log_file)
        
        # Words only in CSV
        csv_only = set(csv_words) - set(txt_words)
        log_output(f"Words only in CSV file ({len(csv_only)}):", log_file)
        for word in sorted(csv_only):
            log_output(f"  - {word}", log_file)
        log_output("", log_file)
        
        # Words only in TXT
        txt_only = set(txt_words) - set(csv_words)
        log_output(f"Words only in TXT file ({len(txt_only)}):", log_file)
        for word in sorted(txt_only):
            log_output(f"  - {word}", log_file)
        log_output("", log_file)
        
        # Check if sequences match
        log_output("Sequence comparison:", log_file)
        if len(csv_words) == len(txt_words):
            log_output("  Both files have the same number of words.", log_file)
            
            mismatches = []
            for i, (csv_word, txt_word) in enumerate(zip(csv_words, txt_words)):
                if csv_word != txt_word:
                    mismatches.append((i+1, csv_word, txt_word))
            
            if mismatches:
                log_output(f"  Found {len(mismatches)} word mismatches:", log_file)
                for line_num, csv_word, txt_word in mismatches[:10]:  # Show first 10
                    log_output(f"    Line {line_num}: CSV='{csv_word}' vs TXT='{txt_word}'", log_file)
                if len(mismatches) > 10:
                    log_output(f"    ... and {len(mismatches) - 10} more mismatches", log_file)
            else:
                log_output("  All words match in sequence!", log_file)
        else:
            log_output(f"  Different word counts: CSV has {len(csv_words)}, TXT has {len(txt_words)}", log_file)
            
            # Find where they start to differ
            min_length = min(len(csv_words), len(txt_words))
            for i in range(min_length):
                if csv_words[i] != txt_words[i]:
                    log_output(f"  First difference at position {i+1}: CSV='{csv_words[i]}' vs TXT='{txt_words[i]}'", log_file)
                    break
            else:
                if len(csv_words) > len(txt_words):
                    log_output(f"  TXT file matches first {len(txt_words)} words of CSV, but CSV has {len(csv_words) - len(txt_words)} extra words", log_file)
                else:
                    log_output(f"  CSV file matches first {len(csv_words)} words of TXT, but TXT has {len(txt_words) - len(csv_words)} extra words", log_file)
    finally:
        if log_file:
            log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two transcript files: a CSV and a TXT.")
    parser.add_argument("csv_path", help="Path to the CSV transcript file")
    parser.add_argument("txt_path", help="Path to the TXT transcript file")
    parser.add_argument("--log-file", help="Optional path to log file for output")
    
    args = parser.parse_args()
    
    compare_transcripts(args.csv_path, args.txt_path, args.log_file)