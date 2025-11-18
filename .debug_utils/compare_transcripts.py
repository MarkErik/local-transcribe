import csv
import re

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

def compare_transcripts(csv_path, txt_path):
    """Compare two transcript files."""
    csv_words = read_csv_words(csv_path)
    txt_words = read_txt_words(txt_path)
    
    print(f"CSV file ({csv_path}) contains {len(csv_words)} words")
    print(f"TXT file ({txt_path}) contains {len(txt_words)} words")
    print()
    
    # Find common words
    common_words = set(csv_words) & set(txt_words)
    print(f"Number of unique common words: {len(common_words)}")
    print()
    
    # Words only in CSV
    csv_only = set(csv_words) - set(txt_words)
    print(f"Words only in CSV file ({len(csv_only)}):")
    for word in sorted(csv_only):
        print(f"  - {word}")
    print()
    
    # Words only in TXT
    txt_only = set(txt_words) - set(csv_words)
    print(f"Words only in TXT file ({len(txt_only)}):")
    for word in sorted(txt_only):
        print(f"  - {word}")
    print()
    
    # Check if sequences match
    print("Sequence comparison:")
    if len(csv_words) == len(txt_words):
        print("  Both files have the same number of words.")
        
        mismatches = []
        for i, (csv_word, txt_word) in enumerate(zip(csv_words, txt_words)):
            if csv_word != txt_word:
                mismatches.append((i+1, csv_word, txt_word))
        
        if mismatches:
            print(f"  Found {len(mismatches)} word mismatches:")
            for line_num, csv_word, txt_word in mismatches[:10]:  # Show first 10
                print(f"    Line {line_num}: CSV='{csv_word}' vs TXT='{txt_word}'")
            if len(mismatches) > 10:
                print(f"    ... and {len(mismatches) - 10} more mismatches")
        else:
            print("  All words match in sequence!")
    else:
        print(f"  Different word counts: CSV has {len(csv_words)}, TXT has {len(txt_words)}")
        
        # Find where they start to differ
        min_length = min(len(csv_words), len(txt_words))
        for i in range(min_length):
            if csv_words[i] != txt_words[i]:
                print(f"  First difference at position {i+1}: CSV='{csv_words[i]}' vs TXT='{txt_words[i]}'")
                break
        else:
            if len(csv_words) > len(txt_words):
                print(f"  TXT file matches first {len(txt_words)} words of CSV, but CSV has {len(csv_words) - len(txt_words)} extra words")
            else:
                print(f"  CSV file matches first {len(csv_words)} words of TXT, but TXT has {len(txt_words) - len(csv_words)} extra words")

if __name__ == "__main__":
    csv_path = "ponly/participant_transcript.csv"
    txt_path = "mfa-test-llmtur3/Intermediate_Outputs/transcription/participant_raw_transcript.txt"
    
    compare_transcripts(csv_path, txt_path)