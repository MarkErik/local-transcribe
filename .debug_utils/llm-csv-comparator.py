#!/usr/bin/env python3

import argparse
import csv
import requests
import sys

def read_csv_words(filepath):
    words = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header if present
        for row in reader:
            if len(row) >= 2:
                words.append(row[1])  # Word column
    return words

def compare_chunks(words1, words2, llm_url, chunk_size=400):
    min_len = min(len(words1), len(words2))
    max_len = max(len(words1), len(words2))
    chunks = []
    for i in range(0, max_len, chunk_size):
        chunk1 = words1[i:i+chunk_size] if i < len(words1) else []
        chunk2 = words2[i:i+chunk_size] if i < len(words2) else []
        chunks.append((chunk1, chunk2))
    
    for idx, (chunk1, chunk2) in enumerate(chunks):
        if not chunk1 and not chunk2:
            continue
        prompt = f"Compare the following two sequences of words from two transcripts:\n\nTranscript 1: {' '.join(chunk1)}\n\nTranscript 2: {' '.join(chunk2)}\n\nSummarize the differences between them."
        try:
            response = requests.post(llm_url, json={"prompt": prompt}, timeout=60)
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "No response")
        except Exception as e:
            summary = f"Error: {str(e)}"
        print(f"Chunk {idx+1}: {summary}")

def main():
    parser = argparse.ArgumentParser(description="Compare two CSV files using LLM")
    parser.add_argument("file1", help="Path to first CSV file")
    parser.add_argument("file2", help="Path to second CSV file")
    parser.add_argument("llm_url", help="URL of the LLM endpoint")
    args = parser.parse_args()
    
    words1 = read_csv_words(args.file1)
    words2 = read_csv_words(args.file2)
    
    compare_chunks(words1, words2, args.llm_url)

if __name__ == "__main__":
    main()