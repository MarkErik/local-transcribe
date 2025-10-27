#!/usr/bin/env python3
"""
Clean runner for local-transcribe that suppresses warnings.
This script redirects stderr to filter out specific warning messages.
"""

import sys
import os
import subprocess
from pathlib import Path

def filter_warnings(process):
    """Filter out specific warning messages from stderr."""
    while True:
        # Read line from stderr
        line = process.stderr.readline()
        if not line:
            break
        
        # Skip lines containing the warnings we want to suppress
        if any(warning in line for warning in [
            "torchcodec is not installed correctly",
            "TRANSFORMERS_CACHE is deprecated",
            "libtorchcodec",
            "Could not load libtorchcodec"
        ]):
            continue
        
        # Print the line to stderr (it will be shown)
        sys.stderr.write(line)
        sys.stderr.flush()

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Build the command to run the actual main.py
    cmd = [sys.executable, str(script_dir / "main.py")] + sys.argv[1:]
    
    # Run the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1  # Line buffered
    )
    
    # Process stdout normally
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    
    # Filter stderr
    filter_warnings(process)
    
    # Wait for process to complete
    process.wait()
    return process.returncode

if __name__ == "__main__":
    sys.exit(main())