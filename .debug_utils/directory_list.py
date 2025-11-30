#!/usr/bin/env python3
"""
Utility script to list the contents of the models directory recursively.
This is useful for debugging when the models directory is too large
to be included in the context window.
"""

import os
from pathlib import Path

def list_models_directory():
    """List all files and directories in the models directory recursively."""
    # Get the project root (parent of .debug-utils)
    project_root = Path(__file__).parent.parent
    models_dir = project_root / ".models"

    print(f"Contents of {models_dir} (recursive):")
    print("=" * 60)
    
    if not models_dir.exists():
        print("Models directory does not exist.")
        return
    
    if not models_dir.is_dir():
        print("Models path exists but is not a directory.")
        return
    
    try:
        # Walk through the directory tree
        for root, dirs, files in os.walk(models_dir):
            # Get the relative path from models_dir
            relative_root = Path(root).relative_to(models_dir)
            
            # Print directory name (unless it's the root)
            if relative_root != Path("."):
                indent_level = len(relative_root.parts) - 1
                indent = "  " * indent_level
                print(f"{indent}{relative_root.name}/")
            else:
                print(".models/")
                indent = ""
            
            # Print files in current directory
            current_indent = indent + "  " if relative_root != Path(".") else "  "
            for file_name in sorted(files):
                print(f"{current_indent}{file_name}")
                
            # Sort directories for consistent output
            dirs.sort()
                
    except PermissionError:
        print("Permission denied accessing models directory.")
    except Exception as e:
        print(f"Error accessing models directory: {e}")

if __name__ == "__main__":
    list_models_directory()