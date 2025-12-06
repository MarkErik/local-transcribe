#!/usr/bin/env python3
"""
Test script for running the granite+mfa plugin on a specific audio segment.

This script can transcribe a full audio file or a cropped segment using the granite model with MFA alignment.
"""

import pathlib
import tempfile
import os
import argparse
import sys

# Add the project root to Python path for local imports
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from local_transcribe.lib.audio_cropper import AudioCropper
from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
from local_transcribe.lib.system_capability_utils import set_system_capability

def main():
    parser = argparse.ArgumentParser(description="Test script for granite+mfa transcription")
    parser.add_argument('--audio', required=True, help='Path to the audio file to transcribe')
    parser.add_argument('--start', '--start-time', type=float, help='Start time in seconds')
    parser.add_argument('--end', '--end-time', type=float, help='End time in seconds')
    parser.add_argument('--chunk-length-seconds', type=float, default=10.0, help='Chunk length in seconds (default: 10.0)')
    parser.add_argument('--model', choices=['2b', '8b'], default='2b', help='Model size to use: 2b or 8b (default: 2b)')
    parser.add_argument('--out-file', default='granite_mfa_test_results.txt', help='Output file for results (default: granite_mfa_test_results.txt)')
    args = parser.parse_args()

    # Set system capability to MPS for GPU acceleration
    set_system_capability("mps")
    
    # Audio file path
    audio_file = pathlib.Path(args.audio)

    # Get full audio duration
    cropper = AudioCropper()
    full_duration = cropper.get_audio_duration(str(audio_file))
    print(f"Full audio duration: {full_duration:.2f}s")

    # Segment to transcribe (in seconds)
    start_seconds = args.start
    end_seconds = args.end
    chunk_length_seconds = args.chunk_length_seconds
    model_size = args.model
    out_file = args.out_file

    model_name = f"granite-{model_size}"

    audio_to_transcribe = None
    cropped_audio_path = None

    # Determine cropping parameters
    if start_seconds is None and end_seconds is None:
        # Transcribe full file
        print("Transcribing full audio file")
        audio_to_transcribe = str(audio_file)
    else:
        # Need to crop
        if start_seconds is None:
            start_seconds = 0.0
        if end_seconds is None:
            end_seconds = full_duration

        # Convert to minutes for the cropper
        start_minutes = start_seconds / 60.0
        end_minutes = end_seconds / 60.0

        print(f"Cropping audio from {start_seconds:.3f}s ({start_minutes:.3f}min) to {end_seconds:.3f}s ({end_minutes:.3f}min)")

        # Create temporary file for cropped audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            cropped_audio_path = temp_file.name

        # Crop the audio segment
        cropper.crop_audio(
            input_path=audio_file,
            output_path=cropped_audio_path,
            start_minutes=start_minutes,
            end_minutes=end_minutes
        )

        # Check the duration of cropped audio
        cropped_duration = cropper.get_audio_duration(cropped_audio_path)
        print(f"Cropped audio duration: {cropped_duration:.2f}s (expected: {end_seconds - start_seconds:.2f}s)")

        audio_to_transcribe = cropped_audio_path

    try:
        # Initialize the transcriber
        provider = GraniteMFATranscriberProvider()

        # Adjust chunk settings
        provider.chunk_length_seconds = chunk_length_seconds
        provider.overlap_seconds = 0.0
        provider.min_chunk_seconds = 0.5  # Allow very short chunks

        # Set up models directory
        project_root = pathlib.Path(__file__).parent
        models_dir = project_root / ".models"

        # Set environment variables to fix model loading and tokenizers warning
        import os
        os.environ["HF_HOME"] = str(models_dir / "transcribers" / "granite")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Ensure models are available
        print("Ensuring models are available...")
        required_models = provider.get_required_models(selected_model=model_name)
        print(f"Required models: {required_models}")
        provider.ensure_models_available(required_models, models_dir)

        # Set up MFA models directory
        provider.mfa_models_dir = models_dir / "aligners" / "mfa"

        # Transcribe with alignment using the selected granite model (smaller for MPS)
        print(f"Starting transcription with {model_name} + MFA...")
        result = provider.transcribe_with_alignment(
            audio_path=audio_to_transcribe,
            transcriber_model=model_name,
            role=None  # No specific role for this test
        )

        # Adjust timestamps if we cropped the audio
        if start_seconds is not None:
            for chunk in result:
                chunk["chunk_start_time"] += start_seconds
                for word in chunk["words"]:
                    word["start"] += start_seconds
                    word["end"] += start_seconds

        # Collect results for output file
        output_lines = []
        output_lines.append("Transcription Results:")
        output_lines.append("=" * 50)

        for chunk in result:
            chunk_id = chunk["chunk_id"]
            chunk_start = chunk["chunk_start_time"]
            words = chunk["words"]

            output_lines.append(f"\nChunk {chunk_id} (starts at {chunk_start:.2f}s):")
            output_lines.append("-" * 30)

            for word in words:
                text = word["text"]
                start = word["start"]
                end = word["end"]
                output_lines.append(f"[{start:.2f}-{end:.2f}] {text}")

        # Also print the full text
        full_text = ""
        for chunk in result:
            for word in chunk["words"]:
                full_text += word["text"] + " "

        output_lines.append(f"\nFull transcript: {full_text.strip()}")

        # Print to console
        print('\n'.join(output_lines))

        # Write to output file
        with open(out_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Results written to {out_file}")

    finally:
        # Clean up temporary file if it was created
        if cropped_audio_path and os.path.exists(cropped_audio_path):
            os.unlink(cropped_audio_path)
            print(f"Cleaned up temporary file: {cropped_audio_path}")

if __name__ == "__main__":
    main()