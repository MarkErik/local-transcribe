#!/usr/bin/env python3
"""
Test script for running the granite+mfa plugin on a specific audio segment.

This script crops the audio file to the specified segment (64.600s - 69.700s)
and then transcribes it using the granite8b model with MFA alignment.
"""

import pathlib
import tempfile
import os
from local_transcribe.lib.audio_cropper import AudioCropper
from local_transcribe.providers.transcribers.granite_mfa import GraniteMFATranscriberProvider
from local_transcribe.lib.system_capability_utils import set_system_capability

def main():
    # Set system capability to MPS for GPU acceleration
    set_system_capability("mps")
    
    # Audio file path
    audio_file = pathlib.Path("samples/audioMA-P28.m4a")

    # Segment to transcribe (in seconds)
    start_seconds = 64.600
    end_seconds = 69.700

    # Convert to minutes for the cropper
    start_minutes = start_seconds / 60.0
    end_minutes = end_seconds / 60.0

    print(f"Cropping audio from {start_seconds:.3f}s ({start_minutes:.3f}min) to {end_seconds:.3f}s ({end_minutes:.3f}min)")

    # Initialize audio cropper
    cropper = AudioCropper()

    # Create temporary file for cropped audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        cropped_audio_path = temp_file.name

    try:
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

        # Initialize the transcriber
        provider = GraniteMFATranscriberProvider()

        # Adjust chunk settings for short audio - process as single chunk
        provider.chunk_length_seconds = 10.0  # Larger than audio duration to process as one chunk
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
        required_models = provider.get_required_models(selected_model="granite-2b")
        print(f"Required models: {required_models}")
        provider.ensure_models_available(required_models, models_dir)

        # Set up MFA models directory
        provider.mfa_models_dir = models_dir / "aligners" / "mfa"

        # Transcribe with alignment using granite-2b model (smaller for MPS)
        print("Starting transcription with granite-2b + MFA...")
        result = provider.transcribe_with_alignment(
            audio_path=cropped_audio_path,
            transcriber_model="granite-2b",
            role=None  # No specific role for this test
        )

        print("\nTranscription Results:")
        print("=" * 50)

        for chunk in result:
            chunk_id = chunk["chunk_id"]
            chunk_start = chunk["chunk_start_time"]
            words = chunk["words"]

            print(f"\nChunk {chunk_id} (starts at {chunk_start:.2f}s):")
            print("-" * 30)

            for word in words:
                text = word["text"]
                start = word["start"]
                end = word["end"]
                print(f"[{start:.2f}-{end:.2f}] {text}")

        # Also print the full text
        full_text = ""
        for chunk in result:
            for word in chunk["words"]:
                full_text += word["text"] + " "

        print(f"\nFull transcript: {full_text.strip()}")

    finally:
        # Clean up temporary file
        if os.path.exists(cropped_audio_path):
            os.unlink(cropped_audio_path)
            print(f"Cleaned up temporary file: {cropped_audio_path}")

if __name__ == "__main__":
    main()