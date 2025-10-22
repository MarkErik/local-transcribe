import wave
import numpy as np
import os

def create_dummy_wav(filename, duration=5, sample_rate=16000):
    """Create a silent WAV file."""
    try:
        # Create silence (zero amplitude)
        silence = np.zeros(int(duration * sample_rate), dtype=np.int16)
        
        # Write to WAV file
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence.tobytes())
        
        print(f"Created dummy audio file: {filename}")
        return True
    except Exception as e:
        print(f"Failed to create dummy audio file: {e}")
        return False

if __name__ == "__main__":
    # Create a dummy mixed audio file
    input_dir = "temp_input"
    output_dir = "temp_output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    mixed_audio_file = os.path.join(input_dir, "mixed.wav")
    
    if create_dummy_wav(mixed_audio_file):
        print(f"\nYou can now run the main script with:")
        print(f"python main.py -c {mixed_audio_file} --outdir {output_dir}")
    else:
        print("Could not create dummy audio file.")