#!/usr/bin/env python3
"""
ASR plugin using IBM Granite with MFA forced alignment.
"""

from typing import List, Optional
import os
import pathlib
import tempfile
import subprocess
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel
from huggingface_hub import hf_hub_download, snapshot_download
import librosa
from local_transcribe.framework.plugins import ASRProvider, WordSegment, registry


class GraniteMFASRProvider(ASRProvider):
    """ASR provider using IBM Granite with MFA forced alignment for word timestamps."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "ibm-granite/granite-speech-3.3-8b"
        self.processor = None
        self.model = None
        self.tokenizer = None
        # MFA setup
        self.mfa_models_dir = None

    @property
    def name(self) -> str:
        return "granite-mfa"

    @property
    def description(self) -> str:
        return "IBM Granite ASR with MFA forced alignment for word timestamps"

    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        return ["ibm-granite/granite-speech-3.3-8b"]

    def get_available_models(self) -> List[str]:
        return ["granite-8b"]

    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Preload Granite models to cache."""
        import os
        import sys
        
        # DEBUG: Log environment state before download attempt
        print(f"DEBUG: HF_HUB_OFFLINE before setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        print(f"DEBUG: HF_HOME: {os.environ.get('HF_HOME')}")
        print(f"DEBUG: HF_TOKEN: {'***' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
        
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        # DEBUG: Confirm environment variable was set
        print(f"DEBUG: HF_HUB_OFFLINE after setting to 0: {os.environ.get('HF_HUB_OFFLINE')}")
        
        # Force reload of huggingface_hub modules to pick up new environment
        print(f"DEBUG: Reloading huggingface_hub modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        # Also reload transformers modules
        print(f"DEBUG: Reloading transformers modules...")
        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
        for module_name in modules_to_reload:
            del sys.modules[module_name]
            print(f"DEBUG: Reloaded {module_name}")
        
        # Now import transformers classes after environment change
        # from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        from huggingface_hub import snapshot_download
        
        try:
            for model in models:
                if model == "ibm-granite/granite-speech-3.3-8b":
                    cache_dir = models_dir / "asr" / "granite"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    # Use snapshot_download to download the entire repo
                    token = os.getenv("HF_TOKEN")
                    snapshot_download(model, cache_dir=str(cache_dir), token=token)
                    print(f"[✓] {model} downloaded successfully.")
        except Exception as e:
            print(f"DEBUG: Download failed with error: {e}")
            print(f"DEBUG: Error type: {type(e)}")
            
            # Additional debug: Check environment at time of error
            print(f"DEBUG: At error time - HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
            print(f"DEBUG: At error time - HF_HOME: {os.environ.get('HF_HOME')}")
            
            raise Exception(f"Failed to download {model}: {e}")
        finally:
            os.environ["HF_HUB_OFFLINE"] = offline_mode

    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """Check which Granite models are available offline without downloading."""
        missing_models = []
        for model in models:
            if model == "ibm-granite/granite-speech-3.3-8b":
                cache_dir = models_dir / "asr" / "granite"
                # Check for model files (this is a simplified check)
                if not any(cache_dir.rglob("*.bin")) and not any(cache_dir.rglob("*.safetensors")):
                    missing_models.append(model)
        return missing_models

    def _load_model(self):
        """Load the Granite model if not already loaded."""
        if self.model is None:
            # Temporarily allow online access to download model if needed
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
            os.environ["HF_HUB_OFFLINE"] = "0"
            
            # Force reload of huggingface_hub and transformers modules
            import sys
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
            for module_name in modules_to_reload:
                del sys.modules[module_name]
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
            for module_name in modules_to_reload:
                del sys.modules[module_name]
            
            try:
                cache_dir = pathlib.Path(os.environ.get("HF_HOME", "./models")) / "asr" / "granite"
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # Temporarily set HF_HOME to our cache directory so PEFT loading finds the files
                original_hf_home = os.environ.get("HF_HOME")
                os.environ["HF_HOME"] = str(cache_dir)
                
                token = os.getenv("HF_TOKEN")
                # During transcription, models should be cached, so use local_files_only=True
                self.processor = AutoProcessor.from_pretrained(self.model_name, local_files_only=False, token=token)
                self.tokenizer = self.processor.tokenizer
                
                # Load base model
                base_model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name, local_files_only=False, token=token).to(self.device)
                
                # Check if this is a PEFT model
                try:
                    self.model = PeftModel.from_pretrained(base_model, self.model_name, token=token).to(self.device)
                except:
                    # If not a PEFT model, use the base model
                    self.model = base_model
            finally:
                # Restore original HF_HOME
                if original_hf_home is not None:
                    os.environ["HF_HOME"] = original_hf_home
                else:
                    os.environ.pop("HF_HOME", None)
                # Restore offline mode
                os.environ["HF_HUB_OFFLINE"] = offline_mode

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Granite model."""
        self._load_model()

        # Load audio
        wav, sr = torchaudio.load(audio_path, normalize=True)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)  # Convert to mono
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)

        # Create text prompt
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            }
        ]

        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        model_inputs = self.processor(
            text,
            wav,
            device=self.device,
            return_tensors="pt",
        ).to(self.device)

        # Generate transcription
        model_outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=200,
            num_beams=4,
            do_sample=False,
            min_length=1,
            top_p=1.0,
            repetition_penalty=3.0,
            length_penalty=1.0,
            temperature=1.0,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Extract generated text
        num_input_tokens = model_inputs["input_ids"].shape[-1]
        new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

        output_text = self.tokenizer.batch_decode(
            new_tokens, add_special_tokens=False, skip_special_tokens=True
        )

        return output_text[0].strip()

    def _align_with_mfa(self, audio_path: str, transcript: str) -> List[WordSegment]:
        """Use Montreal Forced Aligner for precise word timestamps."""
        
        # Create a temporary directory for MFA processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)
            
            # Prepare input files for MFA
            # MFA expects: audio files in one directory and matching .lab (transcript) files
            audio_dir = temp_path / "audio"
            audio_dir.mkdir()
            
            # Copy audio to temp directory with a simple name
            audio_name = "audio.wav"
            audio_file = audio_dir / audio_name
            
            # MFA requires 16kHz mono WAV - audio_path should already be standardized
            # but let's copy it to ensure proper format
            import shutil
            shutil.copy(audio_path, audio_file)
            
            # Create matching transcript file (.lab extension)
            transcript_file = audio_dir / f"{audio_name.rsplit('.', 1)[0]}.lab"
            transcript_file.write_text(transcript, encoding='utf-8')
            
            # Setup output directory for alignments
            output_dir = temp_path / "output"
            output_dir.mkdir()
            
            # Ensure MFA models directory exists
            if self.mfa_models_dir is None:
                self.mfa_models_dir = pathlib.Path(os.getenv("HF_HOME", "./models")) / "mfa"
                self.mfa_models_dir.mkdir(parents=True, exist_ok=True)
            
            # Download MFA models if needed
            self._ensure_mfa_models()
            
            # Run MFA alignment
            try:
                # MFA command: mfa align <audio_dir> <dictionary> <acoustic_model> <output_dir>
                # Using English dictionary and acoustic model
                cmd = [
                    "mfa", "align",
                    str(audio_dir),
                    "english_us_arpa",  # Dictionary
                    "english_us_arpa",  # Acoustic model
                    str(output_dir),
                    "--clean",  # Clean previous runs
                    "--quiet",  # Suppress verbose output
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse TextGrid output
                textgrid_file = output_dir / f"{audio_name.rsplit('.', 1)[0]}.TextGrid"
                
                if not textgrid_file.exists():
                    # Fallback to simple alignment if MFA fails
                    print(f"Warning: MFA did not produce TextGrid output, falling back to simple alignment")
                    return self._simple_alignment(audio_path, transcript)
                
                # Parse TextGrid to extract word timestamps
                segments = self._parse_textgrid(textgrid_file)
                
                return segments
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: MFA alignment failed: {e.stderr}")
                print(f"Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript)
            except FileNotFoundError:
                print(f"Warning: MFA not installed or not in PATH")
                print(f"Install with: conda install -c conda-forge montreal-forced-aligner")
                print(f"Falling back to simple alignment")
                return self._simple_alignment(audio_path, transcript)
    
    def _ensure_mfa_models(self):
        """Ensure MFA acoustic model and dictionary are downloaded."""
        try:
            # Check if models are already downloaded
            result = subprocess.run(
                ["mfa", "model", "list", "acoustic"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "english_us_arpa" not in result.stdout:
                print("[*] Downloading MFA English acoustic model...")
                subprocess.run(
                    ["mfa", "model", "download", "acoustic", "english_us_arpa"],
                    check=True
                )
            
            result = subprocess.run(
                ["mfa", "model", "list", "dictionary"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "english_us_arpa" not in result.stdout:
                print("[*] Downloading MFA English dictionary...")
                subprocess.run(
                    ["mfa", "model", "download", "dictionary", "english_us_arpa"],
                    check=True
                )
                
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not download MFA models: {e}")
    
    def _parse_textgrid(self, textgrid_path: pathlib.Path) -> List[WordSegment]:
        """Parse MFA TextGrid output to extract word timestamps."""
        segments = []
        
        try:
            import textgrid
            
            # Parse TextGrid file
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            
            # Find the words tier (usually the last tier)
            words_tier = None
            for tier in tg.tiers:
                if 'words' in tier.name.lower():
                    words_tier = tier
                    break
            
            if words_tier is None and len(tg.tiers) > 0:
                # Use the last tier as words tier
                words_tier = tg.tiers[-1]
            
            if words_tier is None:
                return []
            
            # Extract word segments
            for interval in words_tier:
                # Skip empty intervals
                if interval.mark.strip() and interval.mark.strip() not in ['', 'sp', 'sil']:
                    segments.append(WordSegment(
                        text=interval.mark.strip(),
                        start=float(interval.minTime),
                        end=float(interval.maxTime),
                        speaker=None
                    ))
            
            return segments
            
        except ImportError:
            print("Warning: textgrid package not installed. Install with: pip install textgrid")
            print("Falling back to manual TextGrid parsing")
            return self._parse_textgrid_manual(textgrid_path)
        except Exception as e:
            print(f"Warning: Error parsing TextGrid: {e}")
            return []
    
    def _parse_textgrid_manual(self, textgrid_path: pathlib.Path) -> List[WordSegment]:
        """Manually parse TextGrid file without external library."""
        segments = []
        
        try:
            content = textgrid_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Simple parser for TextGrid format
            in_words_tier = False
            in_interval = False
            current_start = None
            current_end = None
            current_text = None
            
            for line in lines:
                line = line.strip()
                
                # Check if we're in the words tier
                if 'name = "words"' in line.lower() or 'name = "word"' in line.lower():
                    in_words_tier = True
                    continue
                
                if in_words_tier:
                    # Look for interval boundaries
                    if 'xmin =' in line:
                        current_start = float(line.split('=')[1].strip())
                    elif 'xmax =' in line:
                        current_end = float(line.split('=')[1].strip())
                    elif 'text =' in line:
                        # Extract text between quotes
                        text_part = line.split('=', 1)[1].strip()
                        current_text = text_part.strip('"').strip()
                        
                        # If we have all three, create a segment
                        if current_start is not None and current_end is not None and current_text:
                            # Skip silence markers
                            if current_text not in ['', 'sp', 'sil', '<eps>']:
                                segments.append(WordSegment(
                                    text=current_text,
                                    start=current_start,
                                    end=current_end,
                                    speaker=None
                                ))
                        
                        # Reset for next interval
                        current_start = None
                        current_end = None
                        current_text = None
            
            return segments
            
        except Exception as e:
            print(f"Warning: Error manually parsing TextGrid: {e}")
            return []
    
    def _simple_alignment(self, audio_path: str, transcript: str) -> List[WordSegment]:
        """Fallback to simple even-distribution alignment."""
        # Get audio duration
        y, sr = librosa.load(audio_path, sr=None)
        duration = len(y) / sr

        # Simple word-based alignment
        words = transcript.split()
        if not words:
            return []

        word_duration = duration / len(words)

        segments = []
        current_time = 0.0
        for word in words:
            segments.append(WordSegment(
                text=word,
                start=current_time,
                end=min(current_time + word_duration, duration),
                speaker=None
            ))
            current_time += word_duration

        return segments

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Transcribe audio using Granite + MFA alignment.

        Args:
            audio_path: Path to audio file
            role: Speaker role for dual-track mode
            **kwargs: Additional configuration
        """
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Get transcription from Granite
        transcript = self._transcribe_audio(audio_path)

        # Align with MFA
        segments = self._align_with_mfa(audio_path, transcript)

        # Add speaker role if provided
        if role:
            for segment in segments:
                segment.speaker = role

        return segments

    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """Ensure models are available by preloading them."""
        self.preload_models(models, models_dir)


def register_asr_plugins():
    """Register ASR plugins."""
    registry.register_asr_provider(GraniteMFASRProvider())


# Auto-register on import
register_asr_plugins()