# Granite Model Manager
# Base implementation for shared Granite model management functionality
#
# Note: Heavy imports (torch, transformers) are lazily loaded when needed
# to avoid slow startup times when this provider is not used.

import pathlib
import os
import sys
import re
import gc
from typing import Optional, Any, Dict, List, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

# Import system capability utilities
from local_transcribe.lib.system_capability_utils import get_system_capability, clear_device_cache
from local_transcribe.lib.program_logger import log_progress, log_debug, log_completion

# Type hints for lazy-loaded modules
if TYPE_CHECKING:
    import torch


class GraniteModelManager:
    """Base class for managing Granite models across different providers.
    
    Provides consolidated model management and local transcription functionality
    for all Granite-based transcriber providers.
    """
    
    MODEL_MAPPING = {
        "granite-8b": "ibm-granite/granite-speech-3.3-8b",
        "granite-2b": "ibm-granite/granite-speech-3.3-2b"
    }
    
    def __init__(
        self,
        logger: Any,
        models_dir: Optional[pathlib.Path] = None
    ):
        """
        Initialize the model manager.
        
        Args:
            logger: Logger instance for logging messages
            models_dir: Directory for storing/loading models
        """
        self.logger = logger
        self.models_dir = models_dir
        self.selected_model: Optional[str] = None
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
    
    # =========================================================================
    # Transcription Methods
    # =========================================================================
    
    # Prompt fragment markers to filter from transcription output
    _PROMPT_FRAGMENTS = [
        "make sure to include disfluencies",
        "can you transcribe the speech into a written format",
    ]
    
    @property
    def device(self) -> str:
        """Get the current device for model execution."""
        return get_system_capability()
    
    def _strip_prompt_fragments(self, text: str) -> str:
        """Strip prompt fragments from the transcription output.
        
        Sometimes the model echoes parts of the prompt in its output.
        This method removes those fragments.
        """
        if not text:
            return text
            
        lower_text = text.lower()
        cleaned_text = text
        
        for fragment in self._PROMPT_FRAGMENTS:
            idx = lower_text.find(fragment)
            if idx != -1:
                # Remove the fragment and everything before it
                cleaned_text = cleaned_text[:idx]
                lower_text = cleaned_text.lower()
        
        # Clean up any trailing punctuation or whitespace
        cleaned_text = cleaned_text.rstrip(" .,\n\t")
        
        # If we removed everything, return the original text
        if not cleaned_text.strip():
            return text.strip()
            
        return cleaned_text.strip()
    
    def _clean_transcription_output(self, text: str) -> str:
        """Clean the transcription output by removing dialogue markers and quotation marks.
        
        The Granite model sometimes adds dialogue markers like "User:" or "Assistant:"
        and quotation marks around the transcription. This method removes them.
        """
        # Count labels before removal for debug logging
        user_count = len(re.findall(r'\bUser:\s*', text, flags=re.IGNORECASE))
        assistant_count = len(re.findall(r'\bAI Assistant:\s*', text, flags=re.IGNORECASE))
        assistant_short_count = len(re.findall(r'\bAssistant:\s*', text, flags=re.IGNORECASE))
        total_removed = user_count + assistant_count + assistant_short_count
        
        text = re.sub(r'\bUser:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAI Assistant:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAssistant:\s*', '', text, flags=re.IGNORECASE)
        
        text = text.replace('"', '')
        text = text.replace('\u201C', '')
        text = text.replace('\u201D', '')
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Log count if any labels were removed
        if total_removed > 0:
            log_debug(f"Removed {total_removed} dialogue labels from transcript.")
        
        return text
    
    def _calculate_generation_params(self, segment_duration: float, input_length: int) -> Dict[str, Any]:
        """Calculate generation parameters based on segment duration.
        
        Longer segments need more tokens and benefit from repetition penalty.
        Shorter segments work better with fewer tokens and no logits processor.
        
        Args:
            segment_duration: Duration of audio segment in seconds
            input_length: Length of input token IDs for RepetitionPenaltyLogitsProcessor
            
        Returns:
            Dictionary with max_new_tokens and optional logits_processor
        """
        from transformers import RepetitionPenaltyLogitsProcessor
        
        if segment_duration < 8.0:
            # For short segments: reduce max_new_tokens, exclude logits_processor
            return {
                "max_new_tokens": 128,
                "logits_processor": None
            }
        elif segment_duration < 20.0:
            # For medium segments
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=3.0,
                prompt_ignore_length=input_length,
            )
            return {
                "max_new_tokens": 256,
                "logits_processor": [repetition_penalty_processor]
            }
        else:
            # For long segments (20+ seconds)
            repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(
                penalty=3.0,
                prompt_ignore_length=input_length,
            )
            return {
                "max_new_tokens": 512,
                "logits_processor": [repetition_penalty_processor]
            }
    
    def transcribe_segment(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000,
        **kwargs
    ) -> str:
        """
        Transcribe a single audio segment using local Granite model.
        
        This is the main transcription method that should be used by all Granite-based
        transcriber providers. It handles:
        - Duration-based parameter calculation
        - Memory cleanup
        - Output cleaning and prompt fragment stripping
        
        Args:
            audio: Audio samples as numpy array (mono, float32, 16kHz)
            sample_rate: Sample rate (should be 16000)
            **kwargs: Additional options (unused, for compatibility)
        
        Returns:
            Cleaned transcription text
        """
        return self._transcribe_segment_local(audio, sample_rate)
    
    def _transcribe_segment_local(
        self,
        audio: NDArray[np.floating],
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe a single audio segment using local Granite model.
        
        Args:
            audio: Audio samples as numpy array (mono, float32, 16kHz)
            sample_rate: Sample rate (should be 16000)
        
        Returns:
            Cleaned transcription text
        """
        log_progress("Transcribing audio segment with Granite")
        
        # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Granite model not loaded. Call _load_model() first.")
        if self.processor is None:
            raise RuntimeError("Granite processor not loaded. Call _load_model() first.")
        if self.tokenizer is None:
            raise RuntimeError("Granite tokenizer not loaded. Call _load_model() first.")
        
        try:
            # Lazy import of torch
            import torch
            
            wav_tensor = torch.from_numpy(audio).unsqueeze(0)
            segment_duration = len(audio) / sample_rate
            
            # Build chat prompt with disfluencies request for better transcript quality
            chat = [
                {
                    "role": "system",
                    "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
                },
                {
                    "role": "user",
                    "content": "<|audio|>can you transcribe the speech into a written format? make sure to include disfluencies.",
                }
            ]
            
            text = self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            
            model_inputs = self.processor.__call__(
                text,
                wav_tensor,
                device=self.device,
                return_tensors="pt",
            ).to(self.device)
            
            # Calculate generation parameters based on segment duration
            gen_params = self._calculate_generation_params(
                segment_duration,
                model_inputs["input_ids"].shape[-1]
            )
            
            with torch.no_grad():
                model_outputs = self.model.generate.__call__(
                    **model_inputs,
                    max_new_tokens=gen_params["max_new_tokens"],
                    num_beams=4,
                    do_sample=False,
                    min_length=1,
                    top_p=1.0,
                    length_penalty=1.0,
                    temperature=1.0,
                    early_stopping=True,
                    logits_processor=gen_params["logits_processor"],
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            num_input_tokens = model_inputs["input_ids"].shape[-1]
            new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
            
            output_text = self.tokenizer.batch_decode(
                new_tokens, add_special_tokens=False, skip_special_tokens=True
            )
            
            cleaned_text = self._clean_transcription_output(output_text[0].strip())
            final_text = self._strip_prompt_fragments(cleaned_text)
            
            # Log if prompt fragments were removed
            if final_text != cleaned_text:
                log_debug("Removed prompt fragments from transcription output")
            
            return final_text
            
        finally:
            # Explicit memory cleanup
            if 'wav_tensor' in locals():
                del wav_tensor
            if 'model_inputs' in locals():
                for key in list(model_inputs.keys()):
                    del model_inputs[key]
                del model_inputs
            if 'model_outputs' in locals():
                del model_outputs
            if 'new_tokens' in locals():
                del new_tokens
            
            gc.collect()
            clear_device_cache()
    
    def get_required_models(self, selected_model: Optional[str] = None) -> List[str]:
        """
        Get the list of required model names based on the selected model.
        """
        # Determine which model to use
        model_to_check = selected_model or self.selected_model
        
        # Handle special case: return all available models if "all" is requested
        if model_to_check == "all":
            return list(self.MODEL_MAPPING.values())
        
        # If no model is specified, default to granite-8b
        if not model_to_check:
            self.logger.info("No model selected, defaulting to granite-8b")
            return [self.MODEL_MAPPING["granite-8b"]]
        
        # Validate the selected model
        if model_to_check not in self.MODEL_MAPPING:
            self.logger.error(f"Invalid model selected: {model_to_check}")
            self.logger.info(f"Available models: {list(self.MODEL_MAPPING.keys())}")
            raise ValueError(f"Invalid model '{model_to_check}'. Available models: {list(self.MODEL_MAPPING.keys())}")
        
        # Return the selected model
        return [self.MODEL_MAPPING[model_to_check]]
    
    def validate_model_selection(self, selected_model: Optional[str] = None) -> bool:
        """
        Validate that the selected model exists in MODEL_MAPPING.
        """
        # Determine which model to validate
        model_to_validate = selected_model or self.selected_model
        
        # Handle special case: "all" is always valid
        if model_to_validate == "all":
            self.logger.info("Model validation: 'all' is a valid selection")
            return True
        
        # If no model is specified, default validation passes (will use default in get_required_models)
        if not model_to_validate:
            self.logger.info("Model validation: No model specified, will use default")
            return True
        
        # Check if model exists in mapping
        if model_to_validate in self.MODEL_MAPPING:
            self.logger.info(f"Model validation: '{model_to_validate}' is valid")
            return True
        else:
            self.logger.warning(f"Model validation: '{model_to_validate}' is not valid")
            self.logger.info(f"Available models: {list(self.MODEL_MAPPING.keys())}")
            return False
    
    def check_models_available_offline(self, models: List[str], models_dir: pathlib.Path) -> List[str]:
        """
        Check which models are available offline without downloading.
        """
        missing_models = []
        
        # Get the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        
        for model in models:
            # Check if it's a full model name or a short name
            if model in self.MODEL_MAPPING.values():
                # It's a full model name
                hf_model_name = model.replace("/", "--")
                model_dir = cache_dir / f"models--{hf_model_name}"
            elif model in self.MODEL_MAPPING:
                # It's a short name, convert to full name first
                full_model_name = self.MODEL_MAPPING[model]
                hf_model_name = full_model_name.replace("/", "--")
                model_dir = cache_dir / f"models--{hf_model_name}"
            else:
                # Unknown model
                self.logger.warning(f"Unknown model: {model}, skipping availability check")
                continue
            
            # Check for model files (both .bin and .safetensors formats)
            has_model_files = (
                model_dir.exists() and (
                    any(model_dir.rglob("*.bin")) or
                    any(model_dir.rglob("*.safetensors"))
                )
            )
            
            if not has_model_files:
                missing_models.append(model)
        
        return missing_models
    
    def ensure_models_available(self, models: List[str], models_dir: pathlib.Path) -> None:
        """
        Ensure models are available by preloading them if needed.
        """
        self.logger.info(f"Ensuring models are available: {models}")
        
        # Check which models are missing
        missing_models = self.check_models_available_offline(models, models_dir)
        
        if not missing_models:
            self.logger.info("All models are already available offline")
            return
        
        self.logger.info(f"Missing models that need to be downloaded: {missing_models}")
        
        # Preload the missing models
        self.preload_models(missing_models, models_dir)
        
        self.logger.info("All models are now available offline")
    
    def preload_models(self, models: List[str], models_dir: pathlib.Path) -> None:
        """
        Preload Granite models to cache.
        """
        self.logger.info("Starting model preload for Granite models")
        
        # Create cache directory structure using the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        provider_cache_dir = models_dir / "transcribers" / "granite"
        provider_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original environment variables
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        
        # Ensure offline mode is disabled for downloading
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        try:
            # Reload huggingface_hub modules to ensure fresh configuration
            self._reload_huggingface_modules()
            
            # Import huggingface_hub after module reload
            from huggingface_hub import snapshot_download
            
            # Validate models using existing method
            required_models = self.get_required_models()
            valid_models = []
            
            for model in models:
                if model in self.MODEL_MAPPING.values():
                    valid_models.append(model)
                elif model in self.MODEL_MAPPING:
                    # Convert short name to full name
                    valid_models.append(self.MODEL_MAPPING[model])
                else:
                    self.logger.warning(f"Unknown model: {model}, skipping")
            
            if not valid_models:
                self.logger.error("No valid models to preload")
                return
            
            # Download each model
            for i, model in enumerate(valid_models):
                self.logger.info(f"Downloading model {i+1}/{len(valid_models)}: {model}")
                
                # Set HF_HOME to our cache directory
                os.environ["HF_HOME"] = str(cache_dir)
                
                try:
                    # Get authentication token
                    token = os.getenv("HF_TOKEN")
                    
                    # Download the model
                    snapshot_download(model, token=token)
                    self.logger.info(f"Successfully downloaded {model}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to download {model}: {e}")
                    raise Exception(f"Failed to download {model}: {e}")
                    
        finally:
            # Restore original environment variables
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)
    
    def _reload_huggingface_modules(self) -> None:
        """
        Reload HuggingFace modules to ensure fresh configuration for model loading.
        """
        import importlib
        import sys
        
        self.logger.debug("Reloading HuggingFace modules for fresh configuration")
        
        try:
            # Remove huggingface_hub modules from cache
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('huggingface_hub')]
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    self.logger.debug(f"Removed cached module: {module_name}")
            
            # Remove transformers modules from cache
            modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
            for module_name in modules_to_reload:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                    self.logger.debug(f"Removed cached module: {module_name}")
            
            self.logger.debug("HuggingFace modules reloaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Error while reloading HuggingFace modules: {e}")
            # Continue with execution even if module reload fails
    
    def _download_model(self, model_name: str, cache_dir: pathlib.Path) -> None:
        """
        Download a specific model from HuggingFace Hub to the cache directory.
        """
        self.logger.info(f"Starting download for model: {model_name}")
        
        # Use the resolved cache directory for standard HuggingFace models
        resolved_cache_dir = self._resolve_cache_directory()
        
        # Store original environment variables
        offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
        original_hf_home = os.environ.get("HF_HOME")
        
        # Ensure offline mode is disabled for downloading
        os.environ["HF_HUB_OFFLINE"] = "0"
        
        try:
            # Reload huggingface_hub modules to ensure fresh configuration
            self._reload_huggingface_modules()
            
            # Import huggingface_hub after module reload
            from huggingface_hub import snapshot_download
            
            # Set HF_HOME to our resolved cache directory
            os.environ["HF_HOME"] = str(resolved_cache_dir)
            
            # Get authentication token
            token = os.getenv("HF_TOKEN")
            
            if token:
                self.logger.debug("Using HF_TOKEN for authentication")
            else:
                self.logger.warning("No HF_TOKEN found - downloading without authentication")
            
            # Download the model with progress tracking
            self.logger.info(f"Downloading model to: {resolved_cache_dir}")
            snapshot_download(model_name, token=token)
            
            self.logger.info(f"Successfully downloaded {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to download {model_name}: {e}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
        finally:
            # Restore original environment variables
            os.environ["HF_HUB_OFFLINE"] = offline_mode
            if original_hf_home is not None:
                os.environ["HF_HOME"] = original_hf_home
            else:
                os.environ.pop("HF_HOME", None)
    
    def _resolve_cache_directory(self) -> pathlib.Path:
        """
        Resolve the cache directory for storing models.
        """
        # Check for XDG_CACHE_HOME environment variable first
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            models_root = pathlib.Path(xdg_cache_home)
        else:
            # Fallback to standard HuggingFace cache location
            models_root = pathlib.Path.home() / ".cache" / "huggingface"
        
        # The models are stored in the standard HuggingFace hub structure
        cache_dir = models_root / "huggingface" / "hub"
        
        try:
            # Create directory structure if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Log the resolved cache directory for debugging
            self.logger.debug(f"Resolved cache directory: {cache_dir}")
            
            return cache_dir
            
        except OSError as e:
            error_msg = f"Failed to create cache directory {cache_dir}: {e}"
            self.logger.error(error_msg)
            raise OSError(error_msg)
    
    def _load_model(self, model_name: str) -> None:
        """
        Load a Granite model from the cache directory.
        """
        self.logger.info(f"Loading model: {model_name}")
        
        # Get the resolved cache directory
        cache_dir = self._resolve_cache_directory()
        
        try:
            # Import required modules
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            
            # Get authentication token
            token = os.getenv("HF_TOKEN")
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                local_files_only=True,
                token=token
            )
            # Extract tokenizer from processor with null check
            if self.processor is not None:
                self.tokenizer = getattr(self.processor, 'tokenizer', None)
            else:
                self.tokenizer = None
            
            # Load the model with proper device handling
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                local_files_only=True,
                token=token
            )
            
            # Move model to the appropriate device
            try:
                # Lazy import of torch
                import torch
                device = get_system_capability()
                
                if self.model is not None:
                    if device == "mps" and torch.backends.mps.is_available():
                        # For MPS, we need to explicitly move the model and set device
                        self.model = self.model.to("mps")
                        self.logger.info(f"Model moved to MPS device: {device}")
                    elif device == "cuda" and torch.cuda.is_available():
                        # For CUDA
                        self.model = self.model.to("cuda")
                        self.logger.info(f"Model moved to CUDA device: {device}")
                    else:
                        # For CPU or other devices
                        self.model = self.model.to("cpu")
                        self.logger.info(f"Model moved to CPU device: {device}")
                        
            except Exception as device_error:
                self.logger.warning(f"Device placement failed ({device_error}), falling back to CPU")
                if self.model is not None:
                    self.model = self.model.to("cpu")
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {e}"
            self.logger.error(error_msg)
            self.logger.debug(f"Cache directory exists: {cache_dir.exists()}")
            if cache_dir.exists():
                self.logger.debug(f"Cache directory contents: {list(cache_dir.iterdir())}")
            raise Exception(error_msg)
        