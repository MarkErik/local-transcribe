# Granite Model Loading
# Handles model loading and downloading for Granite models
#
# Note: Heavy imports (torch, transformers) are lazily loaded when needed
# to avoid slow startup times when this provider is not used.

import importlib
import os
import pathlib
import sys
from typing import Any, Optional

from local_transcribe.lib.system_capability_utils import get_system_capability


class ModelLoadingMixin:
    """Mixin class providing model loading functionality for GraniteModelManager.
    
    This mixin provides all model loading-related methods including:
    - Loading models from cache
    - Downloading models from HuggingFace
    - Reloading HuggingFace modules
    """
    
    def _reload_huggingface_modules(self) -> None:
        """
        Reload HuggingFace modules to ensure fresh configuration for model loading.
        """
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
        resolved_cache_dir = self._cache_manager.resolve_cache_directory()
        
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
    
    def _load_model(self, model_name: str) -> None:
        """
        Load a Granite model from the cache directory.
        """
        self.logger.info(f"Loading model: {model_name}")
        
        # Get the resolved cache directory
        cache_dir = self._cache_manager.resolve_cache_directory()
        
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
