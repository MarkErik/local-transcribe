# Granite Model Validation
# Handles model validation, availability checking, and preloading for Granite models

import os
import pathlib
from typing import Any, Dict, List, Optional


class ModelValidationMixin:
    """Mixin class providing model validation functionality for GraniteModelManager.
    
    This mixin provides all model validation-related methods including:
    - Getting required models
    - Validating model selection
    - Checking offline availability
    - Ensuring models are available
    - Preloading models
    """
    
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
        cache_dir = self._cache_manager.resolve_cache_directory()
        
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
        cache_dir = self._cache_manager.resolve_cache_directory()
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
