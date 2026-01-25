# Granite Model Manager
# Main manager class that orchestrates all Granite model functionality

import pathlib
from typing import Any, Optional, TYPE_CHECKING

from local_transcribe.lib.system_capability_utils import get_system_capability

from .cache_management import CacheManager
from .transcription import TranscriptionMixin
from .model_loading import ModelLoadingMixin
from .model_validation import ModelValidationMixin

if TYPE_CHECKING:
    import torch


class GraniteModelManager(TranscriptionMixin, ModelLoadingMixin, ModelValidationMixin):
    """Manager for Granite speech models.
    
    Provides consolidated model management and local transcription functionality
    for all Granite-based transcriber providers.
    
    This class combines functionality from multiple mixins:
    - TranscriptionMixin: Audio transcription methods
    - ModelLoadingMixin: Model loading and downloading
    - ModelValidationMixin: Model validation and availability checking
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
        
        # Initialize cache manager
        self._cache_manager = CacheManager(logger)
    
    @property
    def device(self) -> str:
        """Get the current device for model execution."""
        return get_system_capability()
