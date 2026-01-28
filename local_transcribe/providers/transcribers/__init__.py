#!/usr/bin/env python3
"""
Transcriber providers package.

This package contains all transcriber provider implementations.
Providers are automatically discovered and registered by the plugin system.
"""

# Import all transcriber providers to register them with the plugin registry
from . import granite
from . import granite_mfa
from . import granite_vad_silero_mfa
from . import granite_wav2vec2
from . import faster_whisper
from . import mlx_whisper
from . import openai_whisper
from . import remote_transcriber
from . import whisper_cpp
