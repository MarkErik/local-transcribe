#!/usr/bin/env python3
"""
Local-transcribe providers package.
"""

# Import combined providers to register them
from . import combined

# Import ASR providers to register them
from . import asr

# Import diarization providers to register them
from . import diarization

# Import turn builders to register them
from . import turn_builders

# Import writers to register them
from . import writers