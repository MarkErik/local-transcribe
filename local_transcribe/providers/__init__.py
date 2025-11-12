#!/usr/bin/env python3
"""
Local-transcribe providers package.
"""

# Import ASR providers to register them
from . import asr

# Import diarization providers to register them
from . import diarization

# Import turn builders to register them
from . import turn_builders

# Import writers to register them
from . import writers

# Import unified providers to register them
from . import unified