#!/usr/bin/env python3
"""
Local-transcribe providers package.
"""

# Import transcriber providers to register them
from . import transcribers

# Import aligner providers to register them
from . import aligners

# Import diarization providers to register them
from . import diarization

# Import turn builders to register them
from . import turn_builders

# Import file_writers to register them
from . import file_writers

# Import unified providers to register them
from . import unified