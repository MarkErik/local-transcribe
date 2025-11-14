#!/usr/bin/env python3
"""
Turn builder providers.
"""

# Import all turn builder modules to register them
from . import multi_speaker_turn_builder
from . import single_speaker_gap_based_turn_builder
from . import single_speaker_length_based_turn_builder