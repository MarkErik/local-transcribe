# Common providers for local_transcribe

# Import MFA alignment engine for backward compatibility
# The new modular structure is in mfa_alignment/
from local_transcribe.providers.common.mfa_alignment import MFAAlignmentEngine

# Backward compatibility alias - existing code can still import MFAWordAlignmentEngine
MFAWordAlignmentEngine = MFAAlignmentEngine

__all__ = [
    'MFAAlignmentEngine',
    'MFAWordAlignmentEngine',  # Backward compatibility
]