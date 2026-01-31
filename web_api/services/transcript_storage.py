"""
Transcript storage service.

Handles storing and retrieving transcript data from the database,
providing a clean interface for the web API to work with transcripts.
All transcript data is stored in the database - no file-based storage.
"""

import json
import logging
from typing import Optional, Dict, Any, List

from web_api.database import Database, TranscriptData


logger = logging.getLogger(__name__)


# Stage identifiers used for web interface
STAGE_BASE = "base"
STAGE_DE_IDENTIFIED = "de_identified"
STAGE_CLEANED = "cleaned"

# Legacy stage name mappings (from pipeline stages)
LEGACY_STAGE_MAP = {
    "vad_transcription": STAGE_BASE,
    "de_identification": STAGE_DE_IDENTIFIED,
    "speaker_naming": STAGE_BASE,  # Not used for web, fallback to base
    "transcript_cleanup": STAGE_CLEANED,
}


class TranscriptStorageService:
    """
    Service for storing and retrieving transcript data.
    
    Provides a database-only approach to transcript storage for the web interface.
    All transcript data is stored in and retrieved from the database.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the transcript storage service.
        
        Args:
            db: Database instance
        """
        self.db = db
    
    def store_from_context(self, job_id: str, transcript, stage: str = STAGE_BASE) -> bool:
        """
        Store transcript data directly from a pipeline context's transcript object.
        
        This is the preferred method for web mode where skip_file_outputs=True
        and no files are written to disk.
        
        Args:
            job_id: The job ID
            transcript: A TranscriptFlow object (or any object with to_dict() method)
            stage: The stage identifier (default: 'base')
            
        Returns:
            True if stored successfully, False otherwise
        """
        if transcript is None:
            logger.warning(f"No transcript data to store for job {job_id}")
            return False
        
        try:
            # Convert TranscriptFlow to dict if needed
            if hasattr(transcript, 'to_dict'):
                data = transcript.to_dict()
            elif isinstance(transcript, dict):
                data = transcript
            else:
                logger.error(f"Unknown transcript type: {type(transcript)}")
                return False
            
            self.db.store_transcript_data(
                job_id=job_id,
                stage=stage,
                data_json=json.dumps(data),
                created_by="pipeline",
            )
            logger.info(f"Stored transcript stage '{stage}' for job {job_id} from context")
            return True
        except Exception as e:
            logger.error(f"Failed to store transcript for job {job_id}: {e}")
            return False
    
    def store_transcript(
        self,
        job_id: str,
        stage: str,
        data: Dict[str, Any],
        created_by: str = "system",
    ) -> TranscriptData:
        """
        Store transcript data directly.
        
        Args:
            job_id: The job ID
            stage: The stage identifier
            data: The transcript data as a dictionary
            created_by: Who created this data (pipeline, de_identification, edit, etc.)
            
        Returns:
            The created TranscriptData record
        """
        return self.db.store_transcript_data(
            job_id=job_id,
            stage=stage,
            data_json=json.dumps(data),
            created_by=created_by,
        )
    
    def get_transcript(
        self,
        job_id: str,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get transcript data for a job from the database.
        
        Args:
            job_id: The job ID
            stage: The stage to get (if None, gets the best available)
            
        Returns:
            The transcript data as a dictionary, or None if not found
        """
        # Normalize stage name (handle legacy names)
        normalized_stage = self._normalize_stage(stage) if stage else None
        
        if normalized_stage:
            data = self.db.get_current_transcript(job_id, normalized_stage)
            if data:
                return data
        else:
            # Get best available stage
            for stage_priority in [STAGE_CLEANED, STAGE_DE_IDENTIFIED, STAGE_BASE]:
                data = self.db.get_current_transcript(job_id, stage_priority)
                if data:
                    return data
        
        return None
    
    def get_available_stages(
        self,
        job_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get list of available transcript stages for a job from the database.
        
        Args:
            job_id: The job ID
            
        Returns:
            List of stage info dictionaries
        """
        # Get stages from database
        db_stages = self.db.get_available_transcript_stages(job_id)
        
        # Convert to a standardized format with display names
        stage_info = {s["stage"]: s for s in db_stages}
        
        # Return sorted by stage priority
        priority = {STAGE_BASE: 1, STAGE_DE_IDENTIFIED: 2, STAGE_CLEANED: 3}
        return sorted(
            stage_info.values(),
            key=lambda s: priority.get(s["stage"], 99)
        )
    
    def _normalize_stage(self, stage: Optional[str]) -> Optional[str]:
        """
        Normalize a stage name to the standard format.
        
        Handles both new stage names and legacy pipeline stage names.
        """
        if not stage:
            return None
        
        # Already normalized
        if stage in [STAGE_BASE, STAGE_DE_IDENTIFIED, STAGE_CLEANED]:
            return stage
        
        # Legacy stage name
        if stage in LEGACY_STAGE_MAP:
            return LEGACY_STAGE_MAP[stage]
        
        return stage


def get_transcript_storage_service() -> TranscriptStorageService:
    """Get a transcript storage service instance with the global database."""
    from web_api.database import get_database
    return TranscriptStorageService(get_database())
