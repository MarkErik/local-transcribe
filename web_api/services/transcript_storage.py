"""
Transcript storage service.

Handles storing and retrieving transcript data from the database,
providing a clean interface for the web API to work with transcripts
without relying on file-based storage.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from web_api.database import Database, TranscriptData


logger = logging.getLogger(__name__)


# Stage identifiers used for web interface
STAGE_BASE = "base"
STAGE_DE_IDENTIFIED = "de_identified"
STAGE_CLEANED = "cleaned"

# Mapping from file names to stage identifiers
FILE_TO_STAGE_MAP = {
    "turns.json": STAGE_BASE,
    "turns-de-identified.json": STAGE_DE_IDENTIFIED,
    "turns-cleaned.json": STAGE_CLEANED,
}

# Mapping from stage to file names (for file-based fallback)
STAGE_TO_FILE_MAP = {
    STAGE_BASE: "turns.json",
    STAGE_DE_IDENTIFIED: "turns-de-identified.json",
    STAGE_CLEANED: "turns-cleaned.json",
}

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
    
    Provides a database-centric approach to transcript storage for the web interface,
    with fallback to file-based storage for backward compatibility.
    """
    
    def __init__(self, db: Database):
        """
        Initialize the transcript storage service.
        
        Args:
            db: Database instance
        """
        self.db = db
    
    def store_from_pipeline(self, job_id: str, output_dir: Path) -> List[str]:
        """
        Store transcript data from pipeline output files into the database.
        
        This should be called after the pipeline completes to populate the
        database with transcript data for web access.
        
        Args:
            job_id: The job ID
            output_dir: The pipeline output directory
            
        Returns:
            List of stages that were stored
        """
        stored_stages = []
        
        # Check multiple possible locations for transcript files
        search_dirs = [output_dir]
        transcript_raw_dir = output_dir / "Transcript_Raw"
        if transcript_raw_dir.exists():
            search_dirs.append(transcript_raw_dir)
        
        for filename, stage in FILE_TO_STAGE_MAP.items():
            for search_dir in search_dirs:
                file_path = search_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        self.db.store_transcript_data(
                            job_id=job_id,
                            stage=stage,
                            data_json=json.dumps(data),
                            created_by="pipeline",
                        )
                        stored_stages.append(stage)
                        logger.info(f"Stored transcript stage '{stage}' for job {job_id}")
                        break  # Found in this directory, move to next file
                    except Exception as e:
                        logger.error(f"Failed to store transcript {filename} for job {job_id}: {e}")
        
        return stored_stages
    
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
        output_dir: Optional[Path] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get transcript data for a job.
        
        First tries to get from database, then falls back to file-based storage
        for backward compatibility with jobs created before the database refactor.
        
        Args:
            job_id: The job ID
            stage: The stage to get (if None, gets the best available)
            output_dir: The output directory for file-based fallback
            
        Returns:
            The transcript data as a dictionary, or None if not found
        """
        # Normalize stage name (handle legacy names)
        normalized_stage = self._normalize_stage(stage) if stage else None
        
        # Try database first
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
        
        # Fall back to file-based storage
        if output_dir:
            return self._get_from_file(output_dir, stage)
        
        return None
    
    def get_available_stages(
        self,
        job_id: str,
        output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of available transcript stages for a job.
        
        Combines database stages with file-based stages for backward compatibility.
        
        Args:
            job_id: The job ID
            output_dir: The output directory for file-based fallback
            
        Returns:
            List of stage info dictionaries
        """
        # Get stages from database
        db_stages = self.db.get_available_transcript_stages(job_id)
        
        # Convert to a standardized format with display names
        stage_info = {s["stage"]: s for s in db_stages}
        
        # Check file-based stages for backward compatibility
        if output_dir and output_dir.exists():
            file_stages = self._get_file_stages(output_dir)
            for fs in file_stages:
                normalized = self._normalize_stage(fs["stage"])
                if normalized and normalized not in stage_info:
                    # File exists but not in DB - include it
                    stage_info[normalized] = {
                        "stage": normalized,
                        "version": 1,
                        "has_edits": len(self.db.get_edits_for_job(job_id, normalized)) > 0,
                        "source": "file",
                    }
        
        # Return sorted by stage priority
        priority = {STAGE_BASE: 1, STAGE_DE_IDENTIFIED: 2, STAGE_CLEANED: 3}
        return sorted(
            stage_info.values(),
            key=lambda s: priority.get(s["stage"], 99)
        )
    
    def has_database_transcript(self, job_id: str) -> bool:
        """
        Check if a job has transcript data stored in the database.
        
        This can be used to determine whether to use database or file-based access.
        """
        stages = self.db.get_available_transcript_stages(job_id)
        return len(stages) > 0
    
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
    
    def _get_from_file(
        self,
        output_dir: Path,
        stage: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get transcript data from file-based storage (backward compatibility).
        """
        # Search directories
        search_dirs = [output_dir]
        transcript_raw_dir = output_dir / "Transcript_Raw"
        if transcript_raw_dir.exists():
            search_dirs.append(transcript_raw_dir)
        
        # If specific stage requested
        if stage:
            normalized = self._normalize_stage(stage)
            if normalized:
                filename = STAGE_TO_FILE_MAP.get(normalized)
                if filename:
                    for search_dir in search_dirs:
                        file_path = search_dir / filename
                        if file_path.exists():
                            return self._load_json_file(file_path)
        
        # Get best available
        for stage_priority in [STAGE_CLEANED, STAGE_DE_IDENTIFIED, STAGE_BASE]:
            filename = STAGE_TO_FILE_MAP.get(stage_priority)
            if filename:
                for search_dir in search_dirs:
                    file_path = search_dir / filename
                    if file_path.exists():
                        return self._load_json_file(file_path)
        
        # Fallback: any turns*.json file
        for search_dir in search_dirs:
            turns_files = list(search_dir.glob("*turns*.json"))
            if turns_files:
                return self._load_json_file(turns_files[0])
        
        return None
    
    def _get_file_stages(self, output_dir: Path) -> List[Dict[str, Any]]:
        """
        Get list of stages available as files.
        """
        search_dirs = [output_dir]
        transcript_raw_dir = output_dir / "Transcript_Raw"
        if transcript_raw_dir.exists():
            search_dirs.append(transcript_raw_dir)
        
        # Legacy file mappings
        legacy_files = {
            "turns.json": "vad_transcription",
            "turns-de-identified.json": "de_identification",
            "turns-named.json": "speaker_naming",
            "turns-cleaned.json": "transcript_cleanup",
        }
        
        stages = []
        for filename, stage_name in legacy_files.items():
            for search_dir in search_dirs:
                if (search_dir / filename).exists():
                    stages.append({
                        "stage": stage_name,
                        "file": filename,
                    })
                    break
        
        return stages
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file {file_path}: {e}")
            return None


def get_transcript_storage_service() -> TranscriptStorageService:
    """Get a transcript storage service instance with the global database."""
    from web_api.database import get_database
    return TranscriptStorageService(get_database())
