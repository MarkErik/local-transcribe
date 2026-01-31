"""
Upload cleanup service for managing orphaned files.

This service handles cleanup of uploaded files that are no longer
referenced by any active jobs in the database.

The uploads folder can accumulate orphaned files when:
- Jobs are deleted but their uploaded files remain
- Server crashes during job creation (files uploaded but job not created)
- Failed uploads that were never completed
"""

import shutil
import logging
from pathlib import Path
from typing import Set, List, Tuple
from dataclasses import dataclass

from web_api.config import get_config
from web_api.database import get_database


logger = logging.getLogger(__name__)


@dataclass
class CleanupResult:
    """Result of a cleanup operation."""
    orphaned_dirs_found: int
    orphaned_dirs_removed: int
    bytes_freed: int
    errors: List[str]
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0


class UploadCleanupService:
    """
    Service for cleaning up orphaned upload directories.
    
    Orphaned uploads are directories in the uploads folder that are not
    referenced by any active job's interviewer_file_id or participant_file_id.
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
    
    def get_referenced_file_ids(self) -> Set[str]:
        """
        Get all file IDs that are referenced by active jobs.
        
        Returns:
            Set of file IDs (UUIDs) that should be kept.
        """
        referenced_ids: Set[str] = set()
        
        # Get all jobs and collect their file references
        # We fetch all jobs regardless of status to be conservative
        jobs = self.db.list_jobs(limit=100000)  # High limit to get all
        
        for job in jobs:
            if job.interviewer_file_id:
                referenced_ids.add(job.interviewer_file_id)
            if job.participant_file_id:
                referenced_ids.add(job.participant_file_id)
        
        return referenced_ids
    
    def get_upload_directories(self) -> List[Path]:
        """
        Get all directories in the uploads folder.
        
        Returns:
            List of directory paths in the uploads folder.
        """
        upload_dir = self.config.upload_dir
        if not upload_dir.exists():
            return []
        
        return [
            d for d in upload_dir.iterdir()
            if d.is_dir()
        ]
    
    def find_orphaned_uploads(self) -> List[Path]:
        """
        Find upload directories not referenced by any job.
        
        Returns:
            List of paths to orphaned upload directories.
        """
        referenced_ids = self.get_referenced_file_ids()
        all_upload_dirs = self.get_upload_directories()
        
        orphaned = []
        for upload_dir in all_upload_dirs:
            # The directory name is the file_id (UUID)
            file_id = upload_dir.name
            if file_id not in referenced_ids:
                orphaned.append(upload_dir)
        
        return orphaned
    
    def calculate_directory_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
        except (OSError, PermissionError):
            pass
        return total
    
    def cleanup_orphaned_uploads(self, dry_run: bool = False) -> CleanupResult:
        """
        Remove all orphaned upload directories.
        
        Args:
            dry_run: If True, only report what would be deleted without actually deleting.
        
        Returns:
            CleanupResult with details of the operation.
        """
        orphaned_dirs = self.find_orphaned_uploads()
        
        result = CleanupResult(
            orphaned_dirs_found=len(orphaned_dirs),
            orphaned_dirs_removed=0,
            bytes_freed=0,
            errors=[],
        )
        
        for orphan_path in orphaned_dirs:
            try:
                dir_size = self.calculate_directory_size(orphan_path)
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would remove orphaned upload: {orphan_path.name} ({dir_size} bytes)")
                    result.orphaned_dirs_removed += 1
                    result.bytes_freed += dir_size
                else:
                    shutil.rmtree(orphan_path)
                    logger.info(f"Removed orphaned upload: {orphan_path.name} ({dir_size} bytes)")
                    result.orphaned_dirs_removed += 1
                    result.bytes_freed += dir_size
                    
                    # Also clean up the database record if it exists
                    self._cleanup_database_record(orphan_path.name)
                    
            except Exception as e:
                error_msg = f"Failed to remove {orphan_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
        
        return result
    
    def _cleanup_database_record(self, file_id: str) -> None:
        """
        Remove the database record for an orphaned file.
        
        This ensures the uploaded_files table doesn't contain
        stale records for files that no longer exist on disk.
        """
        try:
            # Check if record exists before deleting
            upload = self.db.get_uploaded_file(file_id)
            if upload:
                self.db.delete_uploaded_file(file_id)
                logger.debug(f"Removed database record for orphaned file: {file_id}")
        except Exception as e:
            logger.warning(f"Could not clean database record for {file_id}: {e}")
    
    def cleanup_upload_for_job(self, job_id: str) -> Tuple[int, int]:
        """
        Clean up uploaded files associated with a specific job.
        
        This should be called when a job is deleted to ensure its
        uploaded files are also removed.
        
        Args:
            job_id: The ID of the job being deleted.
        
        Returns:
            Tuple of (files_removed, bytes_freed).
        """
        job = self.db.get_job(job_id)
        if not job:
            return (0, 0)
        
        files_removed = 0
        bytes_freed = 0
        
        file_ids = []
        if job.interviewer_file_id:
            file_ids.append(job.interviewer_file_id)
        if job.participant_file_id:
            file_ids.append(job.participant_file_id)
        
        # Check if these files are referenced by other jobs
        all_jobs = self.db.list_jobs(limit=100000)
        for file_id in file_ids:
            is_shared = False
            for other_job in all_jobs:
                if other_job.id == job_id:
                    continue
                if (other_job.interviewer_file_id == file_id or 
                    other_job.participant_file_id == file_id):
                    is_shared = True
                    break
            
            if not is_shared:
                # Safe to delete this file
                upload_path = self.config.upload_dir / file_id
                if upload_path.exists():
                    try:
                        dir_size = self.calculate_directory_size(upload_path)
                        shutil.rmtree(upload_path)
                        logger.info(f"Cleaned up upload for deleted job: {file_id}")
                        files_removed += 1
                        bytes_freed += dir_size
                        
                        # Also delete database record
                        self.db.delete_uploaded_file(file_id)
                    except Exception as e:
                        logger.error(f"Failed to clean up upload {file_id}: {e}")
        
        return (files_removed, bytes_freed)


# Global service instance
_cleanup_service: UploadCleanupService | None = None


def get_cleanup_service() -> UploadCleanupService:
    """Get the global upload cleanup service instance."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = UploadCleanupService()
    return _cleanup_service


def cleanup_orphaned_uploads(dry_run: bool = False) -> CleanupResult:
    """
    Convenience function to clean up orphaned uploads.
    
    Args:
        dry_run: If True, only report what would be deleted.
    
    Returns:
        CleanupResult with details of the operation.
    """
    service = get_cleanup_service()
    return service.cleanup_orphaned_uploads(dry_run=dry_run)
