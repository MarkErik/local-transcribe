"""
Job service for managing job lifecycle.

Handles job state transitions and coordination.
"""

from typing import Optional, List
from datetime import datetime

from web_api.database import get_database, Job, JobStatus


class JobService:
    """Service for job management operations."""
    
    def __init__(self):
        self.db = get_database()
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.db.get_job(job_id)
    
    def list_pending_jobs(self) -> List[Job]:
        """Get all pending jobs."""
        return self.db.list_jobs(status=JobStatus.PENDING)
    
    def list_running_jobs(self) -> List[Job]:
        """Get all running jobs."""
        return self.db.list_jobs(status=JobStatus.RUNNING)
    
    def mark_running(self, job_id: str) -> None:
        """Mark a job as running."""
        self.db.update_job_status(job_id, JobStatus.RUNNING)
    
    def mark_completed(self, job_id: str, output_dir: str) -> None:
        """Mark a job as completed."""
        self.db.update_job_status(
            job_id, 
            JobStatus.COMPLETED,
            output_dir=output_dir,
        )
    
    def mark_failed(self, job_id: str, error_message: str) -> None:
        """Mark a job as failed."""
        self.db.update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=error_message,
        )
    
    def calculate_duration(self, job: Job) -> Optional[float]:
        """Calculate job duration in seconds."""
        if not job.started_at or not job.completed_at:
            return None
        
        try:
            start = datetime.fromisoformat(job.started_at)
            end = datetime.fromisoformat(job.completed_at)
            return (end - start).total_seconds()
        except (ValueError, TypeError):
            return None


# Global service instance
_job_service: Optional[JobService] = None


def get_job_service() -> JobService:
    """Get the global job service instance."""
    global _job_service
    if _job_service is None:
        _job_service = JobService()
    return _job_service
