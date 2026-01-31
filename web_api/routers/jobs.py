"""
Jobs router for pipeline execution and monitoring.

Handles job submission, status checking, and SSE progress streaming.
"""

import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse

from web_api.config import get_config
from web_api.database import get_database, Job, JobStatus, UploadStatus
from web_api.models.schemas import (
    JobCreateRequest,
    JobCreateResponse,
    JobResponse,
    JobListResponse,
    StageStartEvent,
    BlockProgressEvent,
    StageCompleteEvent,
    JobCompleteEvent,
    JobErrorEvent,
)
from web_api.services.pipeline_service import PipelineService, get_pipeline_service


router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# In-memory event storage for SSE replay (per job)
# Key: job_id, Value: list of (event_id, event_data) tuples
_job_events: dict[str, list[tuple[int, str]]] = {}
_event_counter: int = 0


def _store_event(job_id: str, event_type: str, data: dict) -> int:
    """Store an event for replay and return its ID."""
    global _event_counter
    _event_counter += 1
    event_id = _event_counter
    
    if job_id not in _job_events:
        _job_events[job_id] = []
    
    # Format as SSE
    event_str = f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"
    _job_events[job_id].append((event_id, event_str))
    
    # Keep only last 1000 events per job (memory management)
    if len(_job_events[job_id]) > 1000:
        _job_events[job_id] = _job_events[job_id][-500:]
    
    return event_id


def _get_events_after(job_id: str, last_event_id: int) -> list[str]:
    """Get all events for a job after the given event ID."""
    if job_id not in _job_events:
        return []
    
    return [
        event_str
        for event_id, event_str in _job_events[job_id]
        if event_id > last_event_id
    ]


@router.post("", response_model=JobCreateResponse)
async def create_job(
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Create a new transcription job.
    
    The job will be queued and executed in the background.
    """
    config = get_config()
    db = get_database()
    
    # Validate file IDs exist and are complete
    interviewer_file = db.get_uploaded_file(request.interviewer_file_id)
    if not interviewer_file:
        raise HTTPException(status_code=400, detail="Interviewer file not found")
    if interviewer_file.upload_status != UploadStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Interviewer file upload not complete")
    
    participant_file = db.get_uploaded_file(request.participant_file_id)
    if not participant_file:
        raise HTTPException(status_code=400, detail="Participant file not found")
    if participant_file.upload_status != UploadStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Participant file upload not complete")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Create output directory
    output_dir = config.output_dir / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store configuration
    config_dict = {
        "mode": request.mode,
        "options": request.options.model_dump(),
        "interviewer_file": interviewer_file.stored_path,
        "participant_file": participant_file.stored_path,
    }
    
    # Create job record
    job = Job(
        id=job_id,
        status=JobStatus.PENDING,
        mode=request.mode,
        config_json=json.dumps(config_dict),
        interviewer_file_id=request.interviewer_file_id,
        participant_file_id=request.participant_file_id,
    )
    db.create_job(job)
    
    # Initialize event storage for this job
    _job_events[job_id] = []
    
    # Queue job for execution
    pipeline_service = get_pipeline_service()
    background_tasks.add_task(
        pipeline_service.execute_job,
        job_id=job_id,
        progress_callback=lambda event_type, data: _store_event(job_id, event_type, data),
    )
    
    return JobCreateResponse(job_id=job_id, status="pending")


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all jobs with optional status filter."""
    db = get_database()
    
    job_status = None
    if status:
        try:
            job_status = JobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    jobs = db.list_jobs(status=job_status, limit=limit, offset=offset)
    
    # Convert to response models
    job_responses = []
    for job in jobs:
        job_dict = job.to_dict()
        
        # Calculate duration if completed
        if job.started_at and job.completed_at:
            try:
                start = datetime.fromisoformat(job.started_at)
                end = datetime.fromisoformat(job.completed_at)
                job_dict["duration_seconds"] = (end - start).total_seconds()
            except (ValueError, TypeError):
                pass
        
        job_responses.append(JobResponse(**job_dict))
    
    return JobListResponse(
        jobs=job_responses,
        total=len(job_responses),  # TODO: Add proper count query
        offset=offset,
        limit=limit,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get details for a specific job."""
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dict = job.to_dict()
    
    # Calculate duration if completed
    if job.started_at and job.completed_at:
        try:
            start = datetime.fromisoformat(job.started_at)
            end = datetime.fromisoformat(job.completed_at)
            job_dict["duration_seconds"] = (end - start).total_seconds()
        except (ValueError, TypeError):
            pass
    
    return JobResponse(**job_dict)


@router.get("/{job_id}/progress")
async def get_job_progress(
    job_id: str,
    last_event_id: Optional[int] = Query(None, alias="Last-Event-Id"),
):
    """
    SSE endpoint for real-time job progress.
    
    Connect to this endpoint to receive progress events as the job executes.
    Supports reconnection via Last-Event-Id header.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the job."""
        
        # Replay events after last_event_id if reconnecting
        if last_event_id is not None:
            for event_str in _get_events_after(job_id, last_event_id):
                yield event_str
        
        # If job is already done, send final status and close
        current_job = db.get_job(job_id)
        if current_job and current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            if current_job.status == JobStatus.COMPLETED:
                event = JobCompleteEvent(
                    job_id=job_id,
                    output_path=current_job.output_dir or "",
                )
            else:
                event = JobErrorEvent(
                    job_id=job_id,
                    error=current_job.error_message or "Job failed",
                )
            yield f"data: {event.model_dump_json()}\n\n"
            return
        
        # Poll for new events
        last_seen = last_event_id or 0
        no_event_count = 0
        
        while True:
            # Check for new events
            new_events = _get_events_after(job_id, last_seen)
            if new_events:
                for event_str in new_events:
                    yield event_str
                # Update last seen
                if job_id in _job_events and _job_events[job_id]:
                    last_seen = _job_events[job_id][-1][0]
                no_event_count = 0
            else:
                no_event_count += 1
                # Send keepalive every 15 seconds
                if no_event_count >= 15:
                    yield ": keepalive\n\n"
                    no_event_count = 0
            
            # Check if job is done
            current_job = db.get_job(job_id)
            if current_job and current_job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                # Drain any remaining events
                for event_str in _get_events_after(job_id, last_seen):
                    yield event_str
                break
            
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or running job."""
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status not in (JobStatus.PENDING, JobStatus.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}"
        )
    
    # TODO: Actually cancel the running pipeline
    db.update_job_status(job_id, JobStatus.CANCELLED, error_message="Cancelled by user")
    
    # Store cancel event
    _store_event(job_id, "job_error", {
        "job_id": job_id,
        "error": "Job cancelled by user",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    return {"status": "cancelled"}


@router.delete("/{job_id}")
async def delete_job(job_id: str, force: bool = Query(False, description="Force delete even if running/pending")):
    """
    Delete a job and all its associated data.
    
    By default, only allows deletion of jobs that are completed, failed, or cancelled.
    Running or pending jobs must be cancelled first, unless force=true is specified.
    
    Use force=true to delete jobs that are stuck in 'running' or 'pending' state
    (e.g., after a server restart).
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Only allow deletion of finished jobs unless force=true
    if job.status in (JobStatus.PENDING, JobStatus.RUNNING) and not force:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete job with status: {job.status.value}. Cancel it first, or use force=true if the job is stale."
        )
    
    # Use force_delete for jobs in any state, or regular delete for completed jobs
    if force or job.status in (JobStatus.PENDING, JobStatus.RUNNING):
        deleted = db.force_delete_job(job_id)
    else:
        deleted = db.delete_job(job_id)
        
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Optionally clean up output directory
    if job.output_dir:
        import shutil
        from pathlib import Path
        output_path = Path(job.output_dir)
        if output_path.exists():
            try:
                shutil.rmtree(output_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    # Clean up event storage
    if job_id in _job_events:
        del _job_events[job_id]
    
    return {"status": "deleted", "job_id": job_id}


@router.post("/{job_id}/rerun")
async def rerun_job(
    job_id: str,
    start_stage: Optional[str] = Query(None, description="Stage to start from"),
    background_tasks: BackgroundTasks = None,
):
    """
    Re-run a job from a checkpoint.
    
    Creates a new job that starts from the specified stage using
    the edited transcript as input.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Can only rerun completed jobs"
        )
    
    # Create a new job ID for the rerun
    new_job_id = str(uuid.uuid4())
    
    # Copy configuration from original job
    config = get_config()
    output_dir = config.output_dir / new_job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_config = json.loads(job.config_json) if job.config_json else {}
    original_config["rerun_from"] = job_id
    original_config["start_stage"] = start_stage
    
    new_job = Job(
        id=new_job_id,
        status=JobStatus.PENDING,
        mode=job.mode,
        config_json=json.dumps(original_config),
        interviewer_file_id=job.interviewer_file_id,
        participant_file_id=job.participant_file_id,
    )
    db.create_job(new_job)
    
    # Initialize event storage
    _job_events[new_job_id] = []
    
    # Queue for execution
    pipeline_service = get_pipeline_service()
    background_tasks.add_task(
        pipeline_service.execute_job_from_checkpoint,
        job_id=new_job_id,
        original_job_id=job_id,
        start_stage=start_stage,
        progress_callback=lambda event_type, data: _store_event(new_job_id, event_type, data),
    )
    
    return JobCreateResponse(job_id=new_job_id, status="pending")
