"""
Transcripts router for retrieving and editing transcripts.

Handles transcript retrieval from different pipeline stages and
serves job output files.
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from web_api.config import get_config
from web_api.database import get_database, JobStatus, Edit
from web_api.models.schemas import (
    EditCreateRequest,
    EditResponse,
)
from web_api.services.edit_applicator import apply_edits_to_transcript


router = APIRouter(prefix="/api/jobs", tags=["transcripts"])


# Mapping of stage names to output file patterns
STAGE_OUTPUT_FILES = {
    "vad_transcription": "turns.json",
    "de_identification": "turns-de-identified.json", 
    "speaker_naming": "turns-named.json",
    "transcript_cleanup": "turns-cleaned.json",
}


def _find_transcript_file(output_dir: Path, stage: Optional[str] = None) -> Optional[Path]:
    """
    Find the transcript file for a given stage.
    
    If no stage specified, returns the most recent/complete transcript.
    Searches both in output_dir directly and in Transcript_Raw/ subdirectory.
    """
    if not output_dir.exists():
        return None
    
    # Directories to search (output_dir and Transcript_Raw subdirectory)
    search_dirs = [output_dir]
    transcript_raw_dir = output_dir / "Transcript_Raw"
    if transcript_raw_dir.exists():
        search_dirs.append(transcript_raw_dir)
    
    if stage and stage in STAGE_OUTPUT_FILES:
        pattern = STAGE_OUTPUT_FILES[stage]
        # Look for files matching the pattern in all search directories
        for search_dir in search_dirs:
            matches = list(search_dir.glob(f"*{pattern}"))
            if matches:
                return matches[0]
    
    # Default: find the best available transcript
    # Priority: cleaned > named > de-identified > raw
    for stage_name in ["transcript_cleanup", "speaker_naming", "de_identification", "vad_transcription"]:
        pattern = STAGE_OUTPUT_FILES.get(stage_name, "")
        if pattern:
            for search_dir in search_dirs:
                matches = list(search_dir.glob(f"*{pattern}"))
                if matches:
                    return matches[0]
    
    # Fallback: any JSON file with "turns" in the name
    for search_dir in search_dirs:
        turns_files = list(search_dir.glob("*turns*.json"))
        if turns_files:
            return turns_files[0]
    
    return None


@router.get("/{job_id}/transcript")
async def get_transcript(
    job_id: str,
    stage: Optional[str] = Query(None, description="Stage name to get transcript from"),
):
    """
    Get the transcript for a completed job.
    
    Optionally specify a stage to get the transcript from that specific
    stage of the pipeline.
    """
    db = get_database()
    config = get_config()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (status: {job.status.value})"
        )
    
    # Find output directory
    output_dir = Path(job.output_dir) if job.output_dir else config.output_dir / job_id
    
    # Find transcript file
    transcript_path = _find_transcript_file(output_dir, stage)
    if not transcript_path:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found{' for stage ' + stage if stage else ''}"
        )
    
    # Load and return transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Apply any edits for this job/stage
        edits = db.get_edits_for_job(job_id, stage)
        if edits:
            transcript_data = apply_edits_to_transcript(transcript_data, edits)
        
        return JSONResponse(content=transcript_data)
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid transcript JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load transcript: {str(e)}")


@router.get("/{job_id}/transcript/stages")
async def get_available_stages(job_id: str):
    """Get list of available transcript stages for a job."""
    db = get_database()
    config = get_config()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (status: {job.status.value})"
        )
    
    output_dir = Path(job.output_dir) if job.output_dir else config.output_dir / job_id
    
    # Directories to search (output_dir and Transcript_Raw subdirectory)
    search_dirs = [output_dir]
    transcript_raw_dir = output_dir / "Transcript_Raw"
    if transcript_raw_dir.exists():
        search_dirs.append(transcript_raw_dir)
    
    available_stages = []
    for stage_name, pattern in STAGE_OUTPUT_FILES.items():
        for search_dir in search_dirs:
            matches = list(search_dir.glob(f"*{pattern}"))
            if matches:
                available_stages.append({
                    "stage": stage_name,
                    "file": matches[0].name,
                    "has_edits": len(db.get_edits_for_job(job_id, stage_name)) > 0,
                })
                break  # Found in this search_dir, no need to check others
    
    return {"stages": available_stages}


@router.get("/{job_id}/outputs/{filename}")
async def get_output_file(job_id: str, filename: str):
    """
    Download a specific output file from a job.
    
    This endpoint serves any file from the job's output directory.
    """
    db = get_database()
    config = get_config()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (status: {job.status.value})"
        )
    
    output_dir = Path(job.output_dir) if job.output_dir else config.output_dir / job_id
    file_path = output_dir / filename
    
    # Security: ensure path doesn't escape output directory
    try:
        file_path = file_path.resolve()
        output_dir_resolved = output_dir.resolve()
        if not str(file_path).startswith(str(output_dir_resolved)):
            raise HTTPException(status_code=400, detail="Invalid filename")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".json": "application/json",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".md": "text/markdown",
    }
    content_type = content_types.get(suffix, "application/octet-stream")
    
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=filename,
    )


@router.get("/{job_id}/outputs")
async def list_output_files(job_id: str):
    """List all output files for a job."""
    db = get_database()
    config = get_config()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (status: {job.status.value})"
        )
    
    output_dir = Path(job.output_dir) if job.output_dir else config.output_dir / job_id
    
    if not output_dir.exists():
        return {"files": []}
    
    files = []
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            files.append({
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "url": f"/api/jobs/{job_id}/outputs/{file_path.name}",
            })
    
    return {"files": sorted(files, key=lambda f: f["name"])}


# ==============================================================================
# Edit endpoints
# ==============================================================================

@router.post("/{job_id}/edits", response_model=EditResponse)
async def create_edit(job_id: str, request: EditCreateRequest):
    """
    Save a transcript edit.
    
    Edits are stored in the database and applied when retrieving transcripts.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Can only edit completed jobs"
        )
    
    # Create edit record
    edit = Edit(
        id=None,
        job_id=job_id,
        stage_name=request.stage_name,
        edit_type=request.edit_type,
        turn_id=request.turn_id,
        start_index=request.start_index,
        end_index=request.end_index,
        original_value=request.original_value,
        new_value=request.new_value,
        target_turn_id=request.target_turn_id,
        annotation_type=request.annotation_type,
    )
    
    saved_edit = db.create_edit(edit)
    
    return EditResponse(
        id=saved_edit.id,
        job_id=saved_edit.job_id,
        stage_name=saved_edit.stage_name,
        edit_type=saved_edit.edit_type,
        turn_id=saved_edit.turn_id,
        start_index=saved_edit.start_index,
        end_index=saved_edit.end_index,
        original_value=saved_edit.original_value,
        new_value=saved_edit.new_value,
        target_turn_id=saved_edit.target_turn_id,
        annotation_type=saved_edit.annotation_type,
        created_at=saved_edit.created_at or "",
    )


@router.get("/{job_id}/edits")
async def list_edits(
    job_id: str,
    stage: Optional[str] = Query(None, description="Filter by stage"),
):
    """Get all edits for a job."""
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    edits = db.get_edits_for_job(job_id, stage)
    
    return {
        "edits": [
            EditResponse(
                id=edit.id,
                job_id=edit.job_id,
                stage_name=edit.stage_name,
                edit_type=edit.edit_type,
                turn_id=edit.turn_id,
                start_index=edit.start_index,
                end_index=edit.end_index,
                original_value=edit.original_value,
                new_value=edit.new_value,
                target_turn_id=edit.target_turn_id,
                annotation_type=edit.annotation_type,
                created_at=edit.created_at or "",
            )
            for edit in edits
        ]
    }


@router.delete("/{job_id}/edits/{edit_id}")
async def delete_edit(job_id: str, edit_id: int):
    """Delete a specific edit (for undo)."""
    db = get_database()
    
    if db.delete_edit(edit_id):
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="Edit not found")
