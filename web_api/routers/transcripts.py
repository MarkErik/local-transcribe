"""
Transcripts router for retrieving and editing transcripts.

Handles transcript retrieval from different pipeline stages and
serves job output files. Uses database-centric storage with file-based fallback.
"""

import json
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from web_api.config import get_config
from web_api.database import get_database, JobStatus, Edit
from web_api.models.schemas import (
    EditCreateRequest,
    EditResponse,
)
from web_api.services.edit_applicator import apply_edits_to_transcript
from web_api.services.transcript_storage import (
    TranscriptStorageService,
    STAGE_BASE,
    STAGE_DE_IDENTIFIED,
    STAGE_CLEANED,
    LEGACY_STAGE_MAP,
)


router = APIRouter(prefix="/api/jobs", tags=["transcripts"])


# Mapping of stage names to output file patterns (for file-based fallback)
STAGE_OUTPUT_FILES = {
    "vad_transcription": "turns.json",
    "de_identification": "turns-de-identified.json", 
    "speaker_naming": "turns-named.json",
    "transcript_cleanup": "turns-cleaned.json",
    # New stage names also supported
    "base": "turns.json",
    "de_identified": "turns-de-identified.json",
    "cleaned": "turns-cleaned.json",
}

# Display names for stages (web UI friendly)
STAGE_DISPLAY_NAMES = {
    "base": "Raw Transcription",
    "vad_transcription": "Raw Transcription",
    "de_identified": "De-identified",
    "de_identification": "De-identified",
    "cleaned": "Cleaned",
    "transcript_cleanup": "Cleaned",
    "speaker_naming": "Speaker Named",
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


def _normalize_stage_name(stage: Optional[str]) -> Optional[str]:
    """Normalize stage name to the new format."""
    if not stage:
        return None
    if stage in LEGACY_STAGE_MAP:
        return LEGACY_STAGE_MAP[stage]
    return stage


@router.get("/{job_id}/transcript")
async def get_transcript(
    job_id: str,
    stage: Optional[str] = Query(None, description="Stage name to get transcript from"),
):
    """
    Get the transcript for a completed job.
    
    Optionally specify a stage to get the transcript from that specific
    stage of the pipeline. Uses database storage with file-based fallback.
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
    
    # Find output directory for file fallback
    output_dir = Path(job.output_dir) if job.output_dir else config.output_dir / job_id
    
    # Normalize stage name
    normalized_stage = _normalize_stage_name(stage)
    
    # Try database-centric approach first
    transcript_storage = TranscriptStorageService(db)
    transcript_data = transcript_storage.get_transcript(job_id, normalized_stage, output_dir)
    
    if not transcript_data:
        # Fallback to file-based approach for backward compatibility
        transcript_path = _find_transcript_file(output_dir, stage)
        if not transcript_path:
            raise HTTPException(
                status_code=404,
                detail=f"Transcript not found{' for stage ' + stage if stage else ''}"
            )
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid transcript JSON")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load transcript: {str(e)}")
    
    # Apply any edits for this job/stage (use normalized stage for edits lookup)
    # Also try with original stage name for backward compatibility
    edits = db.get_edits_for_job(job_id, normalized_stage)
    if not edits and stage and stage != normalized_stage:
        edits = db.get_edits_for_job(job_id, stage)
    
    if edits:
        transcript_data = apply_edits_to_transcript(transcript_data, edits)
    
    return JSONResponse(content=transcript_data)


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
    
    # Try database-centric approach first
    transcript_storage = TranscriptStorageService(db)
    db_stages = transcript_storage.get_available_stages(job_id, output_dir)
    
    if db_stages:
        # Return stages with display names
        available_stages = []
        for stage_info in db_stages:
            stage_name = stage_info["stage"]
            available_stages.append({
                "stage": stage_name,
                "display_name": STAGE_DISPLAY_NAMES.get(stage_name, stage_name),
                "has_edits": stage_info.get("has_edits", False),
                "source": stage_info.get("source", "database"),
            })
        return {"stages": available_stages}
    
    # Fallback to file-based approach for backward compatibility
    search_dirs = [output_dir]
    transcript_raw_dir = output_dir / "Transcript_Raw"
    if transcript_raw_dir.exists():
        search_dirs.append(transcript_raw_dir)
    
    available_stages = []
    seen_stages = set()
    for stage_name, pattern in STAGE_OUTPUT_FILES.items():
        # Skip duplicates (new and legacy names for same stage)
        normalized = _normalize_stage_name(stage_name)
        if normalized in seen_stages:
            continue
        
        for search_dir in search_dirs:
            matches = list(search_dir.glob(f"*{pattern}"))
            if matches:
                seen_stages.add(normalized)
                available_stages.append({
                    "stage": stage_name,
                    "display_name": STAGE_DISPLAY_NAMES.get(stage_name, stage_name),
                    "file": matches[0].name,
                    "has_edits": len(db.get_edits_for_job(job_id, stage_name)) > 0,
                    "source": "file",
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
    """Delete a specific edit (permanently removes it)."""
    db = get_database()
    
    if db.delete_edit(edit_id):
        return {"status": "deleted"}
    else:
        raise HTTPException(status_code=404, detail="Edit not found")


# ==============================================================================
# Undo/Redo endpoints
# ==============================================================================


class UndoRedoResponse(BaseModel):
    """Response for undo/redo operations."""
    status: str
    edit_id: Optional[int] = None
    message: str = ""


class EditHistoryResponse(BaseModel):
    """Response for edit history."""
    job_id: str
    stage: Optional[str] = None
    active_edits: List[EditResponse]
    undone_edits: List[EditResponse]
    can_undo: bool
    can_redo: bool


@router.post("/{job_id}/edits/{edit_id}/undo")
async def undo_edit(job_id: str, edit_id: int):
    """
    Undo a specific edit by marking it as undone.
    
    The edit is not deleted, just marked as undone so it can be redone later.
    """
    db = get_database()
    
    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify edit exists and belongs to this job
    edit = db.get_edit_by_id(edit_id)
    if not edit:
        raise HTTPException(status_code=404, detail="Edit not found")
    if edit.job_id != job_id:
        raise HTTPException(status_code=400, detail="Edit does not belong to this job")
    if edit.is_undone:
        raise HTTPException(status_code=400, detail="Edit is already undone")
    
    if db.undo_edit(edit_id):
        return UndoRedoResponse(
            status="undone",
            edit_id=edit_id,
            message=f"Edit {edit_id} has been undone"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to undo edit")


@router.post("/{job_id}/edits/{edit_id}/redo")
async def redo_edit(job_id: str, edit_id: int):
    """
    Redo an undone edit by restoring it.
    """
    db = get_database()
    
    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify edit exists and belongs to this job
    edit = db.get_edit_by_id(edit_id)
    if not edit:
        raise HTTPException(status_code=404, detail="Edit not found")
    if edit.job_id != job_id:
        raise HTTPException(status_code=400, detail="Edit does not belong to this job")
    if not edit.is_undone:
        raise HTTPException(status_code=400, detail="Edit is not undone")
    
    if db.redo_edit(edit_id):
        return UndoRedoResponse(
            status="redone",
            edit_id=edit_id,
            message=f"Edit {edit_id} has been restored"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to redo edit")


@router.post("/{job_id}/edits/undo-last")
async def undo_last_edit(
    job_id: str,
    stage: Optional[str] = Query(None, description="Stage to undo from"),
):
    """
    Undo the most recent active edit for a job/stage.
    """
    db = get_database()
    
    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Normalize stage
    normalized_stage = _normalize_stage_name(stage) if stage else None
    
    # Find the last active edit
    last_edit = None
    if normalized_stage:
        # Try stage-specific first
        last_edit = db.get_last_edit_for_stage(job_id, normalized_stage, undone=False)
        if not last_edit:
            # Get all active edits and find the last one
            edits = db.get_edits_for_job(job_id, normalized_stage, include_undone=False)
            if edits:
                last_edit = edits[-1]
    else:
        # No stage filter - get all active edits and find the last one
        edits = db.get_edits_for_job(job_id, stage_name=None, include_undone=False)
        if edits:
            last_edit = edits[-1]
    
    if not last_edit:
        raise HTTPException(status_code=404, detail="No edits to undo")
    
    if db.undo_edit(last_edit.id):
        return UndoRedoResponse(
            status="undone",
            edit_id=last_edit.id,
            message=f"Edit {last_edit.id} ({last_edit.edit_type}) has been undone"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to undo edit")


@router.post("/{job_id}/edits/redo-last")
async def redo_last_edit(
    job_id: str,
    stage: Optional[str] = Query(None, description="Stage to redo from"),
):
    """
    Redo the most recently undone edit for a job/stage.
    """
    db = get_database()
    
    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Normalize stage
    normalized_stage = _normalize_stage_name(stage) if stage else None
    
    # Find the last undone edit (most recently undone)
    last_undone = None
    if normalized_stage:
        # Try stage-specific first
        last_undone = db.get_last_edit_for_stage(job_id, normalized_stage, undone=True)
        if not last_undone:
            # Get all edits including undone and find the last undone one
            edits = db.get_edits_for_job(job_id, normalized_stage, include_undone=True)
            undone_edits = [e for e in edits if e.is_undone]
            if undone_edits:
                # Get the most recently undone (by undone_at timestamp)
                last_undone = max(undone_edits, key=lambda e: e.undone_at or "")
    else:
        # No stage filter - get all undone edits and find the most recent
        edits = db.get_edits_for_job(job_id, stage_name=None, include_undone=True)
        undone_edits = [e for e in edits if e.is_undone]
        if undone_edits:
            # Get the most recently undone (by undone_at timestamp)
            last_undone = max(undone_edits, key=lambda e: e.undone_at or "")
    
    if not last_undone:
        raise HTTPException(status_code=404, detail="No edits to redo")
    
    if db.redo_edit(last_undone.id):
        return UndoRedoResponse(
            status="redone",
            edit_id=last_undone.id,
            message=f"Edit {last_undone.id} ({last_undone.edit_type}) has been restored"
        )
    else:
        raise HTTPException(status_code=500, detail="Failed to redo edit")


@router.get("/{job_id}/edits/history")
async def get_edit_history(
    job_id: str,
    stage: Optional[str] = Query(None, description="Filter by stage"),
):
    """
    Get full edit history including undone edits.
    
    Useful for displaying undo/redo state in the UI.
    """
    db = get_database()
    
    # Verify job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Normalize stage
    normalized_stage = _normalize_stage_name(stage) if stage else None
    
    # Get all edits including undone
    all_edits = db.get_edits_for_job(job_id, normalized_stage, include_undone=True)
    
    # Separate active and undone edits
    active_edits = [e for e in all_edits if not e.is_undone]
    undone_edits = [e for e in all_edits if e.is_undone]
    
    def to_response(edit: Edit) -> EditResponse:
        return EditResponse(
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
    
    return EditHistoryResponse(
        job_id=job_id,
        stage=normalized_stage,
        active_edits=[to_response(e) for e in active_edits],
        undone_edits=[to_response(e) for e in undone_edits],
        can_undo=len(active_edits) > 0,
        can_redo=len(undone_edits) > 0,
    )
