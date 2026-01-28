"""
De-identification API endpoints.

Provides endpoints for:
- Running first-pass de-identification
- Reviewing and editing discovered names
- Running second-pass de-identification
- Querying PII replacements (audit trail)
- Manual redaction and override operations
"""

import json
from typing import List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from web_api.database import (
    get_database,
    DeIdentificationState,
    PIIReplacement,
    JobStatus,
)
from web_api.config import get_config


router = APIRouter(prefix="/api/jobs/{job_id}/de-identify", tags=["de-identification"])


# ==============================================================================
# Request/Response Models
# ==============================================================================

class DiscoveredNameResponse(BaseModel):
    """A discovered name from de-identification."""
    name: str
    source_speaker: Optional[str] = None
    occurrences: int = 1
    include: bool = True  # For review - whether to include in second pass


class FirstPassResponse(BaseModel):
    """Response after running first pass de-identification."""
    job_id: str
    discovered_names: List[DiscoveredNameResponse]
    first_pass_complete: bool = True
    total_names: int = 0
    message: str = ""


class NameListUpdateRequest(BaseModel):
    """Request to update the name list before second pass."""
    names: List[DiscoveredNameResponse]


class NameListResponse(BaseModel):
    """Response for name list queries."""
    job_id: str
    discovered_names: List[DiscoveredNameResponse]
    reviewed_names: Optional[List[DiscoveredNameResponse]] = None
    first_pass_complete: bool = False
    second_pass_complete: bool = False


class SecondPassResponse(BaseModel):
    """Response after running second pass de-identification."""
    job_id: str
    second_pass_complete: bool = True
    total_replacements: int = 0
    message: str = ""


class PIIReplacementResponse(BaseModel):
    """A PII replacement record."""
    id: int
    job_id: str
    speaker: Optional[str] = None
    original_text: str
    replacement_text: str = "[NAME]"
    word_index: Optional[int] = None
    turn_id: Optional[int] = None
    pass_number: Optional[int] = None
    is_manual: bool = False
    is_override: bool = False
    timestamp_start: Optional[float] = None
    created_at: str


class PIIReplacementsListResponse(BaseModel):
    """Response for listing PII replacements."""
    job_id: str
    replacements: List[PIIReplacementResponse]
    total: int


class ManualRedactionRequest(BaseModel):
    """Request to manually redact text as PII."""
    turn_id: int = Field(..., description="Turn ID containing the text")
    start_index: int = Field(..., description="Start word index")
    end_index: Optional[int] = Field(None, description="End word index (inclusive), defaults to start_index")
    original_text: str = Field(..., description="Original text being redacted")
    replacement_text: str = Field(default="[NAME]", description="Replacement text")
    speaker: Optional[str] = Field(None, description="Speaker name")


class RestorePIIRequest(BaseModel):
    """Request to restore (un-redact) PII."""
    replacement_id: int = Field(..., description="ID of the PII replacement to restore")


class DeIdentificationStatusResponse(BaseModel):
    """Full de-identification status for a job."""
    job_id: str
    first_pass_complete: bool = False
    second_pass_complete: bool = False
    discovered_names_count: int = 0
    reviewed_names_count: Optional[int] = None
    total_pii_replacements: int = 0
    manual_redactions: int = 0
    overrides: int = 0


# ==============================================================================
# Endpoints
# ==============================================================================

@router.get("/status", response_model=DeIdentificationStatusResponse)
async def get_de_identification_status(job_id: str):
    """Get the current de-identification status for a job."""
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get de-identification state
    state = db.get_de_identification_state(job_id)
    
    # Get PII replacements summary
    all_replacements = db.get_pii_replacements_for_job(job_id)
    manual_count = sum(1 for r in all_replacements if r.is_manual)
    override_count = sum(1 for r in all_replacements if r.is_override)
    
    # Parse discovered names if available
    discovered_count = 0
    reviewed_count = None
    if state:
        if state.discovered_names_json:
            discovered_count = len(json.loads(state.discovered_names_json))
        if state.reviewed_names_json:
            reviewed_count = len(json.loads(state.reviewed_names_json))
    
    return DeIdentificationStatusResponse(
        job_id=job_id,
        first_pass_complete=state.first_pass_complete if state else False,
        second_pass_complete=state.second_pass_complete if state else False,
        discovered_names_count=discovered_count,
        reviewed_names_count=reviewed_count,
        total_pii_replacements=len(all_replacements),
        manual_redactions=manual_count,
        overrides=override_count,
    )


@router.post("/first-pass", response_model=FirstPassResponse)
async def run_first_pass(job_id: str, background_tasks: BackgroundTasks):
    """
    Run the first pass of de-identification.
    
    This discovers names in the transcript but does not run the second pass,
    allowing the user to review and edit the name list before continuing.
    """
    db = get_database()
    config = get_config()
    
    # Check job exists and is in correct state
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail="Job must be completed before running de-identification"
        )
    
    # Check if first pass already done
    state = db.get_de_identification_state(job_id)
    if state and state.first_pass_complete:
        # Return existing results
        names = json.loads(state.discovered_names_json) if state.discovered_names_json else []
        return FirstPassResponse(
            job_id=job_id,
            discovered_names=[
                DiscoveredNameResponse(
                    name=n["name"],
                    source_speaker=n.get("source_speaker"),
                    occurrences=n.get("occurrences", 1),
                    include=True,
                )
                for n in names
            ],
            first_pass_complete=True,
            total_names=len(names),
            message="First pass already complete. Use GET /names to retrieve the name list.",
        )
    
    # Run first pass synchronously for now (could be background task for large transcripts)
    try:
        from local_transcribe.processing.de_identification import (
            DeIdentificationOrchestrator,
        )
        
        # Load transcript and extract speaker segments
        output_dir = Path(job.output_dir) if job.output_dir else None
        if not output_dir or not output_dir.exists():
            raise HTTPException(status_code=400, detail="Job output directory not found")
        
        # Find the transcript file
        transcript_file = output_dir / "turns.json"
        if not transcript_file.exists():
            raise HTTPException(status_code=400, detail="Transcript file not found")
        
        # Load transcript and extract words per speaker
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        
        speaker_segments = _extract_speaker_segments_from_transcript(transcript_data)
        
        if not speaker_segments:
            # No segments to process
            state = DeIdentificationState(
                job_id=job_id,
                first_pass_complete=True,
                discovered_names_json="[]",
            )
            db.upsert_de_identification_state(state)
            
            return FirstPassResponse(
                job_id=job_id,
                discovered_names=[],
                first_pass_complete=True,
                total_names=0,
                message="No speaker segments found to de-identify.",
            )
        
        # Get LLM URL from job config or default
        job_config = json.loads(job.config_json) if job.config_json else {}
        llm_url = job_config.get("llm_de_identifier_url", "http://0.0.0.0:8080")
        
        # Run first pass
        orchestrator = DeIdentificationOrchestrator(
            llm_url=llm_url,
            intermediate_dir=output_dir,
        )
        
        first_pass_results = orchestrator.de_identify_multi_speaker_first_pass(speaker_segments)
        
        # Save first pass results
        first_pass_path = output_dir / "de_identification" / "first_pass_results.json"
        first_pass_path.parent.mkdir(parents=True, exist_ok=True)
        with open(first_pass_path, 'w') as f:
            json.dump(first_pass_results.to_dict(), f, indent=2)
        
        # Convert discovered names to JSON
        discovered_names_data = [
            {
                "name": n.name,
                "source_speaker": n.source_speaker,
                "occurrences": n.occurrences,
            }
            for n in first_pass_results.discovered_names
        ]
        
        # Store PII replacements from first pass
        for speaker, replacements in first_pass_results.speaker_replacements.items():
            pii_records = [
                PIIReplacement(
                    id=None,
                    job_id=job_id,
                    speaker=speaker,
                    original_text=r.original,
                    replacement_text="[NAME]",
                    word_index=r.word_index,
                    pass_number=1,
                    is_manual=False,
                    is_override=False,
                    timestamp_start=r.timestamp,
                )
                for r in replacements
            ]
            db.bulk_create_pii_replacements(pii_records)
        
        # Save state
        state = DeIdentificationState(
            job_id=job_id,
            first_pass_complete=True,
            second_pass_complete=False,
            discovered_names_json=json.dumps(discovered_names_data),
            first_pass_segments_path=str(first_pass_path),
        )
        db.upsert_de_identification_state(state)
        
        return FirstPassResponse(
            job_id=job_id,
            discovered_names=[
                DiscoveredNameResponse(
                    name=n["name"],
                    source_speaker=n.get("source_speaker"),
                    occurrences=n.get("occurrences", 1),
                    include=True,
                )
                for n in discovered_names_data
            ],
            first_pass_complete=True,
            total_names=len(discovered_names_data),
            message=f"First pass complete. Found {len(discovered_names_data)} unique names.",
        )
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"De-identification module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"First pass failed: {e}")


@router.get("/names", response_model=NameListResponse)
async def get_discovered_names(job_id: str):
    """Get the discovered names for a job."""
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get de-identification state
    state = db.get_de_identification_state(job_id)
    if not state:
        return NameListResponse(
            job_id=job_id,
            discovered_names=[],
            first_pass_complete=False,
            second_pass_complete=False,
        )
    
    # Parse discovered names
    discovered = []
    if state.discovered_names_json:
        names_data = json.loads(state.discovered_names_json)
        discovered = [
            DiscoveredNameResponse(
                name=n["name"],
                source_speaker=n.get("source_speaker"),
                occurrences=n.get("occurrences", 1),
                include=True,
            )
            for n in names_data
        ]
    
    # Parse reviewed names if available
    reviewed = None
    if state.reviewed_names_json:
        reviewed_data = json.loads(state.reviewed_names_json)
        reviewed = [
            DiscoveredNameResponse(
                name=n["name"],
                source_speaker=n.get("source_speaker"),
                occurrences=n.get("occurrences", 1),
                include=n.get("include", True),
            )
            for n in reviewed_data
        ]
    
    return NameListResponse(
        job_id=job_id,
        discovered_names=discovered,
        reviewed_names=reviewed,
        first_pass_complete=state.first_pass_complete,
        second_pass_complete=state.second_pass_complete,
    )


@router.put("/names", response_model=NameListResponse)
async def update_name_list(job_id: str, request: NameListUpdateRequest):
    """
    Update the name list before running second pass.
    
    Allows users to:
    - Remove names that aren't actually PII (set include=False)
    - Add names that were missed
    """
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check first pass is complete
    state = db.get_de_identification_state(job_id)
    if not state or not state.first_pass_complete:
        raise HTTPException(
            status_code=400,
            detail="First pass must be complete before updating name list"
        )
    
    if state.second_pass_complete:
        raise HTTPException(
            status_code=400,
            detail="Cannot update name list after second pass is complete"
        )
    
    # Store reviewed names
    reviewed_data = [
        {
            "name": n.name,
            "source_speaker": n.source_speaker,
            "occurrences": n.occurrences,
            "include": n.include,
        }
        for n in request.names
    ]
    
    db.update_de_identification_state(
        job_id=job_id,
        reviewed_names_json=json.dumps(reviewed_data),
    )
    
    # Return updated state
    return await get_discovered_names(job_id)


@router.post("/second-pass", response_model=SecondPassResponse)
async def run_second_pass(job_id: str):
    """
    Run the second pass of de-identification.
    
    Uses the reviewed name list (or discovered names if not reviewed)
    to find additional name occurrences.
    """
    db = get_database()
    config = get_config()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Check first pass is complete
    state = db.get_de_identification_state(job_id)
    if not state or not state.first_pass_complete:
        raise HTTPException(
            status_code=400,
            detail="First pass must be complete before running second pass"
        )
    
    if state.second_pass_complete:
        return SecondPassResponse(
            job_id=job_id,
            second_pass_complete=True,
            message="Second pass already complete.",
        )
    
    try:
        from local_transcribe.processing.de_identification import (
            DeIdentificationOrchestrator,
            FirstPassResults,
            DiscoveredName,
        )
        
        output_dir = Path(job.output_dir) if job.output_dir else None
        if not output_dir:
            raise HTTPException(status_code=400, detail="Job output directory not found")
        
        # Load first pass results
        first_pass_path = Path(state.first_pass_segments_path) if state.first_pass_segments_path else None
        if not first_pass_path or not first_pass_path.exists():
            raise HTTPException(status_code=400, detail="First pass results not found")
        
        with open(first_pass_path, 'r') as f:
            first_pass_data = json.load(f)
        
        first_pass_results = FirstPassResults.from_dict(first_pass_data)
        
        # Build name list for second pass
        if state.reviewed_names_json:
            reviewed_data = json.loads(state.reviewed_names_json)
            # Only include names marked for inclusion
            name_list = [
                DiscoveredName(
                    name=n["name"],
                    source_speaker=n.get("source_speaker"),
                    occurrences=n.get("occurrences", 1),
                )
                for n in reviewed_data
                if n.get("include", True)
            ]
        else:
            # Use discovered names as-is
            name_list = first_pass_results.discovered_names
        
        # Get LLM URL from job config
        job_config = json.loads(job.config_json) if job.config_json else {}
        llm_url = job_config.get("llm_de_identifier_url", "http://0.0.0.0:8080")
        
        # Run second pass
        orchestrator = DeIdentificationOrchestrator(
            llm_url=llm_url,
            intermediate_dir=output_dir,
        )
        
        # Load original segments for audit logging
        transcript_file = output_dir / "turns.json"
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        original_segments = _extract_speaker_segments_from_transcript(transcript_data)
        
        final_results = orchestrator.de_identify_multi_speaker_second_pass(
            first_pass_results,
            name_list,
            original_segments,
        )
        
        # Store second pass PII replacements
        total_second_pass = 0
        for speaker, result in final_results.items():
            pii_records = [
                PIIReplacement(
                    id=None,
                    job_id=job_id,
                    speaker=speaker,
                    original_text=r.original,
                    replacement_text="[NAME]",
                    word_index=r.word_index,
                    pass_number=2,
                    is_manual=False,
                    is_override=False,
                    timestamp_start=r.timestamp,
                )
                for r in result.second_pass_replacements
            ]
            db.bulk_create_pii_replacements(pii_records)
            total_second_pass += len(pii_records)
        
        # Update de-identified transcript file
        _apply_deidentification_to_transcript(output_dir, final_results)
        
        # Update state
        db.update_de_identification_state(
            job_id=job_id,
            second_pass_complete=True,
        )
        
        return SecondPassResponse(
            job_id=job_id,
            second_pass_complete=True,
            total_replacements=total_second_pass,
            message=f"Second pass complete. Found {total_second_pass} additional names.",
        )
        
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"De-identification module not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Second pass failed: {e}")


@router.get("/replacements", response_model=PIIReplacementsListResponse)
async def get_pii_replacements(
    job_id: str,
    speaker: Optional[str] = None,
    pass_number: Optional[int] = None,
    include_overrides: bool = True,
):
    """Get all PII replacements for a job (audit trail)."""
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    replacements = db.get_pii_replacements_for_job(
        job_id=job_id,
        speaker=speaker,
        pass_number=pass_number,
        include_overrides=include_overrides,
    )
    
    return PIIReplacementsListResponse(
        job_id=job_id,
        replacements=[
            PIIReplacementResponse(
                id=r.id,
                job_id=r.job_id,
                speaker=r.speaker,
                original_text=r.original_text,
                replacement_text=r.replacement_text,
                word_index=r.word_index,
                turn_id=r.turn_id,
                pass_number=r.pass_number,
                is_manual=r.is_manual,
                is_override=r.is_override,
                timestamp_start=r.timestamp_start,
                created_at=r.created_at,
            )
            for r in replacements
        ],
        total=len(replacements),
    )


@router.post("/redact", response_model=PIIReplacementResponse)
async def manual_redaction(job_id: str, request: ManualRedactionRequest):
    """Manually redact text as PII."""
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Create PII replacement record
    replacement = PIIReplacement(
        id=None,
        job_id=job_id,
        speaker=request.speaker,
        original_text=request.original_text,
        replacement_text=request.replacement_text,
        word_index=request.start_index,
        turn_id=request.turn_id,
        pass_number=None,  # Manual
        is_manual=True,
        is_override=False,
    )
    
    replacement = db.create_pii_replacement(replacement)
    
    return PIIReplacementResponse(
        id=replacement.id,
        job_id=replacement.job_id,
        speaker=replacement.speaker,
        original_text=replacement.original_text,
        replacement_text=replacement.replacement_text,
        word_index=replacement.word_index,
        turn_id=replacement.turn_id,
        pass_number=replacement.pass_number,
        is_manual=replacement.is_manual,
        is_override=replacement.is_override,
        timestamp_start=replacement.timestamp_start,
        created_at=replacement.created_at,
    )


@router.post("/restore", response_model=PIIReplacementResponse)
async def restore_pii(job_id: str, request: RestorePIIRequest):
    """
    Restore (un-redact) previously redacted PII.
    
    Creates an override record that marks the original replacement as no longer applicable.
    """
    db = get_database()
    
    # Check job exists
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    # Get the original replacement
    all_replacements = db.get_pii_replacements_for_job(job_id)
    original = next((r for r in all_replacements if r.id == request.replacement_id), None)
    
    if not original:
        raise HTTPException(status_code=404, detail=f"PII replacement {request.replacement_id} not found")
    
    # Create override record
    override = PIIReplacement(
        id=None,
        job_id=job_id,
        speaker=original.speaker,
        original_text=original.original_text,
        replacement_text=original.original_text,  # Restoring to original
        word_index=original.word_index,
        turn_id=original.turn_id,
        pass_number=original.pass_number,
        is_manual=False,
        is_override=True,
    )
    
    override = db.create_pii_replacement(override)
    
    return PIIReplacementResponse(
        id=override.id,
        job_id=override.job_id,
        speaker=override.speaker,
        original_text=override.original_text,
        replacement_text=override.replacement_text,
        word_index=override.word_index,
        turn_id=override.turn_id,
        pass_number=override.pass_number,
        is_manual=override.is_manual,
        is_override=override.is_override,
        timestamp_start=override.timestamp_start,
        created_at=override.created_at,
    )


# ==============================================================================
# Helper Functions
# ==============================================================================

def _extract_speaker_segments_from_transcript(transcript_data: dict) -> dict:
    """Extract word segments per speaker from transcript JSON."""
    from local_transcribe.framework.plugin_interfaces import WordSegment
    
    speaker_segments = {}
    turns = transcript_data.get("turns", [])
    
    for turn in turns:
        # Handle both "speaker" and "primary_speaker" keys
        speaker = turn.get("primary_speaker", turn.get("speaker", "Unknown"))
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        
        words = turn.get("words", [])
        for w in words:
            # WordSegment only has text, start, end, speaker fields (no confidence)
            segment = WordSegment(
                text=w.get("word", w.get("text", "")),
                start=w.get("start_time", w.get("start", 0.0)),
                end=w.get("end_time", w.get("end", 0.0)),
                speaker=speaker,
            )
            speaker_segments[speaker].append(segment)
        
        # Also process interjections
        for interjection in turn.get("interjections", []):
            int_speaker = interjection.get("speaker", speaker)
            if int_speaker not in speaker_segments:
                speaker_segments[int_speaker] = []
            
            for w in interjection.get("words", []):
                segment = WordSegment(
                    text=w.get("word", w.get("text", "")),
                    start=w.get("start_time", w.get("start", 0.0)),
                    end=w.get("end_time", w.get("end", 0.0)),
                    speaker=int_speaker,
                )
                speaker_segments[int_speaker].append(segment)
    
    return speaker_segments


def _apply_deidentification_to_transcript(output_dir: Path, results: dict) -> None:
    """Apply de-identification results to the transcript file."""
    transcript_file = output_dir / "turns.json"
    deidentified_file = output_dir / "turns_deidentified.json"
    
    with open(transcript_file, 'r') as f:
        transcript_data = json.load(f)
    
    # Build replacement map from all results
    # Key: (speaker, word_text, approximate_start) -> replacement_text
    replacement_map = {}
    for speaker, result in results.items():
        for replacement in result.all_replacements:
            key = (speaker, replacement.original)
            if key not in replacement_map:
                replacement_map[key] = "[NAME]"
    
    # Apply replacements to transcript
    for turn in transcript_data.get("turns", []):
        speaker = turn.get("speaker", "Unknown")
        for word in turn.get("words", []):
            word_text = word.get("word", word.get("text", ""))
            key = (speaker, word_text)
            if key in replacement_map:
                word["word"] = replacement_map[key]
                if "text" in word:
                    word["text"] = replacement_map[key]
        
        # Also process interjections
        for interjection in turn.get("interjections", []):
            int_speaker = interjection.get("speaker", speaker)
            for word in interjection.get("words", []):
                word_text = word.get("word", word.get("text", ""))
                key = (int_speaker, word_text)
                if key in replacement_map:
                    word["word"] = replacement_map[key]
                    if "text" in word:
                        word["text"] = replacement_map[key]
    
    # Save de-identified version
    with open(deidentified_file, 'w') as f:
        json.dump(transcript_data, f, indent=2)
