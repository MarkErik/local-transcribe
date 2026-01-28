"""
Exports router for generating downloadable transcript exports.

Provides endpoints to export transcripts in various formats using
the existing file writers from the pipeline.
"""

import json
import tempfile
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from web_api.config import get_config
from web_api.database import get_database, JobStatus
from web_api.services.edit_applicator import apply_edits_to_transcript

# Import transcript flow for deserialization
from local_transcribe.processing.turn_building.turn_building_data_structures import TranscriptFlow

# Import file writers  
from local_transcribe.providers.file_writers.txt_writer import (
    write_timestamped_txt, 
    write_plain_txt,
    _extract_turns_as_dicts as txt_extract_turns,
)
from local_transcribe.providers.file_writers.srt_writer import write_srt
from local_transcribe.providers.file_writers.json_writer import (
    write_turns_json,
)
from local_transcribe.providers.file_writers.dialogue_script_writer import write_dialogue_script
from local_transcribe.providers.file_writers.annotated_markdown_writer import write_annotated_markdown


router = APIRouter(prefix="/api/jobs", tags=["exports"])


# Supported export formats and their configurations
EXPORT_FORMATS = {
    "timestamped-txt": {
        "name": "Timestamped Text",
        "description": "Plain text with timestamps for each turn",
        "extension": ".timestamped.txt",
        "content_type": "text/plain",
    },
    "plain-txt": {
        "name": "Plain Text",
        "description": "Plain text without timestamps, grouped by speaker",
        "extension": ".txt",
        "content_type": "text/plain",
    },
    "turns-json": {
        "name": "JSON (Structured)",
        "description": "Full structured JSON with all metadata",
        "extension": ".turns.json",
        "content_type": "application/json",
    },
    "dialogue-script": {
        "name": "Dialogue Script",
        "description": "Screenplay-style format with inline interjections",
        "extension": ".script.txt",
        "content_type": "text/plain",
    },
    "markdown": {
        "name": "Markdown",
        "description": "Rich Markdown with statistics and formatting",
        "extension": ".md",
        "content_type": "text/markdown",
    },
    "srt": {
        "name": "SRT Subtitles",
        "description": "Standard subtitle format for video players",
        "extension": ".srt",
        "content_type": "text/plain",
    },
}


class ExportOptions(BaseModel):
    """Options for customizing export output."""
    include_timestamps: bool = Field(default=True, description="Include timestamps in text formats")
    include_speaker_labels: bool = Field(default=True, description="Include speaker labels")
    include_interjections: bool = Field(default=True, description="Include interjection markers")
    include_metadata: bool = Field(default=True, description="Include summary/metadata section")


class ExportRequest(BaseModel):
    """Request body for export endpoint."""
    format: str = Field(..., description="Export format")
    stage: Optional[str] = Field(None, description="Stage to export from")
    options: ExportOptions = Field(default_factory=ExportOptions, description="Export options")


class ExportFormatInfo(BaseModel):
    """Information about an export format."""
    id: str
    name: str
    description: str
    extension: str


class ExportFormatsResponse(BaseModel):
    """Response listing available export formats."""
    formats: List[ExportFormatInfo]


# Stage output file mapping (same as transcripts.py)
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
    """
    if not output_dir.exists():
        return None
    
    if stage and stage in STAGE_OUTPUT_FILES:
        pattern = STAGE_OUTPUT_FILES[stage]
        matches = list(output_dir.glob(f"*{pattern}"))
        if matches:
            return matches[0]
    
    # Default: find the best available transcript
    for stage_name in ["transcript_cleanup", "speaker_naming", "de_identification", "vad_transcription"]:
        pattern = STAGE_OUTPUT_FILES.get(stage_name, "")
        if pattern:
            matches = list(output_dir.glob(f"*{pattern}"))
            if matches:
                return matches[0]
    
    # Fallback
    turns_files = list(output_dir.glob("*turns*.json"))
    if turns_files:
        return turns_files[0]
    
    return None


def _load_transcript_data(job_id: str, stage: Optional[str] = None) -> dict:
    """Load transcript data for a job, applying any edits."""
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
    transcript_path = _find_transcript_file(output_dir, stage)
    
    if not transcript_path:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript not found{' for stage ' + stage if stage else ''}"
        )
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Apply any edits
        edits = db.get_edits_for_job(job_id, stage)
        if edits:
            transcript_data = apply_edits_to_transcript(transcript_data, edits)
        
        return transcript_data
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid transcript JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load transcript: {str(e)}")


def _transcript_data_to_flow(data: dict) -> TranscriptFlow:
    """Convert transcript JSON data back to a TranscriptFlow object."""
    return TranscriptFlow.from_dict(data)


def _generate_export(
    transcript_data: dict, 
    format: str, 
    options: ExportOptions
) -> tuple[bytes, str, str]:
    """
    Generate export content for a given format.
    
    Returns: (content_bytes, content_type, filename_suffix)
    """
    format_info = EXPORT_FORMATS.get(format)
    if not format_info:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    
    # Create a temporary file to use the existing writers
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"export{format_info['extension']}"
        
        try:
            if format == "turns-json":
                # JSON format - write directly
                content = json.dumps(transcript_data, indent=2, ensure_ascii=False)
                return content.encode('utf-8'), format_info['content_type'], format_info['extension']
            
            elif format == "timestamped-txt":
                # Extract turns as dicts for text writers
                transcript_flow = _transcript_data_to_flow(transcript_data)
                turns = txt_extract_turns(transcript_flow)
                write_timestamped_txt(turns, tmp_path)
            
            elif format == "plain-txt":
                transcript_flow = _transcript_data_to_flow(transcript_data)
                turns = txt_extract_turns(transcript_flow)
                write_plain_txt(turns, tmp_path)
            
            elif format == "srt":
                transcript_flow = _transcript_data_to_flow(transcript_data)
                turns = txt_extract_turns(transcript_flow)
                write_srt(turns, tmp_path)
            
            elif format == "dialogue-script":
                transcript_flow = _transcript_data_to_flow(transcript_data)
                write_dialogue_script(transcript_flow, tmp_path)
            
            elif format == "markdown":
                transcript_flow = _transcript_data_to_flow(transcript_data)
                write_annotated_markdown(transcript_flow, tmp_path)
            
            else:
                raise HTTPException(status_code=400, detail=f"Format not yet implemented: {format}")
            
            # Read the generated file
            content = tmp_path.read_bytes()
            return content, format_info['content_type'], format_info['extension']
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Export generation failed: {str(e)}")


@router.get("/{job_id}/export/formats", response_model=ExportFormatsResponse)
async def get_export_formats(job_id: str):
    """
    Get available export formats for a job.
    
    Returns list of supported export formats with descriptions.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    formats = [
        ExportFormatInfo(
            id=fmt_id,
            name=info["name"],
            description=info["description"],
            extension=info["extension"],
        )
        for fmt_id, info in EXPORT_FORMATS.items()
    ]
    
    return ExportFormatsResponse(formats=formats)


@router.post("/{job_id}/export")
async def export_transcript(
    job_id: str,
    request: ExportRequest,
):
    """
    Export a transcript in the specified format.
    
    Returns the exported file as a download.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Load transcript data
    transcript_data = _load_transcript_data(job_id, request.stage)
    
    # Generate export
    content, content_type, extension = _generate_export(
        transcript_data, 
        request.format,
        request.options,
    )
    
    # Build filename
    stage_suffix = f"-{request.stage}" if request.stage else ""
    filename = f"transcript-{job_id[:8]}{stage_suffix}{extension}"
    
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/{job_id}/export/{format}")
async def export_transcript_get(
    job_id: str,
    format: str,
    stage: Optional[str] = Query(None, description="Stage to export from"),
):
    """
    Export a transcript using GET request (for direct download links).
    
    Simpler alternative to POST endpoint for basic exports.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Load transcript data
    transcript_data = _load_transcript_data(job_id, stage)
    
    # Generate export with default options
    content, content_type, extension = _generate_export(
        transcript_data, 
        format,
        ExportOptions(),
    )
    
    # Build filename
    stage_suffix = f"-{stage}" if stage else ""
    filename = f"transcript-{job_id[:8]}{stage_suffix}{extension}"
    
    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/{job_id}/compare")
async def get_comparison_data(
    job_id: str,
    stage_a: str = Query(..., description="First stage to compare"),
    stage_b: str = Query(..., description="Second stage to compare"),
):
    """
    Get comparison data between two pipeline stages.
    
    Returns both transcripts and diff information for side-by-side comparison.
    """
    db = get_database()
    
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Load both transcripts
    try:
        transcript_a = _load_transcript_data(job_id, stage_a)
        transcript_b = _load_transcript_data(job_id, stage_b)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load transcripts: {str(e)}")
    
    # Extract words for diff computation
    words_a = _extract_words_from_transcript(transcript_a)
    words_b = _extract_words_from_transcript(transcript_b)
    
    # Compute diff using the existing diff engine
    try:
        from web_tools.transcript_compare.diff_engine import compute_diff
        diff_result = compute_diff(words_a, words_b)
        
        # Convert diff result to JSON-serializable format
        diff_data = {
            "total_words_a": diff_result.total_words_a,
            "total_words_b": diff_result.total_words_b,
            "matching_words": diff_result.matching_words,
            "inserted_words": diff_result.inserted_words,
            "deleted_words": diff_result.deleted_words,
            "similarity_ratio": diff_result.similarity_ratio,
            "word_error_rate": diff_result.word_error_rate,
            "segments": [
                {
                    "type": seg.diff_type.value,
                    "words_a": seg.words_a,
                    "words_b": seg.words_b,
                    "position_a": seg.position_a,
                    "position_b": seg.position_b,
                }
                for seg in diff_result.segments
            ]
        }
    except Exception as e:
        # Fallback if diff engine fails
        diff_data = {
            "error": str(e),
            "total_words_a": len(words_a),
            "total_words_b": len(words_b),
        }
    
    return {
        "stage_a": stage_a,
        "stage_b": stage_b,
        "transcript_a": transcript_a,
        "transcript_b": transcript_b,
        "diff": diff_data,
    }


def _extract_words_from_transcript(transcript_data: dict) -> List[str]:
    """Extract all words from a transcript for diff comparison."""
    words = []
    turns = transcript_data.get("turns", [])
    
    for turn in turns:
        text = turn.get("text", "")
        # Simple word tokenization
        turn_words = text.split()
        words.extend(turn_words)
    
    return words
