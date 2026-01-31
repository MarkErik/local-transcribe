"""
File upload and download router.

Handles chunked file uploads for large audio files and serves
files for playback with HTTP Range request support.
"""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Header, Request
from fastapi.responses import FileResponse, StreamingResponse

from web_api.config import get_config
from web_api.database import (
    get_database, 
    UploadedFile, 
    UploadStatus,
)
from web_api.models.schemas import (
    UploadInitRequest,
    UploadInitResponse,
    ChunkUploadResponse,
    UploadCompleteResponse,
    UploadStatusResponse,
)


router = APIRouter(prefix="/api/files", tags=["files"])


# Magic bytes for common audio formats
AUDIO_MAGIC_BYTES = {
    b'\xff\xfb': 'audio/mpeg',  # MP3
    b'\xff\xfa': 'audio/mpeg',  # MP3
    b'\xff\xf3': 'audio/mpeg',  # MP3
    b'\xff\xf2': 'audio/mpeg',  # MP3
    b'ID3': 'audio/mpeg',  # MP3 with ID3 tag
    b'fLaC': 'audio/flac',  # FLAC
    b'OggS': 'audio/ogg',  # OGG
    b'RIFF': 'audio/wav',  # WAV
    b'\x00\x00\x00': 'audio/mp4',  # M4A/MP4 (partial, need more context)
}


def detect_audio_content_type(file_path: Path) -> str:
    """
    Detect the content type of an audio file from its magic bytes.
    
    Returns the appropriate MIME type for the audio file.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
        
        # Check for M4A/MP4 (ftyp box)
        if len(header) >= 8 and header[4:8] == b'ftyp':
            return 'audio/mp4'
        
        # Check other audio formats
        for magic, mime_type in AUDIO_MAGIC_BYTES.items():
            if header.startswith(magic):
                return mime_type
        
        # Fallback based on file extension
        ext = file_path.suffix.lower()
        ext_to_mime = {
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.mp4': 'audio/mp4',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.opus': 'audio/opus',
            '.aac': 'audio/aac',
        }
        return ext_to_mime.get(ext, 'audio/mpeg')
        
    except Exception:
        return 'audio/mpeg'


def validate_audio_file(file_path: Path) -> bool:
    """
    Validate that a file is a valid audio file.
    
    Checks magic bytes at the start of the file.
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(12)
        
        # Check for M4A/MP4 (ftyp box)
        if len(header) >= 8 and header[4:8] == b'ftyp':
            return True
        
        # Check other audio formats
        for magic, _ in AUDIO_MAGIC_BYTES.items():
            if header.startswith(magic):
                return True
        
        # Also accept files that ffprobe can handle (fallback)
        # For now, be lenient and accept if we got past the basic checks
        return True
        
    except Exception:
        return False


@router.post("/upload/init", response_model=UploadInitResponse)
async def init_upload(request: UploadInitRequest):
    """
    Initialize a chunked file upload.
    
    Returns an upload_id and the chunk size to use.
    """
    config = get_config()
    db = get_database()
    
    # Validate file size
    max_size = config.max_file_size_mb * 1024 * 1024
    if request.size_bytes > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {config.max_file_size_mb}MB"
        )
    
    # Generate upload ID
    upload_id = str(uuid.uuid4())
    
    # Calculate number of chunks
    chunk_size = config.chunk_size_bytes
    total_chunks = (request.size_bytes + chunk_size - 1) // chunk_size
    
    # Create upload directory
    upload_dir = config.upload_dir / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Create database record
    uploaded_file = UploadedFile(
        id=upload_id,
        original_filename=request.filename,
        stored_path=str(upload_dir / "chunks"),  # Temporary path during upload
        size_bytes=request.size_bytes,
        content_type=request.content_type,
        upload_status=UploadStatus.PENDING,
        chunks_received=0,
        total_chunks=total_chunks,
    )
    db.create_uploaded_file(uploaded_file)
    
    return UploadInitResponse(
        upload_id=upload_id,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
    )


@router.post("/upload/{upload_id}/chunk/{chunk_num}", response_model=ChunkUploadResponse)
async def upload_chunk(
    upload_id: str,
    chunk_num: int,
    file: UploadFile = File(...),
):
    """
    Upload a single chunk of a file.
    
    Chunks must be uploaded in order (0, 1, 2, ...).
    """
    config = get_config()
    db = get_database()
    
    # Get upload record
    upload = db.get_uploaded_file(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    if upload.upload_status == UploadStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="Upload already complete")
    
    if upload.upload_status == UploadStatus.FAILED:
        raise HTTPException(status_code=400, detail="Upload failed, please start a new upload")
    
    # Validate chunk number
    if chunk_num != upload.chunks_received:
        raise HTTPException(
            status_code=400,
            detail=f"Expected chunk {upload.chunks_received}, got {chunk_num}"
        )
    
    # Save chunk to disk
    chunk_dir = config.upload_dir / upload_id / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_num:05d}"
    
    content = await file.read()
    with open(chunk_path, 'wb') as f:
        f.write(content)
    
    # Update progress
    chunks_received = upload.chunks_received + 1
    status = UploadStatus.UPLOADING
    db.update_upload_progress(upload_id, chunks_received, status)
    
    # Calculate bytes received
    bytes_received = chunks_received * config.chunk_size_bytes
    if chunks_received == upload.total_chunks:
        # Last chunk might be smaller
        bytes_received = upload.size_bytes
    
    return ChunkUploadResponse(
        bytes_received=bytes_received,
        chunks_received=chunks_received,
        next_chunk=chunks_received,  # 0-indexed, so this is the next chunk number
    )


@router.post("/upload/{upload_id}/complete", response_model=UploadCompleteResponse)
async def complete_upload(upload_id: str):
    """
    Finalize a chunked upload.
    
    Assembles chunks into the final file and validates it.
    """
    config = get_config()
    db = get_database()
    
    # Get upload record
    upload = db.get_uploaded_file(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    if upload.upload_status == UploadStatus.COMPLETE:
        # Already complete, return current state
        return UploadCompleteResponse(
            file_id=upload.id,
            filename=upload.original_filename,
            size_bytes=upload.size_bytes or 0,
            stored_path=upload.stored_path,
        )
    
    if upload.chunks_received != upload.total_chunks:
        raise HTTPException(
            status_code=400,
            detail=f"Upload incomplete: received {upload.chunks_received}/{upload.total_chunks} chunks"
        )
    
    # Assemble chunks into final file
    chunk_dir = config.upload_dir / upload_id / "chunks"
    final_path = config.upload_dir / upload_id / upload.original_filename
    
    try:
        with open(final_path, 'wb') as outfile:
            for i in range(upload.total_chunks):
                chunk_path = chunk_dir / f"chunk_{i:05d}"
                with open(chunk_path, 'rb') as chunk_file:
                    shutil.copyfileobj(chunk_file, outfile)
        
        # Verify file size
        actual_size = final_path.stat().st_size
        if actual_size != upload.size_bytes:
            raise HTTPException(
                status_code=400,
                detail=f"File size mismatch: expected {upload.size_bytes}, got {actual_size}"
            )
        
        # Validate audio format
        if not validate_audio_file(final_path):
            # Clean up and fail
            final_path.unlink(missing_ok=True)
            db.update_upload_progress(upload_id, upload.chunks_received, UploadStatus.FAILED)
            raise HTTPException(
                status_code=400,
                detail="Invalid audio file format"
            )
        
        # Clean up chunks
        shutil.rmtree(chunk_dir, ignore_errors=True)
        
        # Update database
        db.update_upload_complete(upload_id, str(final_path), actual_size)
        
        return UploadCompleteResponse(
            file_id=upload_id,
            filename=upload.original_filename,
            size_bytes=actual_size,
            stored_path=str(final_path),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.update_upload_progress(upload_id, upload.chunks_received, UploadStatus.FAILED)
        raise HTTPException(status_code=500, detail=f"Failed to assemble file: {str(e)}")


@router.get("/upload/{upload_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(upload_id: str):
    """Get the current status of an upload."""
    db = get_database()
    
    upload = db.get_uploaded_file(upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    percent = 0.0
    if upload.total_chunks > 0:
        percent = (upload.chunks_received / upload.total_chunks) * 100
    
    return UploadStatusResponse(
        upload_id=upload.id,
        filename=upload.original_filename,
        status=upload.upload_status.value if isinstance(upload.upload_status, UploadStatus) else upload.upload_status,
        size_bytes=upload.size_bytes,
        chunks_received=upload.chunks_received,
        total_chunks=upload.total_chunks,
        percent_complete=round(percent, 1),
    )


@router.get("/{file_id}/audio")
async def get_audio_file(
    file_id: str,
    request: Request,
    range: Optional[str] = Header(None),
):
    """
    Serve an audio file with HTTP Range request support.
    
    Supports partial content (206) for seeking in audio players.
    """
    db = get_database()
    
    upload = db.get_uploaded_file(file_id)
    if not upload:
        raise HTTPException(status_code=404, detail="File not found")
    
    if upload.upload_status != UploadStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="File upload not complete")
    
    file_path = Path(upload.stored_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    file_size = file_path.stat().st_size
    
    # Detect content type from file (more reliable than stored type)
    # This handles cases where content_type wasn't set correctly during upload
    content_type = detect_audio_content_type(file_path)
    
    # Handle range requests for seeking
    if range:
        # Parse range header: "bytes=start-end"
        try:
            range_str = range.replace("bytes=", "")
            parts = range_str.split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if parts[1] else file_size - 1
            
            # Validate range
            if start >= file_size:
                raise HTTPException(status_code=416, detail="Range not satisfiable")
            
            end = min(end, file_size - 1)
            content_length = end - start + 1
            
            def iter_file():
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk_size = min(8192, remaining)
                        data = f.read(chunk_size)
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
            
            headers = {
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "max-age=86400",
                # CORS headers for WebAudio API cross-origin playback
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
            }
            
            return StreamingResponse(
                iter_file(),
                status_code=206,
                media_type=content_type,
                headers=headers,
            )
        except (ValueError, IndexError):
            raise HTTPException(status_code=400, detail="Invalid range header")
    
    # Full file response
    return FileResponse(
        file_path,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "max-age=86400",
            # CORS headers for WebAudio API cross-origin playback
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
        },
    )


@router.head("/{file_id}/audio")
async def head_audio_file(file_id: str):
    """HEAD request for audio file (for getting size before download)."""
    db = get_database()
    
    upload = db.get_uploaded_file(file_id)
    if not upload:
        raise HTTPException(status_code=404, detail="File not found")
    
    if upload.upload_status != UploadStatus.COMPLETE:
        raise HTTPException(status_code=400, detail="File upload not complete")
    
    file_path = Path(upload.stored_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    file_size = file_path.stat().st_size
    
    # Detect content type from file (more reliable than stored type)
    content_type = detect_audio_content_type(file_path)
    
    return FileResponse(
        file_path,
        media_type=content_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Cache-Control": "max-age=86400",
            # CORS headers for WebAudio API cross-origin playback
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Range, Accept-Ranges, Content-Length",
        },
    )


@router.post("/cleanup/orphaned")
async def cleanup_orphaned_uploads(dry_run: bool = False):
    """
    Clean up orphaned upload directories.
    
    Orphaned uploads are directories in the uploads folder that are not
    referenced by any job's interviewer_file_id or participant_file_id.
    
    This is useful for manual maintenance or debugging. Note that orphan
    cleanup also runs automatically on server startup.
    
    Args:
        dry_run: If True, only report what would be deleted without actually deleting.
    
    Returns:
        Cleanup result with counts and any errors.
    """
    from web_api.services.upload_cleanup import get_cleanup_service
    
    cleanup_service = get_cleanup_service()
    result = cleanup_service.cleanup_orphaned_uploads(dry_run=dry_run)
    
    return {
        "dry_run": dry_run,
        "orphaned_found": result.orphaned_dirs_found,
        "orphaned_removed": result.orphaned_dirs_removed,
        "bytes_freed": result.bytes_freed,
        "bytes_freed_mb": round(result.bytes_freed / (1024 * 1024), 2),
        "errors": result.errors,
        "success": result.success,
    }


@router.get("/cleanup/status")
async def get_cleanup_status():
    """
    Get the current status of orphaned uploads without removing them.
    
    This is a non-destructive way to check how many orphaned uploads exist.
    
    Returns:
        Count of orphaned uploads and their total size.
    """
    from web_api.services.upload_cleanup import get_cleanup_service
    
    cleanup_service = get_cleanup_service()
    orphaned = cleanup_service.find_orphaned_uploads()
    
    total_size = sum(
        cleanup_service.calculate_directory_size(d) for d in orphaned
    )
    
    return {
        "orphaned_count": len(orphaned),
        "orphaned_ids": [d.name for d in orphaned],
        "total_bytes": total_size,
        "total_mb": round(total_size / (1024 * 1024), 2),
    }
