# Refactoring Plan: Database-Centric Transcript Storage for Web Interface

## Executive Summary

This document outlines a refactoring plan to move the web interface from file-based JSON transcript storage to database-centric storage. The goal is to improve editing, comparison, tracking, and undo capabilities while maintaining complete separation from the CLI pipeline.

### Key Benefits
1. **Atomic edits**: Each edit is stored as a database record, enabling true undo/redo
2. **Version tracking**: Full history of all changes with timestamps
3. **Efficient comparison**: Compare any two versions without file I/O
4. **No intermediate files**: Eliminate stage-specific JSON files for web workflows
5. **On-demand export**: Generate output files only when user explicitly exports
6. **Cleaner separation**: Web workflow is database-driven; CLI remains file-driven

### Important Clarifications

**Speaker Naming for Web**: The web interface uses `vad_split_audio` mode where users upload separate "Interviewer" and "Participant" audio files. Therefore, **speaker names are known from upload time** and no `speaker_naming` stage is needed for web. The pipeline already uses the file labels as speaker names.

**CLI Unchanged**: The CLI will continue to operate with its existing file-based workflow. This refactoring only affects the web interface.

---

## Current Architecture Analysis

### Current Web Workflow
```
Upload → Pipeline Execution → JSON Files → Load + Apply Edits → Display
                                  ↓
                        turns.json (vad_transcription)
                        turns-de-identified.json (de_identification)
                        turns-named.json (speaker_naming)  ← NOT NEEDED FOR WEB
                        turns-cleaned.json (transcript_cleanup)
```

### Current Files and Their Roles

| File | Stage | Purpose |
|------|-------|---------|
| `turns.json` | vad_transcription | Raw transcript with Interviewer/Participant |
| `turns-de-identified.json` | de_identification | After PII redaction |
| `turns-named.json` | speaker_naming | After speaker name assignment (N/A for web) |
| `turns-cleaned.json` | transcript_cleanup | After LLM cleanup |

### Current Edit Storage
Edits are already stored in the database (`edits` table), but the base transcript data lives in JSON files. This hybrid approach creates complexity:
- Must read JSON file + apply edits on every request
- File system becomes source of truth
- No versioning of base transcript data

---

## Proposed Architecture

### New Web Workflow
```
Upload → Pipeline Execution → Store in DB → Apply Edits in DB → Display
                                  ↓
                           transcript_data table (single source of truth)
                           edits table (change history)
                                  ↓
                           Export on demand → Generate output files
```

### New Database Schema

```sql
-- Store the actual transcript data in the database
-- One row per job per logical stage
CREATE TABLE transcript_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,  -- 'base' | 'de_identified' | 'cleaned'
    version INTEGER NOT NULL DEFAULT 1,
    data_json TEXT NOT NULL,  -- Full TranscriptFlow JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT DEFAULT 'pipeline',  -- 'pipeline' | 'edit' | 'import'
    is_current BOOLEAN DEFAULT TRUE,
    
    FOREIGN KEY (job_id) REFERENCES jobs(id),
    UNIQUE(job_id, stage, version)
);

-- Index for fast lookups
CREATE INDEX idx_transcript_data_job_stage ON transcript_data(job_id, stage, is_current);

-- Enhanced edits table (mostly unchanged, but add version tracking)
CREATE TABLE edits_v2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,  -- Which stage this edit applies to
    edit_type TEXT NOT NULL,
    turn_id INTEGER,
    start_index INTEGER,
    end_index INTEGER,
    original_value TEXT,
    new_value TEXT,
    target_turn_id INTEGER,
    annotation_type TEXT,
    
    -- Version tracking
    applied_to_version INTEGER,  -- Which transcript version this was applied to
    result_version INTEGER,      -- Version created by this edit (if materialized)
    
    -- Undo support
    is_undone BOOLEAN DEFAULT FALSE,
    undone_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES jobs(id)
);
```

### Stages for Web Interface

Since speaker naming is not needed for web (speakers are known from upload):

| Stage ID | Description | When Created |
|----------|-------------|--------------|
| `base` | Raw transcript from pipeline | After pipeline completes |
| `de_identified` | After PII redaction | After de-identification |
| `cleaned` | After LLM cleanup | After transcript cleanup (optional) |

**Note**: We simplify from 4 stages to 3 by removing `speaker_naming`.

---

## Implementation Plan

### Phase 1: Database Schema Updates

**File**: `web_api/database.py`

1. Add `transcript_data` table to schema
2. Add migration to create table for existing databases
3. Add methods:
   - `store_transcript_data(job_id, stage, data_json, created_by='pipeline')`
   - `get_transcript_data(job_id, stage, version=None)` 
   - `get_current_transcript(job_id, stage)`
   - `create_transcript_version(job_id, stage, data_json, created_by='edit')`
   - `get_transcript_history(job_id, stage)`

### Phase 2: Pipeline Integration (Store to DB)

**Files**: 
- `web_api/services/pipeline_service.py`
- New: `web_api/services/transcript_storage.py`

1. Create `TranscriptStorageService` class
2. After pipeline completes, store `turns.json` content in DB as `base` stage
3. Modify de-identification flow to store results in DB instead of file
4. Modify cleanup flow to store results in DB instead of file
5. Keep file generation for CLI mode (check context)

**Key change in pipeline_service.py**:
```python
# After pipeline completes successfully
if result.success:
    # Store transcript data in database for web access
    transcript_storage = TranscriptStorageService(self.db)
    transcript_storage.store_from_pipeline(job_id, output_dir)
```

### Phase 3: Transcript Retrieval from DB

**Files**:
- `web_api/routers/transcripts.py`
- `web_api/services/edit_applicator.py`

1. Modify `get_transcript` to read from DB instead of files
2. Modify `get_available_stages` to query DB for available stages
3. Update edit applicator to work with DB-stored transcripts

**Key changes**:
```python
@router.get("/{job_id}/transcript")
async def get_transcript(job_id: str, stage: Optional[str] = None):
    db = get_database()
    
    # Get transcript from database
    transcript_data = db.get_current_transcript(job_id, stage or 'base')
    
    if not transcript_data:
        # Fallback to file for backward compatibility (legacy jobs)
        return _get_transcript_from_file(job_id, stage)
    
    # Apply any pending edits
    edits = db.get_edits_for_job(job_id, stage)
    if edits:
        transcript_data = apply_edits_to_transcript(transcript_data, edits)
    
    return JSONResponse(content=transcript_data)
```

### Phase 4: De-identification Integration

**File**: `web_api/routers/deidentification.py`

1. Modify first-pass to read from DB
2. Modify second-pass to store results in DB as `de_identified` stage
3. Remove file writes (keep for legacy/CLI compatibility check)

**Key changes**:
```python
def _apply_deidentification_to_transcript(job_id: str, results: dict) -> None:
    """Apply de-identification results and store in database."""
    db = get_database()
    
    # Get current base transcript from DB
    base_data = db.get_current_transcript(job_id, 'base')
    
    # Apply de-identification
    deidentified_data = _apply_replacements(base_data, results)
    
    # Store as new stage
    db.store_transcript_data(
        job_id=job_id,
        stage='de_identified',
        data_json=json.dumps(deidentified_data),
        created_by='de_identification'
    )
```


### Phase 5: Export Service (On-Demand File Generation)

**File**: `web_api/routers/exports.py`

1. Modify to read from DB instead of files
2. Generate files on-demand during export
3. Support all existing export formats

**This is largely already correct** - exports already apply edits before generating. Main change is data source.


### Phase 6: Edit System Enhancement

**Files**:
- `web_api/routers/transcripts.py`
- `web_api/services/edit_applicator.py`

1. Add undo endpoint
2. Add redo endpoint  
3. Add edit history endpoint
4. Implement version snapshotting for large edit batches

**New endpoints**:
```python
@router.post("/{job_id}/edits/{edit_id}/undo")
async def undo_edit(job_id: str, edit_id: int):
    """Undo a specific edit."""
    
@router.post("/{job_id}/edits/undo-last")
async def undo_last_edit(job_id: str, stage: Optional[str] = None):
    """Undo the most recent edit for a stage."""

@router.get("/{job_id}/edits/history")
async def get_edit_history(job_id: str, stage: Optional[str] = None):
    """Get full edit history with undo status."""
```

### Phase 7: Frontend Updates

**Files**:
- `web_ui/src/api/client.ts`
- `web_ui/src/components/StageSelector.tsx`
- `web_ui/src/pages/TranscriptEditor.tsx`

1. Update stage names (remove `speaker_naming` references for web)
2. Add undo/redo UI controls
3. Add edit history view


---

## Web Pipeline Simplification

### Current Web Pipeline Stages
```
1. Audio Standardization
2. Transcription/Alignment  
3. De-identification (optional)
4. Turn Building
5. Speaker Naming         ← REMOVE for web (speakers known)
6. Output Generation      ← DEFER to export
7. Transcript Cleanup     ← Optional
8. Cleaned Output Gen     ← DEFER to export
```

### Simplified Web Pipeline Stages
```
1. Audio Standardization
2. Transcription/Alignment
3. Turn Building
4. [Pipeline Complete - Store 'base' transcript to DB]
5. De-identification (user-triggered, interactive) → Store 'de_identified' to DB
6. Transcript Cleanup (user-triggered, optional) → Store 'cleaned' to DB
7. Export (user-triggered) → Generate output files
```

### Implementation Note

The web pipeline can skip `SpeakerNamingStage` since:
- User uploads `interviewer_file` and `participant_file` separately
- The pipeline uses filenames to assign speakers: "Interviewer" and "Participant"
- No diarization is needed (speakers are already separated by audio source)

Modify `create_pipeline_for_mode` or add a web-specific mode check:
```python
# In pipeline execution for web, skip speaker_naming
if is_web_context:
    stages = [s for s in stages if s.name != 'speaker_naming']
```

Or better, recognize that `vad_split_audio` mode already handles this correctly - the speaker names come from the file labels, not from an interactive naming step.

---

## File Changes Summary

### New Files
- `web_api/services/transcript_storage.py` - Transcript storage service
- `web_api/migrations/migrate_to_db_storage.py` - Migration script
- `docs/REFACTORING_DATABASE_STORAGE.md` - This document

### Modified Files
- `web_api/database.py` - Schema updates, new methods
- `web_api/services/pipeline_service.py` - Store to DB after pipeline
- `web_api/routers/transcripts.py` - Read from DB
- `web_api/routers/deidentification.py` - Store results to DB
- `web_api/routers/exports.py` - Read from DB for export
- `web_api/services/edit_applicator.py` - Work with DB data
- `web_ui/src/components/StageSelector.tsx` - Remove speaker_naming for web
- `web_ui/src/api/client.ts` - Add undo/redo API calls
- `web_ui/src/pages/TranscriptEditor.tsx` - Add undo/redo UI

### Files Unchanged
- All files in `local_transcribe/` - CLI pipeline unchanged
- `main.py` - CLI entry point unchanged

---

## Testing Strategy

### Unit Tests
1. Test transcript storage CRUD operations
2. Test edit application with DB source
3. Test version creation and retrieval
4. Test undo/redo logic

### Integration Tests  
1. Full pipeline execution with DB storage
2. De-identification with DB storage
3. Export from DB storage

### Manual Testing
1. Create new job, verify transcript in DB
2. Apply edits, verify history
3. Undo/redo edits
4. Export to various formats

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking CLI | Strict separation - CLI never touches web DB code |

---

## Conclusion

This refactoring will significantly improve the web editing experience by:
1. Enabling true undo/redo with full history
2. Simplifying the stage model (3 stages instead of 4)
3. Eliminating intermediate file I/O
4. Providing a cleaner separation between web and CLI workflows

The implementation is designed to be incremental, with fallbacks at each stage for backward compatibility. The CLI pipeline remains completely unchanged.
