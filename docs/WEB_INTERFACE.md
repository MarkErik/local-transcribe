# Local Transcribe Web Interface

This document provides comprehensive documentation for the Local Transcribe web interface, a browser-based tool for transcribing audio interviews with interactive editing, de-identification, and export capabilities.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Server Setup](#server-setup)
- [Client Usage](#client-usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Local Transcribe web interface allows sociology researchers to:

- **Upload audio files** for transcription (supports M4A, MP3, WAV, FLAC, OGG)
- **Monitor pipeline progress** in real-time via Server-Sent Events (SSE)
- **View and edit transcripts** with synchronized audio playback
- **Review and correct de-identification** with two-pass name discovery
- **Export transcripts** in multiple formats (TXT, JSON, SRT, Markdown)

### Architecture

```
┌─────────────────────┐     HTTP/SSE      ┌─────────────────────┐
│                     │ ◄──────────────► │                     │
│   Web Browser       │                   │   FastAPI Server    │
│   (React Frontend)  │                   │   (Port 8099)       │
│                     │                   │                     │
└─────────────────────┘                   └─────────────────────┘
                                                   │
                                                   ▼
                                          ┌─────────────────────┐
                                          │   Transcription     │
                                          │   Pipeline          │
                                          │   (VAD + ASR)       │
                                          └─────────────────────┘
```

**Primary Mode:** VAD-split-audio (two separate audio files: Interviewer + Participant)

---

## Quick Start

### 1. Start the Server

```bash
cd /path/to/local-transcribe

# Using the startup script (recommended)
./start_server.sh

# Or manually with uv
uv run uvicorn web_api.main:app --host 0.0.0.0 --port 8099
```

### 2. Access the Interface

Open your browser and navigate to:

- **Production:** http://localhost:8099 (serves built frontend)
- **Development:** http://localhost:5173 (with Vite dev server)
- **API Documentation:** http://localhost:8099/docs (Swagger UI)

### 3. Create a Transcription Job

1. Click "New Job" in the navigation
2. Upload your interviewer audio file
3. Upload your participant audio file
4. Configure options (de-identification, cleanup, output formats)
5. Click "Start Transcription"
6. Monitor progress in real-time

---

## Server Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Node.js 18+ (for frontend development only)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/local-transcribe.git
cd local-transcribe

# Install Python dependencies
uv sync

# (Optional) Build frontend for production
cd web_ui
npm install
npm run build
cd ..
```

### Starting the Server

#### Using the Startup Script

```bash
# Start in development mode (auto-reload enabled)
./start_server.sh

# Start in production mode
./start_server.sh --production

# Specify a custom port
./start_server.sh --port 9000

# Show all options
./start_server.sh --help
```

#### Manual Start

```bash
# Development mode (with auto-reload)
uv run uvicorn web_api.main:app --reload --host 0.0.0.0 --port 8099

# Production mode
uv run uvicorn web_api.main:app --host 0.0.0.0 --port 8099 --workers 4
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_HOST` | `0.0.0.0` | Server bind address |
| `TRANSCRIBE_PORT` | `8099` | Server port |
| `TRANSCRIBE_UPLOAD_DIR` | `./uploads` | Directory for uploaded files |
| `TRANSCRIBE_OUTPUT_DIR` | `./output` | Directory for job outputs |
| `TRANSCRIBE_DATABASE_PATH` | `./data/transcribe.db` | SQLite database path |
| `TRANSCRIBE_MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `TRANSCRIBE_CORS_ORIGINS` | `*` | Allowed CORS origins (comma-separated) |

Example `.env` file:

```env
TRANSCRIBE_PORT=8099
TRANSCRIBE_UPLOAD_DIR=/data/uploads
TRANSCRIBE_OUTPUT_DIR=/data/output
TRANSCRIBE_MAX_FILE_SIZE_MB=1000
```

### Remote Server Configuration

To run the server on a remote machine (e.g., a Tailscale network):

1. Ensure the server is accessible on the network
2. Configure firewall to allow the port (default: 8099)
3. Set `TRANSCRIBE_HOST=0.0.0.0` to listen on all interfaces

```bash
# Example: Server on Tailscale network
TRANSCRIBE_HOST=0.0.0.0 ./start_server.sh --production

# Access from client at http://100.84.208.72:8099
```

---

## Client Usage

### Creating a New Job

#### Step 1: Upload Audio Files

The interface supports chunked uploads for large files:

1. Click "Select Interviewer Audio" and choose the interviewer's audio file
2. Click "Select Participant Audio" and choose the participant's audio file
3. Wait for both uploads to complete (progress bars show status)

**Supported formats:** M4A, MP3, WAV, FLAC, OGG

#### Step 2: Configure Options

| Option | Description |
|--------|-------------|
| **Enable De-identification** | Run two-pass LLM de-identification to redact names |
| **Enable Cleanup** | Apply LLM transcript cleanup for readability |
| **Output Formats** | Select which export formats to generate |

#### Step 3: Submit and Monitor

1. Click "Start Transcription"
2. Watch real-time progress:
   - VAD detection progress (blocks processed)
   - ASR transcription progress (per speaker)
   - De-identification passes (if enabled)
   - Output generation

### Editing Transcripts

After a job completes, click "Edit Transcript" to open the editor:

#### Audio Playback

- **Dual-track player:** Separate waveforms for Interviewer and Participant
- **Synchronized playback:** Both tracks play in sync
- **Controls:** Play/Pause (Space), Seek (arrows), Speed (0.5x-2x)
- **Solo/Mute:** Focus on one speaker at a time

#### Transcript Editing

| Action | How to |
|--------|--------|
| **Edit word** | Double-click the word |
| **Insert word** | Click between words, type in the insertion point |
| **Delete word** | Select word, press Delete |
| **Change speaker** | Click turn, select speaker from dropdown |
| **Merge turns** | Select turn, click "Merge with next" |
| **Split turn** | Click word, click "Split here" |
| **Add annotation** | Click between words, click annotation button |

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `←` / `→` | Seek -5s / +5s |
| `[` / `]` | Skip to previous/next turn |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Ctrl+F` | Find & Replace |
| `?` | Show all shortcuts |

### De-identification Review

If de-identification is enabled:

1. **First pass completes** → Review discovered names
2. **Edit name list:**
   - Remove false positives (non-names flagged as names)
   - Add missed names
   - Toggle which names to redact
3. **Approve and continue** to second pass
4. **Review PII highlighting:**
   - Yellow highlight: LLM-detected names
   - Purple highlight: Manually redacted
   - Click to restore incorrectly redacted text

### Exporting Transcripts

#### Available Formats

| Format | Description | Extension |
|--------|-------------|-----------|
| Timestamped Text | Plain text with timestamps | `.timestamped.txt` |
| Plain Text | Clean text grouped by speaker | `.txt` |
| JSON (Structured) | Full metadata including words | `.turns.json` |
| Dialogue Script | Screenplay-style format | `.script.txt` |
| Markdown | Rich formatted with statistics | `.md` |
| SRT Subtitles | Standard subtitle format | `.srt` |

#### Export Options

1. **Single format:** Click stage, select format, download
2. **Bulk export:** Select multiple formats, download as ZIP
3. **Comparison export:** Side-by-side view of raw vs cleaned

---

## API Reference

The API uses REST endpoints with JSON request/response bodies. Full OpenAPI documentation is available at `/docs`.

### Core Endpoints

#### Health Check

```http
GET /api/health

Response:
{
  "status": "ok",
  "version": "0.1.0",
  "database": "connected"
}
```

#### File Upload

```http
# Initialize upload
POST /api/files/upload/init
{
  "filename": "interview.m4a",
  "size_bytes": 50000000
}

Response:
{
  "upload_id": "abc-123",
  "chunk_size": 5242880,
  "total_chunks": 10
}

# Upload chunk
POST /api/files/upload/{upload_id}/chunk/{chunk_num}
Content-Type: multipart/form-data
Body: chunk file data

# Complete upload
POST /api/files/upload/{upload_id}/complete

Response:
{
  "file_id": "file-xyz",
  "filename": "interview.m4a",
  "path": "/uploads/abc-123/interview.m4a"
}
```

#### Job Management

```http
# Create job
POST /api/jobs
{
  "interviewer_file_id": "file-1",
  "participant_file_id": "file-2",
  "mode": "vad_split_audio",
  "options": {
    "enable_de_identification": true,
    "enable_cleanup": false,
    "output_formats": ["timestamped-txt", "turns-json"]
  }
}

# List jobs
GET /api/jobs?status=completed&limit=20

# Get job
GET /api/jobs/{job_id}

# Get progress (SSE stream)
GET /api/jobs/{job_id}/progress
```

#### Transcript Operations

```http
# Get transcript
GET /api/jobs/{job_id}/transcript?stage=vad_transcription

# Get available stages
GET /api/jobs/{job_id}/transcript/stages

# Create edit
POST /api/jobs/{job_id}/edits
{
  "edit_type": "word_change",
  "turn_id": 1,
  "start_index": 0,
  "new_value": "Hello"
}

# List edits
GET /api/jobs/{job_id}/edits

# Re-run from checkpoint
POST /api/jobs/{job_id}/rerun
{
  "start_stage": "de_identification",
  "apply_edits": true
}
```

#### Export

```http
# List formats
GET /api/jobs/{job_id}/export/formats

# Export single format
POST /api/jobs/{job_id}/export
{
  "format": "timestamped-txt",
  "stage": "speaker_naming"
}

# Bulk export
POST /api/jobs/{job_id}/export/bulk
{
  "formats": ["timestamped-txt", "turns-json", "markdown"]
}
```

### SSE Event Types

When connecting to `/api/jobs/{job_id}/progress`:

```
event: stage_start
data: {"stage": "vad_transcription", "message": "Starting..."}

event: block_progress
data: {"stage": "vad_transcription", "current": 15, "total": 42, "speaker": "Interviewer"}

event: stage_complete
data: {"stage": "vad_transcription", "duration_s": 120.5}

event: job_complete
data: {"job_id": "abc-123", "status": "completed"}

event: job_error
data: {"job_id": "abc-123", "error": "Transcription failed", "stage": "vad_transcription"}
```

---

## Configuration

### Server Configuration

Configuration is loaded from environment variables. See [Environment Variables](#environment-variables).

### Transcription Server

For remote transcription (Whisper server):

```bash
# Set in environment or .env
REMOTE_TRANSCRIPTION_URL=http://100.84.208.72:7070
```

### De-identification Server

For LLM-based de-identification:

```bash
DEID_LLM_URL=http://100.84.208.72:8080
```

---

## Troubleshooting

### Server Won't Start

**Error:** `Address already in use`

```bash
# Find process using port
lsof -i :8099

# Kill the process or use different port
./start_server.sh --port 9000
```

**Error:** `ModuleNotFoundError`

```bash
# Ensure dependencies are installed
uv sync
```

### Upload Fails

**Error:** `File too large`

Increase the maximum file size:

```bash
TRANSCRIBE_MAX_FILE_SIZE_MB=1000 ./start_server.sh
```

**Error:** `Invalid audio format`

Ensure the file is a valid audio format. The server validates magic bytes:
- MP3: `ID3` or `\xff\xfb`
- M4A: `ftyp` box at bytes 4-7
- WAV: `RIFF` header
- FLAC: `fLaC` header
- OGG: `OggS` header

### Pipeline Errors

**Error:** `Transcription server unavailable`

Check that the remote transcription server is running:

```bash
curl http://100.84.208.72:7070/health
```

**Error:** `Out of memory`

The Whisper model requires significant GPU memory. Consider:
- Using a smaller model (tiny, base, small)
- Processing one file at a time
- Increasing system swap space

### Frontend Issues

**Blank page after navigation**

Clear browser cache and reload:

```
Ctrl+Shift+R (Windows/Linux)
Cmd+Shift+R (Mac)
```

**SSE connection drops**

The client automatically reconnects. If issues persist:
1. Check server logs for errors
2. Verify network connectivity
3. Increase SSE timeout in server config

### Database Issues

**Error:** `database is locked`

SQLite allows only one writer at a time. This is expected with concurrent requests. The server retries automatically.

**Corrupted database**

```bash
# Backup existing database
mv data/transcribe.db data/transcribe.db.bak

# Server will create new database on restart
./start_server.sh
```

---

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_api_health.py

# Run with verbose output
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=web_api
```

---

## License

See the main repository LICENSE file for licensing information.
