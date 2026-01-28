# Local Transcribe

A comprehensive transcription pipeline for sociology research interviews, featuring automatic speech recognition (ASR), voice activity detection (VAD), speaker diarization, de-identification, and LLM-based transcript cleanup.

## Features

- **Multiple Pipeline Modes:**
  - `vad_split_audio`: Two separate audio files (Interviewer + Participant)
  - `diarized`: Single audio file with speaker diarization
  - `aligned`: Single audio file with forced alignment

- **Transcription Engines:**
  - Whisper (local or remote)
  - MLX Whisper (Apple Silicon optimized)
  - Faster Whisper

- **De-identification:**
  - Two-pass LLM-based name discovery and redaction
  - Interactive name list review
  - Manual redaction controls

- **Web Interface:**
  - Browser-based UI for job management
  - Real-time progress monitoring
  - Interactive transcript editor with audio sync
  - Multiple export formats

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/local-transcribe.git
cd local-transcribe

# Install dependencies
uv sync
```

### Start the Web Server

```bash
# Start in development mode
./start_server.sh

# Or start in production mode
./start_server.sh --production

# Access the interface
open http://localhost:8099
```

See [docs/WEB_INTERFACE.md](docs/WEB_INTERFACE.md) for detailed web interface documentation.

### Command Line Usage

```bash
# VAD split audio mode (two files)
uv run python -m local_transcribe.main \
    --mode vad_split_audio \
    --audio samples/audioMA-P10_cropped_30.0min.m4a \
    --second-audio samples/audioP10_cropped_30.0min.m4a \
    --enable-de-identification

# Diarized mode (single file)
uv run python -m local_transcribe.main \
    --mode diarized \
    --audio samples/interview.m4a
```

## Web Interface

The web interface provides a browser-based tool for:

1. **Job Management** - Upload audio files, configure options, monitor progress
2. **Transcript Editing** - Edit transcripts with synchronized audio playback
3. **De-identification Review** - Review and edit discovered names
4. **Export** - Download transcripts in multiple formats

### Starting the Server

```bash
./start_server.sh              # Development mode (auto-reload)
./start_server.sh --production  # Production mode (multi-worker)
./start_server.sh --help        # Show all options
```

### Configuration

Server configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSCRIBE_PORT` | `8099` | Server port |
| `TRANSCRIBE_UPLOAD_DIR` | `./uploads` | Upload directory |
| `TRANSCRIBE_OUTPUT_DIR` | `./output` | Output directory |
| `TRANSCRIBE_MAX_FILE_SIZE_MB` | `500` | Max upload size |

### API Documentation

Once the server is running, view the API documentation at:
- **Swagger UI:** http://localhost:8099/docs
- **ReDoc:** http://localhost:8099/redoc

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_api_health.py -v

# Run with coverage
uv run pytest tests/ --cov=web_api
```

## Project Structure

```
local-transcribe/
├── local_transcribe/       # Core pipeline library
│   ├── framework/          # Pipeline framework (stages, plugins)
│   ├── lib/               # Utility libraries
│   ├── processing/        # Audio/text processing modules
│   └── providers/         # ASR, VAD, alignment providers
├── web_api/               # FastAPI web backend
│   ├── routers/           # API endpoint routers
│   ├── services/          # Business logic services
│   └── models/            # Pydantic schemas
├── web_ui/                # React frontend
│   └── src/
│       ├── components/    # UI components
│       ├── pages/         # Page components
│       └── store/         # Zustand stores
├── tests/                 # Test suite
├── docs/                  # Documentation
├── start_server.sh        # Server startup script
└── pyproject.toml         # Python project config
```

## Documentation

- [Web Interface Guide](docs/WEB_INTERFACE.md) - Complete web UI documentation
- [API Reference](http://localhost:8099/docs) - OpenAPI specification (when server running)

## License

[Your License Here]

## Contributing

[Contributing guidelines]