#!/bin/bash
#
# Local Transcribe Web Server Startup Script
#
# Usage:
#   ./start_server.sh                  # Start in development mode
#   ./start_server.sh --production     # Start in production mode
#   ./start_server.sh --port 9000      # Use custom port
#   ./start_server.sh --help           # Show help
#

set -e

# Default configuration
HOST="${TRANSCRIBE_HOST:-0.0.0.0}"
PORT="${TRANSCRIBE_PORT:-8099}"
MODE="development"
WORKERS=1
LOG_LEVEL="info"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Print banner
print_banner() {
    echo ""
    print_msg "$BLUE" "╔══════════════════════════════════════════════════════════╗"
    print_msg "$BLUE" "║         Local Transcribe Web Server                      ║"
    print_msg "$BLUE" "╚══════════════════════════════════════════════════════════╝"
    echo ""
}

# Show help
show_help() {
    print_banner
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --production, -p    Start in production mode (no auto-reload, multiple workers)"
    echo "  --port PORT         Set the server port (default: 8099)"
    echo "  --host HOST         Set the server host (default: 0.0.0.0)"
    echo "  --workers N         Number of worker processes (production only, default: 4)"
    echo "  --log-level LEVEL   Set log level: debug, info, warning, error (default: info)"
    echo "  --build-frontend    Build the frontend before starting"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  TRANSCRIBE_HOST           Server bind address"
    echo "  TRANSCRIBE_PORT           Server port"
    echo "  TRANSCRIBE_UPLOAD_DIR     Upload directory path"
    echo "  TRANSCRIBE_OUTPUT_DIR     Output directory path"
    echo "  TRANSCRIBE_DATABASE_PATH  SQLite database path"
    echo "  TRANSCRIBE_MAX_FILE_SIZE_MB  Maximum upload file size in MB"
    echo ""
    echo "Examples:"
    echo "  $0                           # Development mode on port 8099"
    echo "  $0 --production --port 80    # Production on port 80"
    echo "  $0 --build-frontend -p       # Build frontend, then production mode"
    echo ""
}

# Parse arguments
BUILD_FRONTEND=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --production|-p)
            MODE="production"
            WORKERS=4
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --build-frontend)
            BUILD_FRONTEND=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_msg "$RED" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Print banner
print_banner

# Check if uv is available
if ! command -v uv &> /dev/null; then
    print_msg "$RED" "Error: 'uv' is not installed or not in PATH"
    print_msg "$YELLOW" "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build frontend if requested
if [ "$BUILD_FRONTEND" = true ]; then
    print_msg "$BLUE" "Building frontend..."
    
    if [ ! -d "web_ui" ]; then
        print_msg "$RED" "Error: web_ui directory not found"
        exit 1
    fi
    
    cd web_ui
    
    if ! command -v npm &> /dev/null; then
        print_msg "$RED" "Error: npm is not installed"
        exit 1
    fi
    
    npm install
    npm run build
    cd ..
    
    print_msg "$GREEN" "Frontend built successfully"
    echo ""
fi

# Check if frontend is built (for production mode)
if [ "$MODE" = "production" ]; then
    if [ ! -d "web_ui/dist" ]; then
        print_msg "$YELLOW" "Warning: Frontend not built. Run with --build-frontend or:"
        print_msg "$YELLOW" "  cd web_ui && npm install && npm run build"
        echo ""
    fi
fi

# Create necessary directories
mkdir -p uploads output data

# Print configuration
print_msg "$GREEN" "Configuration:"
echo "  Mode:       $MODE"
echo "  Host:       $HOST"
echo "  Port:       $PORT"
if [ "$MODE" = "production" ]; then
    echo "  Workers:    $WORKERS"
fi
echo "  Log Level:  $LOG_LEVEL"
echo ""

# Print URLs
print_msg "$GREEN" "Server URLs:"
echo "  API:      http://${HOST}:${PORT}/api"
echo "  Docs:     http://${HOST}:${PORT}/docs"
if [ -d "web_ui/dist" ]; then
    echo "  Frontend: http://${HOST}:${PORT}/"
else
    print_msg "$YELLOW" "  Frontend: Not available (build with --build-frontend)"
fi
echo ""

# Start server
print_msg "$BLUE" "Starting server..."
echo ""

if [ "$MODE" = "development" ]; then
    # Development mode with auto-reload
    exec uv run uvicorn web_api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    # Production mode with multiple workers
    exec uv run uvicorn web_api.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi
