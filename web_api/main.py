"""
FastAPI application entry point for local-transcribe web API.

Run with: uv run uvicorn web_api.main:app --reload --port 8099
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from web_api import __version__
from web_api.config import get_config
from web_api.database import init_database
from web_api.routers import files_router, jobs_router, transcripts_router
from web_api.models.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes database and configuration on startup,
    performs cleanup on shutdown.
    """
    # Startup
    config = get_config()
    config.ensure_directories()
    init_database(config.database_path)
    
    yield
    
    # Shutdown
    # Cleanup incomplete uploads
    from web_api.database import get_database
    db = get_database()
    cleaned = db.cleanup_stale_uploads(hours=config.upload_timeout_hours)
    if cleaned > 0:
        print(f"Cleaned up {cleaned} stale uploads")


# Create FastAPI app
app = FastAPI(
    title="Local Transcribe API",
    description="Web API for the local-transcribe transcription pipeline",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files_router)
app.include_router(jobs_router)
app.include_router(transcripts_router)


# Health check endpoint
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Check API health and database connectivity."""
    from web_api.database import get_database
    
    try:
        db = get_database()
        # Simple query to verify database is accessible
        db.list_jobs(limit=1)
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        version=__version__,
        database=db_status,
    )


# Serve static files for frontend (production)
# Only mount if the build directory exists
frontend_build_dir = Path(__file__).parent.parent / "web_ui" / "dist"
if frontend_build_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_build_dir), html=True), name="frontend")


# Development: Add a simple root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Local Transcribe API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "web_api.main:app",
        host=config.host,
        port=config.port,
        reload=True,
    )
