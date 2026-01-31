"""
FastAPI application entry point for local-transcribe web API.

Run with: uv run uvicorn web_api.main:app --reload --port 8299
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from web_api import __version__
from web_api.config import get_config
from web_api.database import init_database
from web_api.routers import files_router, jobs_router, transcripts_router
from web_api.routers.deidentification import router as deidentification_router
from web_api.routers.exports import router as exports_router
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
    
    # Clean up stale jobs (jobs stuck in 'pending' or 'running' from previous runs)
    from web_api.database import get_database
    db = get_database()
    stale_jobs = db.cleanup_stale_jobs()
    if stale_jobs > 0:
        print(f"Marked {stale_jobs} stale job(s) as failed (server restart recovery)")
    
    yield
    
    # Shutdown
    # Cleanup incomplete uploads
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
app.include_router(deidentification_router)
app.include_router(exports_router)


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
frontend_index_path = frontend_build_dir / "index.html"

if frontend_build_dir.exists():
    # Mount static assets directory (CSS, JS, etc.)
    # These are files like /assets/index-xxx.js, /assets/index-xxx.css
    assets_dir = frontend_build_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    
    # Serve favicon and other root static files
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        favicon_path = frontend_build_dir / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return FileResponse(frontend_index_path)
    
    @app.get("/vite.svg", include_in_schema=False)
    async def vite_svg():
        svg_path = frontend_build_dir / "vite.svg"
        if svg_path.exists():
            return FileResponse(svg_path, media_type="image/svg+xml")
        return FileResponse(frontend_index_path)

# SPA fallback: serve index.html for all non-API routes
# This must be defined after all other routes to act as a catch-all
@app.api_route("/{full_path:path}", methods=["GET"], include_in_schema=False)
async def spa_fallback(request: Request, full_path: str):
    """
    SPA fallback handler for client-side routing.
    
    Serves index.html for all non-API routes so that React Router
    can handle client-side navigation (e.g., /jobs/xxx/edit).
    """
    # Don't serve index.html for API routes or built-in FastAPI routes
    excluded_prefixes = ["api/", "docs", "redoc", "openapi.json"]
    for prefix in excluded_prefixes:
        if full_path.startswith(prefix) or full_path == prefix.rstrip("/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
    
    # Check if frontend is built
    if frontend_build_dir.exists() and frontend_index_path.exists():
        # Check if the request is for a static file that exists
        static_file = frontend_build_dir / full_path
        if static_file.is_file() and static_file.exists():
            return FileResponse(static_file)
        
        # Otherwise serve index.html for SPA routing
        return FileResponse(frontend_index_path)
    
    # Frontend not built - return API info
    return {
        "name": "Local Transcribe API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
        "note": "Frontend not built. Run 'npm run build' in web_ui/ directory.",
    }


# Development: Add a simple root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API info."""
    # If frontend is built, serve it
    if frontend_build_dir.exists() and frontend_index_path.exists():
        return FileResponse(frontend_index_path)
    
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
