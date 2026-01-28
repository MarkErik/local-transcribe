"""API routers for the web interface."""

from web_api.routers.files import router as files_router
from web_api.routers.jobs import router as jobs_router
from web_api.routers.transcripts import router as transcripts_router

__all__ = ["files_router", "jobs_router", "transcripts_router"]
