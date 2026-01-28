"""
Pipeline service for executing transcription pipelines.

Wraps the existing pipeline infrastructure with progress callbacks
for the web interface.
"""

import json
import traceback
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime

from web_api.config import get_config
from web_api.database import get_database, JobStatus


# Type for progress callback: (event_type: str, data: dict) -> None
ProgressCallback = Callable[[str, Dict[str, Any]], None]


class PipelineService:
    """
    Service for executing transcription pipelines.
    
    Wraps the existing pipeline infrastructure and provides
    progress callbacks for the web interface.
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = get_database()
    
    async def execute_job(
        self,
        job_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Execute a transcription job.
        
        Args:
            job_id: The job ID to execute
            progress_callback: Callback for progress events
        """
        job = self.db.get_job(job_id)
        if not job:
            return
        
        # Parse job configuration
        config_dict = json.loads(job.config_json) if job.config_json else {}
        mode = config_dict.get("mode", "vad_split_audio")
        options = config_dict.get("options", {})
        interviewer_file = config_dict.get("interviewer_file")
        participant_file = config_dict.get("participant_file")
        
        # Mark job as running
        self.db.update_job_status(job_id, JobStatus.RUNNING)
        
        if progress_callback:
            progress_callback("stage_start", {
                "job_id": job_id,
                "stage": "initialization",
                "message": "Initializing pipeline...",
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        output_dir = self.config.output_dir / job_id
        
        try:
            # Import pipeline components
            from local_transcribe.framework.pipeline_context import PipelineContext
            from local_transcribe.framework.pipeline_runner import PipelineRunner
            from local_transcribe.framework.plugin_manager import PluginManager
            from local_transcribe.lib.program_logger import setup_output_context
            
            # Setup logging context
            setup_output_context(
                output_dir=output_dir,
                log_level="INFO",
                console_output=False,  # Don't spam console in web mode
            )
            
            # Create pipeline context
            context = PipelineContext(
                mode=mode,
                interviewer_audio=Path(interviewer_file) if interviewer_file else None,
                participant_audio=Path(participant_file) if participant_file else None,
                output_dir=output_dir,
                enable_de_identification=options.get("enable_de_identification", True),
                enable_cleanup=options.get("enable_cleanup", False),
                output_formats=options.get("output_formats", ["turns-json", "timestamped-txt"]),
            )
            
            # Create progress wrapper for VADASRProcessor
            def block_progress_wrapper(current: int, total: int, speaker: str):
                if progress_callback:
                    progress_callback("block_progress", {
                        "job_id": job_id,
                        "stage": "vad_transcription",
                        "current": current,
                        "total": total,
                        "speaker": speaker,
                        "timestamp": datetime.utcnow().isoformat(),
                    })
            
            # Create stage progress callback
            def stage_callback(stage_name: str, event: str, data: Dict[str, Any]):
                if progress_callback:
                    if event == "start":
                        progress_callback("stage_start", {
                            "job_id": job_id,
                            "stage": stage_name,
                            "message": data.get("message", f"Starting {stage_name}..."),
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                    elif event == "complete":
                        progress_callback("stage_complete", {
                            "job_id": job_id,
                            "stage": stage_name,
                            "duration_s": data.get("duration_s", 0),
                            "summary": data.get("summary"),
                            "timestamp": datetime.utcnow().isoformat(),
                        })
            
            # Store callbacks on context for stages to use
            context.progress_callbacks = {
                "block_progress": block_progress_wrapper,
                "stage": stage_callback,
            }
            
            # Initialize plugin manager and run pipeline
            plugin_manager = PluginManager()
            runner = PipelineRunner(plugin_manager)
            
            # Execute pipeline
            await self._run_pipeline_async(runner, context, stage_callback)
            
            # Mark complete
            self.db.update_job_status(
                job_id, 
                JobStatus.COMPLETED,
                output_dir=str(output_dir),
            )
            
            if progress_callback:
                progress_callback("job_complete", {
                    "job_id": job_id,
                    "status": "completed",
                    "output_path": str(output_dir),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            
            if progress_callback:
                progress_callback("job_error", {
                    "job_id": job_id,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.utcnow().isoformat(),
                })
    
    async def _run_pipeline_async(self, runner, context, stage_callback):
        """
        Run the pipeline (wrapper for async execution).
        
        The actual pipeline is synchronous, so we run it directly.
        In the future, this could use asyncio.to_thread for true async.
        """
        import asyncio
        
        # Run synchronous pipeline in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: runner.run(context),
        )
    
    async def execute_job_from_checkpoint(
        self,
        job_id: str,
        original_job_id: str,
        start_stage: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        Execute a job starting from a checkpoint.
        
        Uses the edited transcript from the original job as input.
        """
        job = self.db.get_job(job_id)
        original_job = self.db.get_job(original_job_id)
        
        if not job or not original_job:
            return
        
        # Mark job as running
        self.db.update_job_status(job_id, JobStatus.RUNNING)
        
        if progress_callback:
            progress_callback("stage_start", {
                "job_id": job_id,
                "stage": "initialization",
                "message": f"Re-running from checkpoint (start: {start_stage or 'auto'})...",
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        output_dir = self.config.output_dir / job_id
        
        try:
            from local_transcribe.framework.pipeline_reentry import run_pipeline_from_checkpoint
            from local_transcribe.lib.program_logger import setup_output_context
            
            # Find checkpoint file from original job
            original_output = Path(original_job.output_dir) if original_job.output_dir else self.config.output_dir / original_job_id
            
            # Look for turns.json or similar checkpoint file
            checkpoint_path = None
            for pattern in ["*turns.json", "*turns-de-identified.json", "*turns-named.json"]:
                matches = list(original_output.glob(pattern))
                if matches:
                    checkpoint_path = matches[0]
                    break
            
            if not checkpoint_path:
                raise FileNotFoundError(f"No checkpoint file found in {original_output}")
            
            # Load checkpoint and apply edits
            import json as json_module
            from web_api.services.edit_applicator import apply_edits_to_transcript
            
            with open(checkpoint_path, 'r') as f:
                transcript_data = json_module.load(f)
            
            # Get edits for the original job and apply them
            edits = self.db.get_edits_for_job(original_job_id)
            if edits:
                transcript_data = apply_edits_to_transcript(transcript_data, edits)
            
            # Write edited checkpoint to new job's output dir
            edited_checkpoint_path = output_dir / "checkpoint-edited.json"
            with open(edited_checkpoint_path, 'w') as f:
                json_module.dump(transcript_data, f, indent=2)
            
            # Setup logging
            setup_output_context(
                output_dir=output_dir,
                log_level="INFO",
                console_output=False,
            )
            
            # Parse original config for options
            original_config = json.loads(original_job.config_json) if original_job.config_json else {}
            options = original_config.get("options", {})
            
            # Run from checkpoint (using edited checkpoint)
            run_pipeline_from_checkpoint(
                checkpoint_path=edited_checkpoint_path,
                output_dir=output_dir,
                start_stage=start_stage,
                enable_de_identification=options.get("enable_de_identification", True),
                enable_cleanup=options.get("enable_cleanup", False),
            )
            
            # Mark complete
            self.db.update_job_status(
                job_id,
                JobStatus.COMPLETED,
                output_dir=str(output_dir),
            )
            
            if progress_callback:
                progress_callback("job_complete", {
                    "job_id": job_id,
                    "status": "completed",
                    "output_path": str(output_dir),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            
            if progress_callback:
                progress_callback("job_error", {
                    "job_id": job_id,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.utcnow().isoformat(),
                })


# Global service instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get the global pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
