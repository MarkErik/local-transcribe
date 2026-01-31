"""
Pipeline service for executing transcription pipelines.

Wraps the existing pipeline infrastructure with progress callbacks
for the web interface.
"""

import json
import traceback
import argparse
from pathlib import Path
from typing import Optional, Callable, Dict, Any
from datetime import datetime, timezone

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        output_dir = self.config.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Import pipeline components - use the actual module structure
            from local_transcribe.framework.pipeline_context import PipelineContext
            from local_transcribe.framework.stages import PipelineExecutor, create_pipeline_for_mode
            from local_transcribe.framework.plugin_manager import import_pipeline_modules
            from local_transcribe.lib.program_logger import configure_global_logging
            from local_transcribe.lib.create_directories import ensure_session_dirs
            from local_transcribe.lib.environment import repo_root_from_here, set_offline_env, ensure_models_exist, get_available_system_capabilities, validate_system_capability
            from local_transcribe.lib.system_capability_utils import set_system_capability
            from local_transcribe.framework.provider_setup import ProviderSetup
            
            # Get the repo root and set up the environment
            root = repo_root_from_here()
            
            # Set up offline environment for model loading (sets XDG_CACHE_HOME for HuggingFace)
            models_dir = root / ".models"
            set_offline_env(models_dir)
            ensure_models_exist(models_dir)
            
            # Set system capability with auto-detection (MPS > CUDA > CPU preference)
            # This matches the main application's behavior
            requested_system = options.get("system")
            if requested_system:
                # User explicitly requested a specific device
                system_capability = validate_system_capability(requested_system)
            else:
                # Auto-detect best available device (preference: MPS > CUDA > CPU)
                available = get_available_system_capabilities()
                if "mps" in available:
                    system_capability = "mps"
                elif "cuda" in available:
                    system_capability = "cuda"
                else:
                    system_capability = "cpu"
            set_system_capability(system_capability)
            
            # Import pipeline modules to get API and registry
            api = import_pipeline_modules(root)
            
            # Configure logging
            configure_global_logging(log_level="INFO")
            
            # Build args namespace for the pipeline
            args = self._build_args_for_pipeline(
                mode=mode,
                options=options,
                interviewer_file=interviewer_file,
                participant_file=participant_file,
                output_dir=output_dir,
            )
            
            # Determine speaker_files mapping based on mode
            speaker_files = {}
            if mode in ("vad_split_audio", "split_audio"):
                if interviewer_file:
                    speaker_files["Interviewer"] = interviewer_file
                if participant_file:
                    speaker_files["Participant"] = participant_file
            elif mode == "combined_audio":
                # Combined audio uses a single file
                speaker_files["combined_audio"] = interviewer_file or participant_file
            
            # Setup providers using the registry
            registry = api.get("registry")
            if registry is None:
                raise ValueError("Registry not found in api")
            
            provider_setup = ProviderSetup(registry, args)
            # For vad_split_audio, use split_audio provider setup
            provider_mode = "split_audio" if mode == "vad_split_audio" else mode
            providers = provider_setup.setup_providers(provider_mode)
            
            # Setup output directories
            capabilities = {
                "mode": mode,
                "has_builtin_alignment": providers.get('transcriber', {}).has_builtin_alignment if providers.get('transcriber') else False,
                "aligner": providers.get('aligner') is not None,
                "diarization": providers.get('diarization') is not None,
            }
            
            ensure_session_dirs_func = api.get("ensure_session_dirs")
            if ensure_session_dirs_func is None:
                raise ValueError("ensure_session_dirs not found in api")
            
            paths = ensure_session_dirs_func(output_dir, mode, speaker_files, capabilities)
            
            # Create pipeline context with all required fields
            # Web mode: skip file outputs - data is stored in database instead
            context = PipelineContext(
                args=args,
                api=api,
                root=root,
                paths=paths,
                mode=mode,
                speaker_files=speaker_files,
                transcriber_provider=providers.get('transcriber'),
                aligner_provider=providers.get('aligner'),
                diarization_provider=providers.get('diarization'),
                transcript_cleanup_provider=providers.get('transcript_cleanup'),
                models_dir=root / ".models",
                dry_run=False,
                skip_file_outputs=True,  # Web mode: store data in database, not files
            )
            
            # Create and execute pipeline
            stages = create_pipeline_for_mode(mode)
            executor = PipelineExecutor(stages)
            
            # Execute pipeline in thread pool with progress callbacks
            result, final_context = await self._run_pipeline_async(executor, context, progress_callback)
            
            if result.success:
                # Mark complete
                self.db.update_job_status(
                    job_id, 
                    JobStatus.COMPLETED,
                    output_dir=str(output_dir),
                )
                
                # Store transcript data in database for web access
                try:
                    from web_api.services.transcript_storage import TranscriptStorageService, STAGE_BASE
                    transcript_storage = TranscriptStorageService(self.db)
                    
                    # In web mode (skip_file_outputs=True), store directly from context
                    if final_context.transcript is not None:
                        stored = transcript_storage.store_from_context(
                            job_id, 
                            final_context.transcript,
                            stage=STAGE_BASE
                        )
                        if stored:
                            if progress_callback:
                                progress_callback("transcript_stored", {
                                    "job_id": job_id,
                                    "stages": [STAGE_BASE],
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                })
                        else:
                            # Log warning - transcript storage failed
                            import logging
                            logging.getLogger(__name__).warning(
                                f"Failed to store transcript in database for job {job_id}"
                            )
                    else:
                        # No transcript in context - this is an error for web mode
                        import logging
                        logging.getLogger(__name__).error(
                            f"No transcript data in context for job {job_id}"
                        )
                except Exception as e:
                    # Log error - database storage is required for web mode
                    import logging
                    logging.getLogger(__name__).error(
                        f"Failed to store transcript in database for job {job_id}: {e}"
                    )
                
                if progress_callback:
                    progress_callback("job_complete", {
                        "job_id": job_id,
                        "status": "completed",
                        "output_path": str(output_dir),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                error_msg = f"Pipeline failed at stage: {result.failed_stage}"
                if result.error:
                    error_msg += f" - {result.error}"
                self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
                
                if progress_callback:
                    progress_callback("job_error", {
                        "job_id": job_id,
                        "error": error_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            
            if progress_callback:
                progress_callback("job_error", {
                    "job_id": job_id,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
    
    def _build_args_for_pipeline(
        self,
        mode: str,
        options: Dict[str, Any],
        interviewer_file: Optional[str],
        participant_file: Optional[str],
        output_dir: Path,
    ) -> argparse.Namespace:
        """
        Build an argparse.Namespace object for the pipeline.
        
        This creates the args object expected by the pipeline with all
        necessary configuration options.
        """
        args = argparse.Namespace()
        
        # Core settings
        args.outdir = str(output_dir)
        args.log_level = options.get("log_level", "INFO")
        args.system = options.get("system", "cpu")
        
        # Audio files based on mode
        audio_files = []
        if interviewer_file:
            audio_files.append(interviewer_file)
        if participant_file:
            audio_files.append(participant_file)
        args.audio_files = audio_files
        
        # Mode settings
        args.single_speaker_audio = mode == "single_speaker_audio"
        args.vad_pipeline = mode == "vad_split_audio"
        
        # Provider settings
        args.transcriber_provider = options.get("transcriber_provider", "granite")
        args.transcriber_model = options.get("transcriber_model", "granite-8b")
        args.aligner_provider = options.get("aligner_provider")
        args.diarization_provider = options.get("diarization_provider")
        
        # Remote URLs
        args.remote_transcriber_url = options.get("remote_transcriber_url", "http://0.0.0.0:7070")
        args.llm_de_identifier_url = options.get("llm_de_identifier_url", "http://0.0.0.0:8080")
        args.llm_transcript_cleanup_url = options.get("llm_transcript_cleanup_url", "http://0.0.0.0:8080")
        
        # Processing options
        args.de_identify = options.get("enable_de_identification", True)
        args.enable_cleanup = options.get("enable_cleanup", False)
        
        # Set transcript_cleanup_provider when enable_cleanup is True
        # Default to 'llm_transcript_cleanup' if not explicitly specified
        args.transcript_cleanup_provider = options.get("transcript_cleanup_provider")
        if args.enable_cleanup and not args.transcript_cleanup_provider:
            args.transcript_cleanup_provider = "llm_transcript_cleanup"
        args.num_speakers = options.get("num_speakers", 2)
        
        # Output settings
        args.selected_outputs = options.get("output_formats", ["turns-json", "timestamped-txt"])
        args.only_final_transcript = False
        
        # Set this to avoid interactive prompts
        args.interactive = False
        
        # Dry run flag
        args.dry_run = False
        
        return args
    
    async def _run_pipeline_async(self, executor, context, progress_callback=None):
        """
        Run the pipeline (wrapper for async execution).
        
        The actual pipeline is synchronous, so we run it in a thread pool
        to avoid blocking the event loop.
        
        This method wraps the executor to send progress events via the callback.
        
        Returns:
            Tuple of (PipelineResult, updated_context)
        """
        import asyncio
        from datetime import datetime, timezone
        
        def execute_with_progress():
            """Execute the pipeline with progress events."""
            from local_transcribe.framework.stages.base import StageStatus
            from local_transcribe.framework.stages.executor import PipelineResult
            
            result = PipelineResult(success=True)
            current_context = context  # Track the updated context
            
            for stage in executor.stages:
                # Skip stages that don't apply to this mode
                if not stage.applies_to_mode(current_context.mode):
                    result.skipped_stages.append(stage.name)
                    continue
                
                # Notify stage start
                if progress_callback:
                    progress_callback("stage_start", {
                        "stage": stage.name,
                        "message": f"Running stage: {stage.name}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                
                # Execute the stage with progress callback
                updated_context, stage_result = stage.execute_safe(
                    current_context,
                    progress_callback=progress_callback
                )
                result.stage_results.append(stage_result)
                
                # Update context for next stage
                current_context = updated_context
                
                if stage_result.status == StageStatus.COMPLETED:
                    result.completed_stages.append(stage.name)
                    # Notify stage complete
                    if progress_callback:
                        progress_callback("stage_complete", {
                            "stage": stage.name,
                            "message": f"Stage completed: {stage.name}",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                elif stage_result.status == StageStatus.SKIPPED:
                    result.skipped_stages.append(stage.name)
                elif stage_result.status == StageStatus.FAILED:
                    result.success = False
                    result.failed_stage = stage.name
                    result.error = stage_result.error
                    break
            
            return result, current_context
        
        # Run synchronous pipeline in thread pool
        loop = asyncio.get_running_loop()
        result, final_context = await loop.run_in_executor(
            None,
            execute_with_progress,
        )
        return result, final_context
    
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        output_dir = self.config.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from local_transcribe.framework.pipeline_reentry import run_pipeline_from_checkpoint
            from local_transcribe.framework.plugin_manager import import_pipeline_modules
            from local_transcribe.lib.program_logger import configure_global_logging
            from local_transcribe.lib.environment import repo_root_from_here, set_offline_env, ensure_models_exist, get_available_system_capabilities, validate_system_capability
            from local_transcribe.lib.system_capability_utils import set_system_capability
            
            # Find checkpoint file from original job
            original_output = Path(original_job.output_dir) if original_job.output_dir else self.config.output_dir / original_job_id
            
            # Look for turns.json or similar checkpoint file
            checkpoint_file = None
            for pattern in ["*turns.json", "*turns-de-identified.json", "*turns-named.json"]:
                matches = list(original_output.glob(pattern))
                if matches:
                    checkpoint_file = matches[0]
                    break
            
            if not checkpoint_file:
                raise FileNotFoundError(f"No checkpoint file found in {original_output}")
            
            # Load checkpoint and apply edits
            import json as json_module
            from web_api.services.edit_applicator import apply_edits_to_transcript
            
            with open(checkpoint_file, 'r') as f:
                transcript_data = json_module.load(f)
            
            # Get edits for the original job and apply them
            edits = self.db.get_edits_for_job(original_job_id)
            if edits:
                transcript_data = apply_edits_to_transcript(transcript_data, edits)
            
            # Write edited checkpoint to new job's output dir
            edited_checkpoint_path = output_dir / "checkpoint-edited.json"
            with open(edited_checkpoint_path, 'w') as f:
                json_module.dump(transcript_data, f, indent=2)
            
            # Get the repo root and set up the environment
            root = repo_root_from_here()
            
            # Set up offline environment for model loading (sets XDG_CACHE_HOME for HuggingFace)
            models_dir = root / ".models"
            set_offline_env(models_dir)
            ensure_models_exist(models_dir)
            
            # Parse original config for options
            original_config = json.loads(original_job.config_json) if original_job.config_json else {}
            options = original_config.get("options", {})
            
            # Set system capability with auto-detection (MPS > CUDA > CPU preference)
            requested_system = options.get("system")
            if requested_system:
                system_capability = validate_system_capability(requested_system)
            else:
                available = get_available_system_capabilities()
                if "mps" in available:
                    system_capability = "mps"
                elif "cuda" in available:
                    system_capability = "cuda"
                else:
                    system_capability = "cpu"
            set_system_capability(system_capability)
            
            # Import pipeline modules to get API and registry
            api = import_pipeline_modules(root)
            
            # Configure logging
            configure_global_logging(log_level="INFO")
            
            # Build args for the checkpoint reentry
            args = argparse.Namespace()
            args.from_diarized_json = str(edited_checkpoint_path)
            args.outdir = str(output_dir)
            args.log_level = "INFO"
            args.system = system_capability
            args.de_identify = options.get("enable_de_identification", True)
            args.enable_cleanup = options.get("enable_cleanup", False)
            args.selected_outputs = options.get("output_formats", ["turns-json", "timestamped-txt"])
            args.interactive = False
            args.dry_run = False
            args.mode = original_config.get("mode")
            args.speaker_map = None
            args.audio_for_video = None
            args.llm_de_identifier_url = options.get("llm_de_identifier_url", "http://0.0.0.0:8080")
            args.llm_transcript_cleanup_url = options.get("llm_transcript_cleanup_url", "http://0.0.0.0:8080")
            
            # Set transcript_cleanup_provider when enable_cleanup is True
            # Default to 'llm_transcript_cleanup' if not explicitly specified
            args.transcript_cleanup_provider = options.get("transcript_cleanup_provider")
            if args.enable_cleanup and not args.transcript_cleanup_provider:
                args.transcript_cleanup_provider = "llm_transcript_cleanup"
            
            # Run from checkpoint
            import asyncio
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_pipeline_from_checkpoint(args, api, root),
            )
            
            if result == 0:
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
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                error_msg = f"Pipeline failed with exit code: {result}"
                self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
                
                if progress_callback:
                    progress_callback("job_error", {
                        "job_id": job_id,
                        "error": error_msg,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.db.update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
            
            if progress_callback:
                progress_callback("job_error", {
                    "job_id": job_id,
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })


# Global service instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get the global pipeline service instance."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service
