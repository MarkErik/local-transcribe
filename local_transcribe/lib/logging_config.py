#!/usr/bin/env python3
from __future__ import annotations
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import json


class TranscriptionError(Exception):
    """Base exception for transcription-related errors."""
    def __init__(self, message: str, stage: str = "unknown", cause: Optional[Exception] = None):
        super().__init__(message)
        self.stage = stage
        self.cause = cause
        self.timestamp = datetime.now()


class AudioProcessingError(TranscriptionError):
    """Exception for audio processing errors."""
    def __init__(self, message: str, audio_path: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message, stage="audio_processing", cause=cause)
        self.audio_path = audio_path


class TranscriptionError(TranscriptionError):
    """Exception for transcription-related errors."""

    def __init__(self, message: str, stage: str = "transcription", cause: Optional[Exception] = None, model: Optional[str] = None):
        super().__init__(message, stage=stage, cause=cause)
        self.model = model


class DiarizationError(TranscriptionError):
    """Exception for diarization errors."""
    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, stage="diarization", cause=cause)


class OutputError(TranscriptionError):
    """Exception for output generation errors."""
    def __init__(self, message: str, output_path: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message, stage="output", cause=cause)
        self.output_path = output_path


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the message
        formatted = super().format(record)
        
        # Add color
        return f"{color}{formatted}{reset}"


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_output: bool = False
) -> logging.Logger:
    """
    Set up comprehensive logging for the transcription pipeline.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : Optional[str]
        Path to log file. If None, no file logging.
    console_output : bool
        Whether to output to console
    structured_output : bool
        Whether to use structured JSON formatting
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("local_transcribe")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        
        if structured_output:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        
        if structured_output:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_exception(
    logger: logging.Logger,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an exception with context information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    exception : Exception
        Exception to log
    context : Optional[Dict[str, Any]]
        Additional context information
    """
    # Create log record with extra context
    extra = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
    }
    
    # Add stage information for custom exceptions
    if isinstance(exception, TranscriptionError):
        extra["stage"] = exception.stage
        extra["timestamp"] = exception.timestamp.isoformat()
        
        if isinstance(exception, AudioProcessingError) and exception.audio_path:
            extra["audio_path"] = exception.audio_path
        elif isinstance(exception, TranscriptionError) and exception.model:
            extra["model"] = exception.model
        elif isinstance(exception, OutputError) and exception.output_path:
            extra["output_path"] = exception.output_path
    
    # Add custom context
    if context:
        extra.update(context)
    
    logger.error(
        f"Exception in {extra.get('stage', 'unknown')}: {exception}",
        exc_info=True,
        extra=extra
    )


def handle_error(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> None:
    """
    Handle an error by logging it and optionally re-raising.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    error : Exception
        Error to handle
    context : Optional[Dict[str, Any]]
        Additional context information
    reraise : bool
        Whether to re-raise the error after logging
    """
    log_exception(logger, error, context)
    
    if reraise:
        raise error


# Global logger instance
_global_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get or create the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger


def configure_global_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_output: bool = False
) -> None:
    """Configure the global logger instance."""
    global _global_logger
    _global_logger = setup_logging(log_level, log_file, console_output, structured_output)


# Context manager for error handling
class ErrorContext:
    """Context manager for automatic error logging."""
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ):
        self.logger = logger or get_logger()
        self.context = context or {}
        self.reraise = reraise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            handle_error(self.logger, exc_val, self.context, self.reraise)
        return not self.reraise  # Suppress exception if not re-raising


def error_context(
    logger: Optional[logging.Logger] = None,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
):
    """Decorator for adding error handling to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ErrorContext(logger, context, reraise):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class OutputContext:
    """Global context for controlling output verbosity and logging."""
    
    def __init__(self, log_level: str = "WARNING"):
        self.log_level = log_level.upper()
        self.level_value = getattr(logging, self.log_level)
        self.logger = get_logger()
    
    def should_log(self, level: str) -> bool:
        """Check if a message at the given level should be logged."""
        return getattr(logging, level.upper()) >= self.level_value
    
    def log_status(self, message: str, level: str = "INFO") -> None:
        """Log a status message if the level is enabled."""
        if self.should_log(level):
            self.logger.log(getattr(logging, level.upper()), f"[*] {message}")
    
    def log_progress(self, message: str, details: Optional[str] = None) -> None:
        """Log progress information with optional details."""
        if self.should_log("INFO"):
            self.logger.info(f"[i] {message}")
            if details and self.should_log("DEBUG"):
                self.logger.debug(f"    {details}")
    
    def log_intermediate_save(self, path: str, description: str) -> None:
        """Log intermediate file saves."""
        if self.should_log("INFO"):
            self.logger.info(f"[+] {description}: {path}")
    
    def log_completion(self, message: str, stats: Optional[Dict[str, Any]] = None) -> None:
        """Log completion messages with optional statistics."""
        if self.should_log("INFO"):
            self.logger.info(f"[✓] {message}")
            if stats and self.should_log("DEBUG"):
                for key, value in stats.items():
                    self.logger.debug(f"    {key}: {value}")


# Global output context
_global_output_context: Optional[OutputContext] = None


def get_output_context() -> OutputContext:
    """Get or create the global output context."""
    global _global_output_context
    if _global_output_context is None:
        _global_output_context = OutputContext()
    return _global_output_context


def configure_global_logging(
    log_level: str = "WARNING",
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_output: bool = False
) -> None:
    """Configure the global logger and output context."""
    global _global_logger, _global_output_context
    _global_logger = setup_logging(log_level, log_file, console_output, structured_output)
    _global_output_context = OutputContext(log_level)


# Convenience functions for global output control
def log_status(message: str, level: str = "INFO") -> None:
    """Log a status message globally."""
    get_output_context().log_status(message, level)


def log_progress(message: str, details: Optional[str] = None) -> None:
    """Log progress information globally."""
    get_output_context().log_progress(message, details)


def log_intermediate_save(path: str, description: str) -> None:
    """Log intermediate file save globally."""
    get_output_context().log_intermediate_save(path, description)


def log_completion(message: str, stats: Optional[Dict[str, Any]] = None) -> None:
    """Log completion message globally."""
    get_output_context().log_completion(message, stats)
