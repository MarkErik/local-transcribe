#!/usr/bin/env python3
"""
Plugin discovery and loading system for external plugins.
"""

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional
import logging

from .plugins import registry

logger = logging.getLogger(__name__)


class PluginLoader:
    """Handles discovery and loading of external plugins."""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        """
        Initialize plugin loader.

        Args:
            plugin_dirs: List of directories to search for plugins.
                        Defaults to [~/.local-transcribe/plugins, ./plugins]
        """
        if plugin_dirs is None:
            plugin_dirs = [
                Path.home() / ".local-transcribe" / "plugins",
                Path.cwd() / "plugins",
                Path(__file__).parent.parent / "providers"
            ]
        self.plugin_dirs = [Path(d).expanduser().resolve() for d in plugin_dirs]

    def discover_plugins(self) -> List[Path]:
        """
        Discover plugin files in configured directories.

        Returns:
            List of plugin file paths (Python files)
        """
        plugin_files = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                logger.debug(f"Plugin directory {plugin_dir} does not exist, skipping")
                continue

            # Find all .py files in the directory
            for py_file in plugin_dir.glob("*.py"):
                if py_file.is_file():
                    plugin_files.append(py_file)

        return plugin_files

    def load_plugin(self, plugin_path: Path) -> bool:
        """
        Load a single plugin file.

        Args:
            plugin_path: Path to the plugin Python file

        Returns:
            True if plugin loaded successfully, False otherwise
        """
        try:
            # Create a module name from the file path
            module_name = f"plugin_{plugin_path.stem}"

            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not create spec for plugin {plugin_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            logger.info(f"Loaded plugin: {plugin_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def load_all_plugins(self) -> int:
        """
        Load all discovered plugins.

        Returns:
            Number of plugins successfully loaded
        """
        plugin_files = self.discover_plugins()
        loaded_count = 0

        for plugin_file in plugin_files:
            if self.load_plugin(plugin_file):
                loaded_count += 1

        logger.info(f"Loaded {loaded_count} out of {len(plugin_files)} plugins")
        return loaded_count


def create_plugin_template(output_path: Path, plugin_type: str) -> None:
    """
    Create a template plugin file for the specified type.

    Args:
        output_path: Where to write the template file
        plugin_type: Type of plugin ('asr', 'diarization', or 'output')
    """
    templates = {
        'asr': '''#!/usr/bin/env python3
"""
Example ASR Plugin Template
"""

from local_transcribe.core import ASRProvider, WordSegment, registry

class ExampleASRProvider(ASRProvider):
    """Example ASR provider implementation."""

    @property
    def name(self) -> str:
        return "example-asr"

    @property
    def description(self) -> str:
        return "Example ASR provider (replace with your implementation)"

    def transcribe_with_alignment(
        self,
        audio_path: str,
        role: Optional[str] = None,
        **kwargs
    ) -> List[WordSegment]:
        """
        Implement your ASR transcription logic here.

        This should return word-level segments with accurate timestamps.
        """
        # TODO: Implement actual transcription
        return [
            WordSegment(
                text="example",
                start=0.0,
                end=1.0,
                speaker=role
            )
        ]

# Register the plugin
registry.register_asr_provider(ExampleASRProvider())
''',

        'diarization': '''#!/usr/bin/env python3
"""
Example Diarization Plugin Template
"""

from local_transcribe.core import DiarizationProvider, WordSegment, Turn, registry

class ExampleDiarizationProvider(DiarizationProvider):
    """Example diarization provider implementation."""

    @property
    def name(self) -> str:
        return "example-diarization"

    @property
    def description(self) -> str:
        return "Example diarization provider (replace with your implementation)"

    def diarize(
        self,
        audio_path: str,
        words: List[WordSegment],
        **kwargs
    ) -> List[Turn]:
        """
        Implement your speaker diarization logic here.

        This should assign speakers to word segments and group them into turns.
        """
        # TODO: Implement actual diarization
        return [
            Turn(
                speaker="SPEAKER_00",
                start=0.0,
                end=1.0,
                text="example text"
            )
        ]

# Register the plugin
registry.register_diarization_provider(ExampleDiarizationProvider())
''',

        'output': '''#!/usr/bin/env python3
"""
Example Output Writer Plugin Template
"""

from local_transcribe.core import OutputWriter, Turn, registry

class ExampleOutputWriter(OutputWriter):
    """Example output writer implementation."""

    @property
    def name(self) -> str:
        return "example-writer"

    @property
    def description(self) -> str:
        return "Example output writer (replace with your implementation)"

    @property
    def supported_formats(self) -> List[str]:
        return [".example"]

    def write(
        self,
        turns: List[Turn],
        output_path: str,
        **kwargs
    ) -> None:
        """
        Implement your output writing logic here.

        This should write the conversation turns to the specified format.
        """
        # TODO: Implement actual writing logic
        with open(output_path, 'w') as f:
            for turn in turns:
                f.write(f"{turn.speaker}: {turn.text}\\n")

# Register the plugin
registry.register_output_writer(ExampleOutputWriter())
'''
    }

    if plugin_type not in templates:
        raise ValueError(f"Unknown plugin type: {plugin_type}. Must be 'asr', 'diarization', or 'output'")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(templates[plugin_type])

    print(f"Created plugin template: {output_path}")