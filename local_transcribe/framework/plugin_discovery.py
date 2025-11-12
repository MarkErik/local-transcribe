#!/usr/bin/env python3
"""
Plugin discovery and loading system for external plugins.
"""

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional
import logging


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
            for py_file in plugin_dir.rglob("*.py"):
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