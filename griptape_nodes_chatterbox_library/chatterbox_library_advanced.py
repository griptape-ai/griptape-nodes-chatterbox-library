"""Advanced library hooks for Chatterbox TTS library initialization."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes_chatterbox_library")


class ChatterboxLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library hooks for Chatterbox TTS library.

    Checks out the Chatterbox git submodule, which carries the model code the nodes import while
    they execute, and installs it into the execution environment. Its dependencies are declared in
    the manifest and installed by the engine, so the submodule itself goes in with --no-deps.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Check out the Chatterbox submodule and install it for the nodes to import at execution time."""
        logger.info("Loading Chatterbox TTS library: %s", library_data.name)
        # The submodule checkout and install below populate the execution environment, which only
        # the worker imports; the orchestrator has no use for them and must not run them.
        if not GriptapeNodes.LibraryManager().is_worker:
            return
        if self._init_chatterbox_submodule():
            self._install_chatterbox_package()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Log completion of library loading."""
        logger.info(
            "Chatterbox TTS library loaded: %d nodes registered",
            len(library.get_registered_nodes()),
        )

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        """Get the Python executable of the library's execution environment."""
        venv_path = self._get_library_root() / ".venv-exec"

        if GriptapeNodes.OSManager().is_windows():
            return venv_path / "Scripts" / "python.exe"
        return venv_path / "bin" / "python"

    def _init_chatterbox_submodule(self) -> bool:
        """Initialize the Chatterbox git submodule, and return whether its sources are present.

        A failure here only costs execution: the nodes still load and can be edited, and the node
        reports the missing sources when someone runs it. Raising would drop the whole library and
        leave placeholders in every workflow that uses it.
        """
        library_root = self._get_library_root()
        chatterbox_dir = library_root / "chatterbox"

        # Check if submodule is already initialized
        if chatterbox_dir.exists() and any(chatterbox_dir.iterdir()):
            logger.info("Chatterbox submodule already initialized")
            return True

        logger.info("Initializing Chatterbox submodule...")
        # The git CLI rather than pygit2: the engine dropped pygit2 (its bundled TLS trust
        # store breaks on some platforms) and requires git on PATH, so it is the one tool
        # guaranteed to be here.
        git_repo_root = library_root.parent
        try:
            subprocess.check_call(["git", "-C", str(git_repo_root), "submodule", "update", "--init", "--recursive"])
        except (subprocess.CalledProcessError, OSError) as e:
            logger.warning(
                "Chatterbox model code is unavailable: checking out the submodule in %s failed (%s). "
                "Nodes will load but cannot generate speech until "
                "'git submodule update --init --recursive' succeeds there.",
                git_repo_root,
                e,
            )
            return False

        if not chatterbox_dir.exists() or not any(chatterbox_dir.iterdir()):
            logger.warning(
                "Chatterbox model code is unavailable: %s is still empty after checking out the "
                "submodule. Nodes will load but cannot generate speech.",
                chatterbox_dir,
            )
            return False

        logger.info("Chatterbox submodule initialized successfully")
        return True

    def _install_chatterbox_package(self) -> None:
        """Install the submodule into the execution environment as the chatterbox-tts distribution.

        Being importable is not enough: chatterbox reads its own version from the distribution's
        metadata when it is imported, so without an install every node fails with "No package
        metadata was found for chatterbox-tts". A failure here only costs execution, for the same
        reason as the submodule checkout.
        """
        venv_python = self._get_venv_python_path()
        installed = subprocess.run(
            [str(venv_python), "-c", "import importlib.metadata; importlib.metadata.version('chatterbox-tts')"],
            capture_output=True,
        )
        if installed.returncode == 0:
            logger.info("chatterbox-tts already installed in the execution environment")
            return

        chatterbox_dir = self._get_library_root() / "chatterbox"
        logger.info("Installing Chatterbox from submodule into the execution environment...")
        try:
            # The engine builds the execution environment with uv, which seeds no pip.
            if subprocess.run([str(venv_python), "-m", "pip", "--version"], capture_output=True).returncode != 0:
                subprocess.run(
                    [str(venv_python), "-m", "ensurepip", "--upgrade"], check=True, capture_output=True, text=True
                )
            subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--no-deps", str(chatterbox_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, OSError) as e:
            output = getattr(e, "stderr", None) or ""
            logger.warning(
                "Chatterbox model code is unavailable: installing %s into %s failed (%s). "
                "Nodes will load but cannot generate speech.\n%s",
                chatterbox_dir,
                venv_python,
                e,
                output,
            )
            return

        logger.info("Chatterbox installed successfully")
