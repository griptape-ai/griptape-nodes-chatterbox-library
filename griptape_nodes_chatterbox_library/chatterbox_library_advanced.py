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
    they execute. Every package the library needs is declared in the manifest and installed by the
    engine, so nothing is installed here.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Check out the Chatterbox submodule the nodes import at execution time."""
        logger.info("Loading Chatterbox TTS library: %s", library_data.name)
        # The submodule checkout below populates the execution environment, which only the
        # worker imports; the orchestrator has no use for it and must not run it.
        if not GriptapeNodes.LibraryManager().is_worker:
            return
        self._init_chatterbox_submodule()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Log completion of library loading."""
        logger.info(
            "Chatterbox TTS library loaded: %d nodes registered",
            len(library.get_registered_nodes()),
        )

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _init_chatterbox_submodule(self) -> None:
        """Initialize the Chatterbox git submodule.

        A failure here only costs execution: the nodes still load and can be edited, and the node
        reports the missing sources when someone runs it. Raising would drop the whole library and
        leave placeholders in every workflow that uses it.
        """
        library_root = self._get_library_root()
        chatterbox_dir = library_root / "chatterbox"

        # Check if submodule is already initialized
        if chatterbox_dir.exists() and any(chatterbox_dir.iterdir()):
            logger.info("Chatterbox submodule already initialized")
            return

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
            return

        if not chatterbox_dir.exists() or not any(chatterbox_dir.iterdir()):
            logger.warning(
                "Chatterbox model code is unavailable: %s is still empty after checking out the "
                "submodule. Nodes will load but cannot generate speech.",
                chatterbox_dir,
            )
            return

        logger.info("Chatterbox submodule initialized successfully")
