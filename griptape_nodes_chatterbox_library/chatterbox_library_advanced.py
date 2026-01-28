"""Advanced library hooks for Chatterbox TTS library initialization."""

from __future__ import annotations

import logging
import re
import subprocess
import tomllib
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("griptape_nodes_chatterbox_library")


class ChatterboxLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library hooks for Chatterbox TTS library.

    Handles git submodule initialization and dependency installation before nodes are loaded.
    """

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Initialize the Chatterbox git submodule and install dependencies."""
        logger.info("Loading Chatterbox TTS library: %s", library_data.name)

        # Check if dependencies are already installed
        if self._check_dependencies_installed():
            logger.info("Chatterbox already installed, skipping installation")
            return

        logger.info("Chatterbox dependencies not found, beginning installation...")
        self._install_chatterbox_dependencies()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Log completion of library loading and configure PyTorch."""
        logger.info(
            "Chatterbox TTS library loaded: %d nodes registered",
            len(library.get_registered_nodes()),
        )
        self._configure_pytorch_settings()

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _get_venv_python_path(self) -> Path:
        """Get the Python executable path from the library's venv."""
        venv_path = self._get_library_root() / ".venv"

        if GriptapeNodes.OSManager().is_windows():
            venv_python_path = venv_path / "Scripts" / "python.exe"
        else:
            venv_python_path = venv_path / "bin" / "python"

        if not venv_python_path.exists():
            raise RuntimeError(
                f"Library venv Python not found at {venv_python_path}. "
                "The library venv must be initialized before loading."
            )

        logger.debug("Python executable found at: %s", venv_python_path)
        return venv_python_path

    def _configure_pytorch_settings(self) -> None:
        """Configure PyTorch TF32 settings for Ampere+ GPUs."""
        try:
            import torch

            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.debug("PyTorch TF32 settings enabled for GPU acceleration")
        except ImportError:
            logger.warning("PyTorch not available, skipping TF32 configuration")

    def _check_dependencies_installed(self) -> bool:
        """Check if Chatterbox is installed by trying to import it."""
        venv_python = self._get_venv_python_path()

        # Check if chatterbox can be imported in the library venv
        result = subprocess.run(
            [str(venv_python), "-c", "import chatterbox"],
            capture_output=True,
        )

        if result.returncode == 0:
            logger.info("chatterbox already installed in library venv")
            return True

        logger.debug("chatterbox not importable in library venv")
        return False

    def _ensure_pip_installed(self) -> None:
        """Ensure pip is installed in the library's venv."""
        python_path = self._get_venv_python_path()

        result = subprocess.run(
            [str(python_path), "-m", "pip", "--version"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            logger.debug("pip already installed: %s", result.stdout.strip())
            return

        logger.info("pip not found in venv, installing with ensurepip...")
        subprocess.run(
            [str(python_path), "-m", "ensurepip", "--upgrade"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("pip installed successfully")

    def _run_pip_install(self, args: list[str]) -> None:
        """Run pip install with the given arguments."""
        python_path = self._get_venv_python_path()
        cmd = [str(python_path), "-m", "pip", "install", *args]
        logger.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            if result.stdout:
                logger.debug(result.stdout)
            if result.stderr:
                logger.debug(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error("pip install failed with exit code %d", e.returncode)
            if e.stdout:
                logger.error("stdout: %s", e.stdout)
            if e.stderr:
                logger.error("stderr: %s", e.stderr)
            raise

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules using pygit2."""
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        # Recursively update nested submodules
        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_chatterbox_submodule(self) -> Path:
        """Initialize the Chatterbox git submodule."""
        library_root = self._get_library_root()
        chatterbox_dir = library_root / "chatterbox"

        # Check if submodule is already initialized
        if chatterbox_dir.exists() and any(chatterbox_dir.iterdir()):
            logger.info("Chatterbox submodule already initialized")
            return chatterbox_dir

        logger.info("Initializing Chatterbox submodule...")
        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not chatterbox_dir.exists() or not any(chatterbox_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {chatterbox_dir} is empty or does not exist"
            )

        logger.info("Chatterbox submodule initialized successfully")
        return chatterbox_dir

    def _get_filtered_dependencies(self, chatterbox_dir: Path) -> list[str]:
        """Parse pyproject.toml and return dependencies, excluding numpy/torch/torchaudio."""
        pyproject_path = chatterbox_dir / "pyproject.toml"

        if not pyproject_path.exists():
            raise RuntimeError(f"pyproject.toml not found: {pyproject_path}")

        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        dependencies = pyproject.get("project", {}).get("dependencies", [])

        # Filter out numpy, torch, torchaudio, gradio (already installed or not needed)
        skip_packages = {"numpy", "torch", "torchaudio", "gradio"}
        filtered = []

        for dep in dependencies:
            # Extract package name (before any version specifier)
            match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
            if match:
                pkg_name = match.group(1).lower()
                if pkg_name not in skip_packages:
                    filtered.append(dep)

        logger.debug("Filtered dependencies: %s", filtered)
        return filtered

    def _install_chatterbox_dependencies(self) -> None:
        """Install Chatterbox and required dependencies."""
        try:
            logger.info("=" * 80)
            logger.info("Installing Chatterbox TTS Library Dependencies...")
            logger.info("=" * 80)

            # Ensure pip is available
            self._ensure_pip_installed()

            # Step 1/3: Initialize Chatterbox submodule
            logger.info("Step 1/3: Initializing Chatterbox submodule...")
            chatterbox_dir = self._init_chatterbox_submodule()

            # Step 2/3: Install filtered dependencies (excluding numpy/torch/torchaudio)
            logger.info("Step 2/3: Installing Chatterbox dependencies...")
            deps = self._get_filtered_dependencies(chatterbox_dir)
            if deps:
                self._run_pip_install(deps)

            # Step 3/3: Install Chatterbox from submodule (without deps)
            logger.info("Step 3/3: Installing Chatterbox from submodule...")
            self._run_pip_install(["--no-deps", str(chatterbox_dir)])

            logger.info("Chatterbox installation completed successfully!")
            logger.info("=" * 80)

        except Exception as e:
            error_msg = f"Failed to install Chatterbox dependencies: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
