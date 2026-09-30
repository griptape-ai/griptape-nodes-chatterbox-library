"""Chatterbox model loading and speech synthesis.

Imported only from `process()`, so nothing here runs on the orchestrator: torch, torchaudio and
the chatterbox sources live in the library's execution environment, which only the process that
executes nodes can import from.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

logger = logging.getLogger("griptape_nodes_chatterbox_library")

# The submodule is installed with --no-deps rather than declared in the execution set, because
# upstream pins torch==2.6.0 and torchaudio==2.6.0, which cannot resolve against the
# torch 2.8.0+cu128 build current GPUs need. That install supplies only distribution metadata;
# `src` goes first on sys.path so the pinned checkout is the code that runs.
CHATTERBOX_SRC = Path(__file__).parent.parent / "chatterbox" / "src"

TURBO = "turbo"
STANDARD = "standard"
MULTILINGUAL = "multilingual"


def synthesize(
    *,
    variant: str,
    device: str,
    text: str,
    cfg_weight: float,
    exaggeration: float,
    output_path: Path,
    reference_audio_path: Path | None = None,
    language: str | None = None,
) -> None:
    """Generate speech and write it to `output_path` as a wav.

    Args:
        variant: One of `TURBO`, `STANDARD` or `MULTILINGUAL`.
        device: Compute device to load the model onto.
        text: Text to speak.
        cfg_weight: Voice adherence, 0.0-1.0.
        exaggeration: Expressiveness, 0.0-1.0.
        output_path: Where to write the generated wav.
        reference_audio_path: Audio to clone the voice from, if any.
        language: Language code, for the multilingual variant.
    """
    model = _load_model(variant, device)

    gen_kwargs: dict[str, Any] = {
        "text": text,
        "cfg_weight": cfg_weight,
        "exaggeration": exaggeration,
    }
    if reference_audio_path is not None:
        gen_kwargs["audio_prompt_path"] = str(reference_audio_path)
    if language is not None:
        gen_kwargs["language_id"] = language

    logger.info("Generating speech with Chatterbox (variant=%s, device=%s)...", variant, device)
    wav = model.generate(**gen_kwargs)
    torchaudio.save(str(output_path), wav, model.sr)


def _load_model(variant: str, device: str) -> Any:
    """Load one Chatterbox model variant onto `device`."""
    _configure_tf32()
    _put_chatterbox_on_path()

    logger.info("Loading Chatterbox model (variant=%s, device=%s)...", variant, device)

    # Imported here rather than at module scope: the chatterbox package is only importable once
    # `_put_chatterbox_on_path` has run, and each variant lives in its own module, so only the
    # selected one is loaded.
    if variant == TURBO:
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        model = ChatterboxTurboTTS.from_pretrained(device=device)
    elif variant == MULTILINGUAL:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    elif variant == STANDARD:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device=device)
    else:
        msg = f"Unknown Chatterbox model variant: {variant}"
        raise ValueError(msg)

    logger.info("Chatterbox model loaded successfully")
    return model


def _put_chatterbox_on_path() -> None:
    """Make the submodule's chatterbox package importable.

    Inserted at the front of `sys.path` so the package found is the one under `src`, ahead of the
    submodule root directory that shares its name.
    """
    if not (CHATTERBOX_SRC / "chatterbox" / "__init__.py").exists():
        msg = (
            f"Attempted to load the Chatterbox model code from {CHATTERBOX_SRC}. Failed due to: the "
            f"chatterbox submodule is not checked out. Run 'git submodule update --init --recursive' "
            f"in the library's repository."
        )
        raise RuntimeError(msg)

    source_dir = str(CHATTERBOX_SRC)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)


def _configure_tf32() -> None:
    """Enable TF32 matmuls, which Ampere and later GPUs run faster than full fp32."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
