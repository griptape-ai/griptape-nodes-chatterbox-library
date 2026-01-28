"""Chatterbox TTS text-to-speech generation node."""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, ClassVar

from griptape.artifacts import AudioUrlArtifact

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

logger = logging.getLogger("griptape_nodes_chatterbox_library")


class ChatterboxTextToSpeech(SuccessFailureNode):
    """Generate speech from text using Chatterbox TTS with optional voice cloning.

    This node supports two Chatterbox model variants:
    - Turbo: Lightweight 350M parameter model, optimized for low latency
    - Standard: Original 500M parameter model with advanced creative controls

    The standard model can optionally be run in multilingual mode (23+ languages).

    Inputs:
        - model: HuggingFace model selection (Turbo or Standard)
        - multilingual: Enable multilingual mode (standard model only)
        - text: Text to convert to speech
        - reference_audio: Optional audio for voice cloning
        - cfg_weight: Voice adherence (0.0-1.0)
        - exaggeration: Expressiveness control (0.0-1.0)
        - language: Language code (for multilingual mode)

    Outputs:
        - audio: Generated speech audio
        - was_successful: Whether generation succeeded
        - result_details: Details about the result or error
    """

    # HuggingFace repo IDs
    REPO_STANDARD: ClassVar[str] = "ResembleAI/chatterbox"
    REPO_TURBO: ClassVar[str] = "ResembleAI/chatterbox-turbo"
    MODEL_REPOS: ClassVar[list[str]] = [REPO_TURBO, REPO_STANDARD]

    DEFAULT_LANGUAGE: ClassVar[str] = "English (en)"

    # Language code map for multilingual model (display name -> code)
    LANGUAGE_CODE_MAP: ClassVar[dict[str, str]] = {
        "Arabic (ar)": "ar",
        "Chinese (zh)": "zh",
        "Czech (cs)": "cs",
        "Dutch (nl)": "nl",
        DEFAULT_LANGUAGE: "en",
        "French (fr)": "fr",
        "German (de)": "de",
        "Greek (el)": "el",
        "Hindi (hi)": "hi",
        "Hungarian (hu)": "hu",
        "Indonesian (id)": "id",
        "Italian (it)": "it",
        "Japanese (ja)": "ja",
        "Korean (ko)": "ko",
        "Polish (pl)": "pl",
        "Portuguese (pt)": "pt",
        "Romanian (ro)": "ro",
        "Russian (ru)": "ru",
        "Spanish (es)": "es",
        "Swedish (sv)": "sv",
        "Thai (th)": "th",
        "Turkish (tr)": "tr",
        "Vietnamese (vi)": "vi",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "Audio Generation"
        self.description = "Generate speech from text using Chatterbox TTS"

        # HuggingFace model parameter
        self.model_param = HuggingFaceRepoParameter(
            self, repo_ids=self.MODEL_REPOS, parameter_name="model"
        )
        self.model_param.add_input_parameters()

        # Multilingual toggle (only shown for standard model)
        self.add_parameter(
            ParameterBool(
                name="multilingual",
                default_value=False,
                tooltip="Enable multilingual mode (23+ languages). Only available for the standard model.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=True,  # Hidden by default, shown when standard model is selected
                ui_options={"display_name": "Multilingual"},
            )
        )

        # Text input
        self.add_parameter(
            ParameterString(
                name="text",
                tooltip="Text to convert to speech",
                multiline=True,
                placeholder_text="Enter text to convert to speech...",
                allow_output=False,
                ui_options={"display_name": "Text"},
            )
        )

        # Reference audio for voice cloning (optional)
        self.add_parameter(
            ParameterAudio(
                name="reference_audio",
                tooltip="Optional reference audio for voice cloning (6-30 seconds recommended)",
                clickable_file_browser=True,
                allow_output=False,
                ui_options={"display_name": "Reference Audio (Voice Cloning)"},
            )
        )

        # Language selection (for multilingual model)
        self.add_parameter(
            ParameterString(
                name="language",
                default_value=self.DEFAULT_LANGUAGE,
                tooltip="Language for multilingual model",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(self.LANGUAGE_CODE_MAP.keys()))},
                hide=True,  # Hidden by default, shown when multilingual is selected
                ui_options={"display_name": "Language"},
            )
        )

        # Voice adherence (cfg_weight)
        self.add_parameter(
            ParameterFloat(
                name="cfg_weight",
                default_value=0.5,
                min_val=0.0,
                max_val=1.0,
                slider=True,
                tooltip="Voice adherence: lower values = creative interpretation, higher values = strict adherence to reference voice",
                allow_input=True,
                allow_property=True,
                allow_output=False,
                ui_options={"display_name": "Voice Adherence"},
            )
        )

        # Expressiveness (exaggeration)
        self.add_parameter(
            ParameterFloat(
                name="exaggeration",
                default_value=0.5,
                min_val=0.0,
                max_val=1.0,
                slider=True,
                tooltip="Expressiveness: lower values = neutral delivery, higher values = highly expressive",
                allow_input=True,
                allow_property=True,
                allow_output=False,
                ui_options={"display_name": "Expressiveness"},
            )
        )

        # Audio output
        self.add_parameter(
            ParameterAudio(
                name="audio",
                tooltip="Generated speech audio",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
                ui_options={"is_full_width": True, "display_name": "Generated Audio"},
            )
        )

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the speech generation result or errors",
            result_details_placeholder="Generation status will appear here.",
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Update parameter visibility based on model and multilingual selection."""
        super().after_value_set(parameter, value)

        # When model changes, show/hide multilingual toggle
        if parameter.name == "model":
            if value == self.REPO_STANDARD:
                self.show_parameter_by_name("multilingual")
            else:
                # Hide multilingual for turbo model and reset to False
                self.hide_parameter_by_name("multilingual")
                self.parameter_values["multilingual"] = False
                self.hide_parameter_by_name("language")

        # When multilingual toggle changes, show/hide language selector
        if parameter.name == "multilingual":
            if value:
                self.show_parameter_by_name("language")
            else:
                self.hide_parameter_by_name("language")

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate CUDA availability, model download, and required inputs."""
        errors = super().validate_before_node_run() or []

        # Validate HuggingFace model is downloaded
        model_errors = self.model_param.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        # Check CUDA availability
        try:
            import torch

            if not torch.cuda.is_available():
                errors.append(ValueError("Chatterbox TTS requires a CUDA-capable GPU. No CUDA device found."))
        except ImportError:
            errors.append(ValueError("PyTorch is not installed. Please ensure chatterbox-tts dependencies are installed."))

        # Check text input
        text = self.get_parameter_value("text")
        if not text or not text.strip():
            errors.append(ValueError("Text input is required"))

        return errors if errors else None

    def _get_model(self, repo_id: str, multilingual: bool) -> Any:
        """Load the Chatterbox model.

        Args:
            repo_id: HuggingFace repository ID for the model
            multilingual: Whether to use multilingual mode (standard model only)

        Returns:
            Loaded Chatterbox model instance
        """
        logger.info("Loading Chatterbox model: %s (multilingual=%s)...", repo_id, multilingual)

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if repo_id == self.REPO_TURBO:
            from chatterbox.tts import ChatterboxTurboTTS

            model = ChatterboxTurboTTS.from_pretrained(device=device)
        elif repo_id == self.REPO_STANDARD:
            if multilingual:
                from chatterbox.tts import ChatterboxMultilingualTTS

                model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS

                model = ChatterboxTTS.from_pretrained(device=device)
        else:
            msg = f"Unknown model repo: {repo_id}"
            raise ValueError(msg)

        logger.info("Chatterbox model loaded successfully")
        return model

    def _download_reference_audio(self, audio_artifact: Any, temp_dir: Path) -> Path | None:
        """Download reference audio to temporary file if provided."""
        if audio_artifact is None:
            return None

        # Extract URL from artifact
        if hasattr(audio_artifact, "value"):
            url = audio_artifact.value
        elif isinstance(audio_artifact, str):
            url = audio_artifact
        else:
            logger.warning("Unsupported reference audio type: %s", type(audio_artifact))
            return None

        # Handle local paths
        if not url.startswith(("http://", "https://")):
            local_path = Path(url)
            if local_path.exists():
                return local_path
            logger.warning("Local reference audio not found: %s", url)
            return None

        # Download from URL
        import httpx

        temp_file = temp_dir / "reference_audio.wav"

        logger.info("Downloading reference audio from %s", url)
        response = httpx.get(url, timeout=60)
        response.raise_for_status()
        temp_file.write_bytes(response.content)

        return temp_file

    def _generate_speech(self) -> None:
        """Generate speech using Chatterbox TTS."""
        import torchaudio

        # Get parameter values
        repo_id, _revision = self.model_param.get_repo_revision()
        multilingual = self.get_parameter_value("multilingual") or False
        text = self.get_parameter_value("text")
        reference_audio = self.get_parameter_value("reference_audio")
        cfg_weight = self.get_parameter_value("cfg_weight")
        if cfg_weight is None:
            cfg_weight = 0.5
        exaggeration = self.get_parameter_value("exaggeration")
        if exaggeration is None:
            exaggeration = 0.5
        language_display = self.get_parameter_value("language") or self.DEFAULT_LANGUAGE
        language = self.LANGUAGE_CODE_MAP.get(language_display, "en")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Load model
            model = self._get_model(repo_id, multilingual)

            # Prepare reference audio path
            ref_audio_path = None
            if reference_audio:
                ref_audio_path = self._download_reference_audio(reference_audio, temp_path)

            # Build generation kwargs
            gen_kwargs: dict[str, Any] = {
                "text": text,
                "cfg_weight": cfg_weight,
                "exaggeration": exaggeration,
            }

            if ref_audio_path:
                gen_kwargs["audio_prompt_path"] = str(ref_audio_path)

            # Add language for multilingual mode
            if multilingual:
                gen_kwargs["language"] = language

            # Generate speech
            logger.info("Generating speech with Chatterbox (repo=%s, multilingual=%s)...", repo_id, multilingual)
            wav = model.generate(**gen_kwargs)

            # Save to temporary file
            temp_output_path = temp_path / "output.wav"
            torchaudio.save(str(temp_output_path), wav, model.sr)

            # Save to static storage
            filename = f"chatterbox_tts_{int(time.time())}.wav"
            audio_bytes = temp_output_path.read_bytes()

            static_files_manager = GriptapeNodes.StaticFilesManager()
            saved_url = static_files_manager.save_static_file(audio_bytes, filename)

            self.parameter_output_values["audio"] = AudioUrlArtifact(value=saved_url, name=filename)

            model_desc = f"{repo_id} (multilingual)" if multilingual else repo_id
            self._set_status_results(
                was_successful=True,
                result_details=f"SUCCESS: Speech generated with {model_desc} and saved as {filename}",
            )

    def process(self) -> AsyncResult[None]:
        """Process the text-to-speech generation asynchronously."""
        yield lambda: self._process()

    def _process(self) -> None:
        """Main processing logic."""
        self._clear_execution_status()

        try:
            self._generate_speech()
        except Exception as e:
            error_msg = f"Speech generation failed: {e}"
            logger.exception(error_msg)
            self.parameter_output_values["audio"] = None
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_msg}")
            self._handle_failure_exception(e)
