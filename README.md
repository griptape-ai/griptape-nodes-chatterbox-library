# Griptape Nodes: Chatterbox TTS Library

A [Griptape Nodes](https://www.griptapenodes.com/) library providing text-to-speech generation with voice cloning using [Chatterbox TTS](https://github.com/resemble-ai/chatterbox) models.

## Features

- **High-quality text-to-speech** generation powered by Chatterbox TTS
- **Voice cloning** from reference audio (6-30 seconds recommended)
- **Two model variants**:
  - **Turbo**: Lightweight 350M parameter model, optimized for low latency
  - **Standard**: Original 500M parameter model with advanced creative controls
- **Multilingual support**: 23+ languages available with the standard model
- **Creative controls**: Fine-tune voice adherence and expressiveness

## Requirements

- **CUDA-capable GPU** (required for Chatterbox TTS)
- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes) installed and running

## Installation

1. **Navigate to your Griptape Nodes workspace directory**:

   ```bash
   cd `gtn config show workspace_directory`
   ```

2. **Clone the library**:

   ```bash
   git clone https://github.com/griptape-ai/griptape-nodes-chatterbox-library.git
   ```

3. **Add the library in the Griptape Nodes Editor**:

   - Open the Settings menu and navigate to the _Libraries_ settings
   - Click on _+ Add Library_ at the bottom of the settings panel
   - Enter the path to the library JSON file:
     ```
     <workspace_directory>/griptape-nodes-chatterbox-library/griptape_nodes_chatterbox_library/griptape_nodes_library.json
     ```
   - Close the Settings Panel
   - Click on _Refresh Libraries_

4. **Verify installation** by checking that the "Chatterbox TTS" node appears in the Audio category.

> **Note**: On first use, the library will automatically initialize a git submodule and install the Chatterbox TTS dependencies. This may take a few minutes.

## Nodes

### Chatterbox TTS

Generate speech from text using Chatterbox TTS with optional voice cloning.

#### Parameters

| Parameter           | Type     | Default | Description                                                                                    |
| ------------------- | -------- | ------- | ---------------------------------------------------------------------------------------------- |
| **model**           | dropdown | Turbo   | HuggingFace model selection (Turbo or Standard)                                                |
| **text**            | string   | —       | Text to convert to speech (required)                                                           |
| **multilingual**    | boolean  | `false` | Enable multilingual mode (standard model only)                                                 |
| **language**        | dropdown | English | Language selection (available when multilingual is enabled)                                    |
| **reference_audio** | audio    | —       | Optional audio for voice cloning (6-30 seconds recommended)                                    |
| **cfg_weight**      | float    | `0.5`   | Voice adherence: lower = creative interpretation, higher = strict adherence to reference voice |
| **exaggeration**    | float    | `0.5`   | Expressiveness: lower = neutral delivery, higher = highly expressive                           |

#### Outputs

| Output             | Type             | Description                       |
| ------------------ | ---------------- | --------------------------------- |
| **audio**          | AudioUrlArtifact | Generated speech audio            |
| **was_successful** | boolean          | Whether generation succeeded      |
| **result_details** | string           | Details about the result or error |

#### Model Comparison

| Feature           | Turbo | Standard            |
| ----------------- | ----- | ------------------- |
| Parameters        | 350M  | 500M                |
| Latency           | Low   | Higher              |
| Multilingual      | No    | Yes (23+ languages) |
| Voice cloning     | Yes   | Yes                 |
| Creative controls | Yes   | Yes                 |

#### Supported Languages (Multilingual Mode)

The standard model supports 23 languages when multilingual mode is enabled:

Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese

## Usage Tips

### Voice Cloning

For best voice cloning results:

- Use reference audio between 6-30 seconds
- Ensure the reference audio is clear with minimal background noise
- Higher `cfg_weight` values will make the output more closely match the reference voice
- Lower `cfg_weight` values allow more creative interpretation

### Expressiveness

- Use lower `exaggeration` values (0.0-0.3) for neutral, professional delivery
- Use higher `exaggeration` values (0.7-1.0) for dramatic, expressive speech
- Values around 0.5 provide balanced expressiveness

### Model Selection

- Use **Turbo** for:

  - Real-time or near-real-time applications
  - Quick prototyping
  - When low latency is a priority

- Use **Standard** for:
  - Multilingual content
  - Maximum quality output
  - When latency is not a concern

## Troubleshooting

### "No CUDA device found" Error

Chatterbox TTS requires a CUDA-capable GPU. Ensure:

- You have an NVIDIA GPU installed
- NVIDIA drivers are up to date
- CUDA is properly installed

### Library Not Loading

If the library fails to load:

1. Check that the library path is correct in Settings
2. Ensure you have an internet connection (required for submodule initialization)
3. Check the Griptape Nodes logs for detailed error messages

## Additional Resources

- [Chatterbox TTS GitHub](https://github.com/resemble-ai/chatterbox)
- [Griptape Nodes Documentation](https://docs.griptapenodes.com/)
- [Griptape Discord](https://discord.gg/griptape)

## License

This library is provided under the Apache License 2.0.
