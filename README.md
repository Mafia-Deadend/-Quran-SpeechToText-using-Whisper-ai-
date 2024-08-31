# Quran Whisper Transcription

This repository contains a Python script that utilizes the fine-tuned Whisper model to transcribe Quranic recitations from audio files. The Whisper model used in this project has been fine-tuned specifically on Quranic recitation, enabling accurate transcription of Arabic text from audio recordings.

## Features

- **Quran-Specific Transcription**: The model is fine-tuned on Quranic recitations, providing high accuracy in transcribing Arabic verses.
- **Audio Input Handling**: Accepts standard audio file formats (e.g., MP3) and processes them for transcription.
- **TorchScript Conversion**: The model is also converted to TorchScript for easier deployment in production environments.

## Getting Started

### Prerequisites

- Python 3.7+
- `torch`
- `transformers`
- `librosa`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/quran-whisper-transcription.git
    ```
2. Install the required packages:
    ```bash
    pip install torch transformers librosa
    ```

### Usage

1. **Load the model and processor**:
    ```python
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import torch
    import librosa

    processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
    model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran")
    ```

2. **Transcribe an audio file**:
    ```python
    audio_path = 'path_to_your_audio.mp3'
    audio_data, sampling_rate = librosa.load(audio_path, sr=16000)
    audio = torch.tensor(audio_data)

    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    generated_ids = model.generate(inputs["input_features"])

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print("Transcription:", transcription)
    ```

3. **TorchScript Conversion** (Optional):
    ```python
    scripted_model = torch.jit.trace(model, torch.randn(1, 80, 3000))
    scripted_model.save("whisper_tiny_ar_quran_scripted.pt")
    ```

### Notes

- Ensure the audio file is correctly preprocessed (resampled to 16 kHz) for accurate transcription.
- The TorchScript model may have limitations due to unsupported operations in the original model.


## Acknowledgments

- **Tarteel.ai** for providing the fine-tuned Whisper model for Quranic recitations.
- **Hugging Face** for the Transformers library.

---
