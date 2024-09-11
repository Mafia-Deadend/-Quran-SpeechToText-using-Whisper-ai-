import torch
import librosa
import time
import onnx
import pyaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import pyautogui as pg

# Load the model and processor (ensure local_files_only=True for offline usage)
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran", local_files_only=True)
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran", local_files_only=True)

# Parameters
sampling_rate = 16000
chunk_duration = 2  # Duration of each audio chunk to process in seconds
chunk_size = int(sampling_rate * chunk_duration)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the file for saving transcriptions
file_path = "transcription.txt"

# Function to process audio data
def process_audio_data(audio_data):
    # Convert the audio data to a torch tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    
    # Process the audio and generate transcription
    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=sampling_rate)

    # Ensure the model uses the correct inputs for decoding
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

    # Generate transcription with the correct inputs
    generated_ids = model.generate(
        inputs["input_features"],
        decoder_input_ids=decoder_input_ids
    )

    # Decode the transcription
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # Print transcription
    print("Transcription:", transcription)
    
    # Write transcription to file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(transcription[0] + "\n")  # Append transcription to the file
    
    pg.typewrite(transcription)

# Callback function to read microphone input
def callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    process_audio_data(audio_data)
    return (in_data, pyaudio.paContinue)

# Open a stream for recording
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sampling_rate,
                input=True,
                frames_per_buffer=chunk_size,
                stream_callback=callback)

# Start the stream
stream.start_stream()

print("Recording... Press Ctrl+C to stop.")

# Keep the script running
try:
    while stream.is_active():
        time.sleep(0.5)
except KeyboardInterrupt:
    print("Recording stopped.")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()
