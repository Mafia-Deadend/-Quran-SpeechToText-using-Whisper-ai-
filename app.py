import streamlit as st
import sounddevice as sd
import numpy as np
import requests

# URL of the Flask backend (assuming it runs locally on port 5000)
backend_url = "http://127.0.0.1:5000/transcribe"

# API key (use the same key you set in the Flask .env file)
api_key = "TRANSCRIPTION_API_KEY=526b03ed8f25e124265f281e81faef96877c035fbc4339a72c1f744b6ca0002e"

# Parameters
duration = 20  # Duration of recording in seconds
sampling_rate = 16000  # Sampling rate

st.title("Live Audio Transcription")

# Function to record live audio
def record_audio(duration, sampling_rate):
    st.write("Recording...")
    audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    st.write("Recording complete.")
    return audio.flatten()

# Record button
if st.button("Record and Transcribe"):
    # Record live audio
    audio_data = record_audio(duration, sampling_rate)

    # Convert audio to bytes and send to the Flask backend
    audio_bytes = audio_data.tobytes()

    headers = {
        "x-api-key": api_key
    }

    response = requests.post(backend_url, headers=headers, data=audio_bytes)

    if response.status_code == 200:
        # Display the transcription result
        transcription = response.json().get("transcription")
        st.write("Transcription Result:")
        st.write(transcription)
    else:
        st.error(f"Failed to transcribe. Status code: {response.status_code}")
