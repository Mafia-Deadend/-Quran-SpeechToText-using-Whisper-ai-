from flask import Flask, request, jsonify, abort
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the model and processor
processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-tiny-ar-quran", local_files_only=False)
model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-tiny-ar-quran", local_files_only=False)

# Retrieve API key from environment variables
API_KEY = os.getenv('TRANSCRIPTION_API_KEY')

sampling_rate = 16000

def process_audio_data(audio_data):
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
    inputs = processor(audio_tensor, return_tensors="pt", sampling_rate=sampling_rate)
    decoder_start_token_id = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)
    generated_ids = model.generate(inputs["input_features"], decoder_input_ids=decoder_input_ids)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return transcription[0]

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check for API key in the headers
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        abort(403)  # Forbidden if the API key does not match

    audio_data = np.frombuffer(request.data, dtype=np.float32)
    transcription = process_audio_data(audio_data)
    return jsonify({'transcription': transcription})

if __name__ == '__main__':
    app.run(debug=True)
