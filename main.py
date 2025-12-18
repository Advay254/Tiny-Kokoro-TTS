import io
import os
import urllib.request
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# UPDATED DIRECT LINKS
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model_quantized.onnx"
VOICES_URL = "https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/voices.json"

def download_files():
    if not os.path.exists("model.onnx"):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, "model.onnx")
    if not os.path.exists("voices.json"):
        print("Downloading voices...")
        urllib.request.urlretrieve(VOICES_URL, "voices.json")

# This runs when the server starts
download_files()
kokoro = Kokoro("model.onnx", "voices.json")

@app.get("/")  # This is the Health Check path
def home():
    return {"status": "healthy", "service": "Kokoro-TTS"}

@app.get("/tts")
async def generate_tts(text: str, voice: str = "af_heart"):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")
    
