import io
import os
import urllib.request
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# Verified links provided by you
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx?download=true"
VOICES_URL = "https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/voices.json?download=true"

MODEL_PATH = "model.onnx"
VOICES_PATH = "voices.json"

def download_if_missing():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... this takes a moment.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    if not os.path.exists(VOICES_PATH):
        print("Downloading voices...")
        urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)

kokoro = None

@app.on_event("startup")
async def startup_event():
    global kokoro
    download_if_missing()
    # Loading the model
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)

@app.get("/")
def health():
    return {"status": "online", "ready": kokoro is not None}

@app.get("/tts")
async def generate_tts(text: str, voice: str = "af_heart"):
    if not kokoro:
        return Response(content="Model loading...", status_code=503)
    
    # Generate audio
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")
    
