import io
import os
import urllib.request
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# Updated URLs to use the stable binary voice file
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx?download=true"
VOICES_URL = "https://huggingface.co/thewh1teagle/kokoro-onnx/releases/download/v0.1.0/voices-v1.0.bin"

MODEL_PATH = "model.onnx"
VOICES_PATH = "voices.bin"

_kokoro_instance = None

def get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if not os.path.exists(VOICES_PATH):
            print("Downloading voices...")
            urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)
        
        print("Loading Kokoro...")
        _kokoro_instance = Kokoro(MODEL_PATH, VOICES_PATH)
    return _kokoro_instance

@app.get("/")
def health():
    return {"status": "online", "ready": _kokoro_instance is not None}

@app.get("/tts")
async def generate_tts(text: str, voice: str = "af_heart"):
    try:
        model = get_kokoro()
        samples, sample_rate = model.create(text, voice=voice, speed=1.0)
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}
        
