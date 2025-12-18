import io
import os
import urllib.request
import numpy as np
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# STABLE FILES (v1.0)
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODEL_PATH = "model.onnx"
VOICES_PATH = "voices.bin"

_kokoro_instance = None

def get_kokoro():
    global _kokoro_instance
    if _kokoro_instance is None:
        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if not os.path.exists(VOICES_PATH):
            urllib.request.urlretrieve(VOICES_URL, VOICES_PATH)
        _kokoro_instance = Kokoro(MODEL_PATH, VOICES_PATH)
    return _kokoro_instance

@app.get("/")
def health():
    return {"status": "online", "ready": _kokoro_instance is not None}

@app.get("/tts")
async def generate_tts(text: str, voice: str = "af_heart", speed: float = 1.0):
    try:
        model = get_kokoro()
        # FIX: Ensure speed is explicitly a float32 to avoid the Tensor(int32) error
        fixed_speed = float(speed)
        
        samples, sample_rate = model.create(text, voice=voice, speed=fixed_speed)
        
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        return {"error": str(e)}
