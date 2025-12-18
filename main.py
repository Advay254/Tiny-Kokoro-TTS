import io
import os
import urllib.request
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# Verified links
MODEL_URL = "https://huggingface.co/onnx-community/Kokoro-82M-ONNX/resolve/main/onnx/model_quantized.onnx?download=true"
VOICES_URL = "https://huggingface.co/NeuML/kokoro-base-onnx/resolve/main/voices.json?download=true"

MODEL_PATH = "model.onnx"
VOICES_PATH = "voices.json"

# This holds the model in memory ONLY after the first request
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
        
        print("Loading model into memory...")
        # Use only 1 thread to save RAM
        import onnxruntime as ort
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        _kokoro_instance = Kokoro(MODEL_PATH, VOICES_PATH, session_options=sess_options)
    return _kokoro_instance

@app.get("/")
def health():
    return {"status": "online", "info": "Model loads on first /tts request"}

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
        
