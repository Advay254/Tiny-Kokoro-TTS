import io
from fastapi import FastAPI, Response
from kokoro_onnx import Kokoro
import soundfile as sf

app = FastAPI()

# Make sure these filenames match exactly what you downloaded
kokoro = Kokoro("model_quantized.onnx", "voices.json")

@app.get("/")
def home():
    return {"message": "TTS is running! Use /tts?text=hello to hear it."}

@app.get("/tts")
async def generate_tts(text: str, voice: str = "af_heart"):
    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0)
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV")
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")
