from fastapi import FastAPI
from pydantic import BaseModel
from app.tts import synthesize

app = FastAPI(title="Kokoro TTS API")

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_bella"
    speed: float = 1.0

@app.get("/")
def health():
    return {"status": "kokoro online"}

@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    audio_b64 = synthesize(
        text=req.text,
        voice=req.voice,
        speed=req.speed
    )
    return {
        "audio_base64": audio_b64,
        "format": "wav"
    }
