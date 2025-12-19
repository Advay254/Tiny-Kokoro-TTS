from kokoro import TTS
import base64
import io
import soundfile as sf

tts = TTS(
    model_path="models/kokoro-v1.0.onnx",
    voice_data="models/voices-v1.0.bin"
)

def synthesize(text: str, voice: str, speed: float):
    audio = tts.synthesize(
        text=text,
        voice=voice,
        speed=speed
    )

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
