from flask import Flask, request, send_file, jsonify
from kokoro import generate
import os
from io import BytesIO

app = Flask(__name__)

# Available voices
VOICES = ['af', 'af_bella', 'af_sarah', 'am_adam', 'am_michael', 
          'bf_emma', 'bf_isabella', 'bm_george', 'bm_lewis']

@app.route('/')
def home():
    return jsonify({
        "message": "Kokoro TTS API",
        "endpoints": {
            "/tts": "POST - Generate speech",
            "/voices": "GET - List available voices"
        }
    })

@app.route('/voices', methods=['GET'])
def list_voices():
    return jsonify({"voices": VOICES})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'af_bella')
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        if voice not in VOICES:
            return jsonify({"error": f"Invalid voice. Choose from: {VOICES}"}), 400
        
        # Generate audio
        audio_data, sample_rate = generate(text, voice=voice, speed=speed)
        
        # Convert to WAV format in memory
        audio_buffer = BytesIO()
        import wave
        with wave.open(audio_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        audio_buffer.seek(0)
        
        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='speech.wav'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
