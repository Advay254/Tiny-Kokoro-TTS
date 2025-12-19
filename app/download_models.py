import os
import urllib.request

MODEL_DIR = "models"
MODEL_URLS = {
    "kokoro-v1.0.onnx": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin"
}

os.makedirs(MODEL_DIR, exist_ok=True)

for name, url in MODEL_URLS.items():
    path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"{name} downloaded.")
