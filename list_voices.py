
import sys
import json
try:
    from kokoro_onnx import Kokoro
    k = Kokoro("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")
    print(dir(k))
    # Try common attributes if they exist
    if hasattr(k, 'get_voices'):
        print(k.get_voices())
    if hasattr(k, 'voices'):
        print(k.voices.keys())
except Exception as e:
    print(f"Error: {e}")
