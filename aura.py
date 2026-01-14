import ollama
from kokoro_onnx import Kokoro
import sounddevice as sd
import os
import argparse
import sys
import requests
import threading
import time
import tempfile
import numpy as np
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav

# Constants
DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.int8.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICE_FILE = "voices-v1.0.bin"

class AuraAssistant:
    def __init__(self, ollama_model, voice, speed, lang):
        self.ollama_model = ollama_model
        self.voice = voice
        self.speed = speed
        self.lang = lang
        
        self._ensure_models()
        
        print("Loading Aura's voice engine...")
        try:
            self.kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
        except Exception as e:
            print(f"Error loading Kokoro engine: {e}")
            sys.exit(1)

        print("Loading Aura's ears (STT)...")
        try:
            # Using tiny.en for speed, can be changed to base or small
            self.stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Error loading STT model: {e}")
            sys.exit(1)

    def _download_file(self, url, filename):
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            sys.exit(1)

    def _ensure_models(self):
        if not os.path.exists(MODEL_FILE):
            self._download_file(DEFAULT_MODEL_URL, MODEL_FILE)
        
        if not os.path.exists(VOICE_FILE):
            self._download_file(DEFAULT_VOICES_URL, VOICE_FILE)

    def speak(self, text):
        print(f"\nAura: {text}")
        try:
            samples, sample_rate = self.kokoro.create(
                text, 
                voice=self.voice, 
                speed=self.speed, 
                lang=self.lang
            )
            sd.play(samples, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Error playing audio: {e}")

    def run(self):
        print(f"\nAura-Local is active using model '{self.ollama_model}'.")
        print("Choose your interaction mode:")
        print("1. Text Chat (Type your messages)")
        print("2. Voice Chat (Speak to Aura)")
        
        try:
            choice = input("\nSelect mode (1 or 2): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        if choice == '2':
            self.voice_chat_loop()
        else:
            self.text_chat_loop()

    def text_chat_loop(self):
        print("\n--- Text Chat Mode Active ---")
        print("(Type 'exit' to stop)")
        while True:
            try:
                if not self.ask_aura():
                    break
            except KeyboardInterrupt:
                print("\nSwitched to menu.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    def voice_chat_loop(self):
        print("\n--- Voice Chat Mode Active ---")
        print("(Press Enter to record/send, Ctrl+C to exit voice mode)")
        while True:
            try:
                audio_info = self.listen()
                if audio_info:
                    audio_data, fs = audio_info
                    print("Processing...")
                    prompt = self.transcribe(audio_data, fs)
                    if prompt:
                        print(f"You: {prompt}")
                        if prompt.lower() in ['exit', 'quit']:
                            break
                        self.process_query(prompt)
                    else:
                        print("Aura didn't hear anything. Try again.")
                else:
                    # If listen returned None (e.g. error), we might want to break or continue
                    continue
            except KeyboardInterrupt:
                print("\nVoice mode stopped.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")

    def listen(self):
        fs = 16000  # Whisper expects 16kHz
        channels = 1
        recording = []
        stop_event = threading.Event()
        
        def record_callback(indata, frames, time, status):
            if status:
                print(status)
            recording.append(indata.copy())

        print("\n[Press Enter to start recording]")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            return None

        print("Listening... (Press Enter to stop, max 30s)")
        
        stream = sd.InputStream(samplerate=fs, channels=channels, callback=record_callback)
        stream.start()
        
        def wait_for_stop():
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            stop_event.set()

        input_thread = threading.Thread(target=wait_for_stop, daemon=True)
        input_thread.start()
        
        start_time = time.time()
        while not stop_event.is_set() and (time.time() - start_time) < 30:
            time.sleep(0.1)
            
        stream.stop()
        stream.close()
        
        if not recording:
            return None
            
        audio_data = np.concatenate(recording, axis=0)
        return audio_data, fs

    def transcribe(self, audio_data, fs):
        # Ensure audio is float32 for faster-whisper
        audio_data = audio_data.flatten().astype(np.float32)
        
        # We can pass the numpy array directly to transcribe
        segments, info = self.stt_model.transcribe(audio_data, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def process_query(self, prompt):
        print("Thinking...")
        try:
            response = ollama.chat(
                model=self.ollama_model, 
                messages=[{'role': 'user', 'content': prompt}]
            )
            answer = response['message']['content']
            self.speak(answer)
        except ollama.ResponseError as e:
            print(f"Ollama Error: {e}")
            print("Make sure Ollama is running and the model is pulled.")
        except Exception as e:
            print(f"Error getting response: {e}")

    def ask_aura(self):
        try:
            prompt = input("\nYou: ")
        except EOFError:
            return False

        if prompt.lower() in ['exit', 'quit']:
            return False
        
        if not prompt.strip():
            return True

        self.process_query(prompt)
        return True

def main():
    parser = argparse.ArgumentParser(description="Aura-Local: AI Voice Assistant")
    parser.add_argument("--model", type=str, default="artifish/llama3.2-uncensored", help="Ollama model to use")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--lang", type=str, default="en-us", help="Language code")
    
    args = parser.parse_args()
    
    app = AuraAssistant(
        ollama_model=args.model,
        voice=args.voice,
        speed=args.speed,
        lang=args.lang
    )
    app.run()

if __name__ == "__main__":
    main()
