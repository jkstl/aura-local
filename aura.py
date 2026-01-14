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
import psutil
import webbrowser
from datetime import datetime
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav

# Constants
DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v1.0.int8.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices-v1.0.bin"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICE_FILE = "voices-v1.0.bin"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
KNOWLEDGE_DIR = "knowledge"
EMBED_MODEL = "nomic-embed-text"

class KnowledgeBase:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./aura_db")
        self.collection = self.chroma_client.get_or_create_collection(name="aura_knowledge")
        
    def index_files(self):
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
            return

        print("Updating knowledge base...")
        files = os.listdir(KNOWLEDGE_DIR)
        for file in files:
            path = os.path.join(KNOWLEDGE_DIR, file)
            if file.endswith('.txt') or file.endswith('.md'):
                self._index_text_file(path)
            elif file.endswith('.pdf'):
                self._index_pdf_file(path)

    def _index_text_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._add_to_collection(path, content)
        except Exception as e:
            print(f"Error indexing {path}: {e}")

    def _index_pdf_file(self, path):
        try:
            reader = PdfReader(path)
            content = ""
            for page in reader.pages:
                content += page.extract_text() + "\n"
            self._add_to_collection(path, content)
        except Exception as e:
            print(f"Error indexing {path}: {e}")

    def _add_to_collection(self, path, content):
        # Very simple chunking: split by paragraphs
        chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
        if not chunks:
            return

        # Generate embeddings using Ollama
        for i, chunk in enumerate(chunks):
            chunk_id = f"{path}_{i}"
            
            try:
                embedding_resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                embedding = embedding_resp['embedding']
                
                self.collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": path}]
                )
            except Exception as e:
                print(f"Error embedding chunk {i} of {path}: {e}")

    def query(self, text, n_results=3):
        try:
            embedding_resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            embedding = embedding_resp['embedding']
            
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            return []

class AuraAssistant:
    def __init__(self, ollama_model, voice, speed, lang):
        self.ollama_model = ollama_model
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.system_prompt = self._load_system_prompt()
        self.available_tools = self._get_tool_definitions()
        self.kb = KnowledgeBase()
        
        self._ensure_models()
        self.kb.index_files()
        
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

    def _load_system_prompt(self):
        if os.path.exists(SYSTEM_PROMPT_FILE):
            try:
                with open(SYSTEM_PROMPT_FILE, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read system prompt file: {e}")
        return ""

    def _get_tool_definitions(self):
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'get_current_time',
                    'description': 'Returns the current local date and time. ONLY call this if the user explicitly asks for the time or date.',
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'open_url',
                    'description': 'Opens a specific URL in the web browser ONLY when explicitly asked by the user to "open" a site.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'url': {
                                'type': 'string',
                                'description': 'The URL to open (e.g., https://www.google.com)',
                            },
                        },
                        'required': ['url'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'get_system_info',
                    'description': 'Returns current system status like CPU and memory usage. ONLY call this if the user asks about performance, CPU, or RAM.',
                }
            }
        ]

    def _execute_tool(self, tool_call):
        name = tool_call['function']['name']
        args = tool_call['function'].get('arguments', {})
        
        # Ensure args is a dict (Ollama returns it as a dict usually)
        if isinstance(args, str):
            import json
            args = json.loads(args)

        print(f"  [Executing Tool: {name} with args: {args}]")
        
        if name == 'get_current_time':
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        elif name == 'open_url':
            url = args.get('url')
            if url:
                # Basic validation: check for a dot and minimum length
                if '.' not in url or len(url) < 4 or ' ' in url:
                    return f"Error: '{url}' does not look like a valid URL. I will only open specific web addresses."
                
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                webbrowser.open(url)
                return f"Successfully opened {url}"
            return "Error: No URL provided."
        
        elif name == 'get_system_info':
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            return f"CPU Usage: {cpu}% | Memory Usage: {mem}%"
        
        return "Error: Tool not found."

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
        
        # Search Knowledge Base
        context_docs = self.kb.query(prompt)
        context_str = ""
        if context_docs:
            context_str = "\n\nRelevant Context from Knowledge Base:\n" + "\n---\n".join(context_docs)
            print(f"  [Knowledge found! Found {len(context_docs)} snippets]")

        messages = []
        if self.system_prompt:
            messages.append({'role': 'system', 'content': self.system_prompt})
        
        # Inject context as a separate block to avoid confusing the model
        user_prompt_with_context = prompt
        if context_str:
            user_prompt_with_context = f"Context from Knowledge Base:\n{context_str}\n\nUser Question: {prompt}"

        messages.append({'role': 'user', 'content': user_prompt_with_context})

        while True:
            try:
                response = ollama.chat(
                    model=self.ollama_model, 
                    messages=messages,
                    tools=self.available_tools
                )
                
                message = response['message']
                
                # If there are tool calls, execute them and continue the loop
                if message.get('tool_calls'):
                    messages.append(message)
                    for tool in message['tool_calls']:
                        result = self._execute_tool(tool)
                        messages.append({
                            'role': 'tool',
                            'content': str(result),
                        })
                    continue # Re-run chat with tool results

                # Otherwise, speak the final answer
                answer = message['content']
                if answer:
                    self.speak(answer)
                break

            except ollama.ResponseError as e:
                print(f"Ollama Error: {e}")
                print("Make sure Ollama is running and the model supports tools.")
                break
            except Exception as e:
                print(f"Error getting response: {e}")
                break

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
