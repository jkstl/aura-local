import os
# Suppress Hugging Face warnings and telemetry
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

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
import queue
import re
from chromadb.config import Settings
from pypdf import PdfReader
from faster_whisper import WhisperModel
import scipy.io.wavfile as wav
import subprocess
import importlib.util

# Constants
DEFAULT_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
DEFAULT_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
MODEL_FILE = "kokoro-v1.0.int8.onnx"
VOICE_FILE = "voices-v1.0.bin"
SYSTEM_PROMPT_FILE = "system_prompt.txt"
OLLAMA_MODEL = "goekdenizguelmez/JOSIEFIED-Qwen3" 
KNOWLEDGE_DIR = "knowledge"
EMBED_MODEL = "nomic-embed-text"

class KnowledgeBase:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path="./aura_db")
        # Use Cosine Similarity for better semantic relevance
        self.collection = self.chroma_client.get_or_create_collection(
            name="aura_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        
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
        # Chatter Filter: Don't query DB for simple greetings or short phrases
        # This prevents RAG from hallucinating context for "hi"
        low_text = text.lower().strip()
        common_greetings = ["hi", "hello", "hey", "hola", "sup", "yo", "ok", "cool", "thanks", "thank you", "bye", "exit"]
        if len(low_text.split()) < 4 and any(g in low_text for g in common_greetings):
            return []

        try:
            embedding_resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
            embedding = embedding_resp['embedding']
            
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )
            
            # Filter results by distance
            valid_docs = []
            if results['documents'] and results['distances']:
                for doc, dist in zip(results['documents'][0], results['distances'][0]):
                    # With cosine: 0.0 is identical, 1.0 is unrelated, 2.0 is opposite.
                    # Relaxed to 0.55 to catch 'favorite color matches' which sat around 0.46
                    if dist < 0.55: 
                        valid_docs.append(doc)
                    # print(f"      [DEBUG: Distance check - {dist:.4f}]") # Enable for debugging
            
            return valid_docs
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
        
        print("Loading Aura's voice engine (GPU Accelerated)...")
        try:
            # Added providers for GPU acceleration
            self.kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
        except Exception as e:
            print(f"Error loading Kokoro engine: {e}")
            sys.exit(1)

        print("Loading Aura's ears (STT - GPU Accelerated)...")
        try:
            # Using 'base.en' for better accuracy since we have GPU power
            # Using 'cuda' device and 'float16' for speed on RTX 50-series
            self.stt_model = WhisperModel("base.en", device="cuda", compute_type="float16")
        except Exception as e:
            print(f"Error loading STT model: {e}")
            print("Falling back to CPU for STT...")
            self.stt_model = WhisperModel("base.en", device="cpu", compute_type="int8")

        # Session history for long-term memory
        self.session_history = []
        
        # Audio playback queue and thread
        self.audio_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()

    def _playback_worker(self):
        """Background thread that plays audio chunks from the queue."""
        while True:
            samples, sample_rate = self.audio_queue.get()
            if samples is None: # Exit signal
                break
            try:
                sd.play(samples, sample_rate)
                sd.wait()
            except Exception as e:
                print(f"Error playing audio chunk: {e}")
            self.audio_queue.task_done()

    def speak_chunk(self, text):
        """Generates audio for a single sentence and adds to playback queue."""
        if not text.strip():
            return
        try:
            samples, sample_rate = self.kokoro.create(
                text, 
                voice=self.voice, 
                speed=self.speed, 
                lang=self.lang
            )
            self.audio_queue.put((samples, sample_rate))
        except Exception as e:
            print(f"Error generating audio for chunk: {e}")

    def speak(self, text):
        """Deprecated: Use speak_chunk for streaming or call this for one-off messages."""
        self.speak_chunk(text)
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
                    'description': 'Returns the current local date and time.',
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'open_url',
                    'description': 'Opens a given URL in the default web browser.',
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
                    'description': 'Returns current system status like CPU, memory, and battery usage.',
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'list_directory',
                    'description': 'Lists files and folders in a specific directory.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type': 'string',
                                'description': 'The path to list (defaults to \'.\')',
                            },
                        },
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'read_file',
                    'description': 'Reads the content of a text-based file.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type': 'string',
                                'description': 'The path to the file to read.',
                            },
                        },
                        'required': ['path'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'write_file',
                    'description': 'Writes or appends a text snippet to a file.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'path': {
                                'type': 'string',
                                'description': 'The path to the file.',
                            },
                            'content': {
                                'type': 'string',
                                'description': 'The text content to write.',
                            },
                            'mode': {
                                'type': 'string',
                                'description': 'Writing mode: \'w\' (overwrite) or \'a\' (append). Defaults to \'a\'.',
                                'enum': ['w', 'a']
                            }
                        },
                        'required': ['path', 'content'],
                    },
                }
            },
            {
                'type': 'function',
                'function': {
                    'name': 'set_volume',
                    'description': 'Adjusts the system volume.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'level': {
                                'type': 'integer',
                                'description': 'Volume level (0-100).',
                            },
                        },
                        'required': ['level'],
                    },
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

        print(f"  [Executing Tool: {name}]")
        
        if name == 'get_current_time':
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        elif name == 'open_url':
            url = args.get('url')
            if url:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                webbrowser.open(url)
                return f"Successfully opened {url}"
            return "Error: No URL provided."
        
        elif name == 'get_system_info':
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            battery = psutil.sensors_battery()
            batt_str = f", Battery: {battery.percent}%" if battery else ""
            return f"CPU Usage: {cpu}% | Memory Usage: {mem}%{batt_str}"
        
        elif name == 'list_directory':
            path = args.get('path', '.')
            try:
                items = os.listdir(path)
                return f"Contents of {path}: " + ", ".join(items)
            except Exception as e:
                return f"Error listing directory: {e}"

        elif name == 'read_file':
            path = args.get('path')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return f"Content of {path}:\n{content[:2000]}" # Limit to 2k chars
            except Exception as e:
                return f"Error reading file: {e}"

        elif name == 'write_file':
            path = args.get('path')
            content = args.get('content')
            mode = args.get('mode', 'a')
            try:
                with open(path, mode, encoding='utf-8') as f:
                    f.write(content + "\n")
                return f"Successfully {'written' if mode=='w' else 'appended'} to {path}"
            except Exception as e:
                return f"Error writing to file: {e}"

        elif name == 'set_volume':
            level = args.get('level')
            try:
                # Basic Linux amixer call
                subprocess.run(["amixer", "set", "Master", f"{level}%"], check=True)
                return f"System volume set to {level}%"
            except Exception as e:
                return f"Error setting volume: {e}"
        
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
            self._update_long_term_memory()
            print("\nGoodbye!")
            return

        if choice == '2':
            self.voice_chat_loop()
        else:
            self.text_chat_loop()
            
        self._update_long_term_memory()

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
        full_system_prompt = self.system_prompt
        if context_str:
            full_system_prompt += context_str

        if full_system_prompt:
            messages.append({'role': 'system', 'content': full_system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        self.session_history.append({'role': 'user', 'content': prompt})

        while True:
            full_response = ""
            sentence_buffer = ""
            tool_calls = []
            
            try:
                stream = ollama.chat(
                    model=self.ollama_model, 
                    messages=messages,
                    tools=self.available_tools,
                    stream=True,
                )
                
                header_printed = False
                
                for chunk in stream:
                    message = chunk['message']
                    
                    # 1. Handle Tool Calls
                    if message.get('tool_calls'):
                        tool_calls.extend(message['tool_calls'])
                    
                    # 2. Handle Content (Streaming to TTS)
                    if message.get('content'):
                        if not header_printed:
                            print("\nAura: ", end="", flush=True)
                            header_printed = True
                            
                        content = message['content']
                        full_response += content
                        sentence_buffer += content
                        print(content, end="", flush=True)

                        # Split by sentence endings
                        if any(c in content for c in ".!?\n"):
                            parts = re.split(r'(?<=[.!?])\s+|\n', sentence_buffer)
                            for i in range(len(parts) - 1):
                                sentence = parts[i].strip()
                                if sentence:
                                    self.speak_chunk(sentence)
                            sentence_buffer = parts[-1]

                # Finish any logic for this specific stream
                if sentence_buffer.strip():
                    self.speak_chunk(sentence_buffer.strip())
                
                if header_printed:
                    print() # End the Aura line
                    self.session_history.append({'role': 'assistant', 'content': full_response})

                # 3. If there were tool calls, execute them and RE-QUERY
                if tool_calls:
                    messages.append({'role': 'assistant', 'content': full_response, 'tool_calls': tool_calls})
                    for tool in tool_calls:
                        result = self._execute_tool(tool)
                        messages.append({
                            'role': 'tool',
                            'content': str(result),
                        })
                    continue # Re-run chat with tool results to get the final verbal response
                
                break # No tool calls, we are done

            except Exception as e:
                print(f"\nError getting response: {e}")
                break

    def _update_long_term_memory(self):
        """Summarizes the session and saves important facts to the knowledge base."""
        if not self.session_history:
            return

        print("\nUpdating Aura's long-term memory...")
        try:
            # Use a quick summary prompt
            # Load existing memory context (last 4KB to save context window)
            existing_memory = ""
            if os.path.exists(os.path.join(KNOWLEDGE_DIR, "memory.txt")):
                 with open(os.path.join(KNOWLEDGE_DIR, "memory.txt"), 'r', encoding='utf-8') as f:
                     existing_memory = f.read()[-4000:]

            # Strict prompt: Only remember if explicitly asked + Deduplicate
            summary_prompt = f"Below is a chat history between Aura and Jeff.\n\nEXISTING MEMORY:\n{existing_memory}\n\nINSTRUCTION:\nYou are a memory manager. Extract ONLY NEW facts where Jeff explicitly asks you to 'remember', 'save', 'note', or 'remind' him of something. Ignore all other conversation. IMPORTANT: If a fact is ALREADY in the EXISTING MEMORY above, DO NOT include it again. Return ONLY the new, unique facts. If no new explicit memory requests are found, return 'NO_NEW_FACTS'.\n\nNEW CHAT HISTORY:\n"
            for msg in self.session_history:
                summary_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
            resp = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': summary_prompt}]
            )
            
            summary = resp['message']['content'].strip()
            
            if "NO_NEW_FACTS" not in summary.upper():
                memory_file = os.path.join(KNOWLEDGE_DIR, "memory.txt")
                with open(memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n--- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                    f.write(summary + "\n")
                print("✅ Memory updated.")
            else:
                print("ℹ️ No new key facts to remember from this session.")
        except Exception as e:
            print(f"Warning: Could not update long-term memory: {e}")

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

def check_dependencies():
    """Check if required system and python dependencies are installed."""
    print("Checking dependencies...")
    
    # 1. Check for PortAudio (System dependency)
    # sounddevice will raise OSError if PortAudio is missing
    try:
        import sounddevice as sd
    except OSError as e:
        if "PortAudio library not found" in str(e):
            print("\n❌ Error: PortAudio library not found.")
            print("Please install it using your system package manager:")
            print("  Ubuntu/Debian: sudo apt-get update && sudo apt-get install libportaudio2")
            print("  Fedora: sudo dnf install portaudio")
            print("  Arch: sudo pacman -S portaudio")
            sys.exit(1)
        raise e

    # 2. Check Python packages from requirements.txt
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            required = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        missing = []
        # Mapping of package name in requirements.txt to its actual import name
        import_mapping = {
            "onnxruntime-gpu": "onnxruntime",
            "onnxruntime": "onnxruntime",
            "faster-whisper": "faster_whisper",
            "kokoro-onnx": "kokoro_onnx",
            "pypdf": "pypdf"
        }
        
        for package in required:
            clean_name = package.split('==')[0].split('>=')[0].strip()
            import_name = import_mapping.get(clean_name, clean_name.replace('-', '_'))
            
            if importlib.util.find_spec(import_name) is None:
                missing.append(package)
        
        if missing:
            print(f"\n❌ Missing Python packages: {', '.join(missing)}")
            print("Please install them using:")
            print("  pip install -r requirements.txt")
            sys.exit(1)
    
    # 3. Check if Ollama is running
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print("\n⚠️ Warning: Could not connect to Ollama.")
        print("Make sure Ollama is running (https://ollama.com).")

    # 4. Check for CUDA availability
    gpu_detected = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Acceleration (PyTorch): {torch.cuda.get_device_name(0)}")
            gpu_detected = True
    except ImportError:
        pass

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print(f"✅ GPU Acceleration (ONNX): CUDA detected")
            gpu_detected = True
        else:
            print(f"⚠️ Warning: ONNX Runtime CUDA provider not found. Providers: {providers}")
    except ImportError:
        pass

    if not gpu_detected:
        print("\n⚠️ Warning: No GPU acceleration detected (CUDA). Moving forward with CPU default.")

def main():
    parser = argparse.ArgumentParser(description="Aura-Local: AI Voice Assistant")
    parser.add_argument("--model", type=str, default="goekdenizguelmez/JOSIEFIED-Qwen3", help="Ollama model to use")
    parser.add_argument("--voice", type=str, default="af_bella", help="Kokoro voice to use")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--lang", type=str, default="en-us", help="Language code")
    
    args = parser.parse_args()
    
    check_dependencies()
    
    app = AuraAssistant(
        ollama_model=args.model,
        voice=args.voice,
        speed=args.speed,
        lang=args.lang
    )
    app.run()

if __name__ == "__main__":
    main()
