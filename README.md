# Aura-Local

A local AI voice assistant powered by [Ollama](https://ollama.com/) (LLM) and [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx) (TTS).

## Features
- **Local LLM**: Uses `artifish/llama3.2-uncensored` by default for uncensored responses.
- **High-Quality TTS**: Uses Kokoro ONNX for fast, high-quality local speech synthesis.
- **Configurable**: Change voices, models, and more via CLI arguments.

## Prerequisites
1. **Ollama**: You must have [Ollama](https://ollama.com/) installed and running.
   ```bash
   ollama pull artifish/llama3.2-uncensored
   ```
2. **Python 3.10+**
3. **System Dependencies**: You might need `portaudio` (for `sounddevice`).
   - Ubuntu/Debian: `sudo apt-get install libportaudio2`

## Installation

1. Clone or navigate to the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the assistant:
```bash
python aura.py
```

The first time you run it, it will automatically download the required Kokoro model files (~100MB).

### Options
```bash
python aura.py --help
```
- `--model`: Change the Ollama model (default: `artifish/llama3.2-uncensored`)
- `--voice`: Change the TTS voice (default: `af_bella`)
- `--speed`: Adjust speech speed (default: `1.0`)
