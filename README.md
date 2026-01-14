# Aura-Local 🌌

A private, local AI voice assistant powered by **Ollama** (LLM), **Faster-Whisper** (STT), and **Kokoro-ONNX** (TTS). Aura can see your files, use tools on your computer, and talk back to you—all 100% locally.

## ✨ Features

- **Voice-to-Voice**: Full conversational flow (Speak -> STT -> LLM -> TTS).
- **Tool Use (Functions)**: Aura can check the time, open websites in your browser, and monitor your system CPU/RAM.
- **RAG (Knowledge Base)**: Drop `.txt`, `.md`, or `.pdf` files into the `knowledge/` folder, and Aura will index them using **ChromaDB** to answer your questions.
- **Customizable Persona**: Edit `system_prompt.txt` to define Aura's personality and behavior.
- **Privacy First**: Everything runs on your machine. No data leaves your local network.

## 🛠️ Prerequisites

1.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the required models:
    ```bash
    ollama pull artifish/llama3.2-uncensored
    ollama pull nomic-embed-text
    ```
2.  **Python 3.10+**
3.  **System Dependencies**: Aura requires **PortAudio** for sound interaction.
    - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install libportaudio2`
    - **Fedora**: `sudo dnf install portaudio`
    - **Arch**: `sudo pacman -S portaudio`
4.  **GPU Acceleration (Optional but Recommended)**:
    - Ensure you have the latest **NVIDIA Drivers** installed.
    - If using CUDA 12, you may need to install additional libraries for Faster-Whisper:
      ```bash
      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
      ```

> **Note**: Aura now automatically checks for these dependencies on startup and will attempt to use your GPU (CUDA) for STT and TTS.

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/jkstl/aura-local.git
    cd aura-local
    ```
2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Usage

Run Aura:
```bash
python aura.py
```

### Modes
- **1. Text Chat**: Standard terminal chat.
- **2. Voice Chat**: Press **Enter** to start recording, speak (up to 30s), and press **Enter** to send.

### Tools & RAG
- **Tools**: Try asking "What time is it?" or "Open google.com".
- **Knowledge Base**: Place your documents in the `knowledge/` directory. Aura will index them on startup.

## ⚙️ Customization

- **System Prompt**: Edit `system_prompt.txt` to change how Aura responds.
- **Embedding Model**: Default is `nomic-embed-text` (Ollama).
- **CLI Options**:
  ```bash
  python aura.py --model llama3.2 --voice af_bella --speed 1.0
  ```

## 📜 License
MIT
