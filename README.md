# Aura-Local 🌌 (GPU Optimized Edition)

This is a specialized branch of **Aura-Local**, specifically optimized for high-performance interaction using NVIDIA GPUs (CUDA). It features real-time sentence-level streaming for both LLM responses and TTS generation, providing a near-zero latency conversational experience.

## 🚀 GPU Enhancements

- **Sentence-Level Streaming**: Aura starts speaking as soon as the first sentence is generated.
- **GPU Accelerated TTS**: Uses `onnxruntime-gpu` to run Kokoro-ONNX on CUDA.
- **Improved STT**: Uses `faster-whisper` on CUDA with the `base.en` model for better accuracy than the CPU version.
- **Self-Learning Memory**: Aura summarizes your conversations on exit and remembers key facts about you next time she starts.
- **Filesystem & OS Tools**: Aura can list directories, read/write files, and control system volume.
- **Smart RAG**: Enhanced Knowledge Base with Cosine Similarity and relevancy thresholds to prevent hallucinations.

## 🛠️ Prerequisites

1.  **NVIDIA GPU**: Recommended 8GB+ VRAM (Developed on RTX 5060 Ti 16GB).
2.  **Ollama**: Install [Ollama](https://ollama.com/) and pull the models:
    ```bash
    ollama pull goekdenizguelmez/JOSIEFIED-Qwen3
    ollama pull nomic-embed-text
    ```
3.  **Drivers & CUDA**: 
    - Ensure NVIDIA drivers and CUDA Toolkit 12.x are installed.
    - Install additional CUDA libraries for Whisper:
      ```bash
      pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
      ```
4.  **System Library**: 
    - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install libportaudio2`
    - **Fedora**: `sudo dnf install portaudio`
    - **Arch**: `sudo pacman -S portaudio`

## 🚀 Installation

1.  **Clone and Branch**:
    ```bash
    git clone https://github.com/jkstl/aura-local.git
    cd aura-local
    git checkout gpu-version
    ```
2.  **Set up Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
    ```

## 🎮 Usage

Run Aura:
```bash
python aura.py
```

### Automatic Checks
Aura now automatically verifies your environment on startup, checking for:
- PortAudio installation.
- Python package dependencies.
- CUDA availability for both PyTorch and ONNX.
- Ollama connectivity.

## 📜 Credits
- **LLM**: Qwen3 (via Ollama)
- **TTS**: Kokoro-ONNX
- **STT**: Faster-Whisper
- **Vector DB**: ChromaDB
