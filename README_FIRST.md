# ⚠️ READ THIS FIRST: Prerequisites & Installation Notes

This document tracks critical technical constraints, version-specific hacks, and mandatory prerequisites for the Self-Hosted RAG Pipeline. **Read this before performing a fresh installation or upgrading packages.**

---

## 🛠️ Mandatory Prerequisites

### 1. System Dependencies (Host & Docker)
The following must be available on the host (if running locally) and are baked into `Dockerfile.worker`:
- **`ffmpeg`**: Required for WhisperX audio extraction from MP4/MP3.
- **`poppler-utils`**: Required for PDF rendering to images (OCR fallback).
- **`tesseract-ocr`**: Required for the EasyOCR/Tesseract fallback layer.
- **`libgl1` / `libglib2.0-0`**: Required for OpenCV/Vision processing.

### 2. Local Models & Air-Gap (100% Offline Runtime)
To ensure 100% local operation without internet requests:
- **`HF_HUB_OFFLINE=1`**: Hard-coded into Docker image and handlers to kill all HuggingFace network requests.
- **Docling Models**: Baked directly into the Docker image during the build-time **warmup** phase. No runtime download required.
- **WHISPER_MODEL_PATH**: You must still mount your local Whisper models (e.g., `faster-whisper-large-v2`) and point this variable to that directory. The system will look for `model.bin`, `tokenizer.json`, etc., in this path.

### 3. GPU Configuration
- **NVIDIA Container Toolkit**: Must be installed on the host.
- **`shm_size`**: The `ingest-dockercompose.yaml` defines `2gb`. If you process very large videos, ensure your Docker daemon supports this allocation to prevent "Bus Error" crashes.

---

## ⚙️ Environment Configuration

You MUST export the following variables in your shell or `.env` file before launching the stack.

### 1. Mandatory Ingestion Paths
```bash
# The absolute root for all document stages (staging, preprocessing, ingestion, etc.)
export DEFAULT_DOC_INGEST_ROOT=/home/samueldoyle/Projects/GitHub/SamD/selfhosted-rag-doc-chat-prototype/Docs/TESTING

# [DEPRECATED but supported] Standard paths
export INGEST_FOLDER=/home/samueldoyle/Projects/GitHub/SamD/selfhosted-rag-doc-chat-prototype/Docs
export STAGING_FOLDER=/home/samueldoyle/Projects/GitHub/SamD/selfhosted-rag-doc-chat-prototype/Docs/staging

# Absolute path to the e5-large-v2 embedding model directory
export EMBEDDING_MODEL_PATH=/home/samueldoyle/AI_LOCAL/e5-large-v2

# Absolute path to the local Whisper model directory
export WHISPER_MODEL_PATH=/home/samueldoyle/AI_LOCAL/Models/Whisper
```

### 2. Hybrid Inference (Local vs. Remote)
The system uses an OpenAI-compatible client. This allows `LLM_PATH` and `SUPERVISOR_LLM_PATH` to be used in two modes:

#### **A. Local GGUF Mode (llama-cpp-python)**
Set the path to a physical file on your disk.
```bash
export LLM_PATH=/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf
export SUPERVISOR_LLM_PATH=/home/samueldoyle/AI_LOCAL/Models/Phi/microsoft_Phi-4-mini-instruct-Q6_K.gguf
```

#### **B. Remote API Mode (Llama-Server / Ollama / AMD)**
Set an HTTP(S) URL. The system will automatically switch to API mode and connect to the remote host.
```bash
# Example for a remote Ollama or llama-server instance
export LLM_PATH=http://192.168.30.70:11434/v1
export SUPERVISOR_LLM_PATH=http://192.168.30.70:11434/v1
```

---

## 📦 Critical Version Constraints

### 🧨 The "Torch Trap" (2.5.1 vs 2.11.0+)
We strictly pin the Torch family to **v2.5.1**.
- **The Issue**: Newer versions (e.g., v2.11.0+) have moved or removed `torchaudio.AudioMetaData`. 
- **The Consequence**: `pyannote-audio` (a core WhisperX dependency) will crash at import time because it expects this class to be at the root namespace.
- **The Fix**: 
    1. Keep `pyproject.toml` pins active.
    2. We utilize a **Monkey-Patch** in `mp4_handler.py` and `mp3_handler.py` that injects a compatible `NamedTuple` if the environment ever drifts.

### 🧩 Protobuf Compatibility
- **Pinned**: `protobuf<4.21.0`.
- **Reason**: Higher versions cause "TypeError: Descriptors" crashes when using Llama-CPP and Transformers simultaneously.

---

## 🧬 Architectural Guardrails

### 1. Zero-Drop Policy (Mathematical Parity)
Chunks are **Hard-Truncated** at 511 tokens if they exceed the 512-token limit. Never delete the `TextProcessor.validate_chunk` truncation logic.

### 2. Database Hardening
- **Retries**: All DuckDB operations are wrapped in a **20-retry exponential backoff**. 
- **Initialization**: `init_db()` is a startup task only.

### 3. Unified Prompting
Use **User Role ONLY** with **Zero Leading Indentation** for 0.5B / 3B models to prevent "Note:" hallucinations.

---

## 🚀 Fresh Installation Flow
1. Install `uv`: `curl -LsS https://astral.sh/uv/install.sh | sh`
2. Sync Environment: `uv sync --all-extras`
3. Launch Stack: `./doc-ingest-chat/run-compose.sh --build`
