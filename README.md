# Chatbot — Version 3.0.0

A modular chatbot framework for local LLM/VLM inference, featuring RAG over uploaded documents, speech-to-text input, multimodal support, and a clean Streamlit UI backed by a FastAPI server.

---

## What's New in v3.0.0

- **Retrieval-Augmented Generation (RAG):**
  Upload documents (PDF, DOCX, XLSX, PPTX, CSV, HTML, JSON, TXT, MD, RTF…) and the chatbot retrieves relevant excerpts to ground its answers. Powered by ChromaDB with a multilingual embedding model (`intfloat/multilingual-e5-small`) and a hierarchical retrieval pipeline combining semantic and keyword scoring. Includes near-duplicate detection to avoid re-indexing the same document.

- **Speech-to-Text (STT):**
  Record audio directly in the chat input. Transcription runs locally via `faster-whisper` (`large-v3-turbo-int8`) with automatic language detection and VAD filtering.

- **Context Window Management:**
  When a conversation exceeds the model's context limit, history is automatically summarized so the model never loses context mid-conversation.

- **Docker Support:**
  Full Docker + CUDA setup via `docker compose up --build`. ChromaDB and model weights persist via volumes.

- **Path Configuration via `.env`:**
  All system paths (model cache, RAG store, config files) are configurable through a `.env` file. Defaults to `models/` in the project root.

- **Parallel Startup Loading:**
  STT model, embedding model, and chat model all load in parallel at startup via `asyncio.gather`, reducing cold-start time.

- **Bug Fixes:**
  - Fixed VLM tensor device mismatch (`RuntimeError: Expected all tensors on same device`)
  - Fixed token counting and context overflow handling
  - Fixed temp file cleanup after media/audio processing
  - Fixed chat history saving and loading
  - Fixed model unloading and memory release between model switches

---

## Features

- Text, image, and video chat (VLM mode)
- Document Q&A via RAG (PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML, TXT, MD, RTF)
- Voice input with automatic transcription (STT)
- Streaming token-by-token responses
- Automatic conversation summarization when context limit is reached
- Plug-and-play model configuration via `config/model_cfg.json`
- Adjustable generation parameters (temperature, top-p, max tokens, RAG threshold)
- Recent chats sidebar with auto-generated titles
- Local-first, fully offline capable after initial model download
- Docker-ready with GPU passthrough

---

## Architecture

```
chatbot/
├── app.py                  # Streamlit UI
├── api_server.py           # FastAPI backend (inference, RAG, STT, chunking)
├── launch.py               # Launcher — starts API then Streamlit
├── Dockerfile
├── docker-compose.yml
├── .env                    # Path overrides (optional)
├── config/
│   ├── model_cfg.json      # Chat model registry
│   ├── hidden_model_cfg.json  # Background models (STT, embedding)
│   └── system_prompt.txt
├── utils/
│   ├── model.py            # Model loading & inference abstraction
│   ├── api_client.py       # HTTP client for the API server
│   └── template_media.py   # Prompt building, RAG, media processing, document parsing
├── models/                 # GGUF files + HF cache
└── outputs/RAG/            # ChromaDB vector store + document metadata
```

---

## Models

| Name | Type | Device | Framework | Context |
|------|------|--------|-----------|---------|
| **Qwen3-VL-2B-gpu** | VLM | GPU | transformers | 256K |
| **Mistral-7B-v0.2-gpu** | LLM | Hybrid (GPU+CPU) | ctransformers | 4096 |
| **Mistral-7B-v0.2-cpu** | LLM | CPU | ctransformers | 4096 |

Background models loaded at startup:

| Role | Model |
|------|-------|
| Embeddings (RAG) | `intfloat/multilingual-e5-small` |
| Speech-to-Text | `Zoont/faster-whisper-large-v3-turbo-int8-ct2` |

Models are downloaded automatically on first use via Hugging Face into `models/.cache/huggingface/`.

---

## Installation

### Local

```bash
git clone https://github.com/yaghmo/chatbot.git
cd chatbot
conda create -n chatbot python=3.11 -y
conda activate chatbot
pip install -r requirements.txt
python launch.py
```

### Docker (GPU)

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

---

## Configuration

Copy `.env` and adjust paths if needed:

```env
MODELS_DIR=./models
RAG_DIR=./outputs/RAG
CONFIG_DIR=./config
MODEL_CONFIG=./config/model_cfg.json
HIDDEN_CONFIG=./config/hidden_model_cfg.json
```

To add or swap a model, edit `config/model_cfg.json`. To change the STT or embedding model, edit `config/hidden_model_cfg.json`.
