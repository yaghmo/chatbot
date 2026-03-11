# Chatbot — Version 3.0.0

A modular chatbot framework for local LLM/VLM inference, featuring RAG over uploaded documents, speech-to-text input, multimodal support, and a clean Streamlit UI backed by a FastAPI server.

---

## What's New in v3.0.0

- **Retrieval-Augmented Generation (RAG):**
  Upload documents (PDF, DOCX, XLSX, PPTX, CSV, HTML, JSON, TXT, MD, RTF…) and the chatbot will retrieve relevant excerpts to ground its answers. Powered by ChromaDB with a multilingual embedding model (`intfloat/multilingual-e5-small`) and a hierarchical retrieval pipeline (semantic + keyword scoring).

- **Speech-to-Text (STT):**
  Record audio directly in the chat input. Transcription is handled locally by `faster-whisper` (`large-v3-turbo-int8`), with automatic language detection and VAD filtering.

- **Context Window Management:**
  When a conversation grows too long, the history is automatically summarized so the model never loses context.

- **Duplicate Document Detection:**
  Before indexing a new file, the system computes its embedding similarity against already-uploaded documents and skips near-duplicates.

- **Bug Fixes:**
  - Fixed token counting and context overflow handling.
  - Fixed temp file cleanup after media/audio processing.
  - Fixed chat history saving and loading.
  - Fixed model unloading/memory release between model switches.

---

## Features

- Text, image, and video chat (VLM mode)
- Document Q&A via RAG (PDF, DOCX, XLSX, PPTX, CSV, JSON, HTML, TXT, MD, RTF)
- Voice input with automatic transcription (STT)
- Streaming token-by-token response
- Automatic conversation summarization when context limit is reached
- Plug-and-play model configuration via `config/model_cfg.json`
- Adjustable generation parameters (temperature, top-p, max tokens, RAG threshold)
- Recent chats sidebar with auto-generated titles
- Local-first, fully offline capable

---

## Architecture

```
chatbot/
├── app.py              # Streamlit UI
├── api_server.py       # FastAPI backend (inference, RAG, STT)
├── launch.py           # Launcher (starts both servers)
├── config/
│   ├── model_cfg.json  # Model registry
│   └── system_prompt.txt
├── utils/
│   ├── model.py        # Model loading & inference abstraction
│   ├── api_client.py   # HTTP client for the API server
│   └── template_media.py  # Prompt building, RAG retrieval, media processing, document parsing
├── models/             # Local GGUF model files
└── outputs/RAG/        # ChromaDB vector store + document metadata
```

---

## Models

| Name | Type | Device | Framework | Context |
|------|------|--------|-----------|---------|
| **Qwen3-VL-2B-gpu** | VLM | GPU | transformers | 5000 |
| **Mistral-7B-v0.2-gpu** | LLM | Hybrid (GPU+CPU) | ctransformers | 4096 |
| **Mistral-7B-v0.2-cpu** | LLM | CPU | ctransformers | 4096 |

Models are downloaded automatically on first use via Hugging Face.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yaghmo/chatbot.git
cd chatbot
```

### 2. Create environment
```bash
conda create -n chatbot python=3.10 -y
conda activate chatbot
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the app
```bash
python launch.py
```

This starts the FastAPI backend and the Streamlit UI together.

---