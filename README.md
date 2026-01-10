# Chatbot — Version 2.0.0

A modular chatbot framework designed for easy experimentation with LLMs, VLMs, multimodal inputs, and customizable pipelines.  
This project is the evolution of the original `chatbot` repository and introduces GPU‑accelerated vision capabilities, attachment handling, and a cleaner architecture for future development.

---

## What's New in v2.0.0

- **Vision-Language Model Integration (VLM):**  
  Added support for **Qwen V3 2B** running on GPU for fast and accurate multimodal reasoning: image understanding, OCR-like extraction, captioning, and more.

- **Attachment Input Handling:**  
  You can now upload images or documents directly; the system processes them and routes them through the appropriate model pipeline.

- **Improved Modular Architecture:**  
  Cleaner separation between components (UI, inference backend, runtime logic).
---

## Features

- Text‑only and multimodal chat
- GPU‑accelerated inference
- Plug‑and‑play model configuration
- Support for attachments (images and videos)
- Local and cloud‑ready deployment
- Fully open‑source and easy to customize

---

## 🛠 Installation

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

---

## 🧠 Models Included

| Type | Model | Notes |
|------|--------|--------|
| LLM | **Mistral 7B (gguf)** | Local or API-based |
| VLM | **Qwen V3 2B** | GPU‑accelerated multimodal model |

You can replace or extend models via the `models/` directory.

---

## Examples

![alt text](src/image.png)