import gc
import os

from dotenv import load_dotenv

load_dotenv()

_models_dir = os.path.abspath(os.getenv("MODELS_DIR", "models"))
os.environ["HF_HOME"] = os.path.join(_models_dir, ".cache", "huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chromadb
import nltk
import torch
import uvicorn
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

from utils.model import Model
from utils.template_media import extract_keywords, hierarchical_retrieval, similarity_matching

RAG_DIR = os.getenv("RAG_DIR", os.path.join("outputs", "RAG"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.model_cfg = {}

    def load_config(self, cfg_file: str):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                self.model_cfg = json.load(f)
            logger.info(f"Loaded config for {len(self.model_cfg)} models")
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")
            raise

    def get_or_create_model(self, model_name: str) -> Model:
        if model_name not in self.models:
            cfg = self.model_cfg[model_name]
            logger.info(f"Creating new Model instance for {model_name}")
            self.models[model_name] = Model(cfg=cfg, model_name=model_name)
        return self.models[model_name]

    def has_enough_resources(self, model: Model, device_type: str) -> bool:
        try:
            if device_type == "gpu":
                return model.can_fit_on_gpu()
            elif device_type == "hybrid":
                return model.can_fit_on_gpu() and model.can_fit_on_ram()
            return model.can_fit_on_ram()
        except Exception as e:
            logger.exception(f"Error checking resources: {e}")
            return False

    def unload_other_models(self, keep_name: str):
        for name, model in self.models.items():
            if name != keep_name and model is not None and model._model is not None:
                logger.info(f"Unloading model: {name}")
                model.unload()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def load_model_by_name(self, model_name: str):
        """Load a model, unloading others if needed. Raises HTTPException on failure."""
        model = self.get_or_create_model(model_name)
        if model._model is not None:
            self.active_model_name = model_name
            return

        cfg = self.model_cfg[model_name]
        device_type = cfg.get("device", "cpu")

        if not self.has_enough_resources(model, device_type):
            self.unload_other_models(keep_name=model_name)

        if self.has_enough_resources(model, device_type):
            model.model_load()
            self.active_model_name = model_name
            logger.info(f"Loaded model: {model_name}")
        else:
            raise HTTPException(
                status_code=507,
                detail=f"Not enough resources to load {model_name} on {device_type}",
            )

    def chunk_text(self, text: str, model_name: str, chunk_size: int = 216, overlap: int = 10) -> List[str]:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        sentences = nltk.sent_tokenize(text)
        model = self.models[model_name]
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            num_tokens = model.count_tokens(sentence)
            if current_len + num_tokens > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                if overlap > 0:
                    overlap_text = chunks[-1].split()[-overlap:]
                    current_chunk = [" ".join(overlap_text)]
                    current_len = model.count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_len = 0
            current_chunk.append(sentence)
            current_len += num_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


class GlobalModels:
    def __init__(self):
        self.audio_model = None
        self.embedding_model = None
        self.embedding_function = None
        self.chroma_client = None
        self.collections = {}

    def load_embedding_fn_model(self, cfg: dict):
        device = cfg.get("device", "cpu")
        logger.info(f"Loading embedding model on {device}...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            cfg["model_name"], device=device
        )
        self.embedding_model = self.embedding_function._model
        os.makedirs(RAG_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=RAG_DIR, settings=Settings(anonymized_telemetry=False))

    def get_user_collection(self, user: str):
        if user not in self.collections:
            self.collections[user] = self.chroma_client.get_or_create_collection(
                name=f"user_{user}_doc_data",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self.collections[user]

    def load_audio_model(self, cfg: dict):
        device = cfg.get("device", "cpu")
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading stt model on {device}...")
        self.audio_model = WhisperModel(cfg["model_name"], device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.audio_model.transcribe(
            audio_path,
            vad_filter=True,
            beam_size=5,
            language=None,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=True,
        )
        return "".join(seg.text for seg in segments).strip()


global_models = GlobalModels()
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        hidden_cfg_path = os.getenv("HIDDEN_CONFIG", os.path.join("config", "hidden_model_cfg.json"))
        with open(hidden_cfg_path, "r", encoding="utf-8") as f:
            hidden_cfg = json.load(f)

        cfg_path = os.getenv("MODEL_CONFIG", os.path.join("config", "model_cfg.json"))
        model_manager.load_config(cfg_path)

        first_model_name = next(iter(model_manager.model_cfg))

        await asyncio.gather(
            asyncio.to_thread(global_models.load_embedding_fn_model, hidden_cfg["embedding"]),
            asyncio.to_thread(global_models.load_audio_model, hidden_cfg["stt"]),
            asyncio.to_thread(model_manager.load_model_by_name, first_model_name),
        )

        logger.info("✅ API started successfully")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")

    yield

    logger.info("Shutting down...")


app = FastAPI(title="Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response models ────────────────────────────────────────────────────


class LoadModelRequest(BaseModel):
    model_name: str


class GenerateRequest(BaseModel):
    model_name: str
    template: List[dict]
    max_tokens: int = 1024
    temperature: float = 0.3
    top_p: float = 1.0
    stream: bool = True


class GenerateResponse(BaseModel):
    response: Optional[str] = None


class CountTokensRequest(BaseModel):
    model_name: str
    text: Union[str, List[Dict[str, Any]]]


class AddDocumentRequest(BaseModel):
    model_name: str
    user: str
    chat_id: str
    file_name: str
    summary: str = ""
    content: str


class RAGQueryRequest(BaseModel):
    user: str
    query: str
    chat_docs: list = None
    top_n: int = 7
    threshold: float = 0.5


class TranscribeRequest(BaseModel):
    audio_path: str


class SimCheckRequest(BaseModel):
    model_name: str
    user: str
    all_doc: list = None
    text_doc: str = ""


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_loaded_model(model_name: str) -> Model:
    model = model_manager.models.get(model_name)
    if model is None or model._model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    return model


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)


@app.post("/rag/add_document")
async def add_document(request: AddDocumentRequest):
    try:
        collection = global_models.get_user_collection(request.user)
        file_path = os.path.join(RAG_DIR, "DOCUMENT_METADATA.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                document_metadata = json.load(f)
        except FileNotFoundError:
            document_metadata = {}
        except json.JSONDecodeError:
            raise ValueError(f"Failed to load JSON: {file_path}")

        if not isinstance(document_metadata, dict):
            raise ValueError("JSON root is not a dict")

        chunks = model_manager.chunk_text(request.content, request.model_name, chunk_size=232, overlap=32)
        logger.info(f"Created {len(chunks)} chunks")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"{request.file_name}_{timestamp}"
        keywords = extract_keywords(chunks, top_n=10)

        document_metadata[doc_id] = {
            "user": request.user,
            "chat_id": request.chat_id,
            "file_name": request.file_name,
            "summary": request.summary,
            "upload_time": timestamp,
            "total_chunks": len(chunks),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(document_metadata, f, ensure_ascii=False)

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "chat_id": request.chat_id,
                "chunk_index": i,
                "keywords": ",".join(keywords[i]),
            }
            for i in range(len(chunks))
        ]

        collection.add(ids=chunk_ids, documents=chunks, metadatas=chunk_metadatas)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Added document for {request.user} / chat: {request.chat_id}")
        return {"status": "success", "doc_id": doc_id, "chunks": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    try:
        results = hierarchical_retrieval(
            collection=global_models.get_user_collection(request.user),
            embedding_model=global_models.embedding_model,
            chat_docs=request.chat_docs,
            query=request.query,
            top_n=request.top_n,
            threshold=request.threshold,
            user=request.user,
        )
        return {"status": "success", "results": results}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/load")
async def load_model(request: LoadModelRequest):
    try:
        if request.model_name not in model_manager.model_cfg:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found in config")

        await asyncio.to_thread(model_manager.load_model_by_name, request.model_name)

        return {"status": "success", "model_name": request.model_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    try:
        model = _get_loaded_model(request.model_name)

        def generate():
            try:
                is_llama = model_manager.model_cfg[model_manager.active_model_name]["framework"] == "llama"
                for token in model.model_inf(
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=request.stream,
                ):
                    text = token["choices"][0]["text"] if is_llama else token
                    yield f"data: {json.dumps({'token': text})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.exception(f"Error during streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return StreamingResponse(generate(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in stream generate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    try:
        model = _get_loaded_model(request.model_name)

        full_response = "".join(
            model.model_inf(
                template=request.template,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=False,
            )
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return GenerateResponse(response=full_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/count_tokens")
async def count_tokens(request: CountTokensRequest):
    try:
        model = _get_loaded_model(request.model_name)
        return {"status": "success", "num_tokens": model.count_tokens(request.text)}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest):
    try:
        transcription = global_models.transcribe(request.audio_path)
        logger.info(f"Transcribed audio: {transcription}")
        return {"status": "success", "transcription": transcription}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_sim")
async def check_sim(request: SimCheckRequest):
    try:
        chunks = model_manager.chunk_text(
            text=request.text_doc, model_name=request.model_name, chunk_size=232, overlap=32
        )
        is_sim = similarity_matching(
            collection=global_models.get_user_collection(request.user),
            embedding_model=global_models.embedding_model,
            all_doc_ids=request.all_doc,
            chunks=chunks,
            threshold=0.9,
        )
        return {"status": "success", "is_sim": is_sim}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error checking similarity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "embedding_model_loaded": global_models.embedding_model is not None,
        "embedding_function_loaded": global_models.embedding_function is not None,
        "audio_model_loaded": global_models.audio_model is not None,
        "chat_model_loaded": model_manager.active_model_name is not None,
        "active_users": len(global_models.collections),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
