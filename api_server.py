from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional, List, Dict, Union, Any
import json
import os
import torch
from utils.model import Model
from utils.template_media import extract_keywords, hierarchical_retrieval, similarity_matching
import uvicorn
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state for models
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
    
    def unload_other_models(self, keep_name: str):
        import gc
        for name, model in self.models.items():
            if name != keep_name and model is not None and model._model is not None:
                logger.info(f"Unloading model: {name}")
                model.unload()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    
    def chunk_text(self, text: str, model_name, chunk_size: int = 216, overlap: int = 10) -> List[str]:    
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            sentences = nltk.sent_tokenize(text)
            
            chunks = []
            current_chunk =[]
            current_len = 0
            model = self.models[model_name]
            
            for sentence in sentences:
                num_tokens = model.count_tokens(sentence)
                if current_len + num_tokens > chunk_size:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                        if overlap > 0 and chunks:
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
    
    def load_embedding_fn_model(self, embedding_fn_model):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = 'cpu'
        logger.info(f"Loading embedding model on {device}...")
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            embedding_fn_model,
            device=device
        )
        self.embedding_model = self.embedding_function._model
        rag_db_path = os.path.join("outputs", "RAG")
        os.makedirs(rag_db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=rag_db_path,settings=Settings(anonymized_telemetry=False))
    
    def get_user_collection(self, user: str):
        """Get or create collection for a user"""
        if user not in self.collections:
            collection = self.chroma_client.get_or_create_collection(
                name=f"user_{user}_doc_data",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self.collections[user] = collection
        logger.info(f"Collection for the user: {user} is loading...")
        return self.collections[user]

    def load_audio_model(self, model_name):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = 'cpu'
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info(f"Loading stt model on {device}...")
        self.audio_model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path):
        segments, info = self.audio_model.transcribe(
            audio_path,
            vad_filter=True,
            beam_size=5,
            language=None,
            task="transcribe",
            temperature=0.0,
            condition_on_previous_text=True
        )
        return "".join(seg.text for seg in segments).strip()


global_models = GlobalModels()
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        global_models.load_embedding_fn_model("intfloat/multilingual-e5-small")
        # global_models.load_audio_model("Systran/faster-distil-whisper-large-v3")
        global_models.load_audio_model("Zoont/faster-whisper-large-v3-turbo-int8-ct2")
        model_manager.load_config(os.path.join("config", "model_cfg.json"))

        logger.info("✅ API started successfully")
    except Exception as e:
        logger.exception(f"Startup failed: {e}")
    
    yield
    
    logger.info("Shutting down...")

app = FastAPI(title="Chatbot API", lifespan=lifespan)

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
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

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = "static/favicon.ico"
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return Response(status_code=204)  

@app.post("/rag/add_document")
async def add_document(request: AddDocumentRequest):
    """Add document to user's RAG collection"""
    try:
        collection = global_models.get_user_collection(request.user)
        logger.info(f"Received collection for: {request.user}")
        # Use global embedding model for chunking
        from datetime import datetime
        
        dir_path = os.path.join("outputs","RAG")
        file_path = os.path.join(dir_path,"DOCUMENT_METADATA.json")

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    DOCUMENT_METADATA = json.load(f)
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to load JSON: {file_path}")
        else:
            DOCUMENT_METADATA = {}

        if not isinstance(DOCUMENT_METADATA, dict):
            raise ValueError("JSON root is not a dict")


        chunks = model_manager.chunk_text(request.content,request.model_name,chunk_size=232,overlap=32)
        logger.info(f"Created: {len(chunks)} chunks")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"{request.file_name}_{timestamp}"
        chunk_ids = []
        chunk_texts = []
        chunk_metadatas = []
        keywords = extract_keywords(chunks, top_n=10)

        DOCUMENT_METADATA[doc_id] = {
            "user": request.user,
            "chat_id": request.chat_id,
            "file_name": request.file_name,
            "summary": request.summary,
            "upload_time": timestamp,
            "total_chunks": 0  
        }
        DOCUMENT_METADATA[doc_id]["total_chunks"] = len(chunks)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(DOCUMENT_METADATA, f, ensure_ascii=False)
        logger.info(f"Successfully updated {file_path}")


        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"

            chunk_ids.append(chunk_id)
            chunk_texts.append(chunk)
            chunk_metadatas.append({
                "doc_id":doc_id,
                "chunk_id":chunk_id,
                "chat_id":request.chat_id,
                "chunk_index": i,
                "keywords": ",".join(keywords[i])
            })

        collection.add(
            ids=chunk_ids,
            documents=chunk_texts,
            metadatas=chunk_metadatas
        )
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Successfully added document for {request.user}/ chat: {request.chat_id}")
        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    """Query user's RAG collection"""
    try:
        collection = global_models.get_user_collection(request.user)
        
        results = hierarchical_retrieval(
            collection=collection,
            embedding_model=global_models.embedding_model,
            chat_docs=request.chat_docs,
            query=request.query,
            top_n=request.top_n,
            threshold=request.threshold,
            user=request.user,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/load")
async def load_model(request: LoadModelRequest):
    """Load a model into memory"""
    try:
        logger.info(f"Received request to load model: {request.model_name}")
        
        if request.model_name not in model_manager.model_cfg:
            raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found in config")
        

        model = model_manager.get_or_create_model(request.model_name)
        cfg = model_manager.model_cfg[request.model_name]
        
        logger.info(f"Model config: {cfg}")
        

        if model._model is not None:
            logger.info(f"Model {request.model_name} already loaded")
            model_manager.active_model_name = request.model_name
            return {
                "status": "success",
                "model_name": request.model_name,
                "message": f"Model {request.model_name} already loaded"
            }
        
        device_type = cfg.get("device", "cpu")
        logger.info(f"Device type: {device_type}")
        

        def has_enough():
            try:
                if device_type == "gpu":
                    result = model.can_fit_on_gpu()
                    logger.info(f"GPU check: {result}")
                    return result
                elif device_type == "cpu":
                    result = model.can_fit_on_ram()
                    logger.info(f"RAM check: {result}")
                    return result
                elif device_type == "hybrid":
                    gpu_ok = model.can_fit_on_gpu()
                    ram_ok = model.can_fit_on_ram()
                    logger.info(f"Hybrid check - GPU: {gpu_ok}, RAM: {ram_ok}")
                    return gpu_ok and ram_ok
                else:
                    result = model.can_fit_on_ram()
                    logger.info(f"Default RAM check: {result}")
                    return result
            except Exception as e:
                logger.exception(f"Error checking resources: {e}")
                return False
        
        # Try to load
        if has_enough():
            logger.info(f"Enough resources available, loading {request.model_name}")
            model.model_load()
        else:
            logger.info(f"Not enough resources, unloading other models")
            # Unload other models and try again
            model_manager.unload_other_models(keep_name=request.model_name)
            
            if has_enough():
                logger.info(f"After unloading, loading {request.model_name}")
                model.model_load()
            else:
                error_msg = f"Not enough resources to load {request.model_name} on {device_type}"
                logger.error(error_msg)
                raise HTTPException(status_code=507, detail=error_msg)
        
        
        model_manager.active_model_name = request.model_name
        logger.info(f"Successfully loaded {request.model_name}")
        
        return {
            "status": "success",
            "model_name": request.model_name,
            "message": f"Model {request.model_name} loaded successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.exception(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Generate text with streaming response"""
    try:
        logger.info(f"Stream generate - model: {request.model_name}, temp: {request.temperature}, top_p: {request.top_p}, max_tokens: {request.max_tokens}")
        
        if request.model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model = model_manager.models[request.model_name]
        
        if model._model is None:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        def generate():
            try:
                for token in model.model_inf(
                    template=request.template,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=request.stream
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n" if model_manager.model_cfg[model_manager.active_model_name]["framework"] != "llama" else f"data: {json.dumps({'token': token['choices'][0]['text']})}\n\n"

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
    """Generate text (non-streaming)"""
    try:
        logger.info(f"Generate request - model: {request.model_name}")
        
        if request.model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model = model_manager.models[request.model_name]
        
        if model._model is None:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        full_response = ""
        for token in model.model_inf(
            template=request.template,
            max_tokens=request.max_tokens,
            temperature=0.3,
            top_p=1,
            stream=False
        ):
            full_response += token
        
        logger.info(f"Generated: {full_response}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return GenerateResponse(
            response=full_response,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/count_tokens")
async def count_tokens(request: CountTokensRequest):
    """Count tokens in plain text"""
    try:
        if request.model_name not in model_manager.models:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        model = model_manager.models[request.model_name]

        num_tokens = model.count_tokens(request.text)
        
        logger.info(f"Token count for pure text: {num_tokens}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "success",
            "num_tokens": num_tokens,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe")
async def transcrbie(request: TranscribeRequest):
    try:
        transcription = global_models.transcribe(request.audio_path)

        logger.info(f"Transcripted audio: {transcription}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "transcription": transcription
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/check_sim")
async def check_sim(request: SimCheckRequest):
    try:
        chunks = model_manager.chunk_text(text=request.text_doc,model_name=request.model_name,chunk_size=232,overlap=32)
        is_sim = similarity_matching(collection = global_models.get_user_collection(request.user), 
        embedding_model = global_models.embedding_model, 
        all_doc_ids = request.all_doc, 
        chunks = chunks, 
        threshold = 0.9
        )

        return {
            "status": "success",
            "is_sim": is_sim,
        }
        
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
        # "device": str(global_models.embedding_model.device) if global_models.embedding_model else None,
        "active_users": len(global_models.collections)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)