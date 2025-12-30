from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import torch
from utils.model import Model
import uvicorn
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API")

# Enable CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            logger.error(f"Failed to load config: {e}")
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

model_manager = ModelManager()

# Request/Response models
class LoadModelRequest(BaseModel):
    model_name: str

class GenerateRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = True

class GenerateResponse(BaseModel):
    response: str
    tokens_used: int

# API Endpoints
@app.on_event("startup")
async def startup_event():
    try:
        model_manager.load_config("utils/model_cfg.json")
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())


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
                logger.error(f"Error checking resources: {e}")
                logger.error(traceback.format_exc())
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
        logger.error(error_msg)
        logger.error(traceback.format_exc())
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
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ):
                    yield f"data: {json.dumps({'token': token})}\n\n" if model_manager.model_cfg[model_manager.active_model_name]["framework"] != "llama" else f"data: {json.dumps({'token': token['choices'][0]['text']})}\n\n"
                

                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in stream generate: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/generate")
# async def generate(request: GenerateRequest):
#     """Generate text (non-streaming)"""
#     try:
#         logger.info(f"Generate request - model: {request.model_name}, temp: {request.temperature}, top_p: {request.top_p}")
        
#         if request.model_name not in model_manager.models:
#             raise HTTPException(status_code=404, detail="Model not loaded")
        
#         model = model_manager.models[request.model_name]
        
#         if model._model is None:
#             raise HTTPException(status_code=400, detail="Model not loaded")
        
#         full_response = ""
#         for token in model.model_inf(
#             prompt=request.prompt,
#             max_tokens=request.max_tokens,
#             temperature=request.temperature,
#             top_p=request.top_p,
#         ):
#             full_response += token
        
#         tokens_used = model.max_token(request.prompt + full_response)
        
#         logger.info(f"Generated {tokens_used} tokens")
        
#         return GenerateResponse(
#             response=full_response,
#             tokens_used=tokens_used
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating: {e}")
#         logger.error(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)