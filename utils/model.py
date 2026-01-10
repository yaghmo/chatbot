from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForVision2Seq, Qwen3VLForConditionalGeneration
from ctransformers import AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download, hf_hub_url, HfApi
import os
import json
import subprocess
import requests
import logging
import psutil
import gc
import re
from utils.image_encoding import decode_image_msgpack

logger = logging.getLogger(__name__)

class Model:
    def __init__(self, cfg: dict, model_name):
        self.cfg = cfg
        self.model_name = model_name
        self._device = cfg["device"]
        self._model = None
        self._tokenizer = None
        self._processor = None 
        self.size = 0

    def _get_file_size_from_url(self, filename):
        """Get file size by making a HEAD request"""
        try:
            url = hf_hub_url(repo_id=self.cfg["url"], filename=filename)
            response = requests.head(url, allow_redirects=True, timeout=10)
            
            if 'Content-Length' in response.headers:
                return int(response.headers['Content-Length'])
            if 'X-Linked-Size' in response.headers:
                return int(response.headers['X-Linked-Size'])

        except Exception as e:
            logger.exception(f"Error getting size for {filename}: {e}")
        return None

    def _get_model(self):
        if self.cfg["framework"] in ("llama", "ctransformers"):
            os.makedirs("models", exist_ok=True)
            # DL model to local
            hf_hub_download(
                repo_id=self.cfg["url"],
                filename=self.cfg["file_name"],
                local_dir="models",
                local_dir_use_symlinks=False
            )

        # get model info
        config_path = hf_hub_download(repo_id=self.cfg.get("origin"),filename="config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        num_hidden_layers = None
        hidden_size = None
        
        if "num_hidden_layers" in config:
            num_hidden_layers = config.get("num_hidden_layers")
            hidden_size = config.get("hidden_size")
        elif "text_config" in config:
            num_hidden_layers = config["text_config"].get("num_hidden_layers")
            hidden_size = config["text_config"].get("hidden_size")
        elif "num_layers" in config:
            num_hidden_layers = config.get("num_layers")
            hidden_size = config.get("hidden_dim") or config.get("d_model")
        
        # If still not found, use reasonable defaults based on model type
        if num_hidden_layers is None or hidden_size is None:
            logger.warning(f"Could not find num_hidden_layers or hidden_size in config. Using defaults.")
            num_hidden_layers = 32
            hidden_size = 4096

        torch_dtype = config.get("torch_dtype", "float16")
        
        # Handle torch_dtype being a string like "float16" or actual value
        if isinstance(torch_dtype, str):
            digits = re.sub(r"\D", "", torch_dtype)
        else:
            digits = "16"
        
        bytes_per_element = {"32":4,"16":2,"8":1,"4":0.5,"1":0.25}
        bytes_per_element = bytes_per_element.get(digits, 2)

        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id=self.cfg["url"], repo_type="model")
        except Exception as e:
            logger.exception(f"Error accessing repository: {e}")
            
        if self.cfg.get("format") == "gguf":
            sizes = [os.path.getsize(f'models/{self.cfg.get("file_name")}')]
        else:
            sizes = [self._get_file_size_from_url(f) for f in repo_files if f.endswith(self.cfg.get("format"))]
            logger.info("repos :",repo_files)
            if not sizes:
                logger.error("No files found in this repository.")

        total_size_bytes = sum(size for size in sizes if sizes is not None)

        kv_cache_bytes = num_hidden_layers * self.cfg["context_length"] * hidden_size * 2 * bytes_per_element
        self.size = (total_size_bytes + kv_cache_bytes )/ (1024**3)  # in GB
        
        logger.info(f"Model size estimate: {self.size:.2f} GB (layers={num_hidden_layers}, hidden={hidden_size})")

    def can_fit_on_gpu(self) -> bool:
        if self.size == 0:
            try:
                self._get_model()
            except Exception as e:
                logger.error(f"Failed to get model size: {e}")
                return False
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            available_gb = float(result.stdout.strip().split('\n')[0]) / 1024
            logger.info(f"GPU: {available_gb:.2f}GB available, need {self.size:.2f}GB")
            return available_gb >= self.size
        except Exception as e:
            logger.info(f"GPU check error: {e}")
            return False

    def can_fit_on_ram(self) -> bool:
        return True
        # if self.size == 0:
        #     try:
        #         self._get_model()
        #     except Exception as e:
        #         logger.error(f"Failed to get model size: {e}")
        #         return False
        # try:
        #     import gc
        #     gc.collect()
            
        #     mem = psutil.virtual_memory()
        #     available_gb = mem.available / (1024**3)

        #     logger.info(f"RAM: {available_gb:.2f}GB available, need {self.size:.2f}GB.")
        #     return available_gb >= self.size
        # except Exception as e:
        #     logger.info(f"RAM check error: {e}")
        #     return False

    def _load_tokenizer(self):  
        """Load tokenizer or processor based on model mode"""
        try:
            tokenizer_path = self.cfg.get("origin", self.cfg["url"]) if self.cfg["framework"] == "ctransformers" else self.cfg["url"]
            
            if self.cfg.get("mode") == "llm":
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Tokenizer loaded from {tokenizer_path} for {self.model_name}")
            elif self.cfg.get("mode") == "vlm":
                self._processor = AutoProcessor.from_pretrained(tokenizer_path)
                logger.info(f"Processor loaded from {tokenizer_path} for {self.model_name}")
            else:
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Tokenizer loaded from {tokenizer_path} for {self.model_name} (default)")
        except Exception as e:
            logger.error(f"Failed to load tokenizer/processor: {e}")
            raise

    def _ctransformers(self):
        if self.size == 0:
            self._get_model()
        MODEL_PATH = f'models/{self.cfg["file_name"]}'

        gpu_layers = 0 if self._device == "cpu" else 100
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=self.cfg["type"],
            gpu_layers=gpu_layers,
        )
        logger.info(f"Model loaded: {self.model_name}")
    
    def _transformers(self):
        device_map = "cpu"  # default to CPU
        if self._device == "gpu" and torch.cuda.is_available():
            device_map = "cuda:0"
        elif self._device == "cpu":
            device_map = "cpu"
        elif self._device == "hybrid":
            device_map = "auto"
        
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg["url"],
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            dtype="auto"
        )
        logger.info(f"Model loaded: {self.model_name}")
    

    def model_load(self):
        self._load_tokenizer()
        
        if self.cfg["framework"] == "ctransformers":
            self._ctransformers()
        elif self.cfg["framework"] == "transformers":
            self._transformers()

    def model_inf(self, template: list, max_tokens: int, temperature: float, top_p: float, stream: bool = True, files: list = None):
    
        logger.info(f"[Model Inference] framework={self.cfg['framework']}, mode={self.cfg.get('mode')}, temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
        
        framework = self.cfg["framework"]
        mode = self.cfg.get("mode", "llm")
        model_type = self.cfg.get("type")
        
        if framework == "ctransformers":
            # ctransformers needs text prompt
            inputs = self._tokenizer.apply_chat_template(
                template,
                add_generation_prompt=True,
                tokenize=False
            )
            
        elif framework == "transformers":
            if mode == "vlm" and model_type == "qwen":
                inputs = self._processor.apply_chat_template(
                    template,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self._model.device)
            else:
                # Standard transformers: use tokenizer
                prompt = self._tokenizer.apply_chat_template(
                    template,
                    add_generation_prompt=True,
                    tokenize=False
                )
                inputs = self._tokenizer(prompt, return_tensors="pt")
                if self._device == "gpu" and torch.cuda.is_available():
                    inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        
        # Generate 
        if framework == "ctransformers":
            for token in self._model(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=top_p,
                repetition_penalty=1.15,
                stream=stream
            ):
                yield token
                
        elif framework == "transformers":

            if stream:
                # correct code - Use TextIteratorStreamer for real token streaming
                from transformers import TextIteratorStreamer
                from threading import Thread
                
                streamer = TextIteratorStreamer(
                    self._processor.tokenizer if mode == "vlm" else self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                
                # Generation config
                generation_kwargs = dict(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=40,
                    top_p=top_p,
                    repetition_penalty=1.15,
                    do_sample=temperature > 0,
                    streamer=streamer
                )
                
                # Run generation in separate thread
                thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
                thread.start()
                
                # Yield tokens as they come
                for text in streamer:
                    yield text
                
                thread.join()
            else:
                # Non-streaming mode
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=40,
                        top_p=top_p,
                        repetition_penalty=1.15,
                        do_sample=temperature > 0,
                    )
                
                if mode == "vlm":
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] 
                        for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]
                    generated_text = self._processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                else:
                    generated_text = self._tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                
                yield generated_text

            # with torch.no_grad():
            #     outputs = self._model.generate(
            #         **inputs,
            #         max_new_tokens=max_tokens,
            #         temperature=temperature,
            #         top_k=40,
            #         top_p=top_p,
            #         repetition_penalty=1.15,
            #         do_sample=temperature > 0,
            #     )
            
            # if mode == "vlm":
            #     generated_ids_trimmed = [
            #         out_ids[len(in_ids):] 
            #         for in_ids, out_ids in zip(inputs.input_ids, outputs)
            #     ]
            #     generated_text = self._processor.batch_decode(
            #         generated_ids_trimmed,
            #         skip_special_tokens=True,
            #         clean_up_tokenization_spaces=False
            #     )[0]
            # else:
            #     generated_text = self._tokenizer.decode(
            #         outputs[0][inputs['input_ids'].shape[1]:],
            #         skip_special_tokens=True
            #     )
            
            # if stream:
            #     for char in generated_text:
            #         yield char
            # else:
            #     yield generated_text


    def unload(self):
        """Unload model and free memory"""
        if self._model:
            del self._model
            self._model = None
        
        if self._tokenizer:
            del self._tokenizer
            self._tokenizer = None
        
        # correct code - also clean up processor
        if self._processor:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info(f"Model unloaded: {self.model_name}")

    def __del__(self):
        self.unload()