from transformers import AutoTokenizer, AutoModel
from ctransformers import AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download, hf_hub_url
import os
import json
import subprocess
import requests
import psutil
import gc
import re

class Model:
    def __init__(self, cfg: dict, model_name):
        self.cfg = cfg
        self.model_name = model_name
        self._device = cfg["device"]
        self._model = None
        self.tokenizer = None
        self.size = 0

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
        config_path = hf_hub_download(repo_id=self.cfg["origin"],filename="config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        num_hidden_layers = config.get("num_hidden_layers")
        hidden_size = config.get("hidden_size")
        torch_dtype = config.get("torch_dtype")

        digits = re.sub(r"\D", "", torch_dtype)
        bytes_per_element = {"32":4,"16":2,"8":1,"4":0.5,"1":0.25}
        bytes_per_element = bytes_per_element.get(digits)

        url = hf_hub_url(repo_id=self.cfg["url"], filename=self.cfg["file_name"])
        response = requests.head(url, allow_redirects=True, timeout=10)

        kv_cache_bytes = num_hidden_layers * self.cfg["context_length"] * hidden_size * 2 * bytes_per_element
        self.size = int(response.headers.get('Content-Length', 0) )+ kv_cache_bytes / (1024**3)  # in GB

    def can_fit_on_gpu(self, margin_gb=2.0) -> bool:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            available_gb = float(result.stdout.strip().split('\n')[0]) / 1024
            return (available_gb - margin_gb) > self.size
        except Exception as e:
            print(f"GPU check error: {e}")
            return False

    def can_fit_on_ram(self, margin_gb=1.0) -> bool:
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            return (available_gb - margin_gb) > self.size
        except Exception as e:
            print(f"RAM check error: {e}")
            return False

    def _ctransformers(self):
        self._get_model()
        MODEL_PATH = f'models/{self.cfg["file_name"]}'
        
        # Load model with gpu_layers
        gpu_layers = -1 if self._device == "cpu" else 100
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type=self.cfg["type"],
            gpu_layers=gpu_layers,
        )
        print(f"Model loaded: {self.model_name}")

    def model_load(self):
        if self.cfg["framework"] == "ctransformers":
            self._ctransformers()

    def max_token(self, text: str) -> int:
        """Count tokens in text"""
        if self.cfg["framework"] == "ctransformers":
            return len(self._model.tokenize(text))
        return 0

    def model_inf(self, prompt: str, max_tokens: int, temperature: float, top_p: float):
        if self.cfg["framework"] == "ctransformers":
            print(f"[Model Inference] temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")
            
            # Generate with streaming
            for token in self._model(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
            ):
                yield token
        else:
            yield "Framework not supported"

    def unload(self):
        """Unload model and free memory"""
        if self._model:
            del self._model
            self._model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        print(f"Model unloaded: {self.model_name}")

    def __del__(self):
        self.unload()