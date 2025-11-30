from transformers import AutoTokenizer, AutoModel
# from auto_gptq import AutoGPTQForCausalLM
from ctransformers import AutoModelForCausalLM
import torch
from huggingface_hub import hf_hub_download, hf_hub_url
import os
import subprocess
import requests

class Model:
    def __init__(self, cfg:dict):
        self.cfg = cfg
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self.loaded = False
        self.size = 0
        self.temperature = 0.7
        self.top_p = 1.0

    def _get_model(self)->int:
        if self.cfg["framework"] == "llama" or "ctransformers":
            os.makedirs("models", exist_ok=True)
            model_path = hf_hub_download(
                repo_id=self.cfg["url"],
                filename=self.cfg["file_name"],
                local_dir="models",
                local_dir_use_symlinks=False
            )
            available_gb = float(result.stdout.strip().split('\n')[0]) / 1024
            url = hf_hub_url(repo_id=self.cfg["url"], filename=self.cfg["file_name"])
            response = requests.head(url, allow_redirects=True, timeout=10)
            self.size = int(response.headers.get('Content-Length', 0)) / (1024**3) # in Gigabits

    def _can_fit_on_gpu(self, margin_gb=2.0)->float:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True
            )
            available_gb = float(result.stdout.strip().split('\n')[0]) / 1024
            return (available_gb - margin_gb) > self.size

        except Exception as e:
            print(f"Error: {e}")
            return False

    def _can_fit_on_ram(self, margin_gb=1.0)->float:
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            return (available_gb - margin_gb) > self.size
            
        except Exception as e:
            print(f"Error: {e}")
            return False

    def _llama():
        ...
    
    def _ctransformers(self):
        MODEL_PATH = f'models/{self.cfg["model_name"]}'
        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            model_type="mistral",
            gpu_layers=100,
            context_length=4096,
        )


    def model_load(self):
        if self.cfg["framework"] == "ctransformers":
            self._ctransformers()


    def model_inf(self, prompt: str, max_tokens):
        match self.cfg["framework"]:
            case "ctransformers":
                for token in llm(
                    prompt,
                    max_new_tokens=self.cfg["max_tokens"],
                    temperature=self.cfg["temperature"],
                    top_p=self.cfg["top_p"],
                    stream=True,
                ):
                    yield token


    
    def unload(self):
        if self._model:
            del self._model
            self._model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self.loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()

    def __del__(self):
        self.unload()
