from dotenv import load_dotenv

load_dotenv()

import gc
import json
import logging
import os
import re
import subprocess
from threading import Thread

import psutil
import requests
import torch
from ctransformers import AutoModelForCausalLM
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration, TextIteratorStreamer

MODELS_DIR = os.path.abspath(os.getenv("MODELS_DIR", "models"))

os.environ["HF_HOME"] = os.path.join(MODELS_DIR, ".cache", "huggingface")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BYTES_PER_DTYPE = {"32": 4, "16": 2, "8": 1, "4": 0.5, "1": 0.25}


class Model:
    def __init__(self, cfg: dict, model_name: str):
        self.cfg = cfg
        self.model_name = model_name
        self._device = cfg["device"]
        self._model = None
        self._tokenizer = None
        self._processor = None
        self.size = 0

    def _get_file_size_from_url(self, filename: str):
        try:
            url = hf_hub_url(repo_id=self.cfg["url"], filename=filename)
            response = requests.head(url, allow_redirects=True, timeout=10)
            return int(response.headers.get("Content-Length") or response.headers.get("X-Linked-Size", 0)) or None
        except Exception as e:
            logger.exception(f"Error getting size for {filename}: {e}")
        return None

    def _get_model(self):
        if self.cfg["framework"] in ("llama", "ctransformers"):
            os.makedirs(MODELS_DIR, exist_ok=True)
            hf_hub_download(
                repo_id=self.cfg["url"],
                filename=self.cfg["file_name"],
                local_dir=MODELS_DIR,
                local_dir_use_symlinks=False,
            )

        config_path = hf_hub_download(repo_id=self.cfg.get("origin"), filename="config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if "num_hidden_layers" in config:
            num_hidden_layers = config["num_hidden_layers"]
            hidden_size = config.get("hidden_size")
        elif "text_config" in config:
            num_hidden_layers = config["text_config"].get("num_hidden_layers")
            hidden_size = config["text_config"].get("hidden_size")
        elif "num_layers" in config:
            num_hidden_layers = config["num_layers"]
            hidden_size = config.get("hidden_dim") or config.get("d_model")
        else:
            logger.warning("Could not find layer/size info in config. Using defaults.")
            num_hidden_layers, hidden_size = 32, 4096

        torch_dtype = config.get("torch_dtype", "float16")
        digits = re.sub(r"\D", "", torch_dtype) if isinstance(torch_dtype, str) else "16"
        bytes_per_param = _BYTES_PER_DTYPE.get(digits, 2)

        repo_files = []
        try:
            repo_files = list(HfApi().list_repo_files(repo_id=self.cfg["url"], repo_type="model"))
        except Exception as e:
            logger.exception(f"Error accessing repository: {e}")

        if self.cfg.get("format") == "gguf":
            sizes = [os.path.getsize(os.path.join(MODELS_DIR, self.cfg["file_name"]))]
        else:
            sizes = [self._get_file_size_from_url(f) for f in repo_files if f.endswith(self.cfg.get("format", ""))]
            logger.info(f"repo files: {repo_files}")
            if not sizes:
                logger.error("No files found in this repository.")

        total_size_bytes = sum(s for s in sizes if s is not None)
        kv_cache_bytes = num_hidden_layers * self.cfg["context_length"] * hidden_size * 2 * bytes_per_param
        self.size = (total_size_bytes + kv_cache_bytes) / (1024**3)

        logger.info(f"Model size estimate: {self.size:.2f} GB (layers={num_hidden_layers}, hidden={hidden_size})")

    def _ensure_size(self) -> bool:
        if self.size == 0:
            try:
                self._get_model()
            except Exception as e:
                logger.error(f"Failed to get model size: {e}")
                return False
        return True

    def can_fit_on_gpu(self) -> bool:
        if not self._ensure_size():
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            available_gb = float(result.stdout.strip().split("\n")[0]) / 1024
            logger.info(f"GPU: {available_gb:.2f}GB available, need {self.size:.2f}GB")
            return available_gb >= self.size
        except Exception as e:
            logger.info(f"GPU check error: {e}")
            return False

    def can_fit_on_ram(self) -> bool:
        if not self._ensure_size():
            return False
        try:
            gc.collect()
            available_gb = psutil.virtual_memory().available / (1024**3)
            logger.info(f"RAM: {available_gb:.2f}GB available, need {self.size:.2f}GB.")
            return available_gb >= self.size
        except Exception as e:
            logger.info(f"RAM check error: {e}")
            return False

    def _load_tokenizer(self):
        tokenizer_path = (
            self.cfg.get("origin", self.cfg["url"]) if self.cfg["framework"] == "ctransformers" else self.cfg["url"]
        )
        if self.cfg.get("mode") == "vlm":
            self._processor = AutoProcessor.from_pretrained(tokenizer_path)
            logger.info(f"Processor loaded from {tokenizer_path} for {self.model_name}")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"Tokenizer loaded from {tokenizer_path} for {self.model_name}")

    def _ctransformers(self):
        if self.size == 0:
            self._get_model()
        model_path = os.path.join(MODELS_DIR, self.cfg["file_name"])
        gpu_layers = 0 if self._device == "cpu" else 100
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, model_type=self.cfg["type"], gpu_layers=gpu_layers, context_length=self.cfg["context_length"]
        )
        logger.info(f"Model loaded: {self.model_name}")

    def _transformers(self):
        if self._device == "gpu" and torch.cuda.is_available():
            device_map = "cuda:0"
        elif self._device == "hybrid":
            device_map = "auto"
        else:
            device_map = "cpu"

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg["url"],
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        logger.info(f"Model loaded: {self.model_name}")

    def model_load(self):
        self._load_tokenizer()
        if self.cfg["framework"] == "ctransformers":
            self._ctransformers()
        elif self.cfg["framework"] == "transformers":
            self._transformers()

    def model_inf(self, template: list, max_tokens: int, temperature: float, top_p: float, stream: bool = True):
        logger.info(
            f"[Inference] framework={self.cfg['framework']}, mode={self.cfg.get('mode')}, "
            f"temp={temperature}, top_p={top_p}, max_tokens={max_tokens}"
        )

        framework = self.cfg["framework"]
        mode = self.cfg.get("mode", "llm")

        if framework == "ctransformers":
            inputs = self._tokenizer.apply_chat_template(template, add_generation_prompt=True, tokenize=False)
        elif framework == "transformers":
            tokenizer_or_processor = self._processor if mode == "vlm" else self._tokenizer
            inputs = tokenizer_or_processor.apply_chat_template(
                template, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._model.device)
            elif isinstance(inputs, dict):
                inputs = {k: v.to(self._model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        else:
            raise ValueError(f"Unsupported framework: {framework}")

        if framework == "ctransformers":
            yield from self._model(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=top_p,
                repetition_penalty=1.15,
                stream=stream,
            )

        elif framework == "transformers":
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=top_p,
                repetition_penalty=1.15,
                do_sample=temperature > 0,
            )
            if stream:
                streamer = TextIteratorStreamer(
                    self._processor.tokenizer if mode == "vlm" else self._tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                with torch.inference_mode():
                    thread = Thread(target=self._model.generate, kwargs={**generation_kwargs, "streamer": streamer})
                    thread.start()
                    yield from streamer
                    thread.join()
            else:
                with torch.no_grad():
                    outputs = self._model.generate(**generation_kwargs)

                if mode == "vlm":
                    trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
                    yield self._processor.batch_decode(
                        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                else:
                    yield self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def count_tokens(self, text) -> int:
        if not text:
            return 0
        tokenizer = self._tokenizer if self._tokenizer else self._processor.tokenizer
        if isinstance(text, list) and text and isinstance(text[0], dict):
            return len(tokenizer.apply_chat_template(text, tokenize=True, add_generation_prompt=False))
        return len(tokenizer.encode(text, add_special_tokens=False))

    def unload(self):
        for attr in ("_model", "_tokenizer", "_processor"):
            if getattr(self, attr):
                delattr(self, attr)
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Model unloaded: {self.model_name}")

    def __del__(self):
        self.unload()
