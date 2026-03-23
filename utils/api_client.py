import json
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def _extract_error(e: requests.exceptions.RequestException) -> str:
    if hasattr(e, "response") and e.response is not None:
        try:
            return e.response.json().get("detail", str(e))
        except Exception:
            return str(e)
    return str(e)


class APIClient:
    @staticmethod
    def load_model(model_name: str):
        try:
            response = requests.post(f"{API_BASE_URL}/models/load", json={"model_name": model_name}, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to load model: {_extract_error(e)}")

    @staticmethod
    def generate_stream(model_name: str, template: list, max_tokens: int, temperature: float, top_p: float):
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate/stream",
                json={
                    "model_name": model_name,
                    "template": template,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                stream=True,
                timeout=300,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8")[6:])  # strip "data: "
                    if "token" in data:
                        yield data["token"]
                    elif "done" in data:
                        break
                    elif "error" in data:
                        raise Exception(f"Generation error: {data['error']}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to generate: {_extract_error(e)}")

    @staticmethod
    def generate(model_name: str, template: list, max_tokens: int) -> str:
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={"model_name": model_name, "template": template, "max_tokens": max_tokens},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to generate: {_extract_error(e)}")

    @staticmethod
    def count_tokens(model_name: str, text: str) -> int:
        try:
            response = requests.post(
                f"{API_BASE_URL}/count_tokens",
                json={"model_name": model_name, "text": text},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("num_tokens", 0)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to count tokens: {_extract_error(e)}")

    @staticmethod
    def rag_query(user: str, query: str, chat_docs: list, top_n: int = 7, threshold: float = 0.5) -> dict:
        try:
            response = requests.post(
                f"{API_BASE_URL}/rag/query",
                json={"user": user, "query": query, "chat_docs": chat_docs, "top_n": top_n, "threshold": threshold},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("results", {})
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to make query: {_extract_error(e)}")

    @staticmethod
    def add_document(user: str, chat_id: str, file_name: str, summary: str, content: str, model_name: str) -> str:
        try:
            response = requests.post(
                f"{API_BASE_URL}/rag/add_document",
                json={
                    "user": user,
                    "chat_id": chat_id,
                    "file_name": file_name,
                    "summary": summary,
                    "content": content,
                    "model_name": model_name,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("doc_id", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to add document: {_extract_error(e)}")

    @staticmethod
    def transcribe(audio_path: str) -> str:
        try:
            response = requests.post(f"{API_BASE_URL}/transcribe", json={"audio_path": audio_path}, timeout=60)
            response.raise_for_status()
            return response.json().get("transcription", "")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to transcribe: {_extract_error(e)}")

    @staticmethod
    def check_sim(model_name: str, user: str, all_doc: list, text_doc: str) -> bool:
        try:
            response = requests.post(
                f"{API_BASE_URL}/check_sim",
                json={"model_name": model_name, "user": user, "all_doc": all_doc, "text_doc": text_doc},
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("is_sim")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to check similarity: {_extract_error(e)}")
