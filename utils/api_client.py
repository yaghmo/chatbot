import json

import requests

API_BASE_URL = "http://localhost:8000"


class APIClient:
    @staticmethod
    def load_model(model_name: str):
        try:
            response = requests.post(f"{API_BASE_URL}/models/load", json={"model_name": model_name}, timeout=300)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to load model: {error_detail}")

    @staticmethod
    def generate_stream(model_name: str, template: list, max_tokens: int, temperature: float, top_p: float):
        """Generator that yields tokens as they come from the API"""
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
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "token" in data:
                            yield data["token"]
                        elif "done" in data:
                            break
                        elif "error" in data:
                            raise Exception(f"Generation error: {data['error']}")
                            break
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to generate: {error_detail}")

    @staticmethod
    def generate(model_name: str, template: list, max_tokens: int):
        """Chat title generator (non-streaming)"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "model_name": model_name,
                    "template": template,
                    "max_tokens": max_tokens,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to generate: {error_detail}")

    @staticmethod
    def count_tokens(model_name: str, text: str):
        try:
            response = requests.post(
                f"{API_BASE_URL}/count_tokens", json={"model_name": model_name, "text": text}, timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get("num_tokens", 0)

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to count tokens: {error_detail}")

    @staticmethod
    def rag_query(user: str, query: str, chat_docs: list, top_n: int = 7, threshold: float = 0.5):
        try:
            response = requests.post(
                f"{API_BASE_URL}/rag/query",
                json={
                    "user": user,
                    "query": query,
                    "chat_docs": chat_docs,
                    "top_n": top_n,
                    "threshold": threshold,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("results", {})

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to make query: {error_detail}")

    @staticmethod
    def add_document(user: str, chat_id: str, file_name: str, summary: str, content: str, model_name: str):
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

            result = response.json()
            return result.get("doc_id", "")

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to add document: {error_detail}")

    @staticmethod
    def transcribe(audio_path: str):
        try:
            response = requests.post(
                f"{API_BASE_URL}/transcribe",
                json={
                    "audio_path": audio_path,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("transcription", "")

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to transcribe: {error_detail}")

    @staticmethod
    def check_sim(model_name: str, user: str, all_doc: list, text_doc: str):
        try:
            response = requests.post(
                f"{API_BASE_URL}/check_sim",
                json={
                    "model_name": model_name,
                    "user": user,
                    "all_doc": all_doc,
                    "text_doc": text_doc,
                },
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("is_sim")

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except e:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            raise Exception(f"Failed to check similarity: {error_detail}")
