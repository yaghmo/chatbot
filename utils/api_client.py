import requests 
import streamlit as st
import json

API_BASE_URL = "http://localhost:8000"

class APIClient:
    @staticmethod
    def load_model(model_name: str):
        try:
            response = requests.post(
                f"{API_BASE_URL}/models/load",
                json={"model_name": model_name},
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            st.error(f"Failed to load model: {error_detail}")
            return None
    
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
                    "top_p": top_p
                },
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        if 'token' in data:
                            yield data['token']
                        elif 'done' in data:
                            break
                        elif 'error' in data:
                            st.error(f"Generation error: {data['error']}")
                            break
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            st.error(f"Failed to generate: {error_detail}")

    @staticmethod
    def generate(model_name: str, template: list):
        """Chat title generator (non-streaming)"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "model_name": model_name,
                    "template": template,
                },
                timeout=60 
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            

        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json().get("detail")
                except:
                    error_detail = str(e)
            else:
                error_detail = str(e)
            st.error(f"Failed to generate: {error_detail}")
            return ""