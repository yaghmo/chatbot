import requests 
import streamlit as st
import json

API_BASE_URL = "http://localhost:8000"

class APIClient:
    """Client for communicating with FastAPI backend"""
    
    @staticmethod
    def load_model(model_name: str):
        try:
            response = requests.post(
                f"{API_BASE_URL}/models/load",
                json={"model_name": model_name}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_detail = e.response.json().get("detail") if e.response else str(e)
            st.error(f"Failed to load model: {error_detail}")
            return None
    
    @staticmethod
    def generate_stream(model_name: str, prompt: str, max_tokens: int, temperature: float, top_p: float):
        """Generator that yields tokens as they come from the API"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/generate/stream",
                json={
                    "model_name": model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                },
                stream=True
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
            error_detail = e.response.json().get("detail") if e.response else str(e)
            st.error(f"Failed to generate: {error_detail}")
