import os
import requests
import json
from google import genai as gemini
from huggingface_hub import InferenceClient as huggingface
from dotenv import load_dotenv

class init_models:
    
    def __init__(self):
        load_dotenv()
        self.gemini_client = gemini.Client(api_key = os.getenv("GEMINI_API_KEY"))
        self.huggingface_client = huggingface(api_key = os.getenv("HUGGINGFACE_API_KEY"))
    
    def gemini_model(self, model_name, contents):
        response = self.gemini_client.models.generate_content(
            model=model_name,
            contents=contents)
        return response.text
    
    def huggingface_model(self, model_name, contents):
        response = self.huggingface_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": contents
                }
            ],
        )
        return response.choices[0].message.content
    
    def list_gemini_models(self):
        models = self.gemini_client.models.list()
        for model in models:
            print(model.name)

    def get_gemini_embedding(self, contents, model_name="gemini-embedding-001"):
        response = self.gemini_client.models.embed_content(
            model=model_name,
            contents=contents
        )
        return response.embeddings[0].values
    
    def get_nvidia_api_response(self, contents, model_name="google/gemma-3-27b-it"):
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        payload = {
            "messages": [{ "role": "user", "content": contents}],
            "model": model_name,
            "max_tokens": 1024,
            "stream": False,
            "temperature": 0.2,
            "top_p":0.7
        }
        headers = {
            "authorization" : f"""Bearer {os.getenv("NVIDIA_API_KEY")}""",
            "accept" : "application/json",
            "content-type" : "application/json"
        }

        response = requests.post(url, json=payload, headers=headers)
        responseObject = json.loads(response.text)
        if "choices" in responseObject  : 
            return responseObject["choices"][0]["message"]["content"]
        else : 
            return ""