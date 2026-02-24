import os
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