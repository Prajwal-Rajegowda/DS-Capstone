from utils import init_models
from utils import repo_loader
from utils import vector_db

models = init_models.init_models()
# models.list_gemini_models()

# response = models.gemini_model("gemini-2.5-flash", "What's the capital of Karnataka? Let me know top 5 things of this place.")
# response = models.huggingface_model("meta-llama/Llama-3.1-8B-Instruct:novita", "What's the capital of Karnataka? Let me know top 5 things of this place.")
# print(response)
# print(models.list_gemini_models())
print(vector_db.RAGDatabase(models).list_collections())