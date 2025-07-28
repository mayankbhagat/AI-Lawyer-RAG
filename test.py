import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() # Ensure your GOOGLE_API_KEY is loaded from .env

genai.configure(api_key=os.getenv("AIzaSyDrlb7QzXSBQdsjbrYVqmrp7isMSCxx6cY"))

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"Model: {model.name}, Supported: {model.supported_generation_methods}")