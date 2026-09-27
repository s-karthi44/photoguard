from dotenv import load_dotenv
import os
from huggingface_hub import login

load_dotenv()

token = os.getenv("HF_TOKEN")

login(token=token)

print("✅ Hugging Face Login Successful")