from dotenv import load_dotenv
import os

load_dotenv()

DB_DIR = os.getenv("DB_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
CSV_PATH = os.getenv("CSV_PATH")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")