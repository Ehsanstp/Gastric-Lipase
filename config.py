import os
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
DATABASE_URL = os.getenv("DATABASE_URL")

PDF_DIR = "./data"
CHROMA_PATH = "./chromadb"
CHROMA_COLLECTION = "chutop"
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 30
FULLPAGE_PATH = "./fullpage.json"
OLLAMA_MODEL = "qwen2.5:0.5b"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K_RESULTS = 11 # CHUNKS to retrieve
MAX_CONTENT = 4000
HISTORY_TURNS = 7
