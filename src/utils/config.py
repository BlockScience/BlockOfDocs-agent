from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    OLLAMA_API_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "mistral"
    
    # Storage Configuration
    CHROMA_PERSIST_DIR: str = "./data/vector_store"
    
    # File Configuration
    MAX_FILE_SIZE: int = 10485760  # 10MB in bytes
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def load_config() -> Settings:
    """Load and return application settings."""
    return Settings()
