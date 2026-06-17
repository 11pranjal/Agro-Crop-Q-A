"""Configuration management for AGRO QA Chatbot"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    # Configuration for Pydantic v2: ignore extra env vars and set env file
    model_config = ConfigDict(env_file=".env", extra="ignore", case_sensitive=True)
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    FRONTEND_URL: str = "http://localhost:5176"
    DEBUG: bool = True
    
    # OpenAI Configuration (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # Local Model Configuration
    USE_LOCAL_MODEL: bool = True
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./database/conversations.db"
    DATABASE_PATH: Path = Path("database/conversations.db")
    
    # Data paths
    DATA_DIR: Path = Path("data")
    VECTOR_STORE_PATH: Path = Path("data/vector_store")
    DOCUMENTS_PATH: Path = Path("data/documents")
    
    # Text Processing
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = 3
    SIMILARITY_THRESHOLD: float = 0.08
    
    # Session Configuration
    MAX_HISTORY: int = 20
    SESSION_TIMEOUT: int = 3600  # 1 hour in seconds
    
    # No inner Config: using model_config (ConfigDict) instead for pydantic v2


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)
settings.DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
Path("database").mkdir(parents=True, exist_ok=True)
