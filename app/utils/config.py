"""
Contextual Scholar: AI-Powered Research Assistant
Configuration management module
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # Application
    app_name: str = "Contextual Scholar"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API Keys
    gemini_api_key: str
    
    # Neo4j Configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./chroma_db"
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    default_top_k: int = 5
    max_context_length: int = 4000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
