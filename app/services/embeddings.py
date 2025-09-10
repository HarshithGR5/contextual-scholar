"""
Embedding service using SentenceTransformers
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.utils.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize the embedding service."""
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return [embedding for embedding in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings for texts: {e}")
            raise
    
    def embed_texts_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Generate embeddings for texts in batches."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
                embeddings.extend(batch_embeddings)
                
                if i + batch_size < len(texts):
                    logger.info(f"Processed {i + batch_size}/{len(texts)} texts")
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        return self.model.get_sentence_embedding_dimension()


# Global embedding service instance
embedding_service = EmbeddingService()
