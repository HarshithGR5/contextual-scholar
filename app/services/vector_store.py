"""
ChromaDB vector store service for semantic similarity search
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.models.schemas import DocumentChunk, RetrievedSource
from app.services.embeddings import embedding_service
from app.utils.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for vector storage and similarity search using ChromaDB."""
    
    def __init__(self):
        """Initialize the ChromaDB client and collection."""
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Configure ChromaDB settings
            chroma_settings = ChromaSettings(
                persist_directory=settings.chroma_persist_directory,
                anonymized_telemetry=False
            )
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=chroma_settings
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="research_documents",
                metadata={"description": "Research documents for contextual scholar"}
            )
            
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to the vector store."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Prepare data for ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            chunk_embeddings = embedding_service.embed_texts_batch(texts)
            
            for chunk, embedding in zip(chunks, chunk_embeddings):
                chunk_id = f"{chunk.doc_id}_{chunk.chunk_id}"
                ids.append(chunk_id)
                documents.append(chunk.content)
                
                # Prepare metadata
                metadata = {
                    "doc_id": chunk.doc_id,
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata
                }
                
                if chunk.page_number is not None:
                    metadata["page_number"] = chunk.page_number
                
                metadatas.append(metadata)
                embeddings.append(embedding.tolist())
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            
            logger.info(f"Added {len(chunks)} document chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedSource]:
        """Perform similarity search for the given query."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Generate query embedding
            query_embedding = embedding_service.embed_text(query)
            
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=filters
            )
            
            # Convert results to RetrievedSource objects
            sources = []
            
            if results['ids'] and len(results['ids']) > 0:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1 - distance
                    
                    source = RetrievedSource(
                        doc_id=metadata.get('doc_id', doc_id),
                        title=metadata.get('title'),
                        chunk=document,
                        score=score,
                        metadata=metadata
                    )
                    sources.append(source)
            
            logger.info(f"Found {len(sources)} similar documents for query")
            return sources
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            raise
    
    def get_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        if not self.collection:
            return 0
        
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a specific document."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"doc_id": doc_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {doc_id}")
                return True
            
            logger.warning(f"No chunks found for document {doc_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise
    
    def clear_collection(self) -> bool:
        """Clear all documents from the vector store."""
        if not self.collection:
            raise RuntimeError("Vector store not initialized")
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("research_documents")
            self.collection = self.client.get_or_create_collection(
                name="research_documents",
                metadata={"description": "Research documents for contextual scholar"}
            )
            
            logger.info("Vector store cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            raise

    def list_documents(self) -> List[str]:
        """List all unique document IDs in the vector store."""
        if not self.collection:
            return []
        
        try:
            # Get all documents in the collection
            results = self.collection.get()
            
            if not results or not results.get('metadatas'):
                return []
            
            # Extract unique filenames/document IDs
            documents = set()
            for metadata in results['metadatas']:
                if 'original_filename' in metadata:
                    documents.add(metadata['original_filename'])
                elif 'doc_id' in metadata:
                    documents.add(metadata['doc_id'])
            
            return sorted(list(documents))
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []


# Global vector store service instance
vector_store_service = VectorStoreService()
