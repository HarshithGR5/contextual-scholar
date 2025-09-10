"""
Pydantic models for API request/response schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of processed document content."""
    doc_id: str
    chunk_id: str
    content: str
    metadata: Dict[str, Any] = {}
    page_number: Optional[int] = None


class EntityRelation(BaseModel):
    """Knowledge graph entity relationship."""
    entity: str
    relationship: str
    target_entity: Optional[str] = None
    context: Optional[str] = None
    confidence: Optional[float] = None


class RetrievedSource(BaseModel):
    """A retrieved document source with relevance score."""
    doc_id: str
    title: Optional[str] = None
    chunk: str
    score: float
    metadata: Dict[str, Any] = {}


class ResearchQuery(BaseModel):
    """Request model for research queries."""
    question: str = Field(..., min_length=1, description="The research question to answer")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top results to retrieve")
    include_entities: bool = Field(default=True, description="Include related entities from knowledge graph")
    
    
class ResearchResponse(BaseModel):
    """Response model for research queries."""
    answer: str
    sources: List[RetrievedSource]
    related_entities: List[EntityRelation] = []
    confidence: Optional[float] = None
    processing_time: Optional[float] = None


class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion."""
    file_path: str = Field(..., description="Path to the document file")
    doc_id: Optional[str] = Field(None, description="Optional custom document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentIngestionResponse(BaseModel):
    """Response model for document ingestion."""
    doc_id: str
    chunks_processed: int
    entities_extracted: int
    status: str = "success"
    message: Optional[str] = None


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = "healthy"
    timestamp: str
    version: str
    services: Dict[str, str] = {}


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: str
