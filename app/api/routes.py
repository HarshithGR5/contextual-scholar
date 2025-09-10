"""
FastAPI routes for the Contextual Scholar API
"""

import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ResearchQuery,
    ResearchResponse, 
    DocumentIngestionRequest,
    DocumentIngestionResponse,
    HealthCheck,
    ErrorResponse
)
from app.services.rag_pipeline import rag_pipeline
from app.utils.document_processor import document_processor

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post("/query", response_model=ResearchResponse)
async def query_documents(query: ResearchQuery) -> ResearchResponse:
    """
    Query documents with a research question.
    
    This endpoint combines vector similarity search with knowledge graph traversal
    to provide comprehensive, citation-backed responses.
    """
    try:
        logger.info(f"Received query: {query.question[:100]}...")
        
        # Process the query through the RAG pipeline
        response = await rag_pipeline.process_query(query)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> DocumentIngestionResponse:
    """
    Upload and ingest a document file.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file temporarily
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process the document
            chunks = document_processor.process_pdf(temp_file_path, None)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata["original_filename"] = file.filename
            
            # Ingest through RAG pipeline
            result = await rag_pipeline.ingest_document(chunks)
            
            # Prepare response
            response = DocumentIngestionResponse(
                doc_id=chunks[0].doc_id,
                chunks_processed=result["chunks_processed"],
                entities_extracted=result["entities_added"],
                message=f"Successfully uploaded and processed {file.filename}"
            )
            
            return response
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.get("/documents")
async def list_documents():
    """
    List all uploaded documents.
    """
    try:
        # Get document list from vector store
        documents = rag_pipeline.vector_store.list_documents()
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": []}


@router.post("/ingest", response_model=DocumentIngestionResponse)
async def ingest_document(request: DocumentIngestionRequest) -> DocumentIngestionResponse:
    """
    Ingest a document into the system for future querying.
    
    Processes the document by:
    1. Extracting text and creating chunks
    2. Generating embeddings and storing in vector database
    3. Extracting entities and adding to knowledge graph
    """
    try:
        logger.info(f"Ingesting document: {request.file_path}")
        
        # Validate file
        if not document_processor.validate_file(request.file_path):
            raise HTTPException(
                status_code=400,
                detail="Invalid file or unsupported format"
            )
        
        # Process the document
        chunks = document_processor.process_pdf(
            request.file_path, 
            request.doc_id
        )
        
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document"
            )
        
        # Add metadata from request
        for chunk in chunks:
            chunk.metadata.update(request.metadata)
        
        # Ingest through RAG pipeline
        result = await rag_pipeline.ingest_document(chunks)
        
        # Prepare response
        response = DocumentIngestionResponse(
            doc_id=chunks[0].doc_id,
            chunks_processed=result["chunks_processed"],
            entities_extracted=result["entities_added"],
            message=f"Successfully processed {len(chunks)} chunks"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post("/ingest/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    doc_id: str = None,
    metadata: str = "{}"
) -> DocumentIngestionResponse:
    """
    Upload and ingest a document file.
    
    Accepts file uploads and processes them directly without requiring
    a file path on the server.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file temporarily
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse metadata
            metadata_dict = json.loads(metadata) if metadata else {}
            metadata_dict["original_filename"] = file.filename
            
            # Process the document
            chunks = document_processor.process_pdf(temp_file_path, doc_id)
            
            # Add metadata
            for chunk in chunks:
                chunk.metadata.update(metadata_dict)
            
            # Ingest through RAG pipeline
            result = await rag_pipeline.ingest_document(chunks)
            
            # Prepare response
            response = DocumentIngestionResponse(
                doc_id=chunks[0].doc_id,
                chunks_processed=result["chunks_processed"],
                entities_extracted=result["entities_added"],
                message=f"Successfully uploaded and processed {file.filename}"
            )
            
            return response
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading and ingesting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload and ingest document: {str(e)}"
        )


@router.get("/health", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """
    Health check endpoint to verify system status.
    
    Returns the status of all system components including:
    - Vector database
    - Knowledge graph
    - LLM service
    """
    try:
        # Get system status
        status = rag_pipeline.get_system_status()
        
        # Determine overall health
        services = {}
        overall_healthy = True
        
        # Check vector store
        if status.get("vector_store", {}).get("connected"):
            services["vector_store"] = "healthy"
        else:
            services["vector_store"] = "unhealthy"
            overall_healthy = False
        
        # Check knowledge graph
        kg_status = status.get("knowledge_graph", {})
        if kg_status.get("status") == "connected":
            services["knowledge_graph"] = "healthy"
        elif kg_status.get("status") == "disconnected":
            services["knowledge_graph"] = "degraded"  # Not critical
        else:
            services["knowledge_graph"] = "unhealthy"
        
        # Check LLM service
        if status.get("llm_service", {}).get("status") == "available":
            services["llm_service"] = "healthy"
        else:
            services["llm_service"] = "unhealthy"
            overall_healthy = False
        
        return HealthCheck(
            status="healthy" if overall_healthy else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheck(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0",
            services={"error": str(e)}
        )


@router.get("/stats")
async def get_statistics() -> Dict[str, Any]:
    """
    Get detailed statistics about the system.
    
    Returns information about:
    - Number of documents in vector store
    - Knowledge graph statistics
    - System performance metrics
    """
    try:
        status = rag_pipeline.get_system_status()
        
        stats = {
            "vector_store": {
                "document_count": status.get("vector_store", {}).get("document_count", 0)
            },
            "knowledge_graph": status.get("knowledge_graph", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, str]:
    """
    Delete a document and all its associated data.
    
    Removes:
    - Document chunks from vector store
    - Associated entities from knowledge graph
    """
    try:
        # Delete from vector store
        vector_deleted = rag_pipeline.vector_store.delete_document(doc_id)
        
        # Note: Knowledge graph deletion would require additional implementation
        # For now, we just delete from vector store
        
        if vector_deleted:
            return {"status": "success", "message": f"Document {doc_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document {doc_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {doc_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )
