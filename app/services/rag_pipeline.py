"""
Main RAG pipeline orchestrating retrieval, knowledge graph, and generation
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import re

from app.models.schemas import (
    ResearchQuery, 
    ResearchResponse, 
    RetrievedSource, 
    EntityRelation,
    DocumentChunk
)
from app.services.vector_store import vector_store_service
from app.services.knowledge_graph import kg_service
from app.services.llm_service import gemini_service
from app.utils.config import settings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline combining vector search, knowledge graph, and LLM generation."""
    
    def __init__(self):
        """Initialize the RAG pipeline."""
        self.vector_store = vector_store_service
        self.knowledge_graph = kg_service
        self.llm_service = gemini_service
    
    async def process_query(self, query: ResearchQuery) -> ResearchResponse:
        """Process a research query through the complete RAG pipeline."""
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {query.question}")
            
            # Step 1: Retrieve relevant documents using vector similarity
            retrieved_sources = await self._retrieve_documents(query.question, query.top_k)
            
            # Step 2: Get related entities from knowledge graph
            related_entities = []
            if query.include_entities and kg_service.is_connected():
                related_entities = await self._get_related_entities(query.question)
            
            # Step 3: Generate response using LLM with context
            answer = await self._generate_answer(
                query.question, 
                retrieved_sources, 
                related_entities
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response = ResearchResponse(
                answer=answer,
                sources=retrieved_sources,
                related_entities=related_entities,
                processing_time=processing_time
            )
            
            logger.info(f"Query processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def _retrieve_documents(self, query: str, top_k: int) -> List[RetrievedSource]:
        """Retrieve relevant documents using vector similarity search."""
        try:
            # Perform similarity search
            sources = self.vector_store.similarity_search(query, top_k)
            
            logger.info(f"Retrieved {len(sources)} documents for query")
            return sources
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def _get_related_entities(self, query: str) -> List[EntityRelation]:
        """Get related entities from the knowledge graph."""
        try:
            # Extract potential entity names from the query
            query_entities = self._extract_query_entities(query)
            
            related_entities = []
            
            # For each potential entity, find related entities in the graph
            for entity_name in query_entities:
                entity_relations = kg_service.get_related_entities(entity_name, max_depth=2)
                related_entities.extend(entity_relations)
            
            # Also search for entities matching keywords from the query
            keywords = self._extract_keywords(query)
            keyword_entities = kg_service.find_entities_by_keywords(keywords)
            related_entities.extend(keyword_entities)
            
            # Remove duplicates and limit results
            unique_entities = self._deduplicate_entities(related_entities)
            
            logger.info(f"Found {len(unique_entities)} related entities")
            return unique_entities[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error getting related entities: {e}")
            return []
    
    def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entity names from the query text."""
        # Simple heuristic: look for capitalized words and phrases
        # In a production system, you'd use NER models here
        
        entities = []
        
        # Find capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized_words)
        
        # Find quoted phrases
        quoted_phrases = re.findall(r'"([^"]*)"', query)
        entities.extend(quoted_phrases)
        
        # Remove common stop words that might be capitalized
        stop_words = {'The', 'This', 'That', 'What', 'How', 'Why', 'Where', 'When', 'Who'}
        entities = [entity for entity in entities if entity not in stop_words]
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Simple keyword extraction
        # Remove common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'how', 'why', 'where', 'when', 'who', 'which'
        }
        
        # Clean and split query
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords[:5]  # Return top 5 keywords
    
    def _deduplicate_entities(self, entities: List[EntityRelation]) -> List[EntityRelation]:
        """Remove duplicate entities while preserving the most informative ones."""
        seen_entities = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = entity.entity.lower()
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    async def _generate_answer(
        self, 
        question: str,
        sources: List[RetrievedSource],
        entities: List[EntityRelation]
    ) -> str:
        """Generate an answer using the LLM with retrieved context."""
        try:
            # Prepare context passages
            context_passages = []
            for source in sources:
                # Add document ID for citation
                passage = f"[{source.doc_id}] {source.chunk}"
                context_passages.append(passage)
            
            # Prepare entity information
            entity_data = []
            for entity in entities:
                entity_info = {
                    "entity": entity.entity,
                    "relationship": entity.relationship,
                    "context": entity.context
                }
                entity_data.append(entity_info)
            
            # Generate response using Gemini with fallback
            try:
                result = await gemini_service.generate_response(
                    prompt=question,
                    context_passages=context_passages,
                    related_entities=entity_data,
                    max_tokens=1000,
                    temperature=0.3
                )
                
                return result.get("response", "I apologize, but I couldn't generate a response.")
                
            except Exception as gemini_error:
                # Check if it's a quota/rate limit error
                error_msg = str(gemini_error).lower()
                if "429" in error_msg or "quota" in error_msg or "rate limit" in error_msg:
                    logger.warning(f"Gemini API quota exceeded, using fallback: {gemini_error}")
                    
                    # Use fallback LLM service
                    from app.services.fallback_llm import fallback_llm_service
                    fallback_result = await fallback_llm_service.generate_response(
                        prompt=question,
                        context_passages=context_passages,
                        related_entities=entity_data
                    )
                    
                    response = fallback_result.get("response", "")
                    # Add notice about fallback mode
                    response += "\n\n⚠️ Note: Response generated in fallback mode due to API quota limits."
                    
                    return response
                else:
                    # Re-raise non-quota errors
                    raise gemini_error
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    async def ingest_document(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Ingest document chunks into both vector store and knowledge graph."""
        try:
            start_time = time.time()
            
            # Add to vector store
            vector_success = self.vector_store.add_documents(chunks)
            
            # Extract entities and add to knowledge graph
            entities_added = 0
            if kg_service.is_connected():
                entities_added = await self._process_document_entities(chunks)
            
            processing_time = time.time() - start_time
            
            return {
                "chunks_processed": len(chunks),
                "vector_store_success": vector_success,
                "entities_added": entities_added,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            raise
    
    async def _process_document_entities(self, chunks: List[DocumentChunk]) -> int:
        """Extract entities from document chunks and add to knowledge graph."""
        entities_added = 0
        
        try:
            # Group chunks by document
            doc_chunks = {}
            for chunk in chunks:
                if chunk.doc_id not in doc_chunks:
                    doc_chunks[chunk.doc_id] = []
                doc_chunks[chunk.doc_id].append(chunk)
            
            # Process each document
            for doc_id, doc_chunk_list in doc_chunks.items():
                # Combine chunk content for entity extraction
                combined_text = " ".join([chunk.content for chunk in doc_chunk_list[:3]])  # Limit to first 3 chunks
                
                # Extract entities using LLM
                extracted_entities = await gemini_service.extract_entities(combined_text)
                
                # Add entities to knowledge graph
                if extracted_entities:
                    added = kg_service.add_document_entities(doc_id, extracted_entities)
                    entities_added += added
            
            logger.info(f"Added {entities_added} entities to knowledge graph")
            return entities_added
            
        except Exception as e:
            logger.error(f"Error processing document entities: {e}")
            return entities_added
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the status of all system components."""
        try:
            # Vector store status
            vector_status = {
                "connected": True,
                "document_count": self.vector_store.get_document_count()
            }
            
            # Knowledge graph status
            kg_status = kg_service.get_graph_statistics()
            
            return {
                "vector_store": vector_status,
                "knowledge_graph": kg_status,
                "llm_service": {"status": "available", "model": "gemini-2.0-flash"}
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()
