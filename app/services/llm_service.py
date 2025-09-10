"""
Gemini API service for LLM-powered response generation
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional
import httpx
from datetime import datetime

from app.utils.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini 2.0 Flash API."""
    
    def __init__(self):
        """Initialize the Gemini API service."""
        self.api_key = settings.gemini_api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = "gemini-2.0-flash"
        self.headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
    
    async def generate_response(
        self, 
        prompt: str, 
        context_passages: List[str] = None,
        related_entities: List[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate a response using Gemini API."""
        
        try:
            # Construct the full prompt with context
            full_prompt = self._build_prompt(prompt, context_passages, related_entities)
            
            # Prepare request payload
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": full_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.95,
                    "topK": 40
                }
            }
            
            # Make API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/{self.model}:generateContent",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Extract the generated text
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        generated_text = candidate["content"]["parts"][0].get("text", "")
                        
                        return {
                            "response": generated_text,
                            "usage": result.get("usageMetadata", {}),
                            "finish_reason": candidate.get("finishReason", "STOP")
                        }
                
                # Fallback if structure is unexpected
                logger.warning("Unexpected response structure from Gemini API")
                return {
                    "response": "I apologize, but I couldn't generate a proper response. Please try again.",
                    "usage": {},
                    "finish_reason": "ERROR"
                }
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from Gemini API: {e.response.status_code} - {e.response.text}")
            raise Exception(f"API request failed: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Timeout while calling Gemini API")
            raise Exception("Request timeout")
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
    def _build_prompt(
        self, 
        user_question: str, 
        context_passages: List[str] = None,
        related_entities: List[Dict[str, Any]] = None
    ) -> str:
        """Build a comprehensive prompt with context and entities."""
        
        prompt_parts = [
            "You are a research assistant with expertise in academic and scientific literature.",
            "Answer the user's question using the provided context from documents and related entities from the knowledge graph.",
            "Provide accurate, well-structured responses with proper citations.",
            "",
            "IMPORTANT INSTRUCTIONS:",
            "- Use the provided context to answer the question",
            "- Cite sources using [doc_id] format when referencing specific documents",
            "- If the context doesn't contain sufficient information, clearly state this",
            "- Be precise and avoid making unsupported claims",
            "- Structure your response clearly with main points and supporting evidence",
            ""
        ]
        
        # Add document context if available
        if context_passages:
            prompt_parts.extend([
                "CONTEXT FROM DOCUMENTS:",
                "─" * 50
            ])
            
            for i, passage in enumerate(context_passages[:5], 1):  # Limit to top 5
                prompt_parts.append(f"[Document {i}]")
                prompt_parts.append(passage.strip())
                prompt_parts.append("")
        
        # Add knowledge graph entities if available
        if related_entities:
            prompt_parts.extend([
                "RELATED ENTITIES FROM KNOWLEDGE GRAPH:",
                "─" * 50
            ])
            
            for entity in related_entities[:10]:  # Limit to top 10
                entity_info = f"• {entity.get('entity', 'Unknown')}"
                if entity.get('relationship'):
                    entity_info += f" ({entity['relationship']})"
                if entity.get('context'):
                    entity_info += f" - {entity['context']}"
                prompt_parts.append(entity_info)
            
            prompt_parts.append("")
        
        # Add the user question
        prompt_parts.extend([
            "USER QUESTION:",
            "─" * 50,
            user_question,
            "",
            "RESPONSE:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the given text."""
        
        prompt = f"""
        Please provide a concise summary of the following text in approximately {max_length} words:
        
        {text}
        
        Summary:
        """
        
        try:
            result = await self.generate_response(
                prompt, 
                max_tokens=max_length * 2,  # Allow for some buffer
                temperature=0.2  # Lower temperature for more focused summaries
            )
            return result.get("response", "Summary not available")
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Summary generation failed"
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using Gemini."""
        
        prompt = f"""
        Extract key entities from the following text and categorize them. 
        Return the result as a JSON list where each entity has "name", "type", and "context" fields.
        
        Entity types should include: PERSON, ORGANIZATION, CONCEPT, TECHNOLOGY, LOCATION, DATE, etc.
        
        Text: {text}
        
        JSON Response:
        """
        
        try:
            result = await self.generate_response(
                prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            response_text = result.get("response", "[]")
            
            # Try to parse JSON response
            try:
                # Extract JSON from response (handle cases where LLM adds extra text)
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    entities = json.loads(json_str)
                    
                    # Validate and clean entities
                    valid_entities = []
                    for entity in entities:
                        if isinstance(entity, dict) and "name" in entity:
                            valid_entities.append({
                                "name": entity["name"],
                                "type": entity.get("type", "CONCEPT"),
                                "context": entity.get("context", "")
                            })
                    
                    return valid_entities
                
            except json.JSONDecodeError:
                logger.warning("Could not parse entity extraction JSON response")
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return []


# Global Gemini service instance
gemini_service = GeminiService()
