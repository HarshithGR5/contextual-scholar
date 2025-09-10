#!/usr/bin/env python3
"""
Fallback LLM service that works without external API
"""
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FallbackLLMService:
    """Fallback LLM service that provides basic responses without API calls."""
    
    def __init__(self):
        """Initialize fallback service."""
        self.model = "fallback-local"
    
    async def generate_response(
        self, 
        prompt: str, 
        context_passages: List[str] = None,
        related_entities: List[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Generate a fallback response based on context."""
        
        try:
            # Extract key information from context
            if context_passages:
                # Simple keyword extraction and summary
                response = self._generate_context_summary(prompt, context_passages)
            else:
                # Basic response without context
                response = self._generate_basic_response(prompt)
            
            return {
                "response": response,
                "usage": {"total_tokens": len(response.split())},
                "finish_reason": "STOP",
                "source": "fallback_mode"
            }
            
        except Exception as e:
            logger.error(f"Error in fallback LLM: {e}")
            return {
                "response": "I apologize, but I'm currently unable to process your request. Please try again later.",
                "usage": {},
                "finish_reason": "ERROR",
                "source": "fallback_mode"
            }
    
    def _generate_context_summary(self, question: str, context_passages: List[str]) -> str:
        """Generate a response based on context passages."""
        
        # Combine context passages
        combined_context = "\n\n".join(context_passages[:3])  # Limit to first 3
        
        # Extract key sentences that might be relevant
        sentences = []
        for passage in context_passages:
            # Split into sentences and take first few
            passage_sentences = passage.split('. ')
            sentences.extend(passage_sentences[:2])
        
        # Create a structured response
        response_parts = []
        
        response_parts.append("Based on the available documents, here's what I found:")
        response_parts.append("")
        
        # Add relevant context
        if sentences:
            response_parts.append("Key Information:")
            for i, sentence in enumerate(sentences[:3], 1):
                if sentence.strip():
                    response_parts.append(f"• {sentence.strip()}")
        
        response_parts.append("")
        response_parts.append("Note: This response is generated from document context. ")
        response_parts.append("For more detailed analysis, please ensure the AI service is properly configured.")
        
        return "\n".join(response_parts)
    
    def _generate_basic_response(self, question: str) -> str:
        """Generate a basic response without context."""
        
        # Simple keyword-based responses
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'define', 'explain']):
            return f"I understand you're asking about: '{question}'\n\n" \
                   "I'm currently operating in fallback mode without access to the full AI capabilities. " \
                   "To get detailed answers, please ensure the Gemini API is properly configured with sufficient quota."
        
        elif any(word in question_lower for word in ['how', 'steps', 'process']):
            return f"You're asking about the process or method for: '{question}'\n\n" \
                   "I'm currently in fallback mode and cannot provide detailed step-by-step guidance. " \
                   "Please ensure the AI service is properly configured for comprehensive responses."
        
        elif any(word in question_lower for word in ['why', 'reason', 'cause']):
            return f"You're asking about the reasons or causes related to: '{question}'\n\n" \
                   "I'm currently operating in limited mode. " \
                   "For detailed explanations and analysis, please configure the full AI capabilities."
        
        else:
            return f"I received your question: '{question}'\n\n" \
                   "I'm currently operating in fallback mode due to API limitations. " \
                   "While I can access and search through your documents, " \
                   "I cannot provide full AI-powered analysis at the moment.\n\n" \
                   "Please check the API configuration and quota settings to enable full functionality."


# Global fallback service instance
fallback_llm_service = FallbackLLMService()
