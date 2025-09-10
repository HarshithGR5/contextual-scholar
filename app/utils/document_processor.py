"""
Document processing utilities for PDF parsing and text chunking
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import PyPDF2

from app.models.schemas import DocumentChunk
from app.utils.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing documents and creating chunks for the RAG system."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
    
    def process_pdf(self, file_path: str, doc_id: Optional[str] = None) -> List[DocumentChunk]:
        """Process a PDF file and return document chunks."""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path_obj.suffix.lower() == '.pdf':
                raise ValueError("Only PDF files are supported")
            
            # Generate document ID if not provided
            if not doc_id:
                doc_id = self._generate_doc_id(file_path)
            
            # Extract text from PDF
            text_content = self._extract_pdf_text(file_path)
            
            if not text_content.strip():
                raise ValueError("No text content found in PDF")
            
            # Create metadata
            metadata = {
                "file_path": str(file_path_obj),
                "file_name": file_path_obj.name,
                "file_size": file_path_obj.stat().st_size,
                "title": file_path_obj.stem
            }
            
            # Create chunks
            chunks = self._create_chunks(text_content, doc_id, metadata)
            
            logger.info(f"Processed PDF '{file_path}' into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF '{file_path}': {e}")
            raise
    
    def process_text(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """Process raw text and return document chunks."""
        try:
            if not text.strip():
                raise ValueError("Empty text provided")
            
            metadata = metadata or {}
            chunks = self._create_chunks(text, doc_id, metadata)
            
            logger.info(f"Processed text into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            # Clean up the text
                            cleaned_text = self._clean_text(page_text)
                            text_content.append(cleaned_text)
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num}: {e}")
                        continue
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        import re
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _create_chunks(self, text: str, doc_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create overlapping text chunks from the document."""
        chunks = []
        
        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return chunks
        
        current_chunk = ""
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, create a new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    doc_id=doc_id,
                    chunk_id=f"chunk_{chunk_index:04d}",
                    content=current_chunk.strip(),
                    metadata={**metadata, "chunk_index": chunk_index}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
                
                chunk_index += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunk = DocumentChunk(
                doc_id=doc_id,
                chunk_id=f"chunk_{chunk_index:04d}",
                content=current_chunk.strip(),
                metadata={**metadata, "chunk_index": chunk_index}
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristics."""
        import re
        
        # Simple sentence splitting - could be improved with proper NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last `overlap_size` characters from text for chunk overlap."""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at word boundaries
        overlap_text = text[-overlap_size:]
        
        # Find the first space to avoid breaking words
        space_index = overlap_text.find(' ')
        if space_index > 0:
            overlap_text = overlap_text[space_index + 1:]
        
        return overlap_text
    
    def _generate_doc_id(self, file_path: str) -> str:
        """Generate a unique document ID based on file path and content."""
        file_path_obj = Path(file_path)
        
        # Use file name and modification time for ID
        file_stat = file_path_obj.stat()
        id_string = f"{file_path_obj.name}_{file_stat.st_mtime}_{file_stat.st_size}"
        
        # Create hash for shorter ID
        hash_object = hashlib.md5(id_string.encode())
        doc_id = f"doc_{hash_object.hexdigest()[:8]}"
        
        return doc_id
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return [".pdf"]
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file can be processed."""
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                return False
            
            if file_path_obj.suffix.lower() not in self.get_supported_formats():
                return False
            
            # Additional validation could go here
            return True
            
        except Exception:
            return False


# Global document processor instance
document_processor = DocumentProcessor()
