"""
Embeddings-based document boundary detector using semantic similarity.

This detector uses sentence transformers to generate embeddings for text chunks
and identifies boundaries based on semantic shifts between pages.
"""

import logging
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..base_detector import BaseDetector, BoundaryResult, ProcessedPage, DetectionContext, DetectorType, BoundaryType

logger = logging.getLogger(__name__)


class EmbeddingsDetector(BaseDetector):
    """
    Detect document boundaries using semantic similarity of text embeddings.
    
    This detector is particularly effective at catching:
    - Topic/subject changes
    - Style/tone shifts
    - Document type transitions
    - Boundaries without explicit patterns
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.5,
        max_text_length: int = 512,
        confidence_scaling: float = 2.0
    ):
        """
        Initialize the embeddings detector.
        
        Args:
            model_name: Sentence transformer model to use
            similarity_threshold: Threshold below which pages are considered different documents
            max_text_length: Maximum characters to use from each page for embedding
            confidence_scaling: Factor to scale raw similarity scores to confidence
        """
        super().__init__()
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.max_text_length = max_text_length
        self.confidence_scaling = confidence_scaling
        self._model = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model to avoid loading if not needed."""
        if self._model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def get_detector_type(self) -> DetectorType:
        """Return the detector type."""
        return DetectorType.EMBEDDINGS
    
    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None
    ) -> List[BoundaryResult]:
        """
        Detect boundaries by analyzing semantic similarity between consecutive pages.
        
        Args:
            pages: List of processed pages to analyze
            context: Optional detection context
            
        Returns:
            List of boundary detection results
        """
        if len(pages) < 2:
            return []
            
        logger.info(f"Running embeddings detection on {len(pages)} pages")
        
        # Extract meaningful text from each page
        page_texts = [self._extract_semantic_text(page) for page in pages]
        
        # Generate embeddings for all pages at once (more efficient)
        logger.debug("Generating embeddings for all pages")
        embeddings = self.model.encode(page_texts, show_progress_bar=False)
        
        # Calculate similarities between consecutive pages
        results = []
        for i in range(len(pages) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            
            # Low similarity indicates a potential boundary
            is_boundary = similarity < self.similarity_threshold
            
            # Calculate confidence based on how far below threshold we are
            if is_boundary:
                # The lower the similarity, the higher the confidence
                confidence = min(1.0, (self.similarity_threshold - similarity) * self.confidence_scaling)
            else:
                # Not a boundary, but indicate how confident we are
                confidence = min(1.0, (similarity - self.similarity_threshold) * self.confidence_scaling)
                
            result = BoundaryResult(
                page_number=i,
                boundary_type=BoundaryType.DOCUMENT_START if is_boundary else BoundaryType.PAGE_CONTINUATION,
                confidence=confidence,
                detector_type=self.get_detector_type(),
                evidence={
                    "similarity_score": float(similarity),
                    "threshold": self.similarity_threshold,
                    "text_length_p1": len(page_texts[i]),
                    "text_length_p2": len(page_texts[i + 1])
                },
                reasoning=self._generate_reasoning(similarity, is_boundary, pages[i].page_number, pages[i + 1].page_number)
            )
            
            results.append(result)
            
        logger.info(f"Embeddings detection complete. Found {sum(1 for r in results if r.boundary_type == BoundaryType.DOCUMENT_START)} boundaries")
        
        return results
    
    def _extract_semantic_text(self, page: ProcessedPage) -> str:
        """
        Extract the most semantically meaningful text from a page.
        
        Args:
            page: The page to extract text from
            
        Returns:
            Text suitable for embedding generation
        """
        text = page.text.strip()
        
        if not text:
            return ""
            
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)
        
        # Truncate to max length if needed
        if len(text) > self.max_text_length:
            # Try to truncate at a word boundary
            text = text[:self.max_text_length]
            last_space = text.rfind(' ')
            if last_space > self.max_text_length * 0.8:  # Only if we're not losing too much
                text = text[:last_space]
                
        return text
    
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Normalize vectors
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Ensure result is in [0, 1] range (handle floating point errors)
        return float(np.clip(similarity, 0.0, 1.0))
    
    def _generate_reasoning(self, similarity: float, is_boundary: bool, page1_num: int, page2_num: int) -> str:
        """
        Generate human-readable reasoning for the detection result.
        
        Args:
            similarity: Calculated similarity score
            is_boundary: Whether this was detected as a boundary
            page1_num: First page number
            page2_num: Second page number
            
        Returns:
            Reasoning string
        """
        if is_boundary:
            if similarity < 0.2:
                return f"Strong semantic shift detected between pages {page1_num} and {page2_num} (similarity: {similarity:.3f})"
            elif similarity < 0.35:
                return f"Significant semantic difference between pages {page1_num} and {page2_num} (similarity: {similarity:.3f})"
            else:
                return f"Semantic boundary detected between pages {page1_num} and {page2_num} (similarity: {similarity:.3f})"
        else:
            if similarity > 0.8:
                return f"Pages {page1_num} and {page2_num} are semantically very similar (similarity: {similarity:.3f})"
            elif similarity > 0.65:
                return f"Pages {page1_num} and {page2_num} show semantic continuity (similarity: {similarity:.3f})"
            else:
                return f"Pages {page1_num} and {page2_num} are marginally similar (similarity: {similarity:.3f})"
    
    def get_config(self) -> Dict[str, Any]:
        """Get the detector configuration."""
        return {
            "model_name": self.model_name,
            "similarity_threshold": self.similarity_threshold,
            "max_text_length": self.max_text_length,
            "confidence_scaling": self.confidence_scaling
        }