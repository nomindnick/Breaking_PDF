"""
Context-aware boundary detector that analyzes page sequences.

This detector looks at sliding windows of pages to better understand document
flow and structure, improving boundary detection accuracy.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    ProcessedPage,
    BoundaryResult,
    BoundaryType,
    DetectorType,
)
from pdf_splitter.detection.embeddings_detector import EmbeddingsDetector
from pdf_splitter.detection.heuristic_detector.enhanced_heuristic_detector import (
    EnhancedHeuristicDetector,
    create_enhanced_config
)

logger = logging.getLogger(__name__)


@dataclass
class WindowContext:
    """Context information for a window of pages."""
    pages: List[ProcessedPage]
    start_idx: int
    center_idx: int
    embeddings: Optional[np.ndarray] = None
    
    @property
    def size(self) -> int:
        return len(self.pages)
    
    @property
    def before_center(self) -> List[ProcessedPage]:
        """Pages before the center page."""
        center_offset = self.center_idx - self.start_idx
        return self.pages[:center_offset]
    
    @property
    def after_center(self) -> List[ProcessedPage]:
        """Pages after the center page."""
        center_offset = self.center_idx - self.start_idx
        return self.pages[center_offset + 1:]


class ContextAwareDetector(BaseDetector):
    """Detector that uses context windows for better accuracy."""
    
    def get_detector_type(self) -> DetectorType:
        """Return the detector type."""
        return DetectorType.HEURISTIC  # Context-aware is a type of heuristic
    
    def __init__(
        self,
        window_size: int = 5,
        embedding_model: str = 'all-MiniLM-L6-v2',
        use_enhanced_heuristics: bool = True
    ):
        """
        Initialize context-aware detector.
        
        Args:
            window_size: Number of pages to consider in context window
            embedding_model: Model for semantic analysis
            use_enhanced_heuristics: Whether to use enhanced heuristic detector
        """
        self.window_size = window_size
        self.half_window = window_size // 2
        
        # Initialize sub-detectors
        self.embeddings_detector = EmbeddingsDetector(
            model_name=embedding_model,
            similarity_threshold=0.6
        )
        
        if use_enhanced_heuristics:
            self.heuristic_detector = EnhancedHeuristicDetector(create_enhanced_config())
        else:
            from pdf_splitter.detection import HeuristicDetector, get_general_purpose_config
            self.heuristic_detector = HeuristicDetector(get_general_purpose_config())
    
    def detect_boundaries(
        self, pages: List[ProcessedPage], context: Optional[Dict] = None
    ) -> List[BoundaryResult]:
        """Detect boundaries using context windows."""
        if not pages or len(pages) < 2:
            return []
        
        boundaries = []
        
        # Pre-compute embeddings for all pages
        page_texts = [p.text for p in pages]
        all_embeddings = self.embeddings_detector.model.encode(
            page_texts, show_progress_bar=False
        )
        
        # Analyze each potential boundary
        for i in range(len(pages) - 1):
            window = self._get_window(pages, i, all_embeddings)
            
            # Analyze the boundary using context
            result = self._analyze_boundary_with_context(window)
            
            if result and result.confidence >= 0.3:  # Lower threshold for context
                boundaries.append(result)
        
        return boundaries
    
    def _get_window(
        self, 
        pages: List[ProcessedPage], 
        boundary_idx: int,
        embeddings: np.ndarray
    ) -> WindowContext:
        """Get a window of pages around a potential boundary."""
        # Calculate window bounds
        start = max(0, boundary_idx - self.half_window)
        end = min(len(pages), boundary_idx + self.half_window + 2)
        
        # Extract window
        window_pages = pages[start:end]
        window_embeddings = embeddings[start:end]
        
        return WindowContext(
            pages=window_pages,
            start_idx=start,
            center_idx=boundary_idx,
            embeddings=window_embeddings
        )
    
    def _analyze_boundary_with_context(self, window: WindowContext) -> Optional[BoundaryResult]:
        """Analyze a potential boundary using window context."""
        center_offset = window.center_idx - window.start_idx
        
        if center_offset >= len(window.pages) - 1:
            return None
        
        # Get pages around boundary
        prev_page = window.pages[center_offset]
        curr_page = window.pages[center_offset + 1]
        
        signals = {}
        
        # 1. Semantic coherence analysis
        coherence_signal = self._analyze_semantic_coherence(window, center_offset)
        if coherence_signal > 0:
            signals["semantic_shift"] = coherence_signal
        
        # 2. Topic consistency
        topic_signal = self._analyze_topic_consistency(window, center_offset)
        if topic_signal > 0:
            signals["topic_change"] = topic_signal
        
        # 3. Length pattern analysis
        length_signal = self._analyze_length_patterns(window, center_offset)
        if length_signal > 0:
            signals["length_anomaly"] = length_signal
        
        # 4. Style change detection
        style_signal = self._analyze_style_change(prev_page, curr_page, window)
        if style_signal > 0:
            signals["style_change"] = style_signal
        
        # 5. Get heuristic signals
        heuristic_results = self.heuristic_detector.detect_boundaries(
            [prev_page, curr_page]
        )
        if heuristic_results:
            signals["heuristic"] = heuristic_results[0].confidence
        
        # Combine signals
        if signals:
            # Weight different signals
            weights = {
                "semantic_shift": 0.3,
                "topic_change": 0.2,
                "length_anomaly": 0.1,
                "style_change": 0.2,
                "heuristic": 0.2
            }
            
            confidence = sum(
                signals.get(key, 0) * weights.get(key, 0.1)
                for key in signals
            )
            
            # Boost confidence if multiple strong signals
            strong_signals = sum(1 for v in signals.values() if v > 0.7)
            if strong_signals >= 2:
                confidence = min(1.0, confidence * 1.2)
            
            return BoundaryResult(
                page_number=window.center_idx,
                confidence=confidence,
                boundary_type=BoundaryType.DOCUMENT_START,
                evidence={
                    "detector": "context_aware",
                    "window_size": window.size,
                    "signals": signals,
                    "strong_signals": strong_signals
                },
                detector_type=DetectorType.HEURISTIC  # Using heuristic type for now
            )
        
        return None
    
    def _analyze_semantic_coherence(self, window: WindowContext, center_offset: int) -> float:
        """Analyze semantic coherence across the window."""
        if window.embeddings is None or len(window.embeddings) < 3:
            return 0.0
        
        # Calculate average similarity before and after boundary
        before_embeds = window.embeddings[:center_offset+1]
        after_embeds = window.embeddings[center_offset+1:]
        
        if len(before_embeds) < 2 or len(after_embeds) < 1:
            return 0.0
        
        # Coherence within groups
        before_coherence = self._calculate_group_coherence(before_embeds)
        after_coherence = self._calculate_group_coherence(after_embeds) if len(after_embeds) > 1 else 0.8
        
        # Similarity across boundary
        boundary_similarity = np.dot(
            window.embeddings[center_offset],
            window.embeddings[center_offset + 1]
        )
        
        # Strong signal if high coherence within groups but low across boundary
        if before_coherence > 0.7 and after_coherence > 0.7 and boundary_similarity < 0.5:
            return 0.9
        elif boundary_similarity < 0.4:
            return 0.7
        elif boundary_similarity < 0.6 and (before_coherence > 0.8 or after_coherence > 0.8):
            return 0.5
        
        return 0.0
    
    def _calculate_group_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate average pairwise similarity within a group."""
        if len(embeddings) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _analyze_topic_consistency(self, window: WindowContext, center_offset: int) -> float:
        """Analyze topic consistency using keywords."""
        # Extract keywords from before and after
        before_text = " ".join(p.text for p in window.before_center)
        after_text = " ".join(p.text for p in window.after_center[:2])  # Just look ahead a bit
        
        if not before_text or not after_text:
            return 0.0
        
        # Simple keyword extraction (top words)
        before_words = set(w.lower() for w in before_text.split() if len(w) > 4)
        after_words = set(w.lower() for w in after_text.split() if len(w) > 4)
        
        # Calculate overlap
        if not before_words or not after_words:
            return 0.0
        
        overlap = len(before_words & after_words) / min(len(before_words), len(after_words))
        
        # Low overlap suggests topic change
        if overlap < 0.1:
            return 0.8
        elif overlap < 0.3:
            return 0.5
        
        return 0.0
    
    def _analyze_length_patterns(self, window: WindowContext, center_offset: int) -> float:
        """Detect anomalies in page length patterns."""
        lengths = [len(p.text) for p in window.pages]
        
        if len(lengths) < 3:
            return 0.0
        
        # Calculate statistics
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        if std_length == 0:
            return 0.0
        
        # Check if boundary represents a significant change
        prev_length = lengths[center_offset]
        curr_length = lengths[center_offset + 1] if center_offset + 1 < len(lengths) else 0
        
        # Z-scores
        prev_z = abs(prev_length - mean_length) / std_length
        curr_z = abs(curr_length - mean_length) / std_length
        
        # Large change in length
        length_ratio = min(prev_length, curr_length) / max(prev_length, curr_length) if max(prev_length, curr_length) > 0 else 1
        
        if length_ratio < 0.2 and (prev_z > 1.5 or curr_z > 1.5):
            return 0.7
        elif length_ratio < 0.5:
            return 0.4
        
        return 0.0
    
    def _analyze_style_change(
        self, 
        prev_page: ProcessedPage, 
        curr_page: ProcessedPage,
        window: WindowContext
    ) -> float:
        """Detect changes in writing style or format."""
        prev_text = prev_page.text
        curr_text = curr_page.text
        
        if not prev_text or not curr_text:
            return 0.0
        
        # Calculate style metrics
        prev_metrics = self._calculate_style_metrics(prev_text)
        curr_metrics = self._calculate_style_metrics(curr_text)
        
        # Compare metrics
        differences = []
        for key in prev_metrics:
            if key in curr_metrics:
                diff = abs(prev_metrics[key] - curr_metrics[key])
                differences.append(diff)
        
        if not differences:
            return 0.0
        
        # Significant style change
        avg_diff = np.mean(differences)
        if avg_diff > 0.5:
            return 0.7
        elif avg_diff > 0.3:
            return 0.4
        
        return 0.0
    
    def _calculate_style_metrics(self, text: str) -> Dict[str, float]:
        """Calculate writing style metrics."""
        lines = text.split('\n')
        words = text.split()
        
        return {
            "avg_line_length": np.mean([len(l) for l in lines]) if lines else 0,
            "empty_line_ratio": len([l for l in lines if not l.strip()]) / len(lines) if lines else 0,
            "uppercase_ratio": len([c for c in text if c.isupper()]) / len(text) if text else 0,
            "digit_ratio": len([c for c in text if c.isdigit()]) / len(text) if text else 0,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
        }