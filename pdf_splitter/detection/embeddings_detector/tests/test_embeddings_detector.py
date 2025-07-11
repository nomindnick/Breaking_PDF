"""
Tests for the embeddings-based document boundary detector.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pdf_splitter.detection import (
    EmbeddingsDetector,
    ProcessedPage,
    BoundaryType,
    DetectorType,
    DetectionContext,
)


class TestEmbeddingsDetector:
    """Test suite for the EmbeddingsDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return EmbeddingsDetector(
            model_name='all-MiniLM-L6-v2',
            similarity_threshold=0.5,
            max_text_length=512,
            confidence_scaling=2.0
        )
    
    @pytest.fixture
    def sample_pages(self):
        """Create sample pages for testing."""
        return [
            ProcessedPage(
                page_number=0,
                text="This is a letter about project management and deadlines.",
                ocr_confidence=0.95,
                page_type="text"
            ),
            ProcessedPage(
                page_number=1,
                text="The project timeline shows key milestones and deliverables.",
                ocr_confidence=0.95,
                page_type="text"
            ),
            ProcessedPage(
                page_number=2,
                text="Invoice #12345\nBilling Date: 2024-01-15\nAmount Due: $5,000",
                ocr_confidence=0.95,
                page_type="text"
            ),
            ProcessedPage(
                page_number=3,
                text="Payment Terms: Net 30\nThank you for your business.",
                ocr_confidence=0.95,
                page_type="text"
            ),
        ]
    
    def test_init(self, detector):
        """Test detector initialization."""
        assert detector.model_name == 'all-MiniLM-L6-v2'
        assert detector.similarity_threshold == 0.5
        assert detector.max_text_length == 512
        assert detector.confidence_scaling == 2.0
        assert detector._model is None  # Lazy loading
    
    def test_get_detector_type(self, detector):
        """Test detector type identification."""
        assert detector.get_detector_type() == DetectorType.EMBEDDINGS
    
    def test_extract_semantic_text(self, detector):
        """Test semantic text extraction."""
        # Test normal text
        page = ProcessedPage(
            page_number=0,
            text="  This is a test.  \n\n  With multiple lines.  \n  And spaces.  ",
            ocr_confidence=0.95,
            page_type="text"
        )
        result = detector._extract_semantic_text(page)
        assert result == "This is a test. With multiple lines. And spaces."
        
        # Test empty text
        empty_page = ProcessedPage(
            page_number=1,
            text="",
            ocr_confidence=0.0,
            page_type="empty"
        )
        assert detector._extract_semantic_text(empty_page) == ""
        
        # Test text truncation
        long_text = "word " * 200  # 1000 characters
        long_page = ProcessedPage(
            page_number=2,
            text=long_text,
            ocr_confidence=0.95,
            page_type="text"
        )
        result = detector._extract_semantic_text(long_page)
        assert len(result) <= detector.max_text_length
        assert result.endswith("word")  # Should truncate at word boundary
    
    def test_cosine_similarity(self, detector):
        """Test cosine similarity calculation."""
        # Test identical vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        assert detector._cosine_similarity(vec1, vec2) == pytest.approx(1.0)
        
        # Test orthogonal vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        assert detector._cosine_similarity(vec1, vec2) == pytest.approx(0.0)
        
        # Test opposite vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([-1, 0, 0])
        # Cosine similarity can be negative, but we clip to [0, 1]
        assert detector._cosine_similarity(vec1, vec2) == pytest.approx(0.0)
        
        # Test similar vectors
        vec1 = np.array([1, 1, 0])
        vec2 = np.array([1, 0.5, 0])
        similarity = detector._cosine_similarity(vec1, vec2)
        assert 0.8 < similarity < 1.0
    
    def test_generate_reasoning(self, detector):
        """Test reasoning generation."""
        # Test boundary detection reasoning
        reasoning = detector._generate_reasoning(0.1, True, 0, 1)
        assert "Strong semantic shift" in reasoning
        assert "0.100" in reasoning
        
        reasoning = detector._generate_reasoning(0.3, True, 1, 2)
        assert "Significant semantic difference" in reasoning
        
        reasoning = detector._generate_reasoning(0.45, True, 2, 3)
        assert "Semantic boundary detected" in reasoning
        
        # Test non-boundary reasoning
        reasoning = detector._generate_reasoning(0.9, False, 3, 4)
        assert "semantically very similar" in reasoning
        
        reasoning = detector._generate_reasoning(0.7, False, 4, 5)
        assert "semantic continuity" in reasoning
        
        reasoning = detector._generate_reasoning(0.55, False, 5, 6)
        assert "marginally similar" in reasoning
    
    @patch('pdf_splitter.detection.embeddings_detector.embeddings_detector.SentenceTransformer')
    def test_detect_boundaries(self, mock_transformer_class, detector, sample_pages):
        """Test boundary detection with mocked embeddings."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        
        # Create mock embeddings that simulate topic change
        mock_embeddings = np.array([
            [1.0, 0.0, 0.0],  # Page 0 - project letter
            [0.9, 0.1, 0.0],  # Page 1 - similar to page 0
            [0.0, 1.0, 0.0],  # Page 2 - invoice (different)
            [0.1, 0.9, 0.0],  # Page 3 - similar to page 2
        ])
        mock_model.encode.return_value = mock_embeddings
        
        # Run detection
        results = detector.detect_boundaries(sample_pages)
        
        # Verify results
        assert len(results) == 3  # n-1 boundaries for n pages
        
        # Pages 0-1 should be similar (not a boundary)
        assert results[0].boundary_type == BoundaryType.PAGE_CONTINUATION
        assert results[0].confidence > 0.5
        
        # Pages 1-2 should be different (boundary)
        assert results[1].boundary_type == BoundaryType.DOCUMENT_START
        assert results[1].confidence > 0.5
        assert results[1].evidence['similarity_score'] < detector.similarity_threshold
        
        # Pages 2-3 should be similar (not a boundary)
        assert results[2].boundary_type == BoundaryType.PAGE_CONTINUATION
        assert results[2].confidence > 0.5
    
    @patch('pdf_splitter.detection.embeddings_detector.embeddings_detector.SentenceTransformer')
    def test_detect_boundaries_empty_pages(self, mock_transformer_class, detector):
        """Test handling of empty pages."""
        # Mock the model
        mock_model = Mock()
        mock_transformer_class.return_value = mock_model
        mock_model.encode.return_value = np.array([[0.0], [0.0]])
        
        pages = [
            ProcessedPage(page_number=0, text="", ocr_confidence=0.0, page_type="empty"),
            ProcessedPage(page_number=1, text="", ocr_confidence=0.0, page_type="empty"),
        ]
        
        results = detector.detect_boundaries(pages)
        assert len(results) == 1
        # Empty pages should have low confidence
        assert results[0].confidence < 0.5
    
    def test_detect_boundaries_single_page(self, detector):
        """Test with single page (no boundaries)."""
        pages = [ProcessedPage(page_number=0, text="Single page", ocr_confidence=0.95, page_type="text")]
        results = detector.detect_boundaries(pages)
        assert len(results) == 0
    
    def test_detect_boundaries_no_pages(self, detector):
        """Test with no pages."""
        results = detector.detect_boundaries([])
        assert len(results) == 0
    
    def test_get_config(self, detector):
        """Test configuration retrieval."""
        config = detector.get_config()
        assert config['model_name'] == 'all-MiniLM-L6-v2'
        assert config['similarity_threshold'] == 0.5
        assert config['max_text_length'] == 512
        assert config['confidence_scaling'] == 2.0
    
    def test_lazy_model_loading(self, detector):
        """Test that model is loaded lazily."""
        # Model should not be loaded initially
        assert detector._model is None
        
        # Accessing model property should load it
        with patch('pdf_splitter.detection.embeddings_detector.embeddings_detector.SentenceTransformer') as mock_transformer:
            mock_model = Mock()
            mock_transformer.return_value = mock_model
            
            # First access loads the model
            model1 = detector.model
            assert model1 is mock_model
            mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
            
            # Second access returns cached model
            model2 = detector.model
            assert model2 is mock_model
            assert mock_transformer.call_count == 1  # Not called again
    
    @pytest.mark.integration
    def test_real_model_integration(self):
        """Integration test with actual sentence transformer model."""
        # This test requires the model to be downloaded
        detector = EmbeddingsDetector(model_name='all-MiniLM-L6-v2')
        
        pages = [
            ProcessedPage(
                page_number=0,
                text="Machine learning is a subset of artificial intelligence.",
                ocr_confidence=0.95,
                page_type="text"
            ),
            ProcessedPage(
                page_number=1,
                text="Deep learning uses neural networks with multiple layers.",
                ocr_confidence=0.95,
                page_type="text"
            ),
            ProcessedPage(
                page_number=2,
                text="The weather today is sunny with clear skies.",
                ocr_confidence=0.95,
                page_type="text"
            ),
        ]
        
        results = detector.detect_boundaries(pages)
        
        # Pages 0-1 are about ML/AI (similar)
        assert results[0].boundary_type == BoundaryType.PAGE_CONTINUATION
        assert results[0].evidence['similarity_score'] > 0.5
        
        # Pages 1-2 are different topics (ML vs weather)
        assert results[1].boundary_type == BoundaryType.DOCUMENT_START
        assert results[1].evidence['similarity_score'] < 0.5