"""Test LLM detector's ability to respect target pages in context."""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import List

from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.base_detector import (
    ProcessedPage,
    DetectionContext,
    BoundaryResult,
)
from pdf_splitter.core.config import PDFConfig


class TestLLMTargetPages:
    """Test suite for LLM detector target page functionality."""

    def create_test_pages(self, num_pages: int = 10) -> List[ProcessedPage]:
        """Create test pages with dummy content."""
        pages = []
        for i in range(num_pages):
            page = ProcessedPage(
                page_number=i + 1,
                text=f"This is page {i + 1} content. " * 20,
                ocr_confidence=0.95,
            )
            pages.append(page)
        return pages

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_processes_all_pages_without_target(self, mock_prompt):
        """Test that all pages are processed when no target pages specified."""
        # Setup
        mock_prompt.return_value = "test prompt"
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair", return_value=(False, 0.3, "Different documents")) as mock_analyze:
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(5)
                
                # Execute
                results = detector.detect_boundaries(pages)
                
                # Verify all consecutive pairs were analyzed
                assert mock_analyze.call_count == 4  # 5 pages = 4 pairs
                
                # Check the pairs that were analyzed
                call_args = [call[0] for call in mock_analyze.call_args_list]
                expected_pairs = [(pages[i], pages[i+1]) for i in range(4)]
                assert call_args == expected_pairs

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_processes_only_target_pages(self, mock_prompt):
        """Test that only target pages are processed when specified."""
        # Setup
        mock_prompt.return_value = "test prompt"
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair", return_value=(True, 0.9, "Document boundary detected")) as mock_analyze:
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(10)
                
                # Create context with target pages 2, 3, 7 (0-indexed)
                # This should process pairs: (1,2), (2,3), (3,4), (6,7), (7,8)
                context = DetectionContext(
                    config=PDFConfig(),
                    total_pages=10,
                    document_metadata={"target_pages": [2, 3, 7]},
                )
                
                # Execute
                results = detector.detect_boundaries(pages, context)
                
                # Verify only target-related pairs were analyzed
                assert mock_analyze.call_count == 5
                
                # Check which pairs were analyzed
                analyzed_indices = []
                for call in mock_analyze.call_args_list:
                    page1, page2 = call[0]
                    analyzed_indices.append((page1.page_number - 1, page2.page_number - 1))
                
                expected_indices = [(1, 2), (2, 3), (3, 4), (6, 7), (7, 8)]
                assert sorted(analyzed_indices) == sorted(expected_indices)

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_empty_target_pages_processes_none(self, mock_prompt):
        """Test that no pages are processed when target_pages is empty list."""
        # Setup
        mock_prompt.return_value = "test prompt"
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair") as mock_analyze:
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(5)
                
                # Create context with empty target pages
                context = DetectionContext(
                    config=PDFConfig(),
                    total_pages=5,
                    document_metadata={"target_pages": []},
                )
                
                # Execute
                results = detector.detect_boundaries(pages, context)
                
                # Verify no pages were analyzed
                assert mock_analyze.call_count == 0
                assert len(results) == 0

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_target_pages_with_boundaries(self, mock_prompt):
        """Test that boundaries are correctly created for target pages."""
        # Setup
        mock_prompt.return_value = "test prompt"
        
        # Mock different responses for different page pairs
        def analyze_side_effect(page1, page2):
            # Boundary between pages 3-4
            if page1.page_number == 3:
                return (True, 0.95, "Clear document boundary")
            return (False, 0.2, "Same document")
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair", side_effect=analyze_side_effect):
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(8)
                
                # Target pages that will include the boundary
                context = DetectionContext(
                    config=PDFConfig(),
                    total_pages=8,
                    document_metadata={"target_pages": [2, 3, 4, 5]},
                )
                
                # Execute
                results = detector.detect_boundaries(pages, context)
                
                # Verify
                assert len(results) == 1
                assert results[0].page_number == 3
                assert results[0].confidence == 0.95
                assert results[0].next_page_number == 4

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_logging_with_target_pages(self, mock_prompt, caplog):
        """Test that appropriate logging occurs when using target pages."""
        import logging
        # Setup
        mock_prompt.return_value = "test prompt"
        
        # Set log level to capture INFO messages
        caplog.set_level(logging.INFO)
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair", return_value=(False, 0.5, "Test")):
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(5)
                
                context = DetectionContext(
                    config=PDFConfig(),
                    total_pages=5,
                    document_metadata={"target_pages": [1, 3]},
                )
                
                # Execute
                results = detector.detect_boundaries(pages, context)
                
                # Verify logging
                assert "Processing only target pages: [1, 3]" in caplog.text
                assert "LLM detector processed 0 boundaries from target pages" in caplog.text

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_target_pages_edge_cases(self, mock_prompt):
        """Test edge cases for target page handling."""
        mock_prompt.return_value = "test prompt"
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            detector = LLMDetector(cache_enabled=False)
            pages = self.create_test_pages(5)
            
            # Test with None context
            with patch.object(detector, "_analyze_page_pair", return_value=(False, 0.5, "Test")) as mock:
                detector.detect_boundaries(pages, None)
                assert mock.call_count == 4  # All pairs processed
            
            # Test with context but no metadata
            context = DetectionContext(config=PDFConfig(), total_pages=5)
            with patch.object(detector, "_analyze_page_pair", return_value=(False, 0.5, "Test")) as mock:
                detector.detect_boundaries(pages, context)
                assert mock.call_count == 4  # All pairs processed

    @patch("pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt")
    def test_target_pages_indexing(self, mock_prompt):
        """Test that target page indexing works correctly with 0-based indices."""
        mock_prompt.return_value = "test prompt"
        
        with patch.object(LLMDetector, "_check_ollama_availability", return_value=True):
            with patch.object(LLMDetector, "_analyze_page_pair", return_value=(False, 0.5, "Test")) as mock_analyze:
                detector = LLMDetector(cache_enabled=False)
                pages = self.create_test_pages(6)
                
                # Target page 0 should process pair (0,1)
                # Target page 4 should process pairs (3,4) and (4,5)
                context = DetectionContext(
                    config=PDFConfig(),
                    total_pages=6,
                    document_metadata={"target_pages": [0, 4]},
                )
                
                # Execute
                results = detector.detect_boundaries(pages, context)
                
                # Verify
                assert mock_analyze.call_count == 3
                
                # Check which pairs were analyzed
                analyzed_indices = []
                for call in mock_analyze.call_args_list:
                    page1, page2 = call[0]
                    analyzed_indices.append((page1.page_number - 1, page2.page_number - 1))
                
                expected_indices = [(0, 1), (3, 4), (4, 5)]
                assert sorted(analyzed_indices) == sorted(expected_indices)