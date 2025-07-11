"""Comprehensive integration tests for cascade detection strategy."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time
from typing import List, Dict, Any

from pdf_splitter.detection import (
    SignalCombiner,
    SignalCombinerConfig,
    HeuristicDetector,
    LLMDetector,
    VisualDetector,
    ProcessedPage,
    DetectorType,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    get_production_config,
    get_production_cascade_config,
)
from pdf_splitter.core.config import PDFConfig


class TestCascadeIntegration:
    """Test suite for cascade detection strategy integration."""

    def create_test_pages(self, num_pages: int = 10) -> List[ProcessedPage]:
        """Create test pages with varying content."""
        pages = []
        for i in range(num_pages):
            # Create pages with different patterns
            if i == 3 or i == 7:
                # Clear boundary markers (email headers)
                text = f"From: sender@example.com\nTo: recipient@example.com\nSubject: Document {i//4 + 1}\n\nContent..."
            elif i == 5:
                # Medium confidence boundary
                text = f"Chapter {i//3 + 1}\n\nThis is the content of chapter {i//3 + 1}..."
            else:
                # Regular content
                text = f"This is page {i + 1} content. " * 20
            
            page = ProcessedPage(
                page_number=i + 1,
                text=text,
                ocr_confidence=0.95,
            )
            pages.append(page)
        return pages

    def create_mock_detectors(self):
        """Create mock detectors with controlled behavior."""
        # Mock heuristic detector
        mock_heuristic = Mock(spec=HeuristicDetector)
        mock_heuristic.get_detector_type.return_value = DetectorType.HEURISTIC
        
        def heuristic_detect(pages, context=None):
            results = []
            for i, page in enumerate(pages[:-1]):
                # High confidence at pages 3, 7 (email headers)
                if i == 3:
                    results.append(BoundaryResult(
                        page_number=i,
                        boundary_type=BoundaryType.DOCUMENT_END,
                        confidence=0.95,
                        detector_type=DetectorType.HEURISTIC,
                        evidence={'pattern': 'email_header'},
                        reasoning="Email header detected"
                    ))
                elif i == 7:
                    results.append(BoundaryResult(
                        page_number=i,
                        boundary_type=BoundaryType.DOCUMENT_END,
                        confidence=0.92,
                        detector_type=DetectorType.HEURISTIC,
                        evidence={'pattern': 'email_header'},
                        reasoning="Email header detected"
                    ))
                # Medium confidence at page 5
                elif i == 5:
                    results.append(BoundaryResult(
                        page_number=i,
                        boundary_type=BoundaryType.SECTION_BREAK,
                        confidence=0.75,
                        detector_type=DetectorType.HEURISTIC,
                        evidence={'pattern': 'chapter_header'},
                        reasoning="Chapter header detected"
                    ))
                # Low confidence at page 2
                elif i == 2:
                    results.append(BoundaryResult(
                        page_number=i,
                        boundary_type=BoundaryType.UNCERTAIN,
                        confidence=0.4,
                        detector_type=DetectorType.HEURISTIC,
                        evidence={'pattern': 'weak_signal'},
                        reasoning="Weak boundary signal"
                    ))
            return results
        
        mock_heuristic.detect_boundaries.side_effect = heuristic_detect
        
        # Mock LLM detector
        mock_llm = Mock(spec=LLMDetector)
        mock_llm.get_detector_type.return_value = DetectorType.LLM
        
        def llm_detect(pages, context=None):
            # Check if target pages specified
            if context and context.document_metadata and 'target_pages' in context.document_metadata:
                target_pages = context.document_metadata['target_pages']
                # Only process target pages
                results = []
                for target in target_pages:
                    if target == 2:
                        # Confirm boundary at page 2
                        results.append(BoundaryResult(
                            page_number=2,
                            boundary_type=BoundaryType.DOCUMENT_END,
                            confidence=0.95,
                            detector_type=DetectorType.LLM,
                            evidence={'llm_analysis': 'clear_boundary'},
                            reasoning="LLM confirmed document boundary"
                        ))
                return results
            return []
        
        mock_llm.detect_boundaries.side_effect = llm_detect
        
        # Mock visual detector
        mock_visual = Mock(spec=VisualDetector)
        mock_visual.get_detector_type.return_value = DetectorType.VISUAL
        
        def visual_detect(pages, context=None):
            # Detect visual difference at page 5
            return [
                BoundaryResult(
                    page_number=5,
                    boundary_type=BoundaryType.DOCUMENT_END,
                    confidence=0.8,
                    detector_type=DetectorType.VISUAL,
                    evidence={'visual_difference': 'high'},
                    reasoning="Visual layout change detected"
                )
            ]
        
        mock_visual.detect_boundaries.side_effect = visual_detect
        
        return mock_heuristic, mock_llm, mock_visual

    def test_cascade_high_confidence_bypass(self):
        """Test that high confidence heuristic results bypass further verification."""
        mock_heuristic, mock_llm, mock_visual = self.create_mock_detectors()
        
        # Create combiner with cascade config
        detectors = {
            DetectorType.HEURISTIC: mock_heuristic,
            DetectorType.LLM: mock_llm,
            DetectorType.VISUAL: mock_visual,
        }
        config = get_production_cascade_config()
        combiner = SignalCombiner(detectors, config)
        
        # Process pages
        pages = self.create_test_pages(10)
        results = combiner.detect_boundaries(pages)
        
        # Should have boundaries at pages 2, 3, 5, 7
        boundary_pages = sorted([r.page_number for r in results])
        assert 3 in boundary_pages  # High confidence
        assert 7 in boundary_pages  # High confidence
        assert 2 in boundary_pages  # Low confidence -> LLM verified
        assert 5 in boundary_pages  # Medium confidence -> Visual verified
        
        # Check cascade phases
        phase_map = {r.page_number: r.evidence.get('cascade_phase') for r in results}
        assert phase_map[3] == 'high_confidence'
        assert phase_map[7] == 'high_confidence'
        assert phase_map[2] == 'llm_verification'
        assert phase_map[5] in ['visual_verification', 'high_confidence']  # Could be merged
        
        # Verify LLM was only called for low confidence pages
        assert mock_llm.detect_boundaries.called
        llm_context = mock_llm.detect_boundaries.call_args[0][1]
        assert 'target_pages' in llm_context.document_metadata
        assert 2 in llm_context.document_metadata['target_pages']

    def test_cascade_confidence_preservation(self):
        """Test that original confidence is preserved through merging."""
        mock_heuristic, mock_llm, mock_visual = self.create_mock_detectors()
        
        detectors = {
            DetectorType.HEURISTIC: mock_heuristic,
            DetectorType.LLM: mock_llm,
            DetectorType.VISUAL: mock_visual,
        }
        config = SignalCombinerConfig(
            combination_strategy="cascade_ensemble",
            heuristic_confidence_threshold=0.9,
            require_llm_verification_below=0.7,
            confidence_boost_on_agreement=0.1,
        )
        combiner = SignalCombiner(detectors, config)
        
        pages = self.create_test_pages(10)
        results = combiner.detect_boundaries(pages)
        
        # Check that boosted confidence doesn't exceed thresholds
        for result in results:
            if result.original_confidence and result.original_confidence < 0.7:
                # Should not boost above LLM threshold
                assert result.confidence < 0.7
            elif result.original_confidence and result.original_confidence < 0.9:
                # Should not boost above heuristic threshold
                assert result.confidence < 0.9

    def test_cascade_performance_tracking(self):
        """Test that cascade strategy tracks performance correctly."""
        mock_heuristic, mock_llm, mock_visual = self.create_mock_detectors()
        
        # Store original side effects
        original_heuristic_side_effect = mock_heuristic.detect_boundaries.side_effect
        original_llm_side_effect = mock_llm.detect_boundaries.side_effect
        
        # Add timing to mock detectors
        def timed_heuristic_detect(pages, context=None):
            time.sleep(0.01)  # Simulate processing time
            return original_heuristic_side_effect(pages, context)
        
        def timed_llm_detect(pages, context=None):
            time.sleep(0.1)  # Simulate slower LLM
            return original_llm_side_effect(pages, context)
        
        mock_heuristic.detect_boundaries.side_effect = timed_heuristic_detect
        mock_llm.detect_boundaries.side_effect = timed_llm_detect
        
        detectors = {
            DetectorType.HEURISTIC: mock_heuristic,
            DetectorType.LLM: mock_llm,
            DetectorType.VISUAL: mock_visual,
        }
        config = get_production_cascade_config()
        combiner = SignalCombiner(detectors, config)
        
        start_time = time.time()
        pages = self.create_test_pages(10)
        results = combiner.detect_boundaries(pages)
        end_time = time.time()
        
        # Should be faster than running all detectors on all pages
        total_time = end_time - start_time
        assert total_time < 1.5  # Should be much less than 10 pages * 0.1s LLM
        
        # Verify selective LLM usage
        assert mock_llm.detect_boundaries.call_count == 1
        assert mock_heuristic.detect_boundaries.call_count == 1
        assert mock_visual.detect_boundaries.call_count == 1

    def test_cascade_empty_results_handling(self):
        """Test cascade handles empty results from detectors."""
        # Create detectors that return no results
        mock_heuristic = Mock(spec=HeuristicDetector)
        mock_heuristic.get_detector_type.return_value = DetectorType.HEURISTIC
        mock_heuristic.detect_boundaries.return_value = []
        
        mock_llm = Mock(spec=LLMDetector)
        mock_llm.get_detector_type.return_value = DetectorType.LLM
        mock_llm.detect_boundaries.return_value = []
        
        detectors = {
            DetectorType.HEURISTIC: mock_heuristic,
            DetectorType.LLM: mock_llm,
        }
        config = get_production_cascade_config()
        combiner = SignalCombiner(detectors, config)
        
        pages = self.create_test_pages(5)
        results = combiner.detect_boundaries(pages)
        
        # Should handle gracefully
        assert isinstance(results, list)
        # LLM should be called for all pages since no heuristic boundaries
        assert mock_llm.detect_boundaries.called

    def test_cascade_with_real_detectors(self):
        """Test cascade with real detector instances (mocked backends)."""
        # Use real detector classes with mocked LLM/PDF backends
        with patch('pdf_splitter.detection.llm_detector.LLMDetector._check_ollama_availability', return_value=False):
            with patch('pdf_splitter.detection.llm_detector.LLMDetector._get_embedded_prompt', return_value="test"):
                heuristic = HeuristicDetector(get_production_config())
                llm = LLMDetector(cache_enabled=False)
                visual = VisualDetector(pdf_handler=None)
                
                detectors = {
                    DetectorType.HEURISTIC: heuristic,
                    DetectorType.LLM: llm,
                    DetectorType.VISUAL: visual,
                }
                config = get_production_cascade_config()
                combiner = SignalCombiner(detectors, config)
                
                pages = self.create_test_pages(5)
                # Add rendered images for visual detector
                for page in pages:
                    page.rendered_image = b"fake image data"
                
                # Should not crash even with unavailable LLM
                results = combiner.detect_boundaries(pages)
                assert isinstance(results, list)

    def test_cascade_detector_type_tracking(self):
        """Test that detector types are properly tracked through cascade."""
        mock_heuristic, mock_llm, mock_visual = self.create_mock_detectors()
        
        detectors = {
            DetectorType.HEURISTIC: mock_heuristic,
            DetectorType.LLM: mock_llm,
            DetectorType.VISUAL: mock_visual,
        }
        config = get_production_cascade_config()
        combiner = SignalCombiner(detectors, config)
        
        pages = self.create_test_pages(10)
        results = combiner.detect_boundaries(pages)
        
        # Check detector types
        for result in results:
            # Results might be COMBINED if merged, or original type if not merged
            assert result.detector_type in [DetectorType.HEURISTIC, DetectorType.LLM, DetectorType.VISUAL, DetectorType.COMBINED]
            # But evidence should show cascade phase
            assert 'cascade_phase' in result.evidence