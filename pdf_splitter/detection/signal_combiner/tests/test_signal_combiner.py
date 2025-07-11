"""Tests for the signal combiner module."""

import pytest
from unittest.mock import Mock, patch
from typing import List, Optional

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.preprocessing.pdf_handler import PageType
from pdf_splitter.detection.signal_combiner import (
    SignalCombiner,
    SignalCombinerConfig,
    CombinationStrategy,
)


class MockDetector(BaseDetector):
    """Mock detector for testing."""

    def __init__(self, detector_type: DetectorType, results: List[BoundaryResult]):
        super().__init__()
        self.detector_type = detector_type
        self.results = results
        self.call_count = 0

    def get_detector_type(self) -> DetectorType:
        return self.detector_type

    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        self.call_count += 1
        return self.results


@pytest.fixture
def sample_pages():
    """Create sample processed pages for testing."""
    pages = []
    for i in range(10):
        pages.append(
            ProcessedPage(
                page_number=i,
                text=f"This is page {i} content. Lorem ipsum dolor sit amet.",
                ocr_confidence=0.95,
                page_type=PageType.SEARCHABLE,
                metadata={"page_num": i},
            )
        )
    return pages


@pytest.fixture
def heuristic_detector():
    """Create a mock heuristic detector with various confidence levels."""
    results = [
        BoundaryResult(
            page_number=0,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.95,  # High confidence
            detector_type=DetectorType.HEURISTIC,
            evidence={"pattern": "document_start"},
            reasoning="Strong document start indicators",
        ),
        BoundaryResult(
            page_number=3,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.65,  # Low confidence - needs LLM
            detector_type=DetectorType.HEURISTIC,
            evidence={"pattern": "weak_start"},
            reasoning="Weak document start indicators",
        ),
        BoundaryResult(
            page_number=6,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.8,  # Medium confidence - needs visual
            detector_type=DetectorType.HEURISTIC,
            evidence={"pattern": "medium_start"},
            reasoning="Medium document start indicators",
        ),
    ]
    return MockDetector(DetectorType.HEURISTIC, results)


@pytest.fixture
def llm_detector():
    """Create a mock LLM detector."""
    results = [
        BoundaryResult(
            page_number=3,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.9,
            detector_type=DetectorType.LLM,
            evidence={"llm_analysis": "strong_boundary"},
            reasoning="LLM detected clear document transition",
        ),
        BoundaryResult(
            page_number=7,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.85,
            detector_type=DetectorType.LLM,
            evidence={"llm_analysis": "new_document"},
            reasoning="LLM detected new document start",
        ),
    ]
    return MockDetector(DetectorType.LLM, results)


@pytest.fixture
def visual_detector():
    """Create a mock visual detector."""
    results = [
        BoundaryResult(
            page_number=6,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.75,
            detector_type=DetectorType.VISUAL,
            evidence={"visual_similarity": 0.3},
            reasoning="Low visual similarity to previous page",
        ),
    ]
    return MockDetector(DetectorType.VISUAL, results)


class TestSignalCombinerConfig:
    """Test the SignalCombinerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SignalCombinerConfig()

        assert config.heuristic_confidence_threshold == 0.9
        assert config.require_llm_verification_below == 0.7
        assert config.visual_verification_range == (0.7, 0.9)
        assert config.combination_strategy == CombinationStrategy.CASCADE_ENSEMBLE
        assert config.enable_parallel_processing is True

    def test_weight_normalization(self):
        """Test that detector weights are normalized to sum to 1.0."""
        config = SignalCombinerConfig(
            detector_weights={
                DetectorType.HEURISTIC: 1.0,
                DetectorType.LLM: 2.0,
                DetectorType.VISUAL: 1.0,
            }
        )

        total = sum(config.detector_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_invalid_thresholds(self):
        """Test validation of invalid thresholds."""
        with pytest.raises(ValueError):
            SignalCombinerConfig(heuristic_confidence_threshold=1.5)

        with pytest.raises(ValueError):
            SignalCombinerConfig(require_llm_verification_below=-0.1)

        with pytest.raises(ValueError):
            SignalCombinerConfig(visual_verification_range=(0.8, 0.7))


class TestSignalCombiner:
    """Test the SignalCombiner class."""

    def test_initialization(self, heuristic_detector, llm_detector):
        """Test signal combiner initialization."""
        detectors = {
            DetectorType.HEURISTIC: heuristic_detector,
            DetectorType.LLM: llm_detector,
        }

        combiner = SignalCombiner(detectors)
        assert combiner.get_detector_type() == DetectorType.COMBINED
        assert len(combiner.detectors) == 2

    def test_empty_detectors_error(self):
        """Test that empty detectors raises an error."""
        with pytest.raises(ValueError):
            SignalCombiner({})

    def test_cascade_ensemble_strategy(
        self, sample_pages, heuristic_detector, llm_detector, visual_detector
    ):
        """Test cascade ensemble detection strategy."""
        detectors = {
            DetectorType.HEURISTIC: heuristic_detector,
            DetectorType.LLM: llm_detector,
            DetectorType.VISUAL: visual_detector,
        }

        config = SignalCombinerConfig(
            combination_strategy=CombinationStrategy.CASCADE_ENSEMBLE,
            enable_parallel_processing=False,  # For predictable test behavior
        )

        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(sample_pages)

        # Should have results from all detectors
        assert len(results) > 0

        # High confidence heuristic result should be included
        assert any(r.page_number == 0 and r.confidence >= 0.9 for r in results)

        # Low confidence heuristic should be verified by LLM
        assert any(r.page_number == 3 for r in results)

        # Check that detectors were called
        assert heuristic_detector.call_count == 1
        assert llm_detector.call_count == 1  # Called for verification
        assert visual_detector.call_count == 1  # Called for medium confidence

    def test_weighted_voting_strategy(
        self, sample_pages, heuristic_detector, llm_detector, visual_detector
    ):
        """Test weighted voting detection strategy."""
        detectors = {
            DetectorType.HEURISTIC: heuristic_detector,
            DetectorType.LLM: llm_detector,
            DetectorType.VISUAL: visual_detector,
        }

        config = SignalCombinerConfig(
            combination_strategy=CombinationStrategy.WEIGHTED_VOTING,
            detector_weights={
                DetectorType.HEURISTIC: 0.2,
                DetectorType.LLM: 0.6,
                DetectorType.VISUAL: 0.2,
            },
            enable_parallel_processing=False,
        )

        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(sample_pages)

        # All detectors should be called
        assert heuristic_detector.call_count == 1
        assert llm_detector.call_count == 1
        assert visual_detector.call_count == 1

        # Results should be combined by weighted voting
        assert len(results) > 0
        assert all(r.detector_type == DetectorType.COMBINED for r in results)

    def test_consensus_strategy(
        self, sample_pages, heuristic_detector, llm_detector
    ):
        """Test consensus detection strategy."""
        # Create detectors that agree on page 3
        heuristic_results = [
            BoundaryResult(
                page_number=3,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.8,
                detector_type=DetectorType.HEURISTIC,
                evidence={},
                reasoning="Heuristic boundary",
            )
        ]
        
        llm_results = [
            BoundaryResult(
                page_number=3,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.LLM,
                evidence={},
                reasoning="LLM boundary",
            )
        ]

        detectors = {
            DetectorType.HEURISTIC: MockDetector(DetectorType.HEURISTIC, heuristic_results),
            DetectorType.LLM: MockDetector(DetectorType.LLM, llm_results),
        }

        config = SignalCombinerConfig(
            combination_strategy=CombinationStrategy.CONSENSUS,
            min_agreement_threshold=0.66,  # Need 2/2 detectors
            enable_parallel_processing=False,
            add_implicit_start_boundary=False,  # Disable for this test
        )

        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(sample_pages)

        # Should have consensus on page 3
        assert len(results) == 1
        assert results[0].page_number == 3
        assert "Consensus" in results[0].reasoning

    def test_merge_results(self):
        """Test merging of results for the same page."""
        combiner = SignalCombiner({DetectorType.HEURISTIC: Mock()})

        result1 = BoundaryResult(
            page_number=5,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.7,
            detector_type=DetectorType.HEURISTIC,
            evidence={"heuristic": "date_pattern"},
            reasoning="Found date pattern",
        )

        result2 = BoundaryResult(
            page_number=5,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.9,
            detector_type=DetectorType.LLM,
            evidence={"llm": "strong_boundary"},
            reasoning="LLM detected boundary",
        )

        merged = combiner._merge_two_results(result1, result2)

        # Should use higher confidence as base
        assert merged.confidence >= 0.9
        # Should combine evidence
        assert "heuristic" in merged.evidence
        assert "llm" in merged.evidence
        # Should combine reasoning
        assert "date pattern" in merged.reasoning
        assert "LLM detected" in merged.reasoning

    def test_confidence_boost_on_agreement(self):
        """Test confidence boost when detectors agree."""
        config = SignalCombinerConfig(confidence_boost_on_agreement=0.1)
        combiner = SignalCombiner({DetectorType.HEURISTIC: Mock()}, config)

        result1 = BoundaryResult(
            page_number=5,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.85,
            detector_type=DetectorType.HEURISTIC,
            evidence={},
            reasoning="R1",
        )

        result2 = BoundaryResult(
            page_number=5,
            boundary_type=BoundaryType.DOCUMENT_START,
            confidence=0.83,  # Close to result1
            detector_type=DetectorType.LLM,
            evidence={},
            reasoning="R2",
        )

        merged = combiner._merge_two_results(result1, result2)

        # Should have boosted confidence
        assert merged.confidence > 0.85
        assert merged.confidence <= 1.0

    def test_post_processing(self, sample_pages):
        """Test post-processing of results."""
        config = SignalCombinerConfig(
            min_document_pages=2,
            merge_adjacent_threshold=2,
            add_implicit_start_boundary=False,  # Disable for this test
        )
        combiner = SignalCombiner({DetectorType.HEURISTIC: Mock()}, config)

        # Create results that are too close together
        results = [
            BoundaryResult(
                page_number=2,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.COMBINED,
                evidence={},
                reasoning="",
            ),
            BoundaryResult(
                page_number=3,  # Only 1 page after previous
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.8,
                detector_type=DetectorType.COMBINED,
                evidence={},
                reasoning="",
            ),
            BoundaryResult(
                page_number=7,  # 4 pages after previous - OK
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.85,
                detector_type=DetectorType.COMBINED,
                evidence={},
                reasoning="",
            ),
        ]

        processed = combiner._post_process_results(results, sample_pages)

        # Should filter out boundary at page 3 (too close)
        assert len(processed) == 2
        assert processed[0].page_number == 2
        assert processed[1].page_number == 7

    def test_implicit_start_boundary(self, sample_pages):
        """Test that an implicit start boundary is added when needed."""
        combiner = SignalCombiner({DetectorType.HEURISTIC: Mock()})

        # Results starting at page 3
        results = [
            BoundaryResult(
                page_number=3,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.COMBINED,
                evidence={},
                reasoning="",
            )
        ]

        processed = combiner._post_process_results(results, sample_pages)

        # Should add implicit boundary at page 0
        assert len(processed) == 2
        assert processed[0].page_number == 0
        assert processed[0].evidence["reason"] == "implicit_start"
        assert processed[1].page_number == 3

    def test_empty_pages(self):
        """Test handling of empty pages list."""
        combiner = SignalCombiner({DetectorType.HEURISTIC: Mock()})
        results = combiner.detect_boundaries([])
        assert results == []

    def test_parallel_processing(self, sample_pages, heuristic_detector, llm_detector):
        """Test parallel processing of detectors."""
        detectors = {
            DetectorType.HEURISTIC: heuristic_detector,
            DetectorType.LLM: llm_detector,
        }

        config = SignalCombinerConfig(
            combination_strategy=CombinationStrategy.WEIGHTED_VOTING,
            enable_parallel_processing=True,
            max_workers=2,
        )

        combiner = SignalCombiner(detectors, config)
        results = combiner.detect_boundaries(sample_pages)

        # Should get results from parallel execution
        assert len(results) > 0
        assert heuristic_detector.call_count == 1
        assert llm_detector.call_count == 1

    def test_detector_failure_handling(self, sample_pages):
        """Test handling of detector failures."""
        # Create a detector that raises an exception
        failing_detector = Mock()
        failing_detector.detect_boundaries.side_effect = RuntimeError("Detector failed")

        # Create a working detector
        working_results = [
            BoundaryResult(
                page_number=5,
                boundary_type=BoundaryType.DOCUMENT_START,
                confidence=0.9,
                detector_type=DetectorType.HEURISTIC,
                evidence={},
                reasoning="Working detector",
            )
        ]
        working_detector = MockDetector(DetectorType.HEURISTIC, working_results)

        detectors = {
            DetectorType.LLM: failing_detector,
            DetectorType.HEURISTIC: working_detector,
        }

        config = SignalCombinerConfig(
            combination_strategy=CombinationStrategy.WEIGHTED_VOTING,
            enable_parallel_processing=False,
            add_implicit_start_boundary=False,  # Disable for this test
        )

        combiner = SignalCombiner(detectors, config)

        # Should still get results from working detector
        results = combiner.detect_boundaries(sample_pages)
        assert len(results) == 1
        assert results[0].page_number == 5