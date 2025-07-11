"""Signal combiner for intelligent document boundary detection.

This module implements a cascade-ensemble approach that combines multiple
detection signals (heuristic, LLM, visual) to achieve high accuracy while
optimizing for performance.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from pdf_splitter.detection.base_detector import (
    BaseDetector,
    BoundaryResult,
    BoundaryType,
    DetectionContext,
    DetectorType,
    ProcessedPage,
)
from pdf_splitter.core.config import PDFConfig

logger = logging.getLogger(__name__)


class CombinationStrategy(str, Enum):
    """Available strategies for combining detection signals."""

    CASCADE_ENSEMBLE = "cascade_ensemble"
    WEIGHTED_VOTING = "weighted_voting"
    CONSENSUS = "consensus"
    ML_ENSEMBLE = "ml_ensemble"  # Future enhancement


@dataclass
class SignalCombinerConfig:
    """Configuration for the signal combiner.

    Attributes:
        heuristic_confidence_threshold: Minimum confidence to accept heuristic results without verification
        require_llm_verification_below: Confidence below which LLM verification is required
        visual_verification_range: Confidence range where visual verification is useful
        detector_weights: Weights for each detector type in weighted voting
        min_agreement_threshold: Minimum agreement score for consensus strategy
        enable_parallel_processing: Whether to process detectors in parallel
        max_workers: Maximum number of parallel workers
        combination_strategy: Strategy to use for combining signals
        min_document_pages: Minimum pages for a valid document
        merge_adjacent_threshold: Distance threshold for merging adjacent boundaries
        confidence_boost_on_agreement: Confidence boost when detectors agree
    """

    # Cascade thresholds
    heuristic_confidence_threshold: float = 0.9
    require_llm_verification_below: float = 0.7
    visual_verification_range: Tuple[float, float] = (0.7, 0.9)

    # Detector weights for weighted voting
    detector_weights: Dict[DetectorType, float] = field(
        default_factory=lambda: {
            DetectorType.HEURISTIC: 0.3,
            DetectorType.LLM: 0.5,
            DetectorType.VISUAL: 0.2,
        }
    )

    # Consensus settings
    min_agreement_threshold: float = 0.66  # At least 2/3 detectors must agree

    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4

    # Strategy selection
    combination_strategy: CombinationStrategy = CombinationStrategy.CASCADE_ENSEMBLE

    # Post-processing settings
    min_document_pages: int = 1
    merge_adjacent_threshold: int = 2  # Pages
    confidence_boost_on_agreement: float = 0.1
    add_implicit_start_boundary: bool = True  # Whether to add boundary at page 0 if missing

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate thresholds
        if not 0 <= self.heuristic_confidence_threshold <= 1:
            raise ValueError("heuristic_confidence_threshold must be between 0 and 1")
        if not 0 <= self.require_llm_verification_below <= 1:
            raise ValueError("require_llm_verification_below must be between 0 and 1")

        # Validate visual verification range
        low, high = self.visual_verification_range
        if not (0 <= low <= high <= 1):
            raise ValueError("visual_verification_range must be valid range between 0 and 1")

        # Validate weights sum to 1.0 (approximately)
        total_weight = sum(self.detector_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.detector_weights = {
                k: v / total_weight for k, v in self.detector_weights.items()
            }


class SignalCombiner(BaseDetector):
    """Combines multiple detection signals for intelligent boundary detection.

    This class implements a sophisticated approach to document boundary detection
    by intelligently combining signals from multiple detectors. It uses a
    cascade-ensemble approach by default, where fast detectors filter obvious
    cases and expensive detectors verify uncertain boundaries.
    """

    def __init__(
        self,
        detectors: Dict[DetectorType, BaseDetector],
        config: Optional[SignalCombinerConfig] = None,
    ):
        """Initialize the signal combiner.

        Args:
            detectors: Dictionary mapping detector types to detector instances
            config: Configuration for the combiner
        """
        super().__init__()
        self.detectors = detectors
        self.config = config or SignalCombinerConfig()

        # Validate detectors
        if not detectors:
            raise ValueError("At least one detector must be provided")

        # Log configuration
        logger.info(
            f"Initialized SignalCombiner with {len(detectors)} detectors: "
            f"{list(detectors.keys())}"
        )
        logger.info(f"Using strategy: {self.config.combination_strategy}")

    def get_detector_type(self) -> DetectorType:
        """Return the detector type."""
        return DetectorType.COMBINED

    def detect_boundaries(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        """Detect document boundaries using combined signals.

        Args:
            pages: List of processed pages to analyze
            context: Optional detection context

        Returns:
            List of detected boundaries with confidence scores
        """
        if not pages:
            return []

        logger.info(f"Starting boundary detection for {len(pages)} pages")

        # Choose strategy
        if self.config.combination_strategy == CombinationStrategy.CASCADE_ENSEMBLE:
            results = self._cascade_ensemble_detection(pages, context)
        elif self.config.combination_strategy == CombinationStrategy.WEIGHTED_VOTING:
            results = self._weighted_voting_detection(pages, context)
        elif self.config.combination_strategy == CombinationStrategy.CONSENSUS:
            results = self._consensus_detection(pages, context)
        else:
            raise ValueError(f"Unknown strategy: {self.config.combination_strategy}")

        # Post-process results
        results = self._post_process_results(results, pages)

        logger.info(f"Detected {len(results)} boundaries")
        return results

    def _cascade_ensemble_detection(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        """Implement cascade-ensemble detection strategy.

        This strategy uses a fast detector first, then selectively applies
        more expensive detectors for uncertain cases.
        """
        logger.debug("Using cascade-ensemble detection strategy")

        # Phase 1: Fast heuristic detection
        heuristic_results = []
        if DetectorType.HEURISTIC in self.detectors:
            logger.debug("Running heuristic detector for fast screening")
            heuristic_results = self.detectors[DetectorType.HEURISTIC].detect_boundaries(
                pages, context
            )

        # Categorize results by confidence
        high_confidence = []
        needs_llm_verification = []
        needs_visual_verification = []

        # Create a set of page numbers with heuristic boundaries
        heuristic_pages = {r.page_number for r in heuristic_results}

        for result in heuristic_results:
            # Use original confidence for cascade decisions
            orig_conf = result.original_confidence or result.confidence
            
            if orig_conf >= self.config.heuristic_confidence_threshold:
                # Mark as high confidence phase
                result.evidence['cascade_phase'] = 'high_confidence'
                high_confidence.append(result)
            elif orig_conf < self.config.require_llm_verification_below:
                needs_llm_verification.append(result.page_number)
            else:
                # In the visual verification range
                needs_visual_verification.append(result.page_number)

        # Also check pages without heuristic boundaries
        for i, page in enumerate(pages[:-1]):  # Skip last page
            if i not in heuristic_pages:
                # No heuristic boundary found - needs verification
                needs_llm_verification.append(i)

        logger.debug(
            f"Heuristic results: {len(high_confidence)} high confidence, "
            f"{len(needs_llm_verification)} need LLM, "
            f"{len(needs_visual_verification)} need visual"
        )

        # Phase 2: Selective verification
        verified_results = []

        # LLM verification for low confidence or missing boundaries
        if needs_llm_verification and DetectorType.LLM in self.detectors:
            logger.debug(f"Running LLM verification for {len(needs_llm_verification)} pages")
            # Create context for LLM with specific pages to check
            llm_context = context or DetectionContext(
                config=PDFConfig(),
                total_pages=len(pages),
                document_metadata={"target_pages": needs_llm_verification},
            )
            llm_results = self.detectors[DetectorType.LLM].detect_boundaries(
                pages, llm_context
            )
            # Mark these results as from LLM verification phase
            for result in llm_results:
                result.evidence['cascade_phase'] = 'llm_verification'
            verified_results.extend(llm_results)

        # Visual verification for medium confidence
        if needs_visual_verification and DetectorType.VISUAL in self.detectors:
            logger.debug(
                f"Running visual verification for {len(needs_visual_verification)} pages"
            )
            visual_results = self.detectors[DetectorType.VISUAL].detect_boundaries(
                pages, context
            )
            # Filter to only pages we care about and mark phase
            visual_results = [
                r for r in visual_results if r.page_number in needs_visual_verification
            ]
            for result in visual_results:
                result.evidence['cascade_phase'] = 'visual_verification'
            verified_results.extend(visual_results)

        # Phase 3: Combine results
        all_results = high_confidence + verified_results

        # Deduplicate and merge results for same page
        merged_results = self._merge_results_by_page(all_results)

        return list(merged_results.values())

    def _weighted_voting_detection(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        """Implement weighted voting detection strategy.

        This strategy runs all detectors and combines their results using
        weighted voting.
        """
        logger.debug("Using weighted voting detection strategy")

        # Run all detectors
        all_results = []
        if self.config.enable_parallel_processing:
            all_results = self._run_detectors_parallel(pages, context)
        else:
            all_results = self._run_detectors_sequential(pages, context)

        # Group results by page
        results_by_page = {}
        for detector_type, results in all_results:
            for result in results:
                if result.page_number not in results_by_page:
                    results_by_page[result.page_number] = []
                results_by_page[result.page_number].append((detector_type, result))

        # Combine results for each page using weighted voting
        combined_results = []
        for page_num, page_results in results_by_page.items():
            combined_result = self._combine_page_results_weighted(page_results)
            if combined_result:
                combined_results.append(combined_result)

        return combined_results

    def _consensus_detection(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[BoundaryResult]:
        """Implement consensus detection strategy.

        This strategy requires agreement from multiple detectors before
        accepting a boundary.
        """
        logger.debug("Using consensus detection strategy")

        # Run all detectors
        all_results = []
        if self.config.enable_parallel_processing:
            all_results = self._run_detectors_parallel(pages, context)
        else:
            all_results = self._run_detectors_sequential(pages, context)

        # Group results by page
        results_by_page = {}
        for detector_type, results in all_results:
            for result in results:
                if result.page_number not in results_by_page:
                    results_by_page[result.page_number] = []
                results_by_page[result.page_number].append((detector_type, result))

        # Find pages with consensus
        combined_results = []
        total_detectors = len(self.detectors)

        for page_num, page_results in results_by_page.items():
            agreement_ratio = len(page_results) / total_detectors

            if agreement_ratio >= self.config.min_agreement_threshold:
                # Combine results with consensus
                combined_result = self._combine_page_results_consensus(page_results)
                if combined_result:
                    combined_results.append(combined_result)

        return combined_results

    def _run_detectors_parallel(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[Tuple[DetectorType, List[BoundaryResult]]]:
        """Run all detectors in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all detection tasks
            future_to_detector = {
                executor.submit(
                    detector.detect_boundaries, pages, context
                ): detector_type
                for detector_type, detector in self.detectors.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_detector):
                detector_type = future_to_detector[future]
                try:
                    detector_results = future.result()
                    results.append((detector_type, detector_results))
                except Exception as e:
                    logger.error(f"Detector {detector_type} failed: {e}")

        return results

    def _run_detectors_sequential(
        self,
        pages: List[ProcessedPage],
        context: Optional[DetectionContext] = None,
    ) -> List[Tuple[DetectorType, List[BoundaryResult]]]:
        """Run all detectors sequentially."""
        results = []

        for detector_type, detector in self.detectors.items():
            try:
                detector_results = detector.detect_boundaries(pages, context)
                results.append((detector_type, detector_results))
            except Exception as e:
                logger.error(f"Detector {detector_type} failed: {e}")

        return results

    def _merge_results_by_page(
        self, results: List[BoundaryResult]
    ) -> Dict[int, BoundaryResult]:
        """Merge multiple results for the same page."""
        page_results = {}

        for result in results:
            page_num = result.page_number

            if page_num not in page_results:
                page_results[page_num] = result
            else:
                # Merge with existing result
                existing = page_results[page_num]
                merged = self._merge_two_results(existing, result)
                page_results[page_num] = merged

        return page_results

    def _merge_two_results(
        self, result1: BoundaryResult, result2: BoundaryResult
    ) -> BoundaryResult:
        """Merge two boundary results for the same page."""
        # Use the one with higher confidence as base
        if result1.confidence >= result2.confidence:
            base, other = result1, result2
        else:
            base, other = result2, result1

        # Combine evidence
        combined_evidence = base.evidence.copy()
        combined_evidence.update(other.evidence)
        
        # Preserve cascade phase if present
        if 'cascade_phase' not in combined_evidence and 'cascade_phase' in base.evidence:
            combined_evidence['cascade_phase'] = base.evidence['cascade_phase']
        elif 'cascade_phase' not in combined_evidence and 'cascade_phase' in other.evidence:
            combined_evidence['cascade_phase'] = other.evidence['cascade_phase']

        # Calculate boosted confidence for display, but preserve original
        confidence_boost = 0.0
        if abs(result1.confidence - result2.confidence) < 0.2:
            confidence_boost = self.config.confidence_boost_on_agreement
        
        # Cap the boost to prevent exceeding cascade thresholds
        # If original confidence is below a threshold, don't boost above it
        boosted_confidence = base.confidence + confidence_boost
        
        # Get the original confidence values
        orig1 = result1.original_confidence or result1.confidence
        orig2 = result2.original_confidence or result2.confidence
        max_original = max(orig1, orig2)
        
        # Don't boost beyond the next cascade threshold
        if max_original < self.config.require_llm_verification_below:
            # Don't boost above the LLM threshold
            boosted_confidence = min(boosted_confidence, self.config.require_llm_verification_below - 0.01)
        elif max_original < self.config.heuristic_confidence_threshold:
            # Don't boost above the heuristic threshold
            boosted_confidence = min(boosted_confidence, self.config.heuristic_confidence_threshold - 0.01)

        # Create merged result
        return BoundaryResult(
            page_number=base.page_number,
            boundary_type=base.boundary_type,
            confidence=min(1.0, boosted_confidence),
            original_confidence=max_original,  # Track the highest original confidence
            detector_type=DetectorType.COMBINED,
            evidence=combined_evidence,
            reasoning=f"{base.reasoning}; {other.reasoning}",
            is_between_pages=base.is_between_pages,
            next_page_number=base.next_page_number,
        )

    def _combine_page_results_weighted(
        self, page_results: List[Tuple[DetectorType, BoundaryResult]]
    ) -> Optional[BoundaryResult]:
        """Combine results for a page using weighted voting."""
        if not page_results:
            return None

        # Calculate weighted confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        combined_evidence = {}
        reasonings = []

        for detector_type, result in page_results:
            weight = self.config.detector_weights.get(detector_type, 0.0)
            weighted_confidence += result.confidence * weight
            total_weight += weight
            combined_evidence.update(result.evidence)
            reasonings.append(f"{detector_type.value}: {result.reasoning}")

        if total_weight == 0:
            return None

        final_confidence = weighted_confidence / total_weight

        # Use the first result as template
        base_result = page_results[0][1]

        return BoundaryResult(
            page_number=base_result.page_number,
            boundary_type=base_result.boundary_type,
            confidence=final_confidence,
            detector_type=DetectorType.COMBINED,
            evidence=combined_evidence,
            reasoning="; ".join(reasonings),
            is_between_pages=base_result.is_between_pages,
            next_page_number=base_result.next_page_number,
        )

    def _combine_page_results_consensus(
        self, page_results: List[Tuple[DetectorType, BoundaryResult]]
    ) -> Optional[BoundaryResult]:
        """Combine results for a page requiring consensus."""
        if not page_results:
            return None

        # Average confidence across all agreeing detectors
        avg_confidence = sum(r[1].confidence for r in page_results) / len(page_results)

        # Combine evidence and reasoning
        combined_evidence = {}
        reasonings = []

        for detector_type, result in page_results:
            combined_evidence.update(result.evidence)
            reasonings.append(f"{detector_type.value}: {result.reasoning}")

        # Use the first result as template
        base_result = page_results[0][1]

        return BoundaryResult(
            page_number=base_result.page_number,
            boundary_type=base_result.boundary_type,
            confidence=avg_confidence,
            detector_type=DetectorType.COMBINED,
            evidence=combined_evidence,
            reasoning=f"Consensus ({len(page_results)} detectors): " + "; ".join(reasonings),
            is_between_pages=base_result.is_between_pages,
            next_page_number=base_result.next_page_number,
        )

    def _post_process_results(
        self, results: List[BoundaryResult], pages: List[ProcessedPage]
    ) -> List[BoundaryResult]:
        """Post-process detection results for consistency and quality."""
        if not results:
            return results

        # Sort by page number
        results.sort(key=lambda r: r.page_number)

        # Apply minimum document length constraint
        filtered_results = []
        for i, result in enumerate(results):
            # Check if this would create a document that's too short
            if i > 0:
                prev_boundary = filtered_results[-1].page_number
                if result.page_number - prev_boundary < self.config.min_document_pages:
                    # Skip this boundary or merge with previous
                    logger.debug(
                        f"Skipping boundary at page {result.page_number} - "
                        f"too close to previous boundary"
                    )
                    continue

            filtered_results.append(result)

        # Ensure we have a boundary at the start if needed
        # Only add if we have results AND the first result is not at page 0
        if self.config.add_implicit_start_boundary and filtered_results:
            if filtered_results[0].page_number > 0:
                # Add implicit boundary at start
                filtered_results.insert(
                    0,
                    BoundaryResult(
                        page_number=0,
                        boundary_type=BoundaryType.DOCUMENT_START,
                        confidence=1.0,
                        detector_type=DetectorType.COMBINED,
                        evidence={"reason": "implicit_start"},
                        reasoning="Implicit document start",
                    ),
                )

        return filtered_results