#!/usr/bin/env python3
"""Hybrid two-tiered boundary detection with confidence scoring."""

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pdf_splitter.detection.experiments.enhanced_synthetic_tests import \
    EnhancedSyntheticTester
from pdf_splitter.detection.experiments.experiment_runner import OllamaClient


@dataclass
class DetectionResult:
    """Result from a detection attempt."""

    decision: str  # "Same" or "Different"
    confidence: float
    model_used: str
    time_taken: float
    escalated: bool = False
    reasoning: Optional[str] = None


class HybridBoundaryDetector:
    """Two-tiered boundary detector with confidence-based escalation."""

    def __init__(self, confidence_threshold: float = 0.8):
        self.ollama = OllamaClient()
        self.confidence_threshold = confidence_threshold

        # Load prompts
        self.tier1_prompt = self._load_prompt("prompts/gemma3_minimal_confidence.txt")
        self.tier2_prompt = self._load_prompt("prompts/gemma3_optimal_confidence.txt")

        # Model configurations
        self.tier1_model = "gemma3:1b-it-q4_K_M"
        self.tier2_model = "gemma3:latest"

        # Statistics tracking
        self.stats = {
            "total_detections": 0,
            "escalations": 0,
            "tier1_times": [],
            "tier2_times": [],
            "confidence_distribution": [],
            "escalation_reasons": [],
        }

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template."""
        prompt_path = Path(__file__).parent / filename
        with open(prompt_path, "r") as f:
            return f.read()

    def _parse_response(
        self, response_text: str
    ) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Parse decision, confidence, and reasoning from response."""
        decision = None
        confidence = None
        reasoning = None

        # Extract answer
        answer_match = re.search(
            r"<answer>(.*?)</answer>", response_text, re.IGNORECASE
        )
        if answer_match:
            answer = answer_match.group(1).strip().upper()
            if "DIFFERENT" in answer:
                decision = "Different"
            elif "SAME" in answer:
                decision = "Same"

        # Extract confidence
        conf_match = re.search(
            r"<confidence>([\d.]+)</confidence>", response_text, re.IGNORECASE
        )
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except ValueError:
                pass

        # Extract reasoning (if present)
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", response_text, re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            reasoning = thinking_match.group(1).strip()

        return decision, confidence, reasoning

    def detect_boundary(self, page1_bottom: str, page2_top: str) -> DetectionResult:
        """Detect boundary using hybrid approach."""
        self.stats["total_detections"] += 1

        # Tier 1: Fast detection with small model
        tier1_start = time.time()
        prompt = self.tier1_prompt.format(
            page1_bottom=page1_bottom, page2_top=page2_top
        )

        try:
            response = self.ollama.generate(
                model=self.tier1_model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=100,
                stop=["<end_of_turn>"],
                timeout=30,
            )
            tier1_time = time.time() - tier1_start
            self.stats["tier1_times"].append(tier1_time)

            response_text = response.get("response", "")
            decision, confidence, _ = self._parse_response(response_text)

            if decision is None or confidence is None:
                # Failed to parse, escalate
                confidence = 0.0
                escalation_reason = "parse_failure"
            else:
                self.stats["confidence_distribution"].append(
                    {"model": "tier1", "confidence": confidence, "decision": decision}
                )
                escalation_reason = (
                    "low_confidence" if confidence < self.confidence_threshold else None
                )

            # Check if escalation is needed
            if confidence >= self.confidence_threshold and decision is not None:
                # High confidence, return tier 1 result
                return DetectionResult(
                    decision=decision,
                    confidence=confidence,
                    model_used=self.tier1_model,
                    time_taken=tier1_time,
                    escalated=False,
                )

            # Escalate to tier 2
            self.stats["escalations"] += 1
            self.stats["escalation_reasons"].append(
                {
                    "reason": escalation_reason,
                    "tier1_confidence": confidence,
                    "tier1_decision": decision,
                }
            )

        except Exception as e:
            # Tier 1 failed, escalate
            tier1_time = time.time() - tier1_start
            self.stats["tier1_times"].append(tier1_time)
            self.stats["escalations"] += 1
            self.stats["escalation_reasons"].append(
                {"reason": "tier1_error", "error": str(e)}
            )

        # Tier 2: Accurate detection with large model
        tier2_start = time.time()
        prompt = self.tier2_prompt.format(
            page1_bottom=page1_bottom, page2_top=page2_top
        )

        try:
            response = self.ollama.generate(
                model=self.tier2_model,
                prompt=prompt,
                temperature=0.0,
                max_tokens=300,
                stop=["<end_of_turn>"],
                timeout=60,
            )
            tier2_time = time.time() - tier2_start
            self.stats["tier2_times"].append(tier2_time)

            response_text = response.get("response", "")
            decision, confidence, reasoning = self._parse_response(response_text)

            if decision is None:
                decision = "Same"  # Default fallback
            if confidence is None:
                confidence = 0.5  # Default fallback

            self.stats["confidence_distribution"].append(
                {"model": "tier2", "confidence": confidence, "decision": decision}
            )

            total_time = tier1_time + tier2_time

            return DetectionResult(
                decision=decision,
                confidence=confidence,
                model_used=self.tier2_model,
                time_taken=total_time,
                escalated=True,
                reasoning=reasoning,
            )

        except Exception as e:
            # Both tiers failed, return default
            tier2_time = time.time() - tier2_start
            self.stats["tier2_times"].append(tier2_time)
            total_time = tier1_time + tier2_time

            return DetectionResult(
                decision="Same",
                confidence=0.0,
                model_used="fallback",
                time_taken=total_time,
                escalated=True,
                reasoning=f"Error: {str(e)}",
            )

    def get_statistics(self) -> Dict:
        """Get performance statistics."""
        stats = self.stats.copy()

        # Calculate averages
        if stats["tier1_times"]:
            stats["avg_tier1_time"] = sum(stats["tier1_times"]) / len(
                stats["tier1_times"]
            )
        else:
            stats["avg_tier1_time"] = 0

        if stats["tier2_times"]:
            stats["avg_tier2_time"] = sum(stats["tier2_times"]) / len(
                stats["tier2_times"]
            )
        else:
            stats["avg_tier2_time"] = 0

        stats["escalation_rate"] = (
            stats["escalations"] / stats["total_detections"]
            if stats["total_detections"] > 0
            else 0
        )

        # Confidence distribution analysis
        if stats["confidence_distribution"]:
            tier1_confs = [
                c["confidence"]
                for c in stats["confidence_distribution"]
                if c["model"] == "tier1"
            ]
            tier2_confs = [
                c["confidence"]
                for c in stats["confidence_distribution"]
                if c["model"] == "tier2"
            ]

            if tier1_confs:
                stats["tier1_avg_confidence"] = sum(tier1_confs) / len(tier1_confs)
                stats["tier1_confidence_range"] = (min(tier1_confs), max(tier1_confs))

            if tier2_confs:
                stats["tier2_avg_confidence"] = sum(tier2_confs) / len(tier2_confs)
                stats["tier2_confidence_range"] = (min(tier2_confs), max(tier2_confs))

        return stats


def test_hybrid_detector():
    """Test the hybrid detector on synthetic test cases."""
    print("Testing Hybrid Two-Tiered Boundary Detector")
    print("=" * 80)

    # Initialize components
    synthetic_tester = EnhancedSyntheticTester()
    hybrid_detector = HybridBoundaryDetector(confidence_threshold=0.8)

    # Test configurations
    test_cases = synthetic_tester.test_cases

    # Results tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "confidence_threshold": hybrid_detector.confidence_threshold,
        "by_difficulty": {},
        "overall": {
            "correct": 0,
            "total": 0,
            "tier1_only_correct": 0,
            "tier2_escalated_correct": 0,
        },
        "timing": {"all_times": [], "tier1_only_times": [], "escalated_times": []},
        "individual_results": [],
    }

    # Run tests
    for difficulty in range(1, 11):
        diff_cases = [c for c in test_cases if c.difficulty == difficulty]
        if not diff_cases:
            continue

        print(f"\nDifficulty {difficulty}: {len(diff_cases)} cases")

        diff_results = {
            "total": len(diff_cases),
            "correct": 0,
            "escalated": 0,
            "tier1_correct": 0,
            "tier2_correct": 0,
            "cases": [],
        }

        for i, case in enumerate(diff_cases):
            # Run detection
            result = hybrid_detector.detect_boundary(case.page1_bottom, case.page2_top)

            # Check correctness
            is_correct = result.decision == case.expected
            if is_correct:
                diff_results["correct"] += 1
                results["overall"]["correct"] += 1

                if not result.escalated:
                    diff_results["tier1_correct"] += 1
                    results["overall"]["tier1_only_correct"] += 1
                else:
                    diff_results["tier2_correct"] += 1
                    results["overall"]["tier2_escalated_correct"] += 1

            if result.escalated:
                diff_results["escalated"] += 1

            # Track timing
            results["timing"]["all_times"].append(result.time_taken)
            if result.escalated:
                results["timing"]["escalated_times"].append(result.time_taken)
            else:
                results["timing"]["tier1_only_times"].append(result.time_taken)

            # Store detailed result
            case_result = {
                "case_id": case.id,
                "difficulty": difficulty,
                "expected": case.expected,
                "predicted": result.decision,
                "correct": is_correct,
                "confidence": result.confidence,
                "model": result.model_used,
                "time": result.time_taken,
                "escalated": result.escalated,
            }
            diff_results["cases"].append(case_result)
            results["individual_results"].append(case_result)

            # Progress output
            status = "✓" if is_correct else "✗"
            escalation = " (escalated)" if result.escalated else ""
            print(
                f"  Case {i+1}: {status} conf={result.confidence:.2f} time={result.time_taken:.2f}s{escalation}"
            )

        results["overall"]["total"] += diff_results["total"]
        diff_results["accuracy"] = diff_results["correct"] / diff_results["total"]
        diff_results["escalation_rate"] = (
            diff_results["escalated"] / diff_results["total"]
        )

        results["by_difficulty"][difficulty] = diff_results

        print(f"  Accuracy: {diff_results['accuracy']:.1%}")
        print(f"  Escalation rate: {diff_results['escalation_rate']:.1%}")

    # Calculate overall metrics
    if results["overall"]["total"] > 0:
        results["overall"]["accuracy"] = (
            results["overall"]["correct"] / results["overall"]["total"]
        )
        results["overall"]["tier1_accuracy"] = (
            results["overall"]["tier1_only_correct"]
            / (results["overall"]["total"] - hybrid_detector.stats["escalations"])
            if results["overall"]["total"] > hybrid_detector.stats["escalations"]
            else 0
        )

    # Get detector statistics
    detector_stats = hybrid_detector.get_statistics()
    results["detector_stats"] = detector_stats

    # Calculate timing statistics
    if results["timing"]["all_times"]:
        results["timing"]["avg_time"] = sum(results["timing"]["all_times"]) / len(
            results["timing"]["all_times"]
        )
    if results["timing"]["tier1_only_times"]:
        results["timing"]["avg_tier1_only"] = sum(
            results["timing"]["tier1_only_times"]
        ) / len(results["timing"]["tier1_only_times"])
    if results["timing"]["escalated_times"]:
        results["timing"]["avg_escalated"] = sum(
            results["timing"]["escalated_times"]
        ) / len(results["timing"]["escalated_times"])

    return results


def compare_approaches():
    """Compare hybrid approach with single-model approaches."""
    print("\nRunning comparative analysis...")
    print("=" * 80)

    # Run hybrid test
    hybrid_results = test_hybrid_detector()

    # Print summary
    print("\n" + "=" * 80)
    print("HYBRID DETECTOR SUMMARY")
    print("=" * 80)

    overall = hybrid_results["overall"]
    timing = hybrid_results["timing"]
    stats = hybrid_results["detector_stats"]

    print(f"\nOverall Performance:")
    print(f"  Total cases: {overall['total']}")
    print(f"  Correct: {overall['correct']} ({overall['accuracy']:.1%})")
    print(f"  Tier 1 only correct: {overall['tier1_only_correct']}")
    print(f"  Tier 2 escalated correct: {overall['tier2_escalated_correct']}")

    print(f"\nEscalation Statistics:")
    print(f"  Escalations: {stats['escalations']} ({stats['escalation_rate']:.1%})")
    print(f"  Tier 1 avg confidence: {stats.get('tier1_avg_confidence', 0):.2f}")
    print(f"  Tier 2 avg confidence: {stats.get('tier2_avg_confidence', 0):.2f}")

    print(f"\nTiming Performance:")
    print(f"  Average time (all): {timing.get('avg_time', 0):.2f}s")
    print(f"  Average time (tier 1 only): {timing.get('avg_tier1_only', 0):.2f}s")
    print(f"  Average time (escalated): {timing.get('avg_escalated', 0):.2f}s")
    print(f"  Tier 1 avg time: {stats.get('avg_tier1_time', 0):.2f}s")
    print(f"  Tier 2 avg time: {stats.get('avg_tier2_time', 0):.2f}s")

    # Analyze escalation patterns
    print(f"\nEscalation Analysis:")
    if stats["escalation_reasons"]:
        reasons = {}
        for r in stats["escalation_reasons"]:
            reason = r.get("reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count/len(stats['escalation_reasons']):.1%})")

    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"  Hybrid avg time: {timing.get('avg_time', 0):.2f}s")
    print(f"  gemma3:latest only (baseline): 7.99s")
    print(f"  Speedup: {7.99 / timing.get('avg_time', 1):.1f}x")
    print(f"  Hybrid accuracy: {overall['accuracy']:.1%}")
    print(f"  gemma3:latest accuracy (baseline): 73.1%")

    # Save results
    output_file = (
        Path("pdf_splitter/detection/experiments/results")
        / f"hybrid_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(hybrid_results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")

    return hybrid_results


if __name__ == "__main__":
    compare_approaches()
