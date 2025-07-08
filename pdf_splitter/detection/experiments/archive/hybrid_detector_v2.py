#!/usr/bin/env python3
"""Improved hybrid two-tiered boundary detection with better confidence handling."""

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
    tier1_decision: Optional[str] = None
    tier1_confidence: Optional[float] = None


class ImprovedHybridDetector:
    """Improved two-tiered boundary detector with dynamic confidence thresholds."""

    def __init__(
        self, confidence_threshold: float = 0.85, force_diversity: bool = True
    ):
        self.ollama = OllamaClient()
        self.confidence_threshold = confidence_threshold
        self.force_diversity = force_diversity  # Force tier1 to give varied responses

        # Load prompts
        self.tier1_prompt = self._load_prompt(
            "prompts/gemma3_minimal_confidence_v2.txt"
        )
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
            "tier1_decisions": {"Same": 0, "Different": 0},
            "tier2_corrections": [],
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

    def _should_escalate(
        self, decision: str, confidence: float, difficulty_hint: Optional[int] = None
    ) -> bool:
        """Determine if escalation is needed based on confidence and other factors."""
        # Always escalate if parsing failed
        if decision is None or confidence is None:
            return True

        # Basic confidence threshold
        if confidence < self.confidence_threshold:
            return True

        # Dynamic threshold based on decision type
        # Be more cautious with "Different" decisions as they're rarer
        if decision == "Different" and confidence < 0.9:
            return True

        # If we know this is a difficult case (from test), be more conservative
        if difficulty_hint and difficulty_hint >= 7 and confidence < 0.95:
            return True

        return False

    def detect_boundary(
        self, page1_bottom: str, page2_top: str, difficulty_hint: Optional[int] = None
    ) -> DetectionResult:
        """Detect boundary using improved hybrid approach."""
        self.stats["total_detections"] += 1

        # Tier 1: Fast detection with small model
        tier1_start = time.time()
        prompt = self.tier1_prompt.format(
            page1_bottom=page1_bottom, page2_top=page2_top
        )

        tier1_decision = None
        tier1_confidence = None

        try:
            response = self.ollama.generate(
                model=self.tier1_model,
                prompt=prompt,
                temperature=0.1
                if self.force_diversity
                else 0.0,  # Add slight randomness
                max_tokens=100,
                stop=["<end_of_turn>"],
                timeout=30,
            )
            tier1_time = time.time() - tier1_start
            self.stats["tier1_times"].append(tier1_time)

            response_text = response.get("response", "")
            tier1_decision, tier1_confidence, _ = self._parse_response(response_text)

            if tier1_decision:
                self.stats["tier1_decisions"][tier1_decision] += 1

            if tier1_decision is None or tier1_confidence is None:
                escalation_reason = "parse_failure"
                tier1_confidence = 0.0
            else:
                self.stats["confidence_distribution"].append(
                    {
                        "model": "tier1",
                        "confidence": tier1_confidence,
                        "decision": tier1_decision,
                    }
                )
                escalation_reason = (
                    "low_confidence"
                    if self._should_escalate(
                        tier1_decision, tier1_confidence, difficulty_hint
                    )
                    else None
                )

            # Check if escalation is needed
            if not self._should_escalate(
                tier1_decision, tier1_confidence, difficulty_hint
            ):
                # High confidence, return tier 1 result
                return DetectionResult(
                    decision=tier1_decision,
                    confidence=tier1_confidence,
                    model_used=self.tier1_model,
                    time_taken=tier1_time,
                    escalated=False,
                )

            # Escalate to tier 2
            self.stats["escalations"] += 1
            self.stats["escalation_reasons"].append(
                {
                    "reason": escalation_reason,
                    "tier1_confidence": tier1_confidence,
                    "tier1_decision": tier1_decision,
                    "difficulty_hint": difficulty_hint,
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

            # Track if tier2 corrected tier1
            if tier1_decision and decision != tier1_decision:
                self.stats["tier2_corrections"].append(
                    {
                        "tier1": tier1_decision,
                        "tier2": decision,
                        "tier1_conf": tier1_confidence,
                        "tier2_conf": confidence,
                    }
                )

            total_time = tier1_time + tier2_time

            return DetectionResult(
                decision=decision,
                confidence=confidence,
                model_used=self.tier2_model,
                time_taken=total_time,
                escalated=True,
                reasoning=reasoning,
                tier1_decision=tier1_decision,
                tier1_confidence=tier1_confidence,
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

        # Analyze corrections
        if stats["tier2_corrections"]:
            stats["correction_rate"] = (
                len(stats["tier2_corrections"]) / stats["escalations"]
                if stats["escalations"] > 0
                else 0
            )
        else:
            stats["correction_rate"] = 0

        return stats


def test_improved_hybrid():
    """Test the improved hybrid detector."""
    print("Testing Improved Hybrid Detector")
    print("=" * 80)

    # Initialize components
    synthetic_tester = EnhancedSyntheticTester()
    detector = ImprovedHybridDetector(confidence_threshold=0.85)

    # Also test single-model baselines for comparison
    ollama = OllamaClient()

    # Results tracking
    results = {
        "timestamp": datetime.now().isoformat(),
        "hybrid": {"correct": 0, "total": 0, "times": [], "by_difficulty": {}},
        "tier1_only": {"correct": 0, "total": 0, "times": []},
        "tier2_only": {"correct": 0, "total": 0, "times": []},
    }

    print("\n1. Testing Hybrid Approach")
    print("-" * 40)

    for difficulty in range(1, 11):
        diff_cases = [
            c for c in synthetic_tester.test_cases if c.difficulty == difficulty
        ]
        if not diff_cases:
            continue

        print(f"\nDifficulty {difficulty}: {len(diff_cases)} cases")

        diff_correct = 0
        diff_escalated = 0

        for i, case in enumerate(diff_cases):
            # Hybrid detection with difficulty hint
            result = detector.detect_boundary(
                case.page1_bottom, case.page2_top, difficulty_hint=difficulty
            )

            is_correct = result.decision == case.expected
            if is_correct:
                diff_correct += 1
                results["hybrid"]["correct"] += 1

            if result.escalated:
                diff_escalated += 1

            results["hybrid"]["times"].append(result.time_taken)

            # Print detailed info for escalated cases
            if result.escalated:
                print(
                    f"  Case {i+1}: {'✓' if is_correct else '✗'} "
                    f"T1({result.tier1_decision}/{result.tier1_confidence:.2f}) → "
                    f"T2({result.decision}/{result.confidence:.2f}) "
                    f"time={result.time_taken:.2f}s"
                )
            else:
                print(
                    f"  Case {i+1}: {'✓' if is_correct else '✗'} "
                    f"{result.decision} conf={result.confidence:.2f} "
                    f"time={result.time_taken:.2f}s"
                )

        results["hybrid"]["total"] += len(diff_cases)
        results["hybrid"]["by_difficulty"][difficulty] = {
            "total": len(diff_cases),
            "correct": diff_correct,
            "accuracy": diff_correct / len(diff_cases),
            "escalation_rate": diff_escalated / len(diff_cases),
        }

        print(f"  Accuracy: {diff_correct/len(diff_cases):.1%}")
        print(f"  Escalation rate: {diff_escalated/len(diff_cases):.1%}")

    # Test tier1 only for baseline
    print("\n\n2. Testing Tier 1 Only (Baseline)")
    print("-" * 40)

    tier1_prompt = detector.tier1_prompt
    for case in synthetic_tester.test_cases[:10]:  # Sample for speed
        start = time.time()
        prompt = tier1_prompt.format(
            page1_bottom=case.page1_bottom, page2_top=case.page2_top
        )

        response = ollama.generate(
            model=detector.tier1_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=100,
            stop=["<end_of_turn>"],
            timeout=30,
        )

        elapsed = time.time() - start
        results["tier1_only"]["times"].append(elapsed)

        response_text = response.get("response", "")
        decision, _, _ = detector._parse_response(response_text)

        if decision == case.expected:
            results["tier1_only"]["correct"] += 1
        results["tier1_only"]["total"] += 1

    # Calculate final metrics
    results["hybrid"]["accuracy"] = (
        results["hybrid"]["correct"] / results["hybrid"]["total"]
    )
    results["hybrid"]["avg_time"] = sum(results["hybrid"]["times"]) / len(
        results["hybrid"]["times"]
    )

    results["tier1_only"]["accuracy"] = (
        results["tier1_only"]["correct"] / results["tier1_only"]["total"]
    )
    results["tier1_only"]["avg_time"] = sum(results["tier1_only"]["times"]) / len(
        results["tier1_only"]["times"]
    )

    # Get detector statistics
    detector_stats = detector.get_statistics()
    results["detector_stats"] = detector_stats

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nHybrid Approach:")
    print(f"  Accuracy: {results['hybrid']['accuracy']:.1%}")
    print(f"  Average time: {results['hybrid']['avg_time']:.2f}s")
    print(f"  Escalation rate: {detector_stats['escalation_rate']:.1%}")
    print(f"  Tier 2 correction rate: {detector_stats.get('correction_rate', 0):.1%}")

    print(f"\nTier 1 Only:")
    print(f"  Accuracy: {results['tier1_only']['accuracy']:.1%}")
    print(f"  Average time: {results['tier1_only']['avg_time']:.2f}s")

    print(f"\nTier 1 Decision Distribution:")
    print(f"  Same: {detector_stats['tier1_decisions']['Same']}")
    print(f"  Different: {detector_stats['tier1_decisions']['Different']}")

    print(f"\nPerformance vs gemma3:latest baseline:")
    print(f"  Speedup: {7.99 / results['hybrid']['avg_time']:.1f}x")
    print(f"  Accuracy difference: {results['hybrid']['accuracy'] - 0.731:.1%}")

    # Save results
    output_file = (
        Path("pdf_splitter/detection/experiments/results")
        / f"improved_hybrid_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    test_improved_hybrid()
