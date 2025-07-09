# Heuristic Detector Production Usage Guide

## Overview

The heuristic detector provides fast, pattern-based document boundary detection with three optimized configurations based on extensive testing.

## Performance Results

From testing on real-world OCR'd PDFs:

| Configuration | F1 Score | Precision | Recall | Speed |
|--------------|----------|-----------|---------|-------|
| optimized | 0.381 | 0.500 | 0.308 | 0.03ms/page |
| fast_screen | 0.522 | 0.364 | 0.923 | 0.03ms/page |
| high_precision | 0.471 | 1.000 | 0.308 | 0.01ms/page |

## Usage

### 1. Standalone Detection (Balanced)

```python
from pdf_splitter.detection.heuristic_detector import HeuristicDetector
from pdf_splitter.detection.heuristic_detector.optimized_config import get_optimized_config

# Create detector with optimized config
detector = HeuristicDetector(get_optimized_config())

# Detect boundaries
results = detector.detect_boundaries(pages)
```

### 2. Fast Screening (High Recall)

Use when heuristics are the first pass in a cascade architecture:

```python
from pdf_splitter.detection.heuristic_detector.optimized_config import get_fast_screen_config

# Create detector for fast screening
detector = HeuristicDetector(get_fast_screen_config())

# Get all potential boundaries (92.3% recall)
results = detector.detect_boundaries(pages)

# Pass low-confidence results to LLM detector
for result in results:
    if result.confidence < 0.7:  # Threshold for LLM processing
        # Send to LLM detector
        pass
```

### 3. High Precision Mode

Use when false positives are very costly:

```python
from pdf_splitter.detection.heuristic_detector.optimized_config import get_high_precision_config

# Create detector for high precision
detector = HeuristicDetector(get_high_precision_config())

# Only detect very confident boundaries
results = detector.detect_boundaries(pages)
```

## Pattern Performance

Based on experimental results:

| Pattern | Accuracy | Notes |
|---------|----------|-------|
| email_header | 100% | Perfect accuracy when detected |
| page_numbering | 100% | Perfect accuracy when detected |
| terminal_phrases | 50% | Moderate reliability |
| header_footer_change | 46.3% | Disabled in optimized config |
| date_pattern | 25% | Low weight in optimized config |
| document_keywords | 24.5% | Useful but needs context |

## Integration with Hybrid Architecture

The heuristic detector is designed to work in a Hybrid Cascaded-Ensemble Architecture:

```python
from pdf_splitter.detection.heuristic_detector import HeuristicDetector
from pdf_splitter.detection.llm_detector import LLMDetector
from pdf_splitter.detection.heuristic_detector.optimized_config import get_fast_screen_config

class HybridDetector:
    def __init__(self):
        self.heuristic = HeuristicDetector(get_fast_screen_config())
        self.llm = LLMDetector()

    def detect_boundaries(self, pages):
        # First pass: Fast heuristic screening
        heuristic_results = self.heuristic.detect_boundaries(pages)

        final_results = []
        for i, result in enumerate(heuristic_results):
            if result.confidence >= 0.8:
                # High confidence - use heuristic result
                final_results.append(result)
            else:
                # Low confidence - verify with LLM
                llm_result = self.llm.detect_boundary(pages[i], pages[i+1])
                final_results.append(llm_result)

        return final_results
```

## Configuration Customization

You can create custom configurations based on your document types:

```python
from pdf_splitter.detection.heuristic_detector import HeuristicConfig, PatternConfig

config = HeuristicConfig()

# Customize for legal documents
config.patterns["document_keywords"] = PatternConfig(
    name="document_keywords",
    enabled=True,
    weight=0.9,
    confidence_threshold=0.7,
    params={
        "keywords": [
            "AGREEMENT",
            "CONTRACT",
            "EXHIBIT",
            "ADDENDUM",
            "SCHEDULE",
            "AMENDMENT",
        ]
    }
)
```

## Recommendations

1. **General Use**: Start with `get_optimized_config()` for balanced performance
2. **Cascade Architecture**: Use `get_fast_screen_config()` as first pass
3. **Critical Applications**: Use `get_high_precision_config()` when accuracy is paramount
4. **Monitor Performance**: Track pattern accuracy on your specific document types
5. **Customize Patterns**: Add domain-specific keywords and patterns

## Limitations

- Works best with structured business documents (emails, invoices, contracts)
- Requires extractable text (OCR quality affects results)
- Email and page numbering patterns are most reliable
- May need customization for specialized document types
