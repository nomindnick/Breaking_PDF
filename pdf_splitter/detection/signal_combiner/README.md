# Signal Combiner Module

The Signal Combiner is the final component of the detection module that intelligently combines signals from multiple detectors (heuristic, LLM, visual) to achieve high-accuracy document boundary detection while optimizing for performance.

## Features

### Combination Strategies

1. **Cascade-Ensemble** (Default, Recommended)
   - Uses fast heuristic detector first
   - Selectively applies expensive detectors (LLM, visual) only for uncertain cases
   - Achieves optimal balance of speed and accuracy
   - Reduces LLM API calls by up to 70%

2. **Weighted Voting**
   - Runs all detectors and combines results using configurable weights
   - Good for cases where all signals are equally important
   - Provides consistent results across all pages

3. **Consensus**
   - Requires agreement from multiple detectors
   - Highest precision but may miss some boundaries
   - Good for critical applications where false positives are costly

### Key Features

- **Parallel Processing**: Run multiple detectors concurrently for faster results
- **Confidence Boosting**: Increases confidence when detectors agree
- **Post-Processing**: Ensures logical consistency with minimum document lengths
- **Implicit Boundaries**: Optionally adds start boundary at page 0 if missing
- **Flexible Configuration**: Fine-tune thresholds and weights for your use case

## Usage

### Basic Usage

```python
from pdf_splitter.detection import (
    HeuristicDetector,
    SignalCombiner,
    DetectorType,
)

# Initialize detectors
detectors = {
    DetectorType.HEURISTIC: HeuristicDetector(),
}

# Create combiner
combiner = SignalCombiner(detectors)

# Detect boundaries
results = combiner.detect_boundaries(pages)
```

### Advanced Configuration

```python
from pdf_splitter.detection import SignalCombinerConfig

# Configure cascade-ensemble strategy
config = SignalCombinerConfig(
    combination_strategy="cascade_ensemble",
    heuristic_confidence_threshold=0.9,
    require_llm_verification_below=0.7,
    visual_verification_range=(0.7, 0.9),
    enable_parallel_processing=True,
    max_workers=4,
)

combiner = SignalCombiner(detectors, config)
```

### Full Integration

```python
# Initialize all detectors
detectors = {
    DetectorType.HEURISTIC: HeuristicDetector(),
    DetectorType.LLM: LLMDetector(),
    DetectorType.VISUAL: VisualDetector(),
}

# Optimal configuration for production
config = SignalCombinerConfig(
    combination_strategy="cascade_ensemble",
    heuristic_confidence_threshold=0.85,
    detector_weights={
        DetectorType.HEURISTIC: 0.3,
        DetectorType.LLM: 0.5,
        DetectorType.VISUAL: 0.2,
    },
    confidence_boost_on_agreement=0.1,
    min_document_pages=2,
)

combiner = SignalCombiner(detectors, config)
results = combiner.detect_boundaries(pages)
```

## Configuration Options

### SignalCombinerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `combination_strategy` | `"cascade_ensemble"` | Strategy for combining signals |
| `heuristic_confidence_threshold` | `0.9` | Min confidence to accept heuristic results |
| `require_llm_verification_below` | `0.7` | Confidence below which LLM is required |
| `visual_verification_range` | `(0.7, 0.9)` | Range for visual verification |
| `detector_weights` | See below | Weights for weighted voting |
| `min_agreement_threshold` | `0.66` | Min agreement for consensus |
| `enable_parallel_processing` | `True` | Run detectors in parallel |
| `max_workers` | `4` | Max parallel workers |
| `min_document_pages` | `1` | Min pages per document |
| `confidence_boost_on_agreement` | `0.1` | Boost when detectors agree |
| `add_implicit_start_boundary` | `True` | Add boundary at page 0 if missing |

Default detector weights:
- Heuristic: 0.3
- LLM: 0.5
- Visual: 0.2

## Performance Characteristics

### Cascade-Ensemble Strategy
- **Fast Path**: ~80% of boundaries detected in <0.5s using heuristics
- **LLM Usage**: Reduced by 70% compared to running on all pages
- **Total Time**: <5 seconds per page average

### Weighted Voting Strategy
- **Consistent**: All pages processed identically
- **Parallel**: Benefits from parallel processing
- **Time**: Depends on slowest detector (usually LLM)

### Consensus Strategy
- **High Precision**: Few false positives
- **Lower Recall**: May miss some boundaries
- **Time**: Similar to weighted voting

## Testing

The module includes comprehensive unit tests with 92% code coverage:

```bash
pytest pdf_splitter/detection/signal_combiner/tests/
```

Integration tests are also available for testing with real PDF data.

## Best Practices

1. **Start Simple**: Begin with just the heuristic detector for fast results
2. **Add Detectors Gradually**: Add LLM/visual detectors as needed
3. **Use Cascade-Ensemble**: Best balance of speed and accuracy
4. **Tune Thresholds**: Adjust based on your document types
5. **Monitor Performance**: Track processing times and API usage
6. **Enable Caching**: LLM detector uses persistent caching

## Troubleshooting

### Common Issues

1. **Slow Performance**
   - Check if parallel processing is enabled
   - Consider increasing heuristic confidence threshold
   - Verify LLM cache is working

2. **Missing Boundaries**
   - Lower the confidence thresholds
   - Check min_document_pages setting
   - Try weighted voting strategy

3. **Too Many False Positives**
   - Increase confidence thresholds
   - Use consensus strategy
   - Adjust detector weights

### Debug Mode

Enable debug logging to see detailed processing information:

```python
import logging
logging.getLogger("pdf_splitter.detection.signal_combiner").setLevel(logging.DEBUG)
```

## Future Enhancements

- Machine learning ensemble with trained weights
- Active learning from user corrections
- Domain-specific detector plugins
- Real-time performance monitoring dashboard