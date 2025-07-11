# Cascade Strategy Workaround

Until the cascade issues are fixed, here's a working configuration that provides reasonable results:

## Temporary Solution: Weighted Voting

```python
from pdf_splitter.detection import (
    SignalCombiner,
    SignalCombinerConfig,
    CombinationStrategy,
    HeuristicDetector,
    LLMDetector,
    get_production_config,
    DetectorType
)

# Initialize detectors
heuristic = HeuristicDetector(get_production_config())
llm = LLMDetector()  # Uses gemma3:latest

# Skip visual detector (broken)
detectors = {
    DetectorType.HEURISTIC: heuristic,
    DetectorType.LLM: llm,
}

# Use weighted voting instead of cascade
config = SignalCombinerConfig(
    combination_strategy=CombinationStrategy.WEIGHTED_VOTING,
    detector_weights={
        DetectorType.HEURISTIC: 0.3,
        DetectorType.LLM: 0.7,
    },
    add_implicit_start_boundary=False,  # Important!
)

combiner = SignalCombiner(detectors, config)
```

## Why This Works

1. **Weighted Voting** runs both detectors on all pages (no cascade logic to fail)
2. **LLM gets 70% weight** ensuring high accuracy
3. **Heuristic gets 30% weight** to boost confidence on obvious boundaries
4. **No visual detector** avoids the "No PDF loaded" errors

## Performance Expectations

- **Speed**: 15-20 seconds per page (LLM on all pages)
- **Accuracy**: Should achieve F1 > 0.8
- **Consistency**: Predictable behavior

## Caching Considerations

```python
# For faster re-runs, ensure LLM cache is enabled (default)
# To force fresh results:
LLM_CACHE_ENABLED=false python your_script.py

# Or programmatically:
llm = LLMDetector(cache_enabled=False)
```

## Notes

- This is a temporary workaround until cascade is fixed
- Performance will be slower since LLM runs on all pages
- But accuracy should be good since LLM has high weight
- Consider batching similar documents for cache benefits