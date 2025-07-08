# Speed Optimization Plan for PDF Boundary Detection

## Current Performance
- **gemma3:latest**: 7.63s average per page
- **Accuracy**: 76.92%
- **Target**: < 5 seconds per page

## Optimization Strategies

### 1. Model Selection Testing
Test smaller models with the gemma3_optimal prompt:
```python
models_to_test = [
    "qwen3:0.6b",      # 522 MB - likely fastest
    "granite3.3:2b",   # 1.5 GB - good balance
    "deepseek-r1:1.5b", # 1.1 GB - another option
    "qwen3:1.7b"       # 1.4 GB - slightly larger
]
```

### 2. Prompt Optimization
Create variations of gemma3_optimal:
- **Minimal**: Remove thinking tags, reduce to 1 example
- **Compressed**: Shorter instructions, 2 examples
- **Keywords**: Focus on key indicators only

### 3. Hybrid Approach
Combine fast heuristics with selective LLM use:
```python
def hybrid_detection(page1_text, page2_text):
    # Fast heuristic checks first
    if has_obvious_boundary_markers(page1_text, page2_text):
        return "Different"

    if has_strong_continuation(page1_text, page2_text):
        return "Same"

    # Only use LLM for ambiguous cases
    if is_ambiguous(page1_text, page2_text):
        return llm_detection(page1_text, page2_text)
```

### 4. Parallel Processing
- Process multiple pages concurrently
- Batch similar detection tasks
- Pipeline: OCR → Detection → Splitting

### 5. Caching Strategy
- Cache LLM responses for similar patterns
- Store common boundary patterns
- Reuse decisions for repeated document types

## Implementation Priority

1. **Quick Wins** (1-2 days)
   - Test smaller models with existing prompt
   - Create minimal prompt variant
   - Implement basic heuristic pre-filter

2. **Medium Term** (3-5 days)
   - Develop hybrid detection system
   - Optimize prompt based on error analysis
   - Set up model preloading

3. **Long Term** (1-2 weeks)
   - Build pattern library from successful detections
   - Implement smart caching system
   - Create ensemble approach

## Benchmarking Plan

```python
# Test configuration
test_config = {
    "models": ["qwen3:0.6b", "granite3.3:2b", "gemma3:latest"],
    "prompts": ["gemma3_optimal", "minimal", "hybrid"],
    "metrics": ["accuracy", "f1_score", "avg_time", "p95_time"]
}

# Run comparative tests
for model in test_config["models"]:
    for prompt in test_config["prompts"]:
        results = run_benchmark(model, prompt)
        log_results(model, prompt, results)
```

## Expected Outcomes

| Approach | Expected Time | Expected Accuracy |
|----------|--------------|-------------------|
| qwen3:0.6b + minimal | 1-2s | 60-70% |
| granite3.3:2b + optimal | 3-4s | 70-75% |
| Hybrid (heuristic + LLM) | 2-3s avg | 80-85% |
| Ensemble (multiple models) | 5-6s | 85-90% |

## Next Steps

1. Create benchmark script to test model/prompt combinations
2. Implement basic heuristic filters
3. Analyze errors from gemma3_optimal to identify patterns
4. Design hybrid system architecture
5. Test in production-like environment with real PDFs
