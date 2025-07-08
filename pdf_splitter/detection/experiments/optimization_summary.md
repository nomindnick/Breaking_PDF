# LLM Boundary Detection Optimization Summary

## Benchmark Results

### Model/Prompt Performance Comparison

| Model | Prompt | Accuracy | Avg Time | P95 Time | Throughput (pages/sec) |
|-------|--------|----------|----------|----------|------------------------|
| **gemma3:1b-it-q4_K_M** | minimal | **57.7%** | **0.82s** | 1.16s | 1.23 |
| gemma3:1b-it-q4_K_M | compressed | 0.0% | 0.60s | 0.93s | 1.66 |
| gemma3:1b-it-q4_K_M | optimal | 19.2% | 7.89s | 8.77s | 0.13 |
| qwen3:0.6b | minimal | 0.0% | 1.23s | 1.31s | 0.81 |
| granite3.3:2b | minimal | 0.0% | 1.67s | 2.78s | 0.60 |
| **gemma3:latest** | minimal | 42.3% | 2.58s | 3.85s | 0.39 |
| gemma3:latest | compressed | 38.5% | 2.59s | 3.86s | 0.39 |
| **gemma3:latest** | **optimal** | **73.1%** | **7.99s** | 9.48s | 0.13 |

### Model Preloading Impact

| Model | Cold Start | Warm | Speedup |
|-------|------------|------|---------|
| gemma3:1b-it-q4_K_M | 0.39s | 0.16s | 2.5x |
| qwen3:0.6b | 0.97s | 0.56s | 1.7x |
| gemma3:latest | 0.79s | 0.59s | 1.3x |

## Key Findings

1. **Best Accuracy**: gemma3:latest + optimal prompt (73.1%)
2. **Best Speed**: gemma3:1b-it-q4_K_M + minimal prompt (0.82s)
3. **Best Balance**: gemma3:1b-it-q4_K_M + minimal prompt (57.7% accuracy at 0.82s)
4. **Preloading Benefits**: 1.3-2.5x speedup when models are kept warm

## Recommendations

### Immediate Actions

1. **For Production Use**:
   - Use **gemma3:1b-it-q4_K_M + minimal prompt** as the primary detector
   - 57.7% accuracy is reasonable for a first pass
   - Sub-second performance (0.82s) meets the < 5s requirement

2. **For High Accuracy Needs**:
   - Keep **gemma3:latest + optimal prompt** available
   - Use selectively for difficult cases or manual review

3. **System Configuration**:
   ```bash
   # Keep models in memory
   export OLLAMA_MAX_LOADED_MODELS=2
   export OLLAMA_KEEP_ALIVE=30m

   # Allow parallel processing
   export OLLAMA_NUM_PARALLEL=2
   ```

### Implementation Strategy

1. **Two-Tier System** (Recommended):
   ```python
   def detect_boundary(page1, page2):
       # Tier 1: Fast detection with gemma3:1b
       result = fast_detect(page1, page2)  # 0.82s

       # Tier 2: If confidence is low, use accurate model
       if result.confidence < 0.8:
           result = accurate_detect(page1, page2)  # 7.99s

       return result
   ```

2. **Confidence Scoring**:
   - Modify prompts to include confidence scores
   - Use confidence to decide when to escalate to larger model

3. **Batch Processing**:
   - Process multiple page pairs in parallel
   - Keep models warm between batches

## Performance Projections

With gemma3:1b-it-q4_K_M + minimal prompt:
- **Single page**: 0.82s
- **100-page PDF**: ~82 seconds (1.4 minutes)
- **With parallelization (4 workers)**: ~21 seconds

With hybrid approach (assuming 20% need accurate model):
- **Average time**: 0.82s × 0.8 + 7.99s × 0.2 = 2.25s per page
- **100-page PDF**: ~225 seconds (3.75 minutes)

## Next Steps

1. **Improve gemma3:1b accuracy**:
   - Fine-tune the minimal prompt
   - Add 1-2 more examples
   - Test with real PDF data

2. **Implement confidence scoring**:
   - Modify prompts to output confidence
   - Build escalation logic

3. **Create production pipeline**:
   - Model preloading service
   - Batch processing queue
   - Performance monitoring

## Files Created

1. `prompts/gemma3_minimal.txt` - Minimal prompt for fast detection
2. `prompts/gemma3_compressed.txt` - Compressed version with 2 examples
3. `prompts/generic_minimal.txt` - Generic prompt for non-Gemma models
4. `benchmark_optimization.py` - Comprehensive benchmarking tool
5. `test_model_preloading.py` - Preloading effect tester
6. `ollama_performance_tuning.md` - Ollama optimization guide
