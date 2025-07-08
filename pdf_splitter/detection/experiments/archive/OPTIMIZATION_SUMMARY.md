# LLM Boundary Detection Optimization Experiments Summary

## Overview

This document summarizes the optimization experiments conducted on 2025-07-08 to improve the performance of LLM-based PDF boundary detection.

## Initial Baseline

- **Model**: gemma3:latest (3.3GB)
- **Prompt**: gemma3_optimal (with 3 examples and chain-of-thought reasoning)
- **Performance**: 73.1% accuracy at 7.99s per page

## Experiments Conducted

### 1. Model Size Testing

Tested various smaller models to find speed/accuracy trade-offs:

| Model | Size | Minimal Prompt | Optimal Prompt |
|-------|------|----------------|----------------|
| gemma3:1b-it-q4_K_M | 815MB | 57.7% @ 0.82s | 19.2% @ 7.89s |
| qwen3:0.6b | 522MB | 0% @ 1.23s | N/A |
| granite3.3:2b | 1.5GB | 0% @ 1.67s | N/A |
| gemma3:latest | 3.3GB | 42.3% @ 2.58s | 73.1% @ 7.99s |

**Finding**: Only Gemma models performed reasonably. The 1b model showed promise with minimal prompts.

### 2. Prompt Optimization

Created three prompt variants:
- **gemma3_optimal.txt**: Original comprehensive prompt (3 examples, reasoning)
- **gemma3_compressed.txt**: Balanced prompt (2 examples, brief instructions)
- **gemma3_minimal.txt**: Ultra-compact prompt (1 example, direct output)

**Finding**: Minimal prompts significantly reduce processing time but at accuracy cost.

### 3. Model Preloading Effects

| Model | Cold Start | Warm | Speedup |
|-------|------------|------|---------|
| gemma3:1b-it-q4_K_M | 0.39s | 0.16s | 2.5x |
| qwen3:0.6b | 0.97s | 0.56s | 1.7x |
| gemma3:latest | 0.79s | 0.59s | 1.3x |

**Finding**: Keeping models warm provides 1.3-2.5x speedup.

### 4. Hybrid Two-Tiered Approach

Implemented confidence-based escalation system:
- **Tier 1**: gemma3:1b (fast, less accurate)
- **Tier 2**: gemma3:latest (slow, more accurate)
- **Escalation**: When confidence < 0.85

Results:
- **Accuracy**: 65.4% (vs 73.1% baseline)
- **Speed**: 1.47s average (5.4x faster)
- **Escalation Rate**: 3.8%

**Critical Issue**: Tier 1 model shows extreme bias:
- Says "Same" 96.2% of the time
- Misses most document boundaries
- High confidence even when wrong

## Key Findings

### What Worked
1. Gemma models can output confidence scores
2. Small models can be 5-10x faster
3. Model preloading provides meaningful benefits
4. Confidence-based escalation logic functions correctly

### What Didn't Work
1. Non-Gemma models performed poorly on this task
2. Small model has severe "Same" bias (96.2%)
3. Minimal prompts sacrifice too much accuracy
4. Hybrid approach doesn't meaningfully improve on speed/accuracy trade-off

## Recommendations for Future Work

1. **Investigate Small Model Bias**
   - Why does gemma3:1b almost always say "Same"?
   - Test with different prompt formulations
   - Consider fine-tuning for boundary detection

2. **Explore Alternative Approaches**
   - Pattern-based pre-filtering for obvious boundaries
   - Visual detection using page layout changes
   - Ensemble methods combining multiple signals

3. **Test with Real Data**
   - Current tests use synthetic data
   - Real PDFs may have different characteristics
   - Validate findings on production workload

4. **Consider Different Models**
   - Test newer models as they become available
   - Explore models specifically trained for document understanding
   - Consider cloud-based APIs for comparison

## Conclusion

While we achieved significant speed improvements (5.4x), the accuracy trade-off and model bias issues make the current hybrid approach unsuitable for production use. The small model's 96.2% "Same" bias essentially means it's not performing boundary detection at all.

Future work should focus on:
1. Understanding and correcting the small model bias
2. Implementing non-LLM pre-filtering for obvious cases
3. Testing with real PDF data to validate findings

## Files Created

### Prompts
- `prompts/gemma3_optimal.txt` - Original comprehensive prompt
- `prompts/gemma3_compressed.txt` - Balanced 2-example prompt
- `prompts/gemma3_minimal.txt` - Minimal 1-example prompt
- `prompts/gemma3_minimal_confidence.txt` - Minimal with confidence
- `prompts/gemma3_optimal_confidence.txt` - Optimal with confidence
- `prompts/generic_minimal.txt` - For non-Gemma models

### Scripts
- `benchmark_optimization.py` - Comprehensive model/prompt testing
- `test_model_preloading.py` - Model warming analysis
- `hybrid_detector.py` - Initial hybrid implementation
- `hybrid_detector_v2.py` - Improved with better confidence handling
- `debug_confidence.py` - Confidence output debugging

### Documentation
- `optimization_summary.md` - Initial optimization findings
- `ollama_performance_tuning.md` - Ollama configuration guide
- `speed_optimization_plan.md` - Optimization strategy
- `hybrid_analysis_report.md` - Hybrid approach analysis

### Results
Multiple JSON files in `results/` directory with detailed test outcomes.
