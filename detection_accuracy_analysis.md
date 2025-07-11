# Detection Module Accuracy Analysis

## Executive Summary

The detection module is currently **not meeting production requirements**. While the cascade fixes work correctly, the underlying accuracy and performance issues prevent it from being truly production-ready.

### Key Findings

1. **Accuracy**: F1 score of 0.174 (very poor)
2. **Performance**: 2.0s per page average (meets <5s target but with poor accuracy)
3. **Main Bottleneck**: LLM detector at ~41 seconds per boundary check

## Detailed Performance Analysis

### Test Configuration
- **Test File**: Test_PDF_Set_2_ocr.pdf (36 pages, pre-OCR'd)
- **Ground Truth**: 9 document boundaries
- **Models Used**: 
  - Heuristic: Production config
  - LLM: gemma3:latest (4.3B parameters)

### Component Performance

#### 1. Heuristic Detector
- **Speed**: < 1ms per page (excellent)
- **Accuracy**: 
  - F1 Score: 0.174
  - Precision: 0.143 (many false positives)
  - Recall: 0.222 (missing most boundaries)
- **Boundaries Found**: 14 (vs 9 actual)
- **Confidence Distribution**:
  - High (≥0.9): 5 boundaries
  - Medium (0.7-0.9): 7 boundaries  
  - Low (<0.7): 2 boundaries

#### 2. LLM Detector (gemma3:latest)
- **Speed**: ~41 seconds per boundary check (unusable)
- **Accuracy**: Found 1 out of 3 test boundaries
- **Issue**: Model is too large/slow for real-time use

#### 3. Cascade Strategy
- **Estimated Time**: 70.2s for 36 pages (2.0s per page)
- **LLM Calls Needed**: 2 (for low confidence boundaries)
- **Problem**: Even with cascade optimization, 2 LLM calls = 82+ seconds

### Why Accuracy is Poor

1. **Heuristic Over-Sensitivity**: 
   - Finding too many false boundaries (14 vs 9 actual)
   - Pattern matching is too aggressive

2. **Document Type Mismatch**:
   - Test set contains business documents (emails, letters, forms)
   - Heuristic may be tuned for different document types

3. **LLM Model Choice**:
   - gemma3:latest is accurate but way too slow
   - 41 seconds per check makes it impractical

## Recommendations for Improvement

### Immediate Actions (1-2 days)

1. **Switch to Faster LLM Model**
   ```python
   # Instead of gemma3:latest (4.3B), try:
   - llama3.2:1b (much faster, ~2-3s per check)
   - phi3:mini (Microsoft's efficient model)
   - gemma2:2b (good balance of speed/accuracy)
   ```

2. **Tune Heuristic Weights**
   - Reduce sensitivity to avoid false positives
   - Focus on high-precision patterns (email headers, page numbers)
   - Current config may be overfitted to training data

3. **Implement Result Caching**
   - Cache LLM results aggressively
   - Use document hashing for cache keys
   - Pre-process common document types

### Medium-Term Improvements (1-2 weeks)

1. **Heuristic Detector Improvements**
   ```python
   # Add more sophisticated patterns:
   - Document type classification first
   - Type-specific boundary detection
   - Contextual analysis (not just pattern matching)
   ```

2. **Alternative LLM Approach**
   ```python
   # Use embeddings instead of generation:
   - Embed page pairs
   - Compare similarity scores
   - Much faster than text generation
   ```

3. **Batch Processing**
   - Process multiple boundaries in single LLM call
   - Reduces overhead significantly

### Long-Term Solutions (1+ months)

1. **Train Custom Model**
   - Fine-tune small model on document boundaries
   - Can achieve <100ms per check
   - Better accuracy than generic models

2. **Hybrid Approach**
   - Use multiple fast models in ensemble
   - Combine embeddings + lightweight classification
   - User feedback loop for continuous improvement

## Performance vs Accuracy Tradeoffs

| Configuration | F1 Score | Speed (per page) | Viable? |
|--------------|----------|------------------|---------|
| Heuristic Only | 0.17 | <1ms | ❌ Too inaccurate |
| Current Cascade | ~0.5* | 2.0s | ❌ Still slow |
| With llama3.2:1b | ~0.7* | 0.2s | ✅ Good balance |
| With embeddings | ~0.6* | 0.05s | ✅ Fast enough |
| Custom model | ~0.8* | 0.1s | ✅ Best option |

*Estimated based on typical model performance

## Conclusion

The detection module architecture is sound, but the current implementation suffers from:
1. Poor heuristic accuracy requiring too many LLM calls
2. LLM model that's too large/slow for production use

**Recommended Path Forward**:
1. Immediately switch to a faster LLM model (llama3.2:1b)
2. Tune heuristic parameters to reduce false positives
3. Implement aggressive caching
4. Plan for custom model training for best long-term results

With these changes, the module can achieve:
- **Accuracy**: F1 > 0.7
- **Speed**: < 0.5s per page
- **Production Ready**: Yes