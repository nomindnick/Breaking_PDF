# Next Steps for Boundary Detection

## Why LLM Verification Failed

Looking at the results, the LLM verification was counterproductive because:

1. **Context Window Issues**: The LLM only sees limited context around each boundary, making it harder to understand document structure
2. **Prompt Engineering**: We may not be giving the LLM the right instructions for boundary detection
3. **Model Selection**: qwen3:0.6b might be too small for nuanced boundary decisions
4. **Verification vs Detection**: LLMs might be better at detecting boundaries from scratch rather than verifying existing ones

## Current State Analysis

**Embeddings Detector Performance:**
- F1: 0.686 (only 0.064 from target)
- Precision: 0.545 (main issue - too many false positives)
- Recall: 0.923 (excellent)
- Detecting: 22 boundaries
- Expected: 13 boundaries
- **Key insight**: We're over-detecting by ~70%

## Recommended Next Steps

### 1. Immediate: Post-Processing Filters (1-2 days)

Since we're detecting 22 boundaries instead of 13, we need smart filtering:

```python
def apply_post_processing_filters(boundaries, pages):
    """Remove likely false positives."""
    filtered = []
    
    for i, boundary in enumerate(boundaries):
        # Filter 1: Minimum document length
        # Don't create documents shorter than 2 pages
        if i < len(boundaries) - 1:
            next_boundary = boundaries[i + 1]
            if next_boundary.page_number - boundary.page_number < 2:
                continue
        
        # Filter 2: Confidence threshold adjustment
        # Raise threshold for boundaries near start/end
        if boundary.page_number < 3 or boundary.page_number > len(pages) - 3:
            if boundary.confidence < 0.7:
                continue
        
        # Filter 3: Content-based filtering
        # Check if the "new document" has substantial content
        page_text = pages[boundary.page_number + 1].text
        if len(page_text.strip()) < 100:  # Too little content
            continue
            
        filtered.append(boundary)
    
    return filtered
```

### 2. Embeddings Threshold Optimization (1 day)

Test thresholds from 0.45 to 0.65 in 0.01 increments to find the optimal point:

```python
# Current: threshold=0.5 gives 22 boundaries
# We need ~13 boundaries for better precision
# Higher threshold = fewer boundaries = better precision
```

### 3. Ensemble Reconsideration (2-3 days)

Instead of traditional ensemble, try:
- **Embeddings Primary + Heuristic Veto**: Use embeddings but remove boundaries that heuristic strongly disagrees with
- **Sequential Filtering**: Embeddings finds candidates → Heuristic filters obvious false positives
- **Confidence Adjustment**: Use heuristic to adjust embeddings confidence rather than add new boundaries

### 4. Custom Similarity Metrics (3-5 days)

The current cosine similarity might not capture document boundaries well:

```python
def document_boundary_similarity(emb1, emb2, text1, text2):
    """Custom similarity that considers document boundary patterns."""
    
    # Base cosine similarity
    cos_sim = cosine_similarity(emb1, emb2)
    
    # Adjust based on text patterns
    if text2.strip().startswith(('From:', 'To:', 'Subject:')):
        cos_sim *= 0.8  # Likely email start
    
    if text1.strip().endswith(('.', '!', '?')) and text2[0].isupper():
        cos_sim *= 0.95  # Proper sentence boundaries
    
    # Length ratio penalty
    len_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
    if len_ratio < 0.3:  # Very different lengths
        cos_sim *= 0.9
    
    return cos_sim
```

### 5. Learning from Errors (1 week)

Analyze the 9 false positives to find patterns:
- Are they mostly at document starts/ends?
- Are they short pages?
- Are they specific document types?
- Do they have common textual patterns?

### 6. Alternative Approach: Two-Stage Detection

```python
# Stage 1: Over-detect (current approach)
candidates = embeddings_detector.detect_boundaries()  # 22 boundaries

# Stage 2: Precision refinement
# Train a simple classifier on boundary features
features = extract_boundary_features(candidates, pages)
final_boundaries = boundary_classifier.predict(features)  # Target: 13 boundaries
```

## Recommended Priority

1. **Start with post-processing filters** - Quick win, could get us to F1>0.70
2. **Optimize embeddings threshold** - Another quick win
3. **Analyze false positives** - Understand the problem better
4. **Implement custom similarity** - Address root cause

## Expected Timeline

- Week 1: Post-processing + threshold optimization → F1 ~0.72-0.74
- Week 2: Custom similarity + error analysis → F1 ~0.75-0.78
- Week 3: Polish and production hardening → F1 ≥0.75 stable

## Alternative: Accept Current Performance

Given F1=0.686 is already quite good:
- Deploy with current performance
- Flag low-confidence boundaries for human review
- Learn from corrections to improve over time
- Focus engineering effort on other parts of the system