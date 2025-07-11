# Boundary Detection - Overfitting Analysis

## Summary

You were absolutely right to be concerned. The post-processing filters that achieved F1=0.769 on the training set were indeed overfitted. When tested on fresh validation data:

- **Training F1**: 0.769
- **Validation F1**: 0.646 (optimized) / 0.633 (balanced)
- **Performance drop**: ~0.12-0.15

Neither approach meets the F1≥0.75 target on unseen data.

## What Went Wrong

### 1. Position-Based Filter Too Aggressive
The rule "pages 0-2 need confidence ≥0.7" worked well on the training set but fails when:
- Documents have multiple short documents at the beginning
- Early pages contain legitimate document boundaries
- The training set had specific patterns in early pages

### 2. Minimum Length Filter Creates Cascades
When boundary 1 is filtered, boundary 2 gets filtered for creating a 1-page document, creating a cascade effect that removes legitimate boundaries.

### 3. Content Filters Too Specific
The "lowercase start = continuation" rule had too many exceptions in real documents.

## Key Findings

### Training Set Specific Patterns
The training set (Test_PDF_Set_2_ocr.pdf) appears to have:
- Few legitimate boundaries in pages 0-2
- Specific continuation patterns
- Document length distributions that aren't representative

### Validation Performance Varies Wildly
- Corporate documents: F1=0.400 (missing early boundaries)
- Academic papers: F1=1.000 or 0.667 (inconsistent)
- Visual test: F1=0.364 (many edge cases)

### Base Embeddings Are Robust
Without post-processing:
- Training: F1=0.686
- Most validation sets: F1=0.500-0.762
- More consistent across datasets

## Lessons Learned

1. **Post-processing rules must be derived from diverse data**, not a single test set
2. **Simple threshold optimization might be better** than complex rules
3. **The base embeddings detector (F1~0.65-0.70) is quite good** and generalizes well
4. **Achieving F1≥0.75 consistently requires either**:
   - Training on more diverse data
   - Different approach (not just filtering)
   - Accepting F1~0.70 as production target

## Recommendations

### Option 1: Use Base Embeddings (F1~0.65-0.70)
- Most honest and generalizable
- No overfitting
- Fast and simple
- Add UI for manual corrections

### Option 2: Collect Diverse Training Data
- Get 10-20 different PDF types
- Derive post-processing rules from all of them
- Validate on held-out set
- More work but could achieve F1≥0.75

### Option 3: Different Architecture
- Train a small classifier on boundary features
- Use active learning from user corrections
- Combine multiple embedding models
- More complex but potentially more accurate

### Option 4: Adjust Success Criteria
- Accept F1~0.70 as "good enough"
- Focus on specific document types
- Optimize for precision OR recall based on use case

## Conclusion

The post-processing approach worked too well on the training set because it learned specific patterns that don't generalize. This is a classic case of overfitting through manual feature engineering. 

The good news is that the base embeddings approach (F1~0.65-0.70) is solid and generalizes well. The bad news is that achieving F1≥0.75 consistently across diverse documents is harder than it initially appeared.

The most practical path forward is likely to:
1. Use the base embeddings detector
2. Collect feedback from production use
3. Gradually improve with real-world data
4. Accept that perfect boundary detection might require document-specific tuning