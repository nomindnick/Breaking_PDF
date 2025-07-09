# Visual Boundary Detection Experimentation Plan

## Overview
This document outlines the experimental approach for developing an optimal visual document boundary detection system. Based on comprehensive research, we will test multiple computer vision techniques to find the best balance of accuracy, speed, and resource usage.

## Guiding Principles
1. **Data-Driven Decisions**: Every technique will be evaluated on the same test set
2. **CPU-First**: All approaches must run efficiently on CPU-only systems
3. **Progressive Complexity**: Start simple, add complexity only if needed
4. **Production Focus**: Consider deployment constraints from the start

## Test Data Preparation

### Visual Test PDF Requirements
- **Size**: 50-60 pages total
- **Documents**: 12-15 distinct documents
- **Variety**: Mix of document types, layouts, and visual styles
- **Boundaries**: Clear ground truth for evaluation

### Document Types to Include
1. **Business Documents** (15 pages)
   - Corporate letters with letterheads
   - Memos with consistent headers
   - Multi-page reports with page numbers

2. **Technical Content** (10 pages)
   - Code listings with syntax highlighting
   - Technical specifications with diagrams
   - API documentation with tables

3. **Forms and Structured Data** (10 pages)
   - Application forms
   - Invoices with line items
   - Spreadsheet-style layouts

4. **Email and Communications** (10 pages)
   - Email threads with quoted content
   - Chat transcripts
   - Meeting notes

5. **Mixed Content** (10 pages)
   - Presentations with images
   - Infographics
   - Marketing materials

### Visual Variations to Test
- **Layout Changes**: Single vs. multi-column, margins, spacing
- **Typography**: Font families, sizes, weights
- **Color**: Full color, grayscale, black & white
- **Quality**: Clean scans, noisy images, slight rotations
- **Orientation**: Portrait and landscape pages

## Phase 1: Baseline Techniques (Days 1-3)

### 1.1 Histogram Comparison
**Objective**: Establish simplest baseline for visual similarity

**Implementation Details**:
```python
# Approach
1. Convert page to grayscale
2. Compute intensity histogram (256 bins)
3. Calculate histogram distance (Chi-squared, Bhattacharyya)
4. Threshold for boundary detection
```

**Experiments**:
- Compare different distance metrics
- Test bin counts (64, 128, 256)
- Analyze sensitivity to image preprocessing
- Measure processing speed

**Expected Results**:
- Very fast (< 0.05s per page)
- Low accuracy on subtle changes
- Good for dramatic visual shifts

### 1.2 Structural Similarity Index (SSIM)
**Objective**: More sophisticated comparison considering human perception

**Implementation Details**:
```python
# Using skimage.metrics.structural_similarity
1. Resize pages to standard resolution
2. Calculate SSIM between adjacent pages
3. Detect boundaries where SSIM < threshold
```

**Experiments**:
- Full page vs. region-based SSIM
- Different window sizes (7, 11, 15)
- Gaussian weighting effects
- Multi-scale SSIM

**Expected Results**:
- Fast (< 0.1s per page)
- Better layout change detection than histograms
- Sensitive to alignment issues

### 1.3 Perceptual Hashing
**Objective**: Robust baseline with good speed/accuracy balance

**Implementation Details**:
```python
# Using imagehash library
1. Generate perceptual hash for each page
2. Calculate Hamming distance between adjacent pages
3. Boundary when distance > threshold
```

**Experiments**:
- Compare hash algorithms (pHash, aHash, dHash, wHash)
- Hash size optimization (8, 16, 32 bits)
- Preprocessing effects (resize, denoise)
- Threshold sensitivity analysis

**Expected Results**:
- Very fast (< 0.02s per page)
- Good robustness to minor variations
- Moderate accuracy on document boundaries

## Phase 2: Advanced Techniques (Days 4-6)

### 2.1 Local Feature Matching (ORB)
**Objective**: Robust detection invariant to transformations

**Implementation Details**:
```python
# Using OpenCV ORB detector
1. Detect keypoints and compute descriptors
2. Match features between adjacent pages
3. Boundary when match ratio < threshold
```

**Experiments**:
- Number of features (500, 1000, 2000)
- Matching strategies (brute force, FLANN)
- Geometric verification (RANSAC)
- Region-specific matching

**Expected Results**:
- Slower (0.2-0.5s per page)
- High robustness to rotation/scale
- May struggle with text-heavy pages

### 2.2 Deep Learning Embeddings
**Objective**: State-of-the-art accuracy using learned representations

**Models to Test**:
1. **MobileNetV2**
   - 3.5M parameters
   - Excellent CPU performance
   - Apache 2.0 license

2. **EfficientNet-Lite0**
   - 4.0M parameters
   - Best accuracy/speed trade-off
   - Apache 2.0 license

**Implementation Details**:
```python
# Feature extraction approach
1. Load pre-trained model (ImageNet weights)
2. Remove classification head
3. Extract features from penultimate layer
4. Calculate cosine similarity
5. Detect boundaries at similarity drops
```

**Experiments**:
- Input resolution (224, 299, 384)
- Feature layer selection
- Similarity metrics (cosine, L2, dot product)
- Ensemble of multiple models

**Expected Results**:
- Moderate speed (0.3-0.5s per page)
- Highest accuracy
- Best semantic understanding

### 2.3 OCR-Based Layout Analysis
**Objective**: Leverage existing OCR data for layout changes

**Implementation Details**:
```python
# Using bounding boxes from preprocessing
1. Extract text bounding boxes from OCR
2. Compute layout features:
   - Text density
   - Margin sizes
   - Column detection
   - Font size distribution
3. Detect significant layout changes
```

**Experiments**:
- Layout feature engineering
- Change detection algorithms
- Integration with visual features
- Performance on non-text pages

## Phase 3: Hybrid Approaches (Days 7-8)

### 3.1 Cascaded Detection
**Concept**: Use fast methods first, expensive methods for uncertain cases

**Architecture**:
```
Pages → Perceptual Hash → Low confidence? → Deep Embeddings → Decision
            ↓                                      ↓
      High confidence                    Refined boundaries
        boundaries
```

**Implementation**:
1. First pass with perceptual hashing
2. Identify uncertain regions (medium Hamming distances)
3. Apply deep learning only to uncertain pairs
4. Combine results with confidence weighting

### 3.2 Multi-Signal Fusion
**Concept**: Combine multiple techniques for robustness

**Approaches to Test**:
1. **Weighted Voting**: Each technique votes, weighted by confidence
2. **Feature Concatenation**: Combine all features, train simple classifier
3. **Hierarchical Decision**: Rule-based combination based on document type

### 3.3 Region-Based Analysis
**Concept**: Different page regions have different importance

**Implementation**:
1. Divide page into regions (header, body, footer)
2. Apply different techniques to different regions
3. Weight region importance by document type
4. Combine for final decision

## Evaluation Methodology

### Metrics
1. **Accuracy Metrics**
   - Precision: Correct boundaries / Predicted boundaries
   - Recall: Correct boundaries / True boundaries
   - F1 Score: Harmonic mean of precision and recall
   - Boundary offset accuracy (within N pages)

2. **Performance Metrics**
   - Processing time per page
   - Memory usage
   - CPU utilization
   - Initialization time

3. **Robustness Metrics**
   - Performance on different document types
   - Sensitivity to image quality
   - Consistency across runs

### Evaluation Protocol
1. Run each technique on full test set
2. Record all metrics for each boundary decision
3. Generate confusion matrices
4. Analyze failure cases
5. Profile performance bottlenecks

### Statistical Analysis
- Multiple runs for timing consistency
- Confidence intervals for accuracy metrics
- Document-type stratified results
- Cross-validation on different PDF sets

## Implementation Timeline

### Week 1: Foundation and Baselines
- **Day 1**: Test PDF creation and ground truth
- **Day 2**: Histogram and SSIM implementation
- **Day 3**: Perceptual hashing and initial results

### Week 2: Advanced Techniques
- **Day 4**: ORB feature matching
- **Day 5**: Deep learning setup and testing
- **Day 6**: OCR layout analysis

### Week 3: Optimization and Production
- **Day 7**: Hybrid approaches
- **Day 8**: Performance optimization
- **Day 9**: Production implementation
- **Day 10**: Integration and documentation

## Success Criteria

### Must Have
- F1 Score > 0.90 on test set
- Processing < 0.5s per page on CPU
- Memory usage < 500MB
- Works with existing preprocessing output

### Nice to Have
- F1 Score > 0.95
- Processing < 0.2s per page
- Graceful degradation on edge cases
- Explanable boundary decisions

## Risk Mitigation

### Technical Risks
1. **Deep learning too slow on CPU**
   - Mitigation: Use smaller models, quantization, or cascade approach

2. **Poor accuracy on specific document types**
   - Mitigation: Document-specific tuning, ensemble methods

3. **Memory constraints**
   - Mitigation: Streaming processing, smaller models

### Implementation Risks
1. **Integration complexity**
   - Mitigation: Clear interfaces, comprehensive testing

2. **Hyperparameter sensitivity**
   - Mitigation: Robust default values, auto-tuning

## Next Steps
1. Create comprehensive test PDF
2. Implement base experiment runner
3. Start with perceptual hashing baseline
4. Iterate based on results

---
*Plan created: 2025-07-09*
*Target completion: 2-3 weeks*
