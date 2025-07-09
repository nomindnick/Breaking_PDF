# Visual Detector Module Status

## Overview
The Visual Detector module implements document boundary detection using computer vision techniques. It analyzes visual changes between consecutive pages to identify document boundaries.

## Current Status: 🚧 Experimental Phase

### Architecture
Based on research findings, implementing the **Pairwise Similarity Paradigm**:
- Compare adjacent pages for visual similarity
- Detect boundaries where similarity drops significantly
- Multiple techniques from simple to advanced

### Planned Techniques (Priority Order)
1. **Perceptual Hashing** (Baseline)
   - Fast, unsupervised approach
   - Using imagehash library
   - Target: < 0.1s per page

2. **SSIM (Structural Similarity)**
   - More nuanced than hashing
   - Better for layout changes
   - Target: < 0.2s per page

3. **Deep Learning Embeddings**
   - Most accurate approach
   - Using MobileNetV2 or EfficientNet-Lite0
   - Target: < 0.5s per page

4. **Hybrid Multi-tier**
   - Fast hashing for initial screening
   - Deep embeddings for uncertain cases
   - Optimal balance of speed and accuracy

### Implementation Progress
- [ ] Basic infrastructure setup
- [ ] Test PDF creation with ground truth
- [ ] Experiment runner framework
- [ ] Baseline techniques implementation
- [ ] Advanced techniques implementation
- [ ] Performance optimization
- [ ] Production implementation
- [ ] Integration with detection module

### Key Design Decisions
1. **Unsupervised Approach**: No training data required
2. **CPU-Only**: All techniques optimized for CPU performance
3. **Modular Design**: Each technique as a pluggable component
4. **Progressive Enhancement**: Start simple, add complexity as needed

### Performance Targets
- **Speed**: < 0.5 seconds per page (all techniques combined)
- **Accuracy**: > 90% F1 score on test set
- **Memory**: < 500MB for model and processing

### Dependencies
All libraries are permissively licensed for commercial use:
- OpenCV (Apache 2.0)
- Pillow (HPND/MIT)
- scikit-image (Modified BSD)
- imagehash (MIT)
- TensorFlow/PyTorch (Apache 2.0) - for deep learning

### Testing Strategy
1. Create comprehensive test PDF with varied document types
2. Test each technique individually
3. Measure accuracy, speed, and resource usage
4. Compare results to select optimal approach
5. Implement production version of best performer

### Integration Points
- Receives `ProcessedPage` objects from preprocessing
- Returns `BoundaryResult` objects like other detectors
- Can leverage OCR bounding boxes from preprocessing
- Will be combined with LLM and heuristic detectors

### Next Steps
1. Complete test PDF creation
2. Implement experiment runner
3. Test baseline techniques
4. Iterate based on results

---
*Status last updated: 2025-07-09*
