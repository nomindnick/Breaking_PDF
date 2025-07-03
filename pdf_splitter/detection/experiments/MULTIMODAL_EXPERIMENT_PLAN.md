# Multi-Modal Boundary Detection Experiment Plan

## Objective
Test whether Gemma's multi-modal capabilities can improve document boundary detection accuracy and speed by analyzing page images instead of or in addition to text.

## Experiment Overview

### Phase 1: Model Capability Assessment
1. **Verify Gemma Vision Model Availability**
   - Check if `gemma3:latest` supports image inputs
   - Identify specific Gemma vision models (e.g., `gemma:7b-vision`, `gemma2:2b-vision`)
   - Test basic image input functionality with Ollama

2. **Image Processing Pipeline**
   - Leverage existing PDF to image conversion from preprocessing module
   - Test optimal image resolution (balance quality vs processing speed)
   - Determine image format requirements (PNG, JPEG, base64 encoding)

### Phase 2: Experiment Design

#### Test Configurations
1. **Baseline Text-Only** (current approach)
   - Model: `gemma3:latest`
   - Input: Text snippets (200 chars from page bottom/top)
   - Current performance: 100% recall, 9 FPs, 2.47s

2. **Image-Only Approach**
   - Model: Gemma vision variant
   - Input: Full page images or cropped regions
   - Prompts to test:
     ```
     a) "You are being provided with two pages. Your task is to determine
        whether these two pages are part of the same document or whether
        the first page is the end of the document and the second page is
        the first page of a different document.
        Respond only with 'Same Document' or 'Different Documents'."

     b) Current successful prompt adapted for images:
        "Your task is to determine if these two page images are part of a
        single document or are different documents. Look for visual cues like:
        - Letterheads or headers
        - Formatting changes
        - Signatures at page end
        - New document headers
        Please only respond with 'Same Document' or 'Different Documents'"
     ```

3. **Hybrid Text + Image**
   - Model: Gemma vision variant
   - Input: Both text context AND page images
   - Leverage both textual and visual signals

4. **Focused Region Analysis**
   - Model: Gemma vision variant
   - Input: Cropped images focusing on:
     - Bottom 25% of page 1
     - Top 25% of page 2
   - Reduce processing overhead while maintaining visual context

### Phase 3: Metrics and Evaluation

#### Primary Metrics
- **Accuracy**: Precision, Recall, F1 Score
- **Speed**: Time per boundary check
- **Resource Usage**: Memory consumption, model size

#### Comparison Dimensions
1. **vs Text-Only Baseline**
   - Accuracy improvement
   - Speed difference
   - False positive reduction

2. **Visual Signal Analysis**
   - Which visual cues are most effective?
   - Does it better detect:
     - Letterhead changes
     - Signature blocks
     - Format/layout changes
     - Document headers

3. **Edge Case Performance**
   - Scanned documents with poor OCR
   - Documents with similar layouts
   - Handwritten sections
   - Forms vs letters vs reports

### Phase 4: Implementation Plan

#### 1. Create Multi-Modal Experiment Runner
```python
# New file: pdf_splitter/detection/experiments/multimodal_experiment.py
class MultiModalExperiment:
    def __init__(self):
        self.vision_models = ['gemma:7b-vision', 'gemma2:2b-vision']
        self.strategies = ['image_only', 'text_image_hybrid', 'focused_region']

    def prepare_image_input(self, pdf_page, strategy):
        # Convert PDF page to image
        # Apply strategy-specific processing
        pass

    def run_boundary_detection(self, page1_img, page2_img, prompt):
        # Call Ollama with image inputs
        pass
```

#### 2. Extend Existing Framework
- Add image preprocessing capabilities
- Create new prompt templates for multi-modal
- Extend ExperimentConfig for image parameters

#### 3. Test Dataset Preparation
- Use existing test PDFs
- Generate page images at multiple resolutions
- Create ground truth for visual boundaries

### Phase 5: Expected Outcomes

#### Success Criteria
1. **Accuracy**: Reduce false positives from 9 to ≤5 while maintaining 100% recall
2. **Speed**: Keep processing under 3 seconds per boundary
3. **Robustness**: Better handling of visual document boundaries

#### Potential Advantages of Multi-Modal
1. **Visual Pattern Recognition**
   - Letterhead detection
   - Signature identification
   - Layout change detection

2. **OCR-Independent**
   - Works on poor quality scans
   - Handles handwritten sections
   - Language agnostic

3. **Richer Context**
   - Combines textual and visual signals
   - Better understanding of document structure

### Phase 6: Risk Mitigation

#### Potential Challenges
1. **Model Availability**: Gemma vision models might not be in Ollama
   - Fallback: Use LLaVA or other vision models
   - Alternative: Use Gemma via API if needed

2. **Processing Speed**: Image processing might be slower
   - Optimize: Use lower resolution images
   - Strategy: Focus on critical regions only

3. **Memory Usage**: Vision models are typically larger
   - Monitor: Track resource consumption
   - Optimize: Batch processing strategies

## Implementation Timeline

### Week 1: Setup and Validation
- [ ] Verify Gemma vision model availability
- [ ] Test basic image input with Ollama
- [ ] Set up image preprocessing pipeline

### Week 2: Core Implementation
- [ ] Implement multi-modal experiment runner
- [ ] Create image-based prompts
- [ ] Run initial experiments

### Week 3: Analysis and Optimization
- [ ] Analyze results vs baseline
- [ ] Optimize for speed/accuracy
- [ ] Document findings

## Quick Start Commands

```bash
# Check available vision models
ollama list | grep -i vision

# Pull Gemma vision model (if available)
ollama pull gemma:7b-vision

# Run multi-modal experiments
python -m pdf_splitter.detection.experiments.multimodal_experiment \
    --models gemma:7b-vision \
    --strategies image_only text_image_hybrid \
    --prompt-style simple visual_focused

# Compare with text baseline
python -m pdf_splitter.detection.experiments.compare_multimodal \
    --baseline gemma3:latest \
    --vision gemma:7b-vision
```

## Decision Points

1. **If vision models unavailable**: Proceed with LLaVA or explore cloud APIs
2. **If slower than 5s**: Focus on region-based analysis
3. **If accuracy improves >20%**: Consider as primary detection method
4. **If minimal improvement**: Use as supplementary signal only

## Next Steps
1. Verify Gemma vision model availability in Ollama
2. Create proof-of-concept script to test image input
3. Implement full experiment framework if POC shows promise
