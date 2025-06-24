# OCR Accuracy Improvement Plan

## Problem Analysis

The OCR module is achieving only 0.8% F1 score on computer-generated PDFs, despite the PDFs being high quality. The issue is NOT with the PDF quality, but with our implementation.

### Symptoms
- Garbled text output: "o: .yle ot" instead of "From: Lyle Bolte"
- Low word count: 54 words extracted vs 222 expected
- High confidence (0.86) despite poor accuracy
- Commercial PDF software can OCR the same file successfully

## Root Cause Hypotheses

### 1. **Low DPI Rendering** (High Probability)
- Currently using 150 DPI default
- Commercial OCR typically uses 300 DPI
- Low DPI may cause character details to be lost

### 2. **Inappropriate Preprocessing** (Very High Probability)
- Preprocessing pipeline designed for scanned documents
- Operations like blur, thresholding may degrade crisp computer text
- Adaptive thresholding could create artifacts on clean text

### 3. **PaddleOCR Configuration** (Medium Probability)
- Default settings may not be optimal for computer-generated text
- Detection thresholds may be too high
- Missing important parameters like `use_space_char`

### 4. **PDF Rendering Method** (Medium Probability)
- PyMuPDF rendering might not preserve text quality
- Color space conversion might affect clarity
- Anti-aliasing settings could blur text edges

## Investigation Plan

### Phase 1: Diagnostic Testing (Immediate)

1. **Run Diagnostic Scripts**
   ```bash
   python scripts/diagnose_ocr_problem.py
   python scripts/test_pdf_rendering.py
   ```
   - Test different DPI settings (150, 200, 250, 300)
   - Test with/without preprocessing
   - Save intermediate images for visual inspection

2. **Visual Inspection**
   - Compare rendered images at different DPIs
   - Check if preprocessing degrades image quality
   - Look for rendering artifacts or blurriness

3. **Direct Comparison**
   - Render a page from Test_PDF_Set_2_ocr.pdf
   - Compare quality with Test_PDF_Set_1.pdf rendering
   - Check if rendering process introduces issues

### Phase 2: Targeted Fixes (Based on Diagnostics)

#### Fix 1: Increase Default DPI
```python
# In core/config.py
default_dpi: int = Field(300, ge=72, le=600, description="Default DPI for rendering")
```

#### Fix 2: Smart Preprocessing
```python
# In ocr_processor.py
def _should_preprocess(self, image: np.ndarray, page_type: PageType) -> bool:
    """Determine if preprocessing is needed based on image quality."""
    # Skip preprocessing for computer-generated PDFs
    quality = self._assess_image_quality(image)
    if quality > 0.8:  # High quality, likely computer-generated
        return False
    return True

def process_image(self, image: np.ndarray, ...):
    # Conditionally apply preprocessing
    if self.config.preprocessing_enabled and self._should_preprocess(image, page_type):
        preprocessing_result = self.preprocess_image(image)
        processed_image = preprocessing_result.image
    else:
        processed_image = image
```

#### Fix 3: Optimize PaddleOCR Settings
```python
# In _initialize_engine for PaddleOCR
engine = PaddleOCR(
    use_angle_cls=True,
    lang=self.config.paddle_lang,
    use_gpu=self.config.paddle_use_gpu,
    # New settings for better accuracy
    det_db_thresh=0.1,  # Lower detection threshold
    det_db_box_thresh=0.3,  # Lower box threshold
    rec_batch_num=1,  # Process one at a time for accuracy
    use_space_char=True,  # Better space detection
    drop_score=0.3,  # Lower confidence threshold
    # Potentially use different model
    det_model_dir=None,  # Use default or specify better model
    rec_model_dir=None,  # Use default or specify better model
)
```

#### Fix 4: Optimize PDF Rendering
```python
# In pdf_handler.py render_page method
def render_page(self, page_num: int, dpi: Optional[int] = None) -> np.ndarray:
    # ... existing code ...

    # Use grayscale for better OCR
    pix = page.get_pixmap(
        matrix=mat,
        alpha=False,
        colorspace=fitz.csGRAY,  # Grayscale often better for OCR
        annots=False,  # Skip annotations
    )
```

### Phase 3: Advanced Improvements

1. **Adaptive DPI Selection**
   - Start with 300 DPI
   - If OCR confidence is low, retry with 400 DPI
   - Balance quality vs performance

2. **Multi-Model Approach**
   - Try PaddleOCR's different models (PP-OCRv3, PP-OCRv4)
   - Test server models vs mobile models
   - Consider using different models for detection and recognition

3. **Post-Processing**
   - Implement character correction for common OCR errors
   - Use context to fix obvious mistakes
   - Apply domain-specific corrections

## Implementation Priority

1. **High Priority (Immediate)**
   - Run diagnostic scripts
   - Increase default DPI to 300
   - Disable preprocessing for high-quality images
   - Update PaddleOCR configuration

2. **Medium Priority (Next Sprint)**
   - Implement adaptive DPI
   - Test different PaddleOCR models
   - Add quality-based preprocessing decisions

3. **Low Priority (Future)**
   - Multi-engine voting system
   - Post-processing corrections
   - Domain-specific optimizations

## Success Criteria

- Achieve >90% word accuracy on Test_PDF_Set_1.pdf
- Match the text extraction quality of commercial PDF software
- Maintain performance under 2 seconds per page
- Confidence scores should correlate with actual accuracy

## Testing Protocol

1. **Baseline Test**
   - Current accuracy: 0.8% F1 score
   - Document current settings and results

2. **Incremental Testing**
   - Test each change individually
   - Measure impact on accuracy and performance
   - Keep best-performing configuration

3. **Validation**
   - Test on both test PDFs
   - Verify no regression on actual scanned documents
   - Test on variety of computer-generated PDFs

## Risk Mitigation

- Keep all changes configurable
- Maintain backward compatibility
- Test thoroughly before changing defaults
- Document all configuration changes
