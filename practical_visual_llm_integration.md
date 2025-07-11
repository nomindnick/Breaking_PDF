# Practical Integration of Visual and LLM Detectors

## Current Situation

- **Embeddings detector**: F1 ~0.65-0.70 (consistent, fast)
- **Visual detector**: Implemented but requires more analysis
- **LLM detector**: High accuracy (F1 ~0.89) but very slow (11-40s per boundary)

## Practical Use Cases for Visual Detector

### 1. Scanned PDF Support
The visual detector could be the **primary detector for scanned PDFs** where embeddings fail:

```python
def detect_boundaries(pages, pdf_has_text):
    if not pdf_has_text:
        # Use visual detector for scanned PDFs
        return visual_detector.detect_boundaries(pages)
    else:
        # Use embeddings for text PDFs
        return embeddings_detector.detect_boundaries(pages)
```

### 2. High-Confidence Verification
Use visual signals to **increase confidence** in uncertain embeddings boundaries:

```python
def verify_boundary_with_visual(page_num, embeddings_confidence):
    visual_result = visual_detector.check_boundary(page_num)
    
    if embeddings_confidence > 0.4 and embeddings_confidence < 0.6:
        # Uncertain embeddings - use visual to confirm
        if visual_result.confidence > 0.7:
            return True  # Both agree
        elif visual_result.confidence < 0.3:
            return False  # Visual strongly disagrees
    
    return embeddings_confidence > 0.5
```

### 3. Format Change Detection
Visual detector excels at finding **dramatic layout changes**:
- Landscape → Portrait orientation
- Text document → Full-page diagram
- Single column → Multi-column layout
- Different document templates

These are often strong boundary signals that embeddings might miss.

## Practical Use Cases for LLM Detector

### 1. Human-in-the-Loop Verification
Instead of automatic verification, use LLM when **users flag uncertain boundaries**:

```python
def user_requested_verification(page_num):
    # Only run expensive LLM when user explicitly asks
    llm_result = llm_detector.verify_boundary(page_num)
    return {
        'is_boundary': llm_result.is_boundary,
        'explanation': llm_result.reasoning,
        'confidence': llm_result.confidence
    }
```

### 2. Batch Processing for High-Value Documents
For critical documents, run LLM verification **overnight or in background**:

```python
def process_high_value_pdf(pdf_path):
    # Quick initial detection
    quick_boundaries = embeddings_detector.detect_boundaries(pages)
    
    # Queue for background LLM verification
    job_id = queue_llm_verification(pdf_path, quick_boundaries)
    
    # Return quick results immediately
    return {
        'initial_boundaries': quick_boundaries,
        'verification_job_id': job_id,
        'status': 'verification_pending'
    }
```

### 3. Learning Dataset Generation
Use LLM to **generate training data** for improving other detectors:

```python
def generate_training_data(pdf_samples):
    training_data = []
    
    for pdf in pdf_samples:
        # Get high-quality labels from LLM
        llm_boundaries = llm_detector.detect_all_boundaries(pdf)
        
        # Extract features at those boundaries
        for boundary in llm_boundaries:
            features = extract_features(pdf, boundary)
            training_data.append({
                'features': features,
                'is_boundary': True,
                'confidence': boundary.confidence
            })
    
    return training_data
```

### 4. Document Type Classification
LLMs are excellent at identifying **document types**, which helps boundary detection:

```python
def classify_and_detect(pages):
    # Sample a few pages
    sample_pages = pages[::5]  # Every 5th page
    
    # Get document type from LLM
    doc_type = llm_detector.classify_document_type(sample_pages)
    
    # Use type-specific detection strategy
    if doc_type == 'email_thread':
        return detect_email_boundaries(pages)
    elif doc_type == 'legal_filing':
        return detect_legal_boundaries(pages)
    else:
        return embeddings_detector.detect_boundaries(pages)
```

## Recommended Architecture

```python
class SmartBoundaryDetector:
    def __init__(self):
        self.embeddings = EmbeddingsDetector()
        self.visual = VisualDetector()
        self.llm = LLMDetector()
    
    def detect_boundaries(self, pages, options=None):
        # 1. Check if PDF has text
        has_text = any(page.text.strip() for page in pages)
        
        if not has_text:
            # Scanned PDF - use visual
            return self.visual.detect_boundaries(pages)
        
        # 2. Primary detection with embeddings
        boundaries = self.embeddings.detect_boundaries(pages)
        
        # 3. Optional visual verification for uncertain boundaries
        if options and options.get('use_visual_verification'):
            boundaries = self._verify_with_visual(boundaries, pages)
        
        # 4. LLM only for specific cases
        if options and options.get('use_llm_for_pages'):
            llm_pages = options['use_llm_for_pages']
            boundaries = self._verify_with_llm(boundaries, pages, llm_pages)
        
        return boundaries
    
    def _verify_with_visual(self, boundaries, pages):
        # Use visual to filter uncertain embeddings boundaries
        verified = []
        visual_results = self.visual.detect_boundaries(pages)
        
        for boundary in boundaries:
            if boundary.confidence < 0.6:
                # Check visual agreement
                visual_match = next(
                    (v for v in visual_results if v.page_number == boundary.page_number),
                    None
                )
                if visual_match and visual_match.confidence > 0.5:
                    verified.append(boundary)
            else:
                # High confidence - keep it
                verified.append(boundary)
        
        return verified
```

## Key Principles

1. **Use the right tool for the job**:
   - Embeddings: General text-based detection
   - Visual: Scanned PDFs and layout changes
   - LLM: High-stakes verification and learning

2. **Avoid automatic LLM calls** due to speed/cost:
   - User-triggered verification
   - Batch processing
   - Training data generation

3. **Combine signals intelligently**:
   - Don't just vote - use each detector's strengths
   - Visual for dramatic changes
   - Embeddings for semantic shifts
   - LLM for difficult cases

4. **Progressive enhancement**:
   - Start with fast embeddings
   - Add visual for specific cases
   - Reserve LLM for when it's worth the wait

This approach would give you:
- **F1 ~0.70** baseline (embeddings)
- **F1 ~0.75+** with visual for appropriate docs
- **F1 ~0.85+** with selective LLM verification
- **Practical speed** (mostly sub-second)
- **Flexibility** for different document types