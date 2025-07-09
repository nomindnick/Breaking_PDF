# Visual Boundary Detector

This module implements visual-based document boundary detection using computer vision techniques. It provides an experimentation framework for testing different approaches and finding the optimal configuration.

## Overview

The visual detector analyzes visual similarity between consecutive pages to identify document boundaries. It implements the **Pairwise Similarity Paradigm** recommended in the research, comparing adjacent pages using various techniques from simple histogram comparison to advanced deep learning embeddings.

## Current Techniques

### 1. Histogram Comparison
- **Speed**: Very fast (< 0.05s/page)
- **Method**: Compares grayscale intensity distributions
- **Best for**: Detecting dramatic visual changes
- **Parameters**: `bins` (default: 256), `threshold` (default: 0.8)

### 2. Structural Similarity (SSIM)
- **Speed**: Fast (< 0.1s/page)
- **Method**: Perceptual similarity considering luminance, contrast, structure
- **Best for**: Layout and structural changes
- **Parameters**: `window_size` (default: 11), `threshold` (default: 0.7)

### 3. Perceptual Hashing
- **Speed**: Very fast (< 0.02s/page)
- **Method**: Generates compact fingerprints, compares Hamming distance
- **Best for**: Robust baseline with good speed/accuracy balance
- **Parameters**: `hash_size` (default: 8), `threshold` (default: 10)

## Usage

### Running Experiments

Use the CLI tool to test techniques:

```bash
# Test all techniques
python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
    test_files/Test_PDF_Set_1.pdf \
    test_files/Test_PDF_Set_Ground_Truth.json

# Test specific technique
python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
    test_files/Test_PDF_Set_1.pdf \
    test_files/Test_PDF_Set_Ground_Truth.json \
    --technique phash \
    --threshold 15

# Compare techniques
python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
    test_files/Test_PDF_Set_1.pdf \
    test_files/Test_PDF_Set_Ground_Truth.json \
    --compare

# Threshold optimization
python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
    test_files/Test_PDF_Set_1.pdf \
    test_files/Test_PDF_Set_Ground_Truth.json \
    --technique ssim \
    --threshold-sweep
```

### Programmatic Usage

```python
from pdf_splitter.detection.visual_detector.experiments import (
    VisualExperimentRunner,
    create_technique
)

# Create runner
runner = VisualExperimentRunner()

# Create and test a technique
technique = create_technique("phash", threshold=10, hash_size=8)
result = runner.run_experiment(
    technique,
    Path("test.pdf"),
    Path("ground_truth.json")
)

# Access results
print(f"F1 Score: {result.metrics.f1_score:.3f}")
print(f"Processing time: {result.metrics.avg_time_per_page:.3f}s/page")
```

## Creating Test Data

### Test PDF Requirements
Create a diverse PDF with:
- Multiple document types (letters, forms, emails, etc.)
- Various layouts (single/multi-column, headers/footers)
- Different visual styles (fonts, margins, colors)
- 50-60 pages total, 12-15 distinct documents

### Ground Truth Format
```json
{
  "documents": [
    {
      "pages": "1-4",
      "type": "Business Letter",
      "visual_markers": {
        "has_letterhead": true,
        "consistent_margins": true
      }
    }
  ],
  "visual_boundaries": [
    {
      "after_page": 4,
      "confidence": 0.95,
      "signals": ["letterhead_change", "margin_shift"]
    }
  ]
}
```

## Experiment Workflow

1. **Prepare Test Data**
   - Create comprehensive test PDF
   - Create ground truth JSON

2. **Run Initial Tests**
   ```bash
   # Test all techniques with defaults
   python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
       visual_test_pdf.pdf visual_test_ground_truth.json --compare
   ```

3. **Optimize Thresholds**
   ```bash
   # For best performing technique
   python -m pdf_splitter.detection.visual_detector.experiments.run_visual_experiments \
       visual_test_pdf.pdf visual_test_ground_truth.json \
       --technique phash --threshold-sweep
   ```

4. **Analyze Results**
   - Check `experiments/results/` for detailed JSON results
   - Review threshold analysis plots
   - Examine false positives/negatives

5. **Implement Production Version**
   - Use best technique configuration
   - Create `VisualDetector` class inheriting from `BaseDetector`

## Results Tracking

Results are saved in `experiments/results/` with:
- Individual experiment JSONs
- Threshold analysis plots
- Comparison summaries

Track progress in `EXPERIMENT_RESULTS.md`.

## Next Steps

1. **Create Test PDF**: Build comprehensive test dataset
2. **Run Baseline Experiments**: Test all three techniques
3. **Implement Advanced Techniques**: Add ORB features and deep learning
4. **Optimize Performance**: Find best speed/accuracy trade-off
5. **Production Implementation**: Create final `VisualDetector` class

## Technical Notes

- All techniques work on CPU only
- Image rendering uses PyMuPDF at 150 DPI by default
- All libraries (OpenCV, scikit-image, imagehash) are permissively licensed
- Techniques are modular and pluggable for easy extension

## Contributing

When adding new techniques:
1. Inherit from `BaseVisualTechnique`
2. Implement `compute_similarity()` method
3. Add to `create_technique()` factory
4. Create corresponding tests
5. Document parameters and performance
