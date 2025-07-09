# Visual Boundary Detection Experiment Results

This document tracks all experimental results for visual boundary detection techniques. Each experiment is documented with parameters, results, and observations.

## Summary Dashboard

| Technique | F1 Score | Precision | Recall | Speed (s/page) | Status |
|-----------|----------|-----------|---------|----------------|---------|
| Histogram Comparison | 0.000 | 0.000 | 0.000 | 0.001 | ❌ Failed |
| SSIM | 0.000 | 0.000 | 0.000 | 0.143 | ❌ Failed |
| Perceptual Hash (pHash) | 0.604 | 0.471 | 0.842 | 0.032 | ✅ Working |
| **Combined Hash (Voting 2/3)** | **0.667** | **0.500** | **1.000** | **0.031** | **✅ Best** |
| Combined Hash (Weighted) | 0.207 | 0.300 | 0.158 | 0.032 | ❌ Too Conservative |
| ORB Features | - | - | - | - | 🔄 Pending |
| MobileNetV2 | - | - | - | - | 🔄 Pending |
| EfficientNet-Lite0 | - | - | - | - | 🔄 Pending |
| Hybrid Cascade | - | - | - | - | 🔄 Pending |

## Detailed Results

### Baseline Techniques

---

#### Experiment 1: Histogram Comparison
- **Date**: 2025-07-09
- **Technique**: Grayscale Histogram Comparison
- **Parameters**:
  - Bins: 256
  - Distance Metric: Correlation coefficient
  - Threshold: 0.8 (also tested 0.3-0.95 sweep)
- **Results**:
  - Precision: 0.000
  - Recall: 0.000
  - F1 Score: 0.000
  - Avg Processing Time: 0.001s/page
  - Memory Usage: ~50 MB
- **Confusion Matrix**:
  ```
  True Positives: 0
  False Positives: 0
  True Negatives: 19
  False Negatives: 19
  ```
- **Observations**:
  - Complete failure - all similarity scores are 0.99+ (pages appear identical)
  - Histogram comparison is too coarse for document boundaries
  - All pages have similar grayscale distributions (mostly white with black text)
  - Even with aggressive threshold tuning, no boundaries detected
- **Next Steps**:
  - Try different color spaces (HSV, LAB)
  - Use localized histograms instead of global
  - Consider edge/gradient histograms

---

#### Experiment 2: SSIM (Structural Similarity)
- **Date**: 2025-07-09
- **Technique**: Full-page SSIM
- **Parameters**:
  - Window Size: 11
  - Gaussian Weights: True
  - Threshold: 0.7
- **Results**:
  - Precision: 0.000
  - Recall: 0.000
  - F1 Score: 0.000
  - Avg Processing Time: 0.143s/page
  - Memory Usage: ~100 MB
- **Observations**:
  - Complete failure - no boundaries detected
  - SSIM scores also very high (likely 0.95+) between all pages
  - Much slower than histogram (143x slower)
  - SSIM is designed for image quality assessment, not document comparison
- **Next Steps**:
  - Try region-based SSIM (top/bottom halves)
  - Experiment with different window sizes
  - Consider MSSIM (multi-scale SSIM)

---

#### Experiment 3: Perceptual Hashing
- **Date**: 2025-07-09
- **Technique**: pHash (DCT-based)
- **Parameters**:
  - Hash Size: 64 bits (8x8)
  - Hamming Distance Threshold: 10
- **Results**:
  - Precision: 0.471
  - Recall: 0.842
  - F1 Score: 0.604
  - Avg Processing Time: 0.032s/page
  - Memory Usage: ~50 MB
- **Observations**:
  - Best performing technique so far!
  - Good recall (84%) but many false positives (53% precision)
  - Hamming distances range from 6 to 26
  - Fast performance (32ms/page)
  - Also computed ahash and dhash for comparison
- **Next Steps**:
  - Optimize threshold (current: 10)
  - Try larger hash sizes (16x16, 32x32)
  - Test different hash algorithms
  - Combine multiple hash types

---

#### Experiment 4: Combined Hash (Voting Mode)
- **Date**: 2025-07-09
- **Technique**: Combined pHash, aHash, dHash with voting
- **Parameters**:
  - Hash Size: 64 bits (8x8)
  - Voting Mode: True
  - Required Votes: 2/3
  - Individual Thresholds: pHash=10, aHash=12, dHash=12
- **Results**:
  - Precision: 0.500
  - Recall: 1.000
  - F1 Score: 0.667
  - Avg Processing Time: 0.031s/page
  - Memory Usage: ~50 MB
- **Observations**:
  - Best performing technique overall!
  - Perfect recall (100%) - catches all true boundaries
  - Improved precision over single pHash (50% vs 47%)
  - Voting approach more effective than weighted average
  - False positives occur at pages with significant layout changes
- **Next Steps**:
  - Fine-tune individual hash thresholds
  - Analyze false positive patterns
  - Test with larger hash sizes

---

### Advanced Techniques

---

#### Experiment 4: ORB Feature Matching
- **Date**: [YYYY-MM-DD]
- **Technique**: ORB keypoint matching
- **Parameters**:
  - Max Features: 1000
  - Scale Factor: 1.2
  - Match Ratio Threshold: [value]
- **Results**:
  - Precision: 0.XX
  - Recall: 0.XX
  - F1 Score: 0.XX
  - Avg Processing Time: X.XXs/page
  - Memory Usage: XX MB
- **Observations**:
  - [Key findings]
- **Next Steps**:
  - [Improvements to try]

---

#### Experiment 5: Deep Learning - MobileNetV2
- **Date**: [YYYY-MM-DD]
- **Technique**: MobileNetV2 feature extraction
- **Parameters**:
  - Input Size: 224x224
  - Feature Layer: global_average_pooling2d
  - Similarity Metric: Cosine
  - Threshold: [value]
- **Results**:
  - Precision: 0.XX
  - Recall: 0.XX
  - F1 Score: 0.XX
  - Avg Processing Time: X.XXs/page
  - Memory Usage: XX MB
- **Observations**:
  - [Key findings]
- **Next Steps**:
  - [Improvements to try]

---

#### Experiment 6: Deep Learning - EfficientNet-Lite0
- **Date**: [YYYY-MM-DD]
- **Technique**: EfficientNet-Lite0 feature extraction
- **Parameters**:
  - Input Size: 224x224
  - Feature Layer: top_activation
  - Similarity Metric: Cosine
  - Threshold: [value]
- **Results**:
  - Precision: 0.XX
  - Recall: 0.XX
  - F1 Score: 0.XX
  - Avg Processing Time: X.XXs/page
  - Memory Usage: XX MB
- **Observations**:
  - [Key findings]
- **Next Steps**:
  - [Improvements to try]

---

### Hybrid Approaches

---

#### Experiment 7: Cascaded Detection
- **Date**: [YYYY-MM-DD]
- **Technique**: pHash → EfficientNet cascade
- **Parameters**:
  - Stage 1: pHash with loose threshold
  - Stage 2: EfficientNet for uncertain cases
  - Uncertainty Range: Hamming distance 5-15
- **Results**:
  - Precision: 0.XX
  - Recall: 0.XX
  - F1 Score: 0.XX
  - Avg Processing Time: X.XXs/page
  - Memory Usage: XX MB
  - Stage 2 Usage: XX% of pages
- **Observations**:
  - [Key findings]
- **Next Steps**:
  - [Improvements to try]

---

## Performance Analysis

### Speed Comparison
```
Technique               | Avg Time | Min Time | Max Time | Std Dev
------------------------|----------|----------|----------|--------
Histogram Comparison    |          |          |          |
SSIM                   |          |          |          |
Perceptual Hash        |          |          |          |
ORB Features           |          |          |          |
MobileNetV2            |          |          |          |
EfficientNet-Lite0     |          |          |          |
Hybrid Cascade         |          |          |          |
```

### Accuracy by Document Type
```
Document Type    | Histogram | SSIM | pHash | ORB | MobileNet | EfficientNet
-----------------|-----------|------|-------|-----|-----------|-------------
Business Letters |           |      |       |     |           |
Technical Docs   |           |      |       |     |           |
Forms           |           |      |       |     |           |
Email Chains    |           |      |       |     |           |
Mixed Content   |           |      |       |     |           |
```

## Key Findings

### What Works Well
1. **Combined Hash Voting** achieves best F1=0.667 with perfect recall (100%)
2. **Speed** is excellent for all hash techniques (~30ms/page)
3. **Voting approach** more effective than weighted averaging for combining signals
4. **Hash-based approaches** handle document variations better than pixel comparisons

### Challenges Identified
1. **Histogram/SSIM complete failure**: Document pages too similar in global statistics
2. **False positive rate**: Even best approach has 50% precision
3. **Threshold sensitivity**: Hash distance thresholds critical for performance
4. **Within-document variation**: Some pages within same document trigger false boundaries

### Surprising Results
1. **Complete failure of histogram/SSIM**: All similarities >0.95, no discrimination
2. **Voting superiority**: 2/3 voting (F1=0.667) vastly outperforms weighted average (F1=0.207)
3. **Perfect recall achievable**: Combined voting catches 100% of true boundaries
4. **Minimal speed penalty**: Combined approach as fast as single hash

## Real-World Performance

### Test on Production PDFs (Test_PDF_Set_1 & Test_PDF_Set_2)
- **Date**: 2025-07-09
- **Test Set**: 36-page real construction documents (emails, invoices, RFIs, plans)
- **True Boundaries**: 9 document boundaries

#### Combined Hash with Different Thresholds
| Threshold | F1 Score | Precision | Recall | False Positives |
|-----------|----------|-----------|---------|-----------------|
| 2 votes   | 0.409    | 0.257     | 1.000   | 26              |
| 1 vote    | 0.439    | 0.281     | 1.000   | 23              |
| 0.3 votes | 0.514    | 0.346     | 1.000   | 17              |

**Key Finding**: Visual detection alone struggles with real-world documents due to significant layout variations within multi-page documents.

## Recommendations

### For Production Implementation
1. **Visual Detection Limitations**:
   - Works well on synthetic/uniform documents (F1=0.667)
   - Poor precision on real-world documents (F1=0.514 at best)
   - Should NOT be used as primary detector for production

2. **Recommended Approach**:
   - Use visual detection as supplementary signal only
   - Combine with LLM detector (primary) for semantic understanding
   - Visual signals can help with confidence scoring
   - Configuration for supplementary use:
     ```python
     CombinedHash(
         hash_size=8,
         voting_mode=True,
         threshold=1,  # More sensitive for supplementary role
         phash_threshold=10,
         ahash_threshold=12,
         dhash_threshold=12
     )
     ```

### Future Improvements
1. **Hybrid approach**: Visual features should be combined with semantic analysis
2. **Document-type specific thresholds**: Different document types need different visual sensitivity
3. **Focus on specific visual cues**: Headers, footers, logos, page numbers
4. **Consider visual detection for specific use cases**:
   - Detecting scanned vs digital PDFs
   - Finding major layout changes (portrait to landscape)
   - Identifying completely blank pages

## Appendix

### Test Set Statistics
- Total Pages: 39
- Total Documents: 20
- True Boundaries: 19
- Document Types: Business letters, technical docs, invoices, emails, forms, reports, presentations, legal documents, memos, newsletters, resumes, contracts, meeting minutes, project proposals, budget reports, training manuals, policy documents, research papers, executive summaries, audit reports

### Hardware Specifications
- CPU: AMD Ryzen or Intel (varies by system)
- RAM: 16+ GB
- OS: Linux
- Python: 3.12

### Software Versions
- OpenCV: 4.11.0
- scikit-image: 0.25.2
- imagehash: 4.3.2
- PyMuPDF: 1.24.14

---
*Results last updated: 2025-07-09*

## Conclusions

Visual boundary detection shows a significant performance gap between synthetic and real-world documents:

1. **Synthetic documents** (uniform, generated): F1=0.667 with good precision/recall balance
2. **Real-world documents** (varied layouts): F1=0.514 max with very poor precision

This suggests that visual-only detection is insufficient for production use. The technique successfully identifies visual changes but cannot distinguish between:
- True document boundaries
- Layout changes within the same document (e.g., different pages of a multi-page form)
- Style variations (e.g., letterhead on first page only)

**Final Recommendation**: Implement visual detection as a supplementary signal to enhance LLM-based detection, not as a standalone solution.
