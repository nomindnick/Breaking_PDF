# Utility Scripts

This directory contains utility scripts for testing and benchmarking the PDF splitter.

## OCR Performance Scripts

### benchmark_ocr_simple.py
Basic OCR performance benchmarking tool for comparing different OCR engines and settings.

**Usage:**
```bash
python scripts/benchmark_ocr_simple.py
```

### test_worker_performance.py
Tests optimal worker count for parallel OCR processing.

**Usage:**
```bash
python scripts/test_worker_performance.py
```

### ocr_accuracy_validation.py
Validates OCR accuracy against ground truth text files.

**Usage:**
```bash
python scripts/ocr_accuracy_validation.py
```

### create_optimized_ocr_pipeline.py
Demonstrates the optimized OCR pipeline with document type detection and specialized processing.

**Usage:**
```bash
python scripts/create_optimized_ocr_pipeline.py
```

## Important Notes

1. All scripts assume the virtual environment is activated
2. Test PDFs should be placed in `test_files/` directory
3. Ground truth files should be in JSON format
4. Results are saved to `analysis_output/` directory

## Cleanup Policy

To maintain a clean codebase:
- Diagnostic scripts (test_*, debug_*, etc.) are not committed
- Only production-ready utilities are kept
- See `.gitignore` for excluded patterns
