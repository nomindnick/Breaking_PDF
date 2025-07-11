# Utility Scripts

This directory contains utility scripts for the PDF splitter.

## Available Scripts

### benchmark_ocr_simple.py
Basic OCR performance benchmarking tool for comparing different OCR engines and settings.

**Usage:**
```bash
python scripts/benchmark_ocr_simple.py
```

### check_ollama_setup.py
Verifies that Ollama is properly installed and configured for LLM-based detection.

**Usage:**
```bash
python scripts/check_ollama_setup.py
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
