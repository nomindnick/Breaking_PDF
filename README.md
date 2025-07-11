# PDF Splitter Application

An intelligent PDF splitter that automatically identifies and separates individual documents within large, multi-document PDF files.

## Features

- **Embeddings-Based Detection**: Uses semantic similarity to identify document boundaries
- **OCR Support**: Handles both searchable and image-based PDFs with 90% accuracy
- **High Performance**: Processes documents at < 5 seconds per page
- **Advanced Caching**: 10-100x performance improvement for repeated operations
- **Modular Architecture**: Clean separation of preprocessing, detection, and splitting

## Current Status

### Completed Modules ✅
- **Preprocessing Module**: PDF handling, text extraction, OCR processing with 90% accuracy
- **Core Module**: Configuration, logging, exception handling
- **Test Infrastructure**: Comprehensive test suite with shared fixtures and utilities

### Completed Modules ✅ (continued)
- **Detection Module**: Simple embeddings-based boundary detection
  - Production-ready with F1=0.65-0.70
  - Fast performance: 0.063s per page
  - Simple and reliable approach avoids overfitting
- **Splitting Module**: Intelligent PDF document separation
  - Smart filename generation based on content
  - Document type detection (12+ types)
  - Session management for stateful operations
  - 48 comprehensive tests with 92-100% coverage

### Planned 📋
- **API Module**: FastAPI web service
- **Frontend Module**: Web user interface

## Quick Start

### Prerequisites

- Python 3.9+
- 32GB RAM recommended
- CPU-only (no GPU required)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Breaking_PDF
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment configuration:
```bash
cp .env.example .env
```

5. Run the application:
```bash
python main.py
```

### Basic Usage

```python
from pdf_splitter.detection import create_production_detector
from pdf_splitter.splitting import PDFSplitter

# Detect document boundaries
detector = create_production_detector()
pages = detector.detect_boundaries("path/to/multi_doc.pdf")

# Split the PDF
splitter = PDFSplitter()
proposal = splitter.generate_proposal("path/to/multi_doc.pdf", pages)
result = splitter.split_pdf(proposal, "output_directory")

# Files are created with intelligent names like:
# - Email_2024-03-15_Project_Update.pdf
# - Invoice_2024-03-20_INV-12345.pdf
# - Letter_2024-03-10_Insurance_Claim.pdf
```

## Project Structure

```
pdf_splitter/
├── core/           # Shared utilities and configuration
├── preprocessing/  # PDF loading and text extraction
├── detection/      # Document boundary detection
├── splitting/      # PDF splitting and output management
├── api/           # FastAPI web service
└── frontend/      # Web user interface
```

## Development

### Setup Development Environment

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pdf_splitter --cov-report=html

# Run specific module tests
pytest pdf_splitter/preprocessing/tests/

# Run excluding slow tests
pytest -m "not slow"

# Run with verbose output
pytest -v
```

The project includes comprehensive test infrastructure:
- **Shared Fixtures**: See `conftest.py` for reusable test fixtures
- **Test Utilities**: See `pdf_splitter/test_utils.py` for helper functions
- **Example Tests**: See `examples/test_example_usage.py` for usage patterns

### Code Style

This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## License

[To be determined]

## Contributing

[Contributing guidelines to be added]
