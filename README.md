# PDF Splitter Application

An intelligent PDF splitter that automatically identifies and separates individual documents within large, multi-document PDF files.

## Features

- **Multi-Signal Detection**: Combines LLM, visual, and heuristic analysis to identify document boundaries
- **OCR Support**: Handles both searchable and image-based PDFs
- **Manual Review Interface**: Web-based UI for reviewing and adjusting split points
- **High Performance**: Processes documents at < 5 seconds per page
- **Modular Architecture**: Designed for future integration into RAG-based systems

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
pytest
```

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
