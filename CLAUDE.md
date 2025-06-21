# Project Context for AI Assistants

## Project Overview
**PDF Splitter** - An intelligent PDF document splitter that automatically identifies and separates individual documents within large, multi-document PDF files using multi-signal detection (LLM, visual, heuristic analysis).

### Primary Purpose
- Process large PDF files (like CPRA requests) containing multiple concatenated documents
- Automatically identify document boundaries with high accuracy
- Provide web interface for manual review and adjustment
- Target performance: < 5 seconds per page

### Future Vision
This project will serve as a foundational component for a larger RAG-based construction claims assistant system.

## Technical Stack

### Core Technologies
- **Language**: Python 3.12
- **Web Framework**: FastAPI with HTMX for frontend
- **PDF Processing**: PyMuPDF, pikepdf
- **OCR Engine**: PaddleOCR (primary), pytesseract (fallback)
- **LLM**: Transformers library (production), Ollama (development)
- **Database**: SQLite (development), PostgreSQL (production)

### Development Tools
- **Code Quality**: Black, isort, flake8, mypy
- **Testing**: pytest with async support
- **Documentation**: Sphinx
- **Version Control**: Git with pre-commit hooks

## Architecture Decisions

### Modular Design
The project follows a strict modular architecture:
```
pdf_splitter/
├── core/           # Shared utilities and configuration
├── preprocessing/  # PDF loading and text extraction
├── detection/      # Document boundary detection
├── splitting/      # PDF splitting and output
├── api/           # FastAPI web service
└── frontend/      # Web user interface
```

### Key Design Principles
1. **Test-Driven Development**: Tests co-located with modules in `/tests` directories
2. **Module Independence**: Each module should be "rock solid" before moving to the next
3. **Progressive Enhancement**: Start with basic functionality, enhance iteratively
4. **Performance First**: Optimize for < 5 seconds per page processing

## Development Guidelines

### Code Style
- Use type hints for all function signatures
- Follow PEP 8 with Black formatting (88 char line length)
- Write descriptive docstrings for all public functions/classes
- Keep functions small and focused (single responsibility)

### Testing Requirements
- Write unit tests for all new functionality
- Aim for > 80% code coverage
- Use pytest fixtures for common test data
- Test both success and failure cases

### Git Workflow
- Feature branches for new development
- Descriptive commit messages
- Pre-commit hooks run automatically (black, isort, flake8, mypy)
- Keep commits atomic and focused

## Current Development Status

### Completed ✅
- [x] Project structure and initial setup
- [x] Virtual environment with core dependencies
- [x] PaddleOCR installation and configuration
- [x] Pre-commit hooks configured
- [x] Basic FastAPI application skeleton

### In Progress 🚧
- [ ] OCR processor module (preprocessing/)
- [ ] PDF text extraction and quality assessment

### Upcoming 📋
- [ ] LLM-based boundary detection
- [ ] Visual pattern analysis for document breaks
- [ ] Heuristic detection rules
- [ ] Signal combination algorithm
- [ ] Web interface for manual review
- [ ] Performance optimization

## Module Development Order

1. **Preprocessing Module** (Current Focus)
   - PDF validation and loading
   - OCR processing with PaddleOCR
   - Text extraction from searchable PDFs
   - Quality assessment

2. **LLM Detection Module**
   - Context overlap analysis (30% strategy)
   - Prompt engineering
   - Boundary confidence scoring

3. **Visual & Heuristic Detection**
   - Layout analysis
   - Date pattern detection
   - Header/footer identification

4. **Signal Combination**
   - Weighted scoring algorithm
   - Confidence thresholds
   - Final boundary decisions

5. **Integration & Frontend**
   - API endpoints
   - Progress tracking
   - Manual review interface

## Performance Targets
- **OCR**: 1-2 seconds per page (when needed)
- **LLM Detection**: 1-2 seconds per boundary check
- **Visual/Heuristic**: < 0.5 seconds per page
- **Total**: < 5 seconds per page

## Test Data
- `Test_PDF_Set_1.pdf`: 32-page non-OCR test file
- `Test_PDF_Set_2_ocr.pdf`: 32-page OCR'd test file
- `Test_PDF_Set_Ground_Truth.json`: Expected boundaries and document types

## Environment Variables
Key settings in `.env`:
- `OCR_ENGINE`: Primary OCR engine (paddleocr)
- `LLM_PROVIDER`: LLM backend (transformers/ollama)
- `DEBUG`: Development mode flag
- `LOG_LEVEL`: Logging verbosity

## Common Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python main.py --reload

# Run tests
pytest

# Run pre-commit checks
pre-commit run --all-files

# Install new dependencies
pip install <package> && pip freeze > requirements.txt
```

## Notes for AI Assistants
- Always run tests after making changes
- Use the modular structure - don't cross module boundaries
- Follow the established patterns in existing code
- Consider performance implications of all changes
- Update this file when making significant architectural decisions
