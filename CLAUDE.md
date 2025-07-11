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
- **PDF Processing**: PyMuPDF (⚠️ AGPL license - requires commercial license), pikepdf
- **OCR Engine**: PaddleOCR (primary), EasyOCR/Tesseract (fallback)
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
- Use shared fixtures from global conftest.py
- Leverage test_utils.py for common testing patterns

### Git Workflow
- Feature branches for new development
- Descriptive commit messages
- Pre-commit hooks run automatically (black, isort, flake8, mypy)
- Keep commits atomic and focused
- **Clean up diagnostic/test files before committing** (see Technical Debt section)

## Current Development Status

### Completed Modules ✅

1. **Preprocessing Module** (100% complete, production-ready)
   - PDFHandler: High-performance PDF processing (0.02-0.05s/page)
   - TextExtractor: Advanced text extraction with layout analysis
   - Advanced Cache: 10-100x performance improvement for repeated access
   - OCR Processor: Multi-engine OCR with document type classification
   - Comprehensive Testing: 90%+ code coverage, OCR accuracy validated

2. **Detection Module** (100% complete, production-ready)
   - ✅ Base architecture complete (BaseDetector abstract class)
   - ✅ Data models defined (ProcessedPage, BoundaryResult, DetectionContext)
   - ✅ **Embeddings Detection** - PRODUCTION SOLUTION (F1=0.65-0.70)
     - Simple and reliable: all-MiniLM-L6-v2 with threshold=0.5
     - No complex post-processing or ensembles needed
     - Fast: ~0.063s per page
     - Good enough accuracy for production use
   - ✅ **Supporting Detectors** (available for future use):
     - Heuristic Detection: Basic pattern matching
     - Visual Detection: For scanned PDFs
     - LLM Detection: For research/analysis only
   - 📊 Production approach: **Simple embeddings, F1=0.65-0.70**

3. **Splitting Module** (100% complete, production-ready)
   - ✅ Data models (DocumentSegment, SplitProposal, SplitSession)
   - ✅ PDFSplitter service with smart filename generation
   - ✅ Session management with SQLite persistence
   - ✅ Preview generation for document segments
   - ✅ Comprehensive test suite (48 tests, all passing)
   - 📊 Coverage: 92-100% for core components

4. **Integration Testing** (100% complete)
   - ✅ Full Pipeline Tests: End-to-end workflow validation
   - ✅ Edge Case Tests: Handling unusual scenarios
   - ✅ Performance Tests: All targets exceeded
   - ✅ Concurrent Processing: Thread safety verified
   - ✅ Test Infrastructure: Automated runners and documentation

### Upcoming 📋
- [ ] API Module
- [ ] Frontend Module

## Module Development Order

1. **Preprocessing Module** ✅ COMPLETE
   - PDF validation and loading (PyMuPDF-based)
   - OCR with 90% accuracy (PaddleOCR + document classification)
   - Text extraction with layout analysis
   - Advanced caching system

2. **Detection Module** ✅ COMPLETE
   - **EmbeddingsDetector** ✅ PRODUCTION SOLUTION
     - Simple embeddings approach: all-MiniLM-L6-v2, threshold=0.5
     - F1=0.65-0.70 - Good enough for production
     - Speed: 0.063s per page (excellent)
     - No complex rules or ensembles needed
   - **Key Learning**: Simple embeddings work well enough
     - Complex post-processing led to overfitting on test set
     - Production focus: reliability over perfect metrics
     - Available detectors for future enhancement if needed

3. **Splitting Module** ✅ COMPLETE
   - **PDFSplitter Service** ✅ PRODUCTION READY
     - Intelligent document type detection (12+ types)
     - Smart filename generation with dates/identifiers
     - Preview generation for document segments
     - Thread-safe PDF operations with pikepdf
   - **Session Management** ✅ IMPLEMENTED
     - SQLite-based persistence
     - State tracking and expiration
     - Transaction-safe operations
   - **Key Features**:
     - Pattern-based document recognition
     - Metadata preservation
     - Batch processing support
     - 48 comprehensive tests (all passing)

4. **API Module**
   - FastAPI endpoints
   - Progress tracking
   - Async processing

5. **Frontend Module**
   - HTMX-based UI
   - Manual review interface
   - Real-time updates

## Performance Targets
- **OCR**: 1-2 seconds per page (when needed) ✅ Achieved: 0.693s avg
- **Boundary Detection**: < 0.1 seconds per page ✅ Achieved: 0.063s
- **F1 Score**: 0.65-0.70 with simple EmbeddingsDetector (production ready)
- **Overall Processing**: < 5 seconds per page ✅ Easily achieved

## Test Data
- `Test_PDF_Set_1.pdf`: 32-page non-OCR test file
- `Test_PDF_Set_2_ocr.pdf`: 32-page OCR'd test file
- `Test_PDF_Set_Ground_Truth.json`: Expected boundaries and document types
- `comprehensive_test_pdf.pdf`: 21-page mixed content test file (created for OCR testing)
- `comprehensive_test_pdf_ground_truth.json`: Ground truth for comprehensive test

## Environment Variables
Key settings in `.env`:
- `OCR_ENGINE`: Primary OCR engine (paddleocr)
- `LLM_PROVIDER`: LLM backend (transformers/ollama)
- `DEBUG`: Development mode flag
- `LOG_LEVEL`: Logging verbosity
- `LLM_CACHE_ENABLED`: Enable/disable LLM response caching (default: true)
- `LLM_CACHE_PATH`: Custom cache location (default: ~/.cache/pdf_splitter/llm_cache.db)

## Testing Infrastructure

### Shared Test Fixtures (conftest.py)
The project includes a comprehensive set of shared fixtures:
- **Configuration**: `pdf_config`, `ocr_config` - Default test configurations
- **Handlers**: `pdf_handler`, `loaded_pdf_handler` - Pre-configured handlers
- **Mocks**: `mock_pdf_page`, `mock_pdf_document` - For unit testing
- **Images**: `test_image_rgb`, `noisy_test_image` - Test image data
- **Utilities**: `temp_dir`, `performance_timer` - Helper fixtures

### Test Utilities (test_utils.py)
Common testing functions available:
- `create_test_pdf()` - Generate test PDFs programmatically
- `assert_pdf_valid()` - Validate PDF files
- `assert_text_quality()` - Check extracted text quality
- `measure_performance()` - Performance benchmarking
- `create_mock_*()` - Mock object creators

### Testing Patterns
```python
# Example using shared fixtures
def test_with_config(pdf_config):
    assert pdf_config.default_dpi == 150

# Example using test utilities
from pdf_splitter.test_utils import create_test_pdf, assert_pdf_valid

def test_pdf_creation(temp_dir):
    pdf_path = create_test_pdf(num_pages=5, output_path=temp_dir / "test.pdf")
    assert_pdf_valid(pdf_path)
```

### Important Testing Notes
- **PDFConfig in Tests**: When creating PDFConfig instances in test fixtures, only pass valid configuration parameters. The model uses Pydantic validation and will reject unknown fields.
- **Resource Cleanup**: The caching system now properly closes PIL Images when evicting entries to prevent resource leaks.
- **OCR Cleanup**: OCR engines have cleanup methods that are called on deletion to free resources.
- **LLM Cache in Tests**: The LLM detector uses persistent SQLite caching that can interfere with testing:
  - Tests may return cached results instead of calling Ollama
  - Use `cache_enabled=False` or `detector.clear_cache()` for fresh results
  - Set `LLM_CACHE_ENABLED=false` to disable caching globally

## Common Commands
```bash
# Activate virtual environment
source venv/bin/activate

# Run the application
python main.py --reload

# Run tests
pytest

# Run tests with coverage
pytest --cov=pdf_splitter --cov-report=html

# Run specific test module
pytest pdf_splitter/preprocessing/tests/

# Run tests excluding slow tests
pytest -m "not slow"

# Run ALL tests including OCR and integration tests
RUN_OCR_TESTS=true RUN_INTEGRATION_TESTS=true pytest

# Run tests with LLM cache disabled (for fresh results)
LLM_CACHE_ENABLED=false pytest

# Clear LLM cache before testing
rm ~/.cache/pdf_splitter/llm_cache.db

# Run pre-commit checks
pre-commit run --all-files

# Install new dependencies
pip install <package> && pip freeze > requirements.txt
```

## Pre-commit Hooks Configuration

The project uses pre-commit hooks to maintain code quality. Configuration has been optimized for developer productivity while maintaining standards.

### Configuration Files
- **`pyproject.toml`**: Unified configuration for Black, isort, flake8, mypy, and pytest
- **`.pre-commit-config.yaml`**: Pre-commit hook definitions

### Key Settings
- **Line Length**: 100 characters (increased from 88 for better readability)
- **Type Checking**: Mypy is optional (manual stage) to avoid blocking quick commits
- **Python Version**: 3.12
- **Fail Fast**: Disabled - all hooks run even if one fails

### Usage During Development

#### Quick Development (Skip Hooks)
```bash
# Skip all hooks when needed
git commit --no-verify -m "WIP: Quick fix"

# Skip only mypy (it's in manual stage by default)
git commit -m "feat: Add new feature"
```

#### Run Specific Hooks
```bash
# Run all hooks including mypy
pre-commit run --all-files --hook-stage manual

# Run specific hook
pre-commit run black --all-files
pre-commit run flake8 --all-files
pre-commit run mypy --all-files
```

#### Fix Common Issues
```bash
# Auto-fix formatting issues
black .
isort .

# Check specific files
flake8 path/to/file.py
mypy path/to/file.py

# Update pre-commit hooks to latest versions
pre-commit autoupdate

# Reinstall hooks after configuration changes
pre-commit install
```

### Hook Details
1. **General Checks**: trailing whitespace, EOF fixes, YAML/JSON validation
2. **Black**: Code formatting (automatic fixes)
3. **isort**: Import sorting (automatic fixes)
4. **flake8**: Linting with reasonable line length (100 chars)
5. **mypy**: Type checking (optional/manual stage)

### Tips for CI/CD
In CI pipelines, run all hooks including manual stages:
```bash
pre-commit run --all-files --show-diff-on-failure --hook-stage manual
```

### Running All Tests
Some tests are skipped by default to speed up development:
- **OCR Tests**: Require `RUN_OCR_TESTS=true` environment variable
- **Integration Tests**: Require `RUN_INTEGRATION_TESTS=true` environment variable

These tests validate OCR accuracy and end-to-end functionality but take longer to run due to:
- OCR engine initialization overhead
- Processing actual PDF files
- Comprehensive accuracy measurements

To run the complete test suite with all tests enabled:
```bash
RUN_OCR_TESTS=true RUN_INTEGRATION_TESTS=true pytest
```

## Critical Technical Decisions

1. **PyMuPDF License**: AGPL v3 - requires commercial license for production use
2. **OCR Settings**:
   - `paddle_enable_mkldnn=False` - Critical for accuracy (91x improvement)
   - Default DPI: 300 (not 150) - Updated based on testing
   - Document type classification for optimized processing
3. **Performance Optimization**:
   - `OMP_THREAD_LIMIT=1` for containerized environments
   - Parallel processing with 4 workers optimal
4. **Caching**: Advanced multi-tier caching system is critical for performance
5. **Detection Configuration**:
   - **IMPORTANT**: Use simple EmbeddingsDetector for production
   - Plain embeddings achieves F1=0.65-0.70 at 0.063s/page
   - No complex post-processing or ensembles needed
   - Key settings: all-MiniLM-L6-v2 model, threshold=0.5
   - Focus on reliability over perfect metrics

## Avoiding Technical Debt

### During Development
1. **Diagnostic Scripts**: Create in `scripts/` with descriptive names (e.g., `test_ocr_accuracy.py`)
2. **Temporary Files**: Use `.gitignore` patterns, never commit PNG/debug files
3. **Experimentation**: Use feature branches for major experiments

### Before Committing
1. **Clean Up Diagnostic Files**:
   ```bash
   # Remove test scripts
   rm scripts/test_*.py scripts/debug_*.py scripts/diagnose_*.py

   # Remove temporary images
   rm *.png analysis_output/*.png

   # Keep only essential utilities
   ```

2. **Consolidate Findings**:
   - Document key discoveries in `development_progress.md`
   - Create summary documents for major investigations
   - Update configuration files with optimal settings

3. **Review Checklist**:
   - [ ] All tests passing
   - [ ] No diagnostic/temporary files
   - [ ] Documentation updated
   - [ ] Key findings recorded
   - [ ] Configuration optimized

## Notes for AI Assistants
- Always run tests after making changes
- Use the modular structure - don't cross module boundaries
- Follow the established patterns in existing code
- Consider performance implications of all changes
- Update this file when making significant architectural decisions
- **Clean up all diagnostic code before suggesting commits**
- Document findings in `development_progress.md` not in scattered files
- When doing deep investigations, plan for cleanup from the start
