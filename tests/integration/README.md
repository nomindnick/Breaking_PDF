# PDF Splitter Integration Tests

This directory contains comprehensive integration tests for the PDF splitting pipeline. These tests verify that all modules work together correctly and meet performance requirements.

## Test Files

### 1. `test_full_pipeline.py`
Tests the complete workflow from PDF loading through splitting:
- **Complete pipeline test (non-OCR)**: Full 32-page PDF processing
- **Complete pipeline with OCR**: Tests OCR-required PDFs
- **Error handling**: Verifies graceful failure modes
- **Preview generation**: Tests PDF preview functionality
- **Session persistence**: Verifies session data survives restarts
- **Performance benchmarking**: Ensures < 5 seconds per page target

### 2. `test_edge_cases.py`
Tests edge cases and error scenarios:
- **Single-page documents**: Each page as separate document
- **No boundaries detected**: Handling empty results
- **Overlapping boundaries**: Duplicate detection handling
- **Empty pages**: Processing whitespace-only content
- **Invalid page ranges**: Boundaries beyond page count
- **Special characters**: Filename sanitization
- **Corrupt PDFs**: Error handling for invalid files
- **Session edge cases**: Expired sessions, invalid states
- **Extreme document counts**: 100+ single-page documents

### 3. `test_performance.py`
Comprehensive performance benchmarking:
- **OCR performance**: Verifies 1-2 seconds per page target
- **Boundary detection**: Ensures < 0.1 seconds per page
- **Full pipeline**: Tests < 5 seconds per page overall
- **Memory usage**: Tracks scaling with PDF size
- **Cache effectiveness**: Measures performance improvements
- **Parallel processing**: Multi-worker performance
- **Stress testing**: Large PDFs (100+ pages)
- **Concurrent processing**: Multiple PDFs simultaneously

### 4. `test_concurrent_processing.py`
Thread safety and concurrent operation tests:
- **Thread safety**: All major components tested
- **Session management**: Concurrent session operations
- **Parallel processing**: Multiple PDFs in parallel
- **Race conditions**: Detection and prevention
- **Resource contention**: File handles, memory, CPU
- **Cleanup scenarios**: Proper resource release
- **Load testing**: High-volume operations
- **Deadlock detection**: Timeout-based prevention

## Running the Tests

### Basic Usage
```bash
# Run all integration tests
RUN_INTEGRATION_TESTS=true pytest tests/integration/

# Run specific test file
RUN_INTEGRATION_TESTS=true pytest tests/integration/test_full_pipeline.py

# Run with verbose output
RUN_INTEGRATION_TESTS=true pytest tests/integration/ -v

# Run excluding slow tests
RUN_INTEGRATION_TESTS=true pytest tests/integration/ -m "not slow"
```

### With OCR Tests
```bash
# Enable both OCR and integration tests
RUN_OCR_TESTS=true RUN_INTEGRATION_TESTS=true pytest tests/integration/
```

### Performance Benchmarking
```bash
# Run with benchmark details
RUN_INTEGRATION_TESTS=true pytest tests/integration/test_performance.py --benchmark-verbose

# Save benchmark results
RUN_INTEGRATION_TESTS=true pytest tests/integration/test_performance.py --benchmark-save=baseline

# Compare with baseline
RUN_INTEGRATION_TESTS=true pytest tests/integration/test_performance.py --benchmark-compare=baseline
```

## Test Data Requirements

The integration tests require test PDFs to be present:
- `test_data/Test_PDF_Set_1.pdf` - 32-page non-OCR PDF
- `test_data/Test_PDF_Set_2_ocr.pdf` - 32-page OCR-required PDF
- `test_data/Test_PDF_Set_Ground_Truth.json` - Expected boundaries

Tests will skip if these files are not available.

## Performance Targets

The integration tests verify these performance requirements:

| Operation | Target | Actual (Typical) |
|-----------|--------|------------------|
| OCR Processing | 1-2 sec/page | ~0.7 sec/page |
| Boundary Detection | < 0.1 sec/page | ~0.063 sec/page |
| Overall Pipeline | < 5 sec/page | ~0.5-1.0 sec/page |
| Memory Usage | < 500MB for 50 pages | ~200-300MB |
| Cache Hit Rate | > 80% | 85-95% |

## Test Coverage

The integration tests provide comprehensive coverage:

1. **Happy Path**: Normal operation with valid inputs
2. **Edge Cases**: Boundary conditions and unusual inputs
3. **Error Handling**: Invalid inputs, corrupted data
4. **Performance**: Speed and resource usage
5. **Concurrency**: Thread safety and parallel operations
6. **Persistence**: Session and cache durability
7. **Scalability**: Large documents and high load

## Known Issues

1. **Cache effectiveness test**: The 80% improvement threshold may need adjustment based on actual cache warming behavior
2. **OCR tests**: Require `RUN_OCR_TESTS=true` due to initialization overhead
3. **Large PDF tests**: Marked as `@pytest.mark.slow` due to processing time

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Fast CI tests (no OCR, no slow tests)
RUN_INTEGRATION_TESTS=true pytest tests/integration/ -m "not slow"

# Full CI tests (all tests enabled)
RUN_OCR_TESTS=true RUN_INTEGRATION_TESTS=true pytest tests/integration/
```

## Future Enhancements

1. **Visual regression tests**: Compare output PDFs visually
2. **API integration tests**: Test REST endpoints
3. **Frontend integration tests**: Selenium/Playwright tests
4. **Cross-platform tests**: Windows/macOS specific scenarios
5. **Docker integration tests**: Containerized environment
