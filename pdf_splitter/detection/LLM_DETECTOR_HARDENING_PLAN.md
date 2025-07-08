# LLM Detector Hardening Plan

## Objective
Ensure the llm_detector.py implementation is production-ready, optimized, and thoroughly tested before proceeding to other detection modules.

## Current State Assessment
- **Performance**: ~33 seconds per page (bottleneck)
- **Accuracy**: F1: 0.889, Precision: 100%, Recall: 80%
- **Testing**: Basic unit tests with mocked Ollama
- **Error Handling**: Basic retry logic and timeout handling
- **Caching**: Simple hash-based in-memory cache

## Phase 1: Deep Code Analysis & Optimization (Priority: HIGH)

### 1.1 Performance Profiling
- [ ] Profile the current implementation to identify bottlenecks
- [ ] Measure memory usage with large documents
- [ ] Analyze Ollama API call overhead vs processing time
- [ ] Test concurrent request handling capabilities

### 1.2 Text Extraction Optimization
- [ ] Experiment with different window sizes (currently 15 lines)
- [ ] Test adaptive window sizing based on content density
- [ ] Validate that bottom/top extraction is optimal vs middle sections
- [ ] Consider overlapping windows for better context

### 1.3 Prompt Optimization
- [ ] Test compressed versions while maintaining accuracy
- [ ] Remove redundant examples if possible
- [ ] Optimize token usage without sacrificing performance
- [ ] Test prompt caching at Ollama level

### 1.4 Caching Enhancement
- [ ] Implement persistent cache (SQLite/Redis)
- [ ] Add similarity-based caching (not just exact matches)
- [ ] Implement cache size limits and eviction policies
- [ ] Add cache warming strategies

## Phase 2: Comprehensive Testing (Priority: HIGH)

### 2.1 Edge Case Test Suite
- [ ] Empty pages handling
- [ ] Single-line pages
- [ ] Pages with only headers/footers
- [ ] Non-text content (tables, images references)
- [ ] Corrupted/garbled text
- [ ] Multiple languages
- [ ] Special characters and encoding issues

### 2.2 Integration Tests
- [ ] Test with real Ollama instance
- [ ] Test model switching (gemma2, llama3, etc.)
- [ ] Test Ollama restart/crash scenarios
- [ ] Network failure simulation
- [ ] Timeout and retry behavior
- [ ] Concurrent request handling

### 2.3 Performance Tests
- [ ] Load testing with 100+ page documents
- [ ] Memory leak detection
- [ ] Response time consistency
- [ ] Cache hit rate analysis
- [ ] Throughput testing

### 2.4 Real PDF Validation
- [ ] Test with actual CPRA documents
- [ ] Test with mixed document types
- [ ] Validate against ground truth data
- [ ] Compare with manual splitting results

## Phase 3: Robustness & Error Handling (Priority: MEDIUM)

### 3.1 Enhanced Error Handling
- [ ] Graceful degradation when Ollama unavailable
- [ ] Better parsing error recovery
- [ ] Implement circuit breaker pattern
- [ ] Add fallback strategies

### 3.2 Response Validation
- [ ] Strict XML parsing with multiple fallbacks
- [ ] Handle incomplete responses
- [ ] Validate reasoning quality
- [ ] Detect and handle model hallucinations

### 3.3 Resource Management
- [ ] Implement request queuing
- [ ] Add rate limiting
- [ ] Memory-aware caching
- [ ] Connection pooling

## Phase 4: Configuration & Flexibility (Priority: LOW)

### 4.1 Configuration Enhancement
- [ ] Externalize all magic numbers
- [ ] Support for multiple prompt templates
- [ ] Model-specific configurations
- [ ] Runtime parameter tuning

### 4.2 Monitoring & Metrics
- [ ] Add performance metrics collection
- [ ] Implement health checks
- [ ] Add debug mode with detailed logging
- [ ] Create diagnostics endpoint

### 4.3 Documentation
- [ ] Comprehensive API documentation
- [ ] Performance tuning guide
- [ ] Troubleshooting guide
- [ ] Model selection guide

## Implementation Strategy

### Week 1: Performance & Optimization
1. Profile current implementation
2. Optimize text extraction windows
3. Enhance caching system
4. Implement batch processing prototype

### Week 2: Testing & Validation
1. Create comprehensive test suite
2. Run real PDF validation
3. Perform load testing
4. Document findings

### Week 3: Robustness & Polish
1. Enhance error handling
2. Add monitoring/metrics
3. Create documentation
4. Final optimization pass

## Success Criteria

### Performance Targets
- Reduce processing time to < 10 seconds per page (70% improvement)
- Maintain or improve current accuracy (F1 > 0.889)
- Support concurrent processing of 5+ documents
- Cache hit rate > 30% in typical usage

### Quality Metrics
- 100% test coverage for critical paths
- Zero crashes in 1000+ page processing
- Graceful handling of all error scenarios
- Clear performance/accuracy trade-offs documented

### Validation Requirements
- Tested on 10+ real PDF documents
- Validated with 3+ different model variations
- Stress tested with 1000+ pages
- Integration tested with full pipeline

## Risk Mitigation

### Performance Risks
- **Risk**: Cannot achieve < 10s target
- **Mitigation**: Implement hybrid approach with pre-filtering

### Accuracy Risks
- **Risk**: Optimizations reduce accuracy
- **Mitigation**: Maintain multiple prompt versions, A/B testing

### Integration Risks
- **Risk**: Ollama instability affects production
- **Mitigation**: Implement fallback detector, queuing system

## Next Steps
1. Begin with performance profiling to establish baseline
2. Create test infrastructure for real PDF validation
3. Implement most promising optimizations
4. Iterate based on test results
