# Detection Module - Experimental Approach

## Overview

The detection module is being developed using an experimental approach, aligning with the project's philosophy of making each component "rock solid" before moving forward. Rather than implementing all detection methods simultaneously, we're focusing first on perfecting LLM-based detection through systematic experimentation.

## Why Experimental?

1. **Accuracy is Paramount**: Document boundary detection is the core functionality - it must be highly accurate
2. **Multiple Variables**: Different models, prompts, and strategies can drastically affect results
3. **Data-Driven Decisions**: We want empirical evidence for our architectural choices
4. **Cost-Benefit Analysis**: Understanding the trade-offs between accuracy and latency

## Current Architecture

### Base Layer (✅ Complete)
- `base_detector.py`: Abstract base class defining the interface for all detectors
- Data models: `ProcessedPage`, `BoundaryResult`, `DetectionContext`
- Shared utilities for context extraction, text similarity, pattern matching
- 97% test coverage ensuring solid foundation

### Experimentation Framework (✅ Complete)
```
experiments/
├── experiment_runner.py      # Core framework with metrics tracking
├── run_experiments.py        # CLI for running experiments
├── prompts/                  # Prompt templates
│   ├── default.txt
│   ├── focused_boundary.txt
│   └── context_analysis.txt
└── results/                  # JSON results with detailed metrics
```

## Experimentation Strategy

### Models to Test
1. **Llama3:8b-instruct-q5_K_M** - Most capable, baseline for accuracy
2. **Gemma3:latest** - Google's model, good for reasoning
3. **Phi4-mini:3.8b** - Latest small model, balance of speed/accuracy
4. **Phi3:mini** - Older small model for comparison

### Strategies to Evaluate

#### 1. Context Overlap (Primary)
- Sliding window with configurable overlap (20%, 30%, 40%)
- Balances context availability with processing efficiency
- Most similar to human document review

#### 2. Type-First Classification
- Two-pass approach: classify document type, then detect boundaries
- May improve accuracy for heterogeneous document sets
- Higher latency but potentially more accurate

#### 3. Chain-of-Thought (CoT)
- Step-by-step reasoning about boundary decisions
- Provides interpretability and debugging insights
- Highest latency but potentially highest accuracy

#### 4. Multi-Signal (Future)
- Placeholder for integrating visual and heuristic signals
- Will combine LLM with other detection methods

### Metrics Being Tracked
- **Accuracy**: Precision, Recall, F1-score (with 1-page tolerance)
- **Performance**: Total time, time per boundary, time per page
- **Consistency**: Variance across multiple runs
- **Errors**: Failed detections, parsing errors

## Running Experiments

### Basic Usage
```bash
# Test default configuration
python -m pdf_splitter.detection.experiments.run_experiments

# Test multiple models and strategies
python -m pdf_splitter.detection.experiments.run_experiments \
    --models llama3:8b-instruct-q5_K_M gemma3:latest phi4-mini:3.8b \
    --strategies context_overlap chain_of_thought type_first

# Compare results
python -m pdf_splitter.detection.experiments.run_experiments \
    --compare-only \
    --models llama3:8b-instruct-q5_K_M gemma3:latest
```

### Experiment Workflow
1. Load test PDF and ground truth
2. Process PDF to extract text
3. Run detection with specified model/strategy
4. Compare predictions to ground truth
5. Calculate metrics and save results
6. Aggregate results for comparison

## Key Questions We're Answering

1. **Which model provides the best accuracy?**
   - Is the larger Llama3 significantly better than smaller models?
   - Do specialized models (Gemma) offer advantages?

2. **What's the optimal context strategy?**
   - How much overlap is needed for accurate detection?
   - Is more context always better?

3. **Do advanced prompting techniques help?**
   - Does Chain-of-Thought improve accuracy?
   - Is type classification a useful preprocessing step?

4. **What are the performance trade-offs?**
   - Can we achieve <2s per boundary with high accuracy?
   - Which approach offers the best accuracy/speed balance?

## Success Criteria

Before moving to production implementation:
- **Accuracy**: >95% F1 score on test set
- **Latency**: <2 seconds per boundary check
- **Consistency**: <5% variance across runs
- **Robustness**: Handles all document types in test set

## Next Steps

1. **Complete Initial Experiments** (Current)
   - Run all model/strategy combinations
   - Analyze results and identify best performers

2. **Optimize Best Approach**
   - Fine-tune prompts based on error analysis
   - Optimize parameters (window size, overlap, temperature)

3. **Implement Production Detector**
   - Create `llm_detector.py` with winning configuration
   - Add caching and optimization
   - Comprehensive error handling

4. **Add Complementary Signals**
   - Visual detector for layout changes
   - Heuristic detector for patterns
   - Signal combiner for consensus

## Benefits of This Approach

1. **Evidence-Based**: Every decision backed by data
2. **Optimal Configuration**: We'll know we have the best setup
3. **Understanding Trade-offs**: Clear view of accuracy vs performance
4. **Future Flexibility**: Framework supports testing new models/strategies
5. **Documentation**: Results provide justification for architectural decisions

## Conclusion

By taking an experimental approach to the detection module, we ensure that the core functionality of the PDF splitter is as accurate and reliable as possible. This methodical approach aligns with the project's philosophy of building "rock solid" components and will result in a detection system we can trust for production use.
