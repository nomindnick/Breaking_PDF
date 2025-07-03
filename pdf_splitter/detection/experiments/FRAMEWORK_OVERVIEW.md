# LLM Detection Framework Overview

## Current State (January 2025)

The experiments directory has been cleaned and reorganized around a systematic prompt engineering framework for LLM-based document boundary detection.

## Directory Structure

```
experiments/
├── README.md                      # Main documentation
├── FRAMEWORK_OVERVIEW.md          # This file
├── PROMPT_ENGINEERING_GUIDE.md    # Detailed prompt engineering guide
├── HISTORICAL_FINDINGS.md         # Consolidated learnings from past experiments
│
├── Core Framework
│   ├── experiment_runner.py       # Base experiment framework
│   ├── synthetic_boundary_tests.py # Original synthetic tests
│   ├── enhanced_synthetic_tests.py # Extended test suite
│   ├── systematic_prompt_test.py  # Main testing orchestrator
│   ├── verify_prompt_system.py    # System verification
│   └── run_experiments.py         # Original runner (backward compatibility)
│
├── prompts/                       # Prompt templates
│   ├── A1_asymmetric.txt         # Conservative single-token
│   ├── A2_high_confidence.txt    # High confidence requirement
│   ├── B1_json_confidence.txt    # JSON with confidence
│   ├── B2_confidence_threshold.txt # Explicit confidence
│   ├── C1_silent_checklist.txt   # Structured checklist
│   ├── C2_self_check.txt         # Double-check mechanism
│   ├── D1_conservative_few_shot.txt # Few-shot examples
│   └── (legacy templates)
│
├── results/                       # Experiment results (JSON)
├── configs/                       # Experiment configurations (future use)
└── analysis/                      # Analysis notebooks (future use)
```

## Key Components

### 1. Progressive Testing Approach
- Start with easy synthetic cases (difficulty 1-3)
- Progress to medium (4-6) only if model performs well
- Test hard cases (7-10) for top performers
- Finally test on real PDFs

### 2. Systematic Evaluation
- Multiple prompt strategies (A1-D1)
- Multiple models tested in parallel
- Automated metric calculation
- Best combination identification

### 3. Extensible Design
- Easy to add new prompts (just add .txt file)
- Simple to test new models
- Results saved for analysis
- Framework handles all orchestration

## Next Steps for Experimentation

1. **Test More Models**
   - Add new Ollama models as they become available
   - Test different model sizes and architectures

2. **Refine Prompts**
   - Create variations of successful prompts
   - Test role-based prompts ("You are a document analyst...")
   - Experiment with different output formats

3. **Expand Test Cases**
   - Add more edge cases
   - Create domain-specific test sets
   - Test with different document types

4. **Optimize Performance**
   - Find the sweet spot between accuracy and speed
   - Test prompt compression techniques
   - Explore batching strategies

## Best Practices

1. Always run `verify_prompt_system.py` before major tests
2. Start with `--quick-test` to validate changes
3. Use `--no-real-pdf` for rapid iteration
4. Document findings in result analysis
5. Version control prompt iterations

## Current Best Results

From initial testing:
- **Best Model**: gemma3:latest (100% accuracy on single-char output)
- **Best Prompt Style**: Single character (S/D) classification
- **Key Issue**: Models tend to over-explain instead of classify

See test results in `results/` directory for detailed metrics.
