# LLM Detection Experiments

This module provides a systematic framework for experimenting with LLM-based document boundary detection using progressive difficulty testing and comprehensive prompt engineering.

## Overview

The experiments framework uses synthetic test cases of varying difficulty to systematically evaluate different models and prompting strategies before testing on real PDFs. This approach ensures robust, data-driven optimization of the LLM detection system.

## Quick Start

```bash
# 1. Verify system setup
python pdf_splitter/detection/experiments/verify_prompt_system.py

# 2. Run a quick test with one model/prompt
python pdf_splitter/detection/experiments/systematic_prompt_test.py \
  --quick-test phi4-mini:3.8b A1_asymmetric

# 3. Run full systematic testing
python pdf_splitter/detection/experiments/systematic_prompt_test.py \
  --models phi4-mini:3.8b gemma3:latest
```

## Framework Components

### Core Modules

1. **experiment_runner.py** - Base experiment framework with Ollama integration
2. **synthetic_boundary_tests.py** - Original synthetic test cases (difficulty 1-10)
3. **enhanced_synthetic_tests.py** - Extended test suite with progressive testing
4. **systematic_prompt_test.py** - Main testing orchestrator
5. **verify_prompt_system.py** - System verification tool

### Prompt Templates

Located in `prompts/` directory:

- **A1_asymmetric** - Conservative single-token output
- **A2_high_confidence** - High confidence requirement
- **B1_json_confidence** - JSON output with confidence scores
- **B2_confidence_threshold** - Explicit confidence rating
- **C1_silent_checklist** - Structured checklist approach
- **C2_self_check** - Double-check mechanism
- **D1_conservative_few_shot** - Few-shot with conservative examples

### Test Data

Synthetic test cases span difficulty levels 1-10:
- **Easy (1-3)**: Obvious boundaries (signatures, headers)
- **Medium (4-6)**: Ambiguous cases requiring context
- **Hard (7-10)**: Edge cases and challenging transitions

## Adding New Experiments

### 1. Create a New Prompt Template

Add a `.txt` file to `prompts/`:

```
prompts/my_new_prompt.txt
```

Use template variables:
- `{page1_bottom}` - Bottom text of first page
- `{page2_top}` - Top text of second page

### 2. Add Prompt Configuration

In your test script, define the prompt configuration:

```python
prompts = {
    "my_prompt": {
        "template": "...",  # or load from file
        "config": {
            "temperature": 0.0,
            "max_tokens": 10,
            "stop": ["S", "D"]  # optional
        },
        "post_process": "custom"  # optional
    }
}
```

### 3. Test Progressively

Use the enhanced synthetic tester for progressive testing:

```python
from enhanced_synthetic_tests import EnhancedSyntheticTester

tester = EnhancedSyntheticTester()
results = tester.test_easy_medium_hard_progression(
    models=["phi4-mini:3.8b"],
    confidence_threshold=0.4
)
```

## Running Experiments

### Systematic Testing

```bash
# Test all prompts on synthetic data, then test winners on real PDFs
python systematic_prompt_test.py --models phi4-mini:3.8b gemma3:latest

# Skip real PDF testing
python systematic_prompt_test.py --no-real-pdf

# Adjust confidence threshold for B1/B2 prompts
python systematic_prompt_test.py --confidence-threshold 0.6
```

### Direct Testing

```bash
# Test enhanced synthetic framework directly
python enhanced_synthetic_tests.py

# Run original synthetic tests
python synthetic_boundary_tests.py
```

## Analyzing Results

Results are saved as JSON in `results/` with:
- Configuration used
- Performance metrics (precision, recall, F1)
- Timing information
- Detailed predictions
- Error logs

### Interpreting Metrics

- **Precision**: How many detected boundaries were correct
- **Recall**: How many true boundaries were found
- **F1 Score**: Harmonic mean of precision and recall
- **Target**: High recall (>90%) with acceptable precision (>40%)

## Best Practices

1. **Start Simple**: Test basic prompts before complex ones
2. **Progressive Testing**: Validate on easy cases before testing hard ones
3. **Multiple Models**: Test across different model sizes and architectures
4. **Document Everything**: Keep notes on what works and what doesn't
5. **Version Control**: Track prompt iterations and results

## Documentation

- `HISTORICAL_FINDINGS.md` - Learnings from initial experiments
- `EXPERIMENTAL_PROGRESS.md` - Timeline and progress tracking
- `PROMPT_TESTING_RESULTS.md` - Detailed results from prompt engineering tests
- `FULL_TEST_RESULTS.md` - Comprehensive performance test results (Jan 4, 2025)
- `PROMPT_ENGINEERING_GUIDE.md` - Guide for creating effective prompts
- `FRAMEWORK_OVERVIEW.md` - Technical architecture details
- `KEY_ACHIEVEMENTS.md` - Major milestones and breakthroughs

## Troubleshooting

### Ollama Issues
```bash
# Ensure Ollama is running
ollama serve

# Pull required models
ollama pull phi4-mini:3.8b
ollama pull gemma3:latest
```

### Poor Results
1. Verify models can follow simple instructions first
2. Check response parsing matches model output format
3. Try lower temperatures (0.0-0.1)
4. Ensure sufficient context (200-300 chars recommended)

## Next Steps

1. Continue testing new prompt strategies
2. Add more challenging synthetic test cases
3. Test on diverse PDF types
4. Optimize for production deployment
5. Integrate with visual and heuristic detectors
