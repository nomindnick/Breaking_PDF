# LLM Detection Experiments

This directory contains the experimental framework and results that led to the production LLMDetector implementation.

## Overview

Through systematic experimentation, we achieved:
- **F1 Score: 0.889** with balanced dataset
- **100% Precision** (no false boundaries)
- **Production-ready approach** using gemma3:latest model

## Directory Structure

```
experiments/
├── FINAL_RESULTS_SUMMARY.md    # Consolidated findings and recommendations
├── experiment_runner.py        # Main experiment framework
├── model_formatting.py         # Model-specific prompt formatting (used in production)
├── prompts/                    # Tested prompt templates
│   ├── gemma3_optimal.txt     # Best performing prompt (used in production)
│   ├── phi4_optimal.txt       # Optimal for Phi models
│   └── ...                    # Other tested prompts
├── results/                    # Raw experiment results
├── archive/                    # Historical experiments and scripts
└── *.md                       # Various documentation and findings
```

## Key Files for Production

1. **model_formatting.py** - Required for LLMDetector, handles model-specific formatting
2. **prompts/gemma3_optimal.txt** - Production prompt template
3. **FINAL_RESULTS_SUMMARY.md** - Key findings and configuration

## Running New Experiments

To run new experiments with the framework:
```python
python experiment_runner.py --model gemma3:latest --prompt prompts/new_prompt.txt
```

## Historical Documentation

The extensive documentation files chronicle the journey to the optimal solution:

- **FINAL_RESULTS_SUMMARY.md** - Executive summary and production recommendations
- **EXPERIMENT_SUMMARY_JAN5_2025.md** - Key milestone achieving target performance
- **OPTIMAL_PROMPTS_RESULTS.md** - Analysis of winning prompt strategies
- **BALANCED_DATASET_RESULTS.md** - Critical dataset balance findings
- **PROMPT_ENGINEERING_GUIDE.md** - Lessons learned for prompt design
- **LESSONS_LEARNED.md** - What worked and what didn't

## Archive Contents

The `archive/` directory contains 30+ experimental scripts that explored:
- Different prompting strategies (asymmetric, chain-of-thought, few-shot)
- Model comparisons (Llama3, Gemma3, Phi4, etc.)
- Performance optimizations (preloading, batching)
- Dataset balance analysis
- Constrained generation attempts
- Hybrid detection approaches

These experiments informed the final production implementation in `../llm_detector.py`.

## Key Learnings

1. **Model-specific formatting is critical** - Generic prompts reduce accuracy by 60%
2. **Dataset balance matters** - Imbalanced data masked true performance
3. **Conservative bias preferred** - Users prefer missing boundaries over false splits
4. **Simple prompts with examples work best** - Complex reasoning increased errors

## Production Configuration

Based on experiments, the optimal configuration is:
```python
model = "gemma3:latest"
prompt = "gemma3_optimal"
temperature = 0.1
confidence_threshold = 0.8
context_chars = 500
```

This configuration achieves:
- F1 Score: 0.889
- Precision: 100%
- Recall: 80%
- Processing: ~33s per boundary
