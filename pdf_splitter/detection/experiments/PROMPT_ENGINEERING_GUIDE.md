# Prompt Engineering Guide for LLM Boundary Detection

## Overview

This guide describes the enhanced prompt engineering system for improving LLM-based document boundary detection. The system implements a progressive testing approach using synthetic test cases before evaluating on real PDFs.

## System Components

### 1. Enhanced Synthetic Test Cases (`enhanced_synthetic_tests.py`)
- Extended test suite with 20+ test cases across difficulty levels 1-10
- Categories: obvious_same, obvious_different, medium_same, medium_different, hard_same, hard_different, edge cases
- Progressive testing: Easy (1-3) → Medium (4-6) → Hard (7-10)

### 2. Prompt Templates (in `prompts/` directory)

#### Group A: Output Control + Conservative Bias
- **A1_asymmetric**: Single token output (S/D) with conservative bias
- **A2_high_confidence**: Requires high confidence for "different" classification

#### Group B: Confidence Scoring
- **B1_json_confidence**: JSON output with confidence scores
- **B2_confidence_threshold**: Explicit confidence rating before classification

#### Group C: Structured Decision Making
- **C1_silent_checklist**: Mental checklist before classification
- **C2_self_check**: Double-check mechanism with conservative fallback

#### Group D: Few-Shot Learning
- **D1_conservative_few_shot**: Strategic examples biasing toward "same"

### 3. Experiment Runner Integration
- Added `synthetic_pairs` strategy to process page pairs
- Support for stop tokens and custom post-processing
- Handles all prompt template formats

### 4. Systematic Testing Script (`systematic_prompt_test.py`)
- Automated progressive difficulty testing
- Identifies best model/prompt combinations
- Tests winning combinations on real PDFs

## Usage Guide

### Quick Verification
```bash
# Verify system setup
python pdf_splitter/detection/experiments/verify_prompt_system.py
```

### Quick Test Single Prompt
```bash
# Test a specific model/prompt combination
python pdf_splitter/detection/experiments/systematic_prompt_test.py \
  --quick-test phi4-mini:3.8b A1_asymmetric
```

### Run Full Systematic Test
```bash
# Test all prompts with multiple models
python pdf_splitter/detection/experiments/systematic_prompt_test.py \
  --models phi4-mini:3.8b gemma3:latest qwen3:8b qwen3:1.7b
```

### Test Synthetic Only
```bash
# Skip real PDF testing
python pdf_splitter/detection/experiments/systematic_prompt_test.py \
  --no-real-pdf
```

### Run Enhanced Synthetic Tests Directly
```bash
# Run just the synthetic tests with detailed analysis
python pdf_splitter/detection/experiments/enhanced_synthetic_tests.py
```

## Prompt Engineering Strategies

### 1. Conservative Bias (A1, A2)
- Assumes pages are consecutive by default
- Requires strong evidence for boundary detection
- Reduces false positives

### 2. Confidence Thresholding (B1, B2)
- Models provide confidence scores
- Post-processing applies thresholds (default: 0.4)
- Allows fine-tuning precision/recall trade-off

### 3. Structured Reasoning (C1, C2)
- Forces systematic evaluation
- Reduces impulsive classifications
- Self-checking mechanism

### 4. Few-Shot Examples (D1)
- Strategic examples guide model behavior
- Conservative examples reduce false positives
- Helps with ambiguous cases

## Expected Results

### Synthetic Test Performance
- **Easy cases (1-3)**: Should achieve 90-100% accuracy
- **Medium cases (4-6)**: Target 70-90% accuracy
- **Hard cases (7-10)**: Accept 50-70% accuracy

### Real PDF Performance
- **Recall**: Target 90-100% (don't miss boundaries)
- **Precision**: Accept 40-60% (false positives are okay)
- **Speed**: < 3 seconds per boundary

## Interpreting Results

### Systematic Test Output
```json
{
  "best_combinations": [
    {
      "model": "phi4-mini:3.8b",
      "prompt": "A1_asymmetric",
      "easy_accuracy": 0.95,
      "medium_accuracy": 0.80,
      "hard_accuracy": 0.60,
      "overall_score": 0.82
    }
  ]
}
```

### Key Metrics
- **Overall Score**: Weighted average (Easy:3, Medium:2, Hard:1)
- **F1 Score**: Balance of precision and recall on real PDFs
- **False Positives**: Over-segmentation (acceptable)
- **False Negatives**: Missed boundaries (minimize these)

## Next Steps

1. **Run systematic tests** to identify best model/prompt combinations
2. **Analyze failure patterns** in results to refine prompts
3. **Test on diverse PDF sets** to ensure generalization
4. **Implement winning combination** in production detector
5. **Add visual and heuristic signals** to reduce false positives

## Troubleshooting

### Ollama Not Running
```bash
# Start Ollama service
ollama serve

# Pull required models
ollama pull phi4-mini:3.8b
ollama pull gemma3:latest
```

### Poor Performance
1. Check if models are following instructions (use verify script)
2. Adjust confidence thresholds for B1/B2 prompts
3. Try different temperature settings (0.0-0.1 recommended)
4. Ensure sufficient context (300 chars recommended)

### Memory Issues
- Test fewer models at once
- Use smaller models (qwen3:1.7b, phi4-mini:3.8b)
- Process PDFs in batches

## Model Recommendations

Based on previous experiments:
- **Best Overall**: phi4-mini:3.8b (fast, good instruction following)
- **Backup Option**: gemma3:latest (consistent performance)
- **Avoid**: qwen3:8b (poor instruction following despite good accuracy)
