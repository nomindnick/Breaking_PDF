# Prompt Engineering Test Results

## Overview

This document summarizes the implementation and testing of advanced prompt engineering techniques for LLM-based document boundary detection, based on research into optimal prompting strategies for Small Language Models (SLMs).

## Implementation Summary

### 1. Model-Specific Optimal Prompts

#### Phi4 Optimal (`phi4_optimal.txt`)
- **Format**: Uses Phi4's `<|im_start|>` / `<|im_end|>` tokens
- **Structure**: system → user → assistant roles
- **Key Features**:
  - Hyper-specific persona: "meticulous document analyst specializing in automated document segmentation"
  - Chain-of-Draft reasoning with XML tags (`<thinking>` and `<answer>`)
  - Three carefully crafted few-shot examples
  - Stop tokens: `["<|im_end|>"]`

#### Gemma3 Optimal (`gemma3_optimal.txt`)
- **Format**: Uses Gemma's `<start_of_turn>` / `<end_of_turn>` tokens
- **Structure**: All instructions embedded in user turn (no system role)
- **Key Features**:
  - Same persona and reasoning structure as Phi4
  - Adapted for Gemma's conversation format
  - Stop tokens: `["<end_of_turn>"]`

### 2. Chain-of-Draft (CoD) Prompts

#### E1 CoD Reasoning (`E1_cod_reasoning.txt`)
- **Purpose**: Structured reasoning with explicit steps
- **Format**: 
  ```
  1. Page 1 Analysis: (content type in 3-5 words)
  2. Page 2 Analysis: (content type in 3-5 words)
  3. Logical Connection: (relationship in 5-10 words)
  4. Decision: SAME or DIFFERENT
  ```

#### E2 CoD Minimal (`E2_cod_minimal.txt`)
- **Purpose**: Ultra-concise reasoning for speed
- **Format**:
  ```
  P1: [doc type]
  P2: [doc type]
  Link: [same/different topic]
  → [S or D]
  ```

### 3. Supporting Infrastructure

#### Model Formatting (`model_formatting.py`)
- Detects model family from name
- Applies appropriate formatting tokens
- Handles system prompt conversion
- Extracts clean responses

#### Enhanced Testing Framework (`enhanced_synthetic_tests.py`)
- Dynamic prompt loading from files
- Model-specific prompt filtering
- XML and CoD response parsing
- Multiple post-processing strategies

#### Constrained Generation Support (`experiment_runner.py`)
- Configuration for constrained vocabulary
- Post-processing simulation of constrained generation
- Support for model-specific stop tokens

## Test Results

### Quick Demo Results (3 Test Cases)

| Test Case | Description | Baseline | Phi4 Optimal | Gemma3 Optimal |
|-----------|-------------|----------|--------------|----------------|
| Clear Different | Letter closing → Invoice | ✓ | ✗ | ✓ |
| Clear Same | Efficiency statement continuation | ✗ | ✓ | ✓ |
| Chapter Break | Chapter 3 → Next chapter | ✗ | ✓ | ✓ |
| **Overall Accuracy** | | **33%** | **67%** | **100%** |

### Performance Metrics

| Metric | Baseline | Phi4 Optimal | Gemma3 Optimal |
|--------|----------|--------------|----------------|
| Avg Response Time | ~3s | 7.24s | 5.95s |
| Reasoning Provided | No | Yes (XML) | Yes (XML) |
| Consistent Format | No | Yes | Yes |

### Key Observations

1. **Accuracy Improvements**:
   - Baseline (simple prompt): 33% - tends to over-predict "DIFFERENT"
   - Phi4 with optimal prompt: 67% - good understanding but missed one case
   - Gemma3 with optimal prompt: 100% - perfect classification

2. **Reasoning Quality**:
   - Both optimal prompts produce clear Chain-of-Draft reasoning
   - Reasoning is concise and relevant
   - XML structure makes parsing reliable

3. **Response Consistency**:
   - Optimal prompts always produce structured output
   - Baseline prompt produces variable output requiring complex parsing

## Prompt Strategies Tested

### Group A: Output Control + Conservative Bias
- **A1_asymmetric**: Conservative bias with single-token output
- **A2_high_confidence**: Requires high confidence for "different"

### Group B: Confidence Scoring
- **B1_json_confidence**: JSON output with confidence scores
- **B2_confidence_threshold**: Explicit confidence rating

### Group C: Structured Decision Making
- **C1_silent_checklist**: Silent checklist evaluation
- **C2_self_check**: Double-check mechanism

### Group D: Few-Shot Learning
- **D1_conservative_few_shot**: Strategic examples with conservative bias

### Group E: Chain-of-Draft (New)
- **E1_cod_reasoning**: Full CoD with structured steps
- **E2_cod_minimal**: Minimal CoD for speed

### Optimal Prompts (New)
- **phi4_optimal**: Research-based optimal for Phi models
- **gemma3_optimal**: Research-based optimal for Gemma models

## Technical Implementation Details

### XML Response Parsing
```python
# Extract answer from XML tags
answer_match = re.search(r"<answer>\s*([^<]+)\s*</answer>", response, re.IGNORECASE)
if answer_match:
    answer = answer_match.group(1).strip().upper()
```

### Model-Specific Formatting
```python
# Phi4 format
formatted = f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
formatted += f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
formatted += "<|im_start|>assistant\n"

# Gemma3 format (no system role)
formatted = f"<start_of_turn>user\n{system_prompt}\n\n{user_prompt}\n<end_of_turn>\n"
formatted += "<start_of_turn>model\n"
```

### Post-Processing Strategies
1. **Simple S/D extraction**: For baseline prompts
2. **JSON parsing**: For confidence-based prompts
3. **XML extraction**: For optimal prompts
4. **CoD parsing**: For Chain-of-Draft prompts

## Lessons Learned

1. **Model-Specific Formatting Matters**: Using the correct conversation format significantly improves response quality and consistency.

2. **Few-Shot Examples Are Critical**: Well-chosen examples that cover edge cases (like chapter breaks) dramatically improve accuracy.

3. **Structured Output Formats**: XML tags or other structured formats make response parsing reliable and enable transparent reasoning.

4. **Chain-of-Draft Works**: Even for small models, structured reasoning improves decision quality without excessive verbosity.

5. **Model Differences**: Gemma3 performed better than Phi4 on these tasks, possibly due to training data or architecture differences.

## Recommendations

1. **For Production**: Use `gemma3_optimal` prompt for highest accuracy (100% on test cases)

2. **For Speed**: Consider `E2_cod_minimal` with constrained generation for faster responses

3. **For Debugging**: The XML reasoning tags provide valuable insight into model decisions

4. **For Integration**: Build on the `enhanced_synthetic_tests.py` framework for continued experimentation

## Next Steps

1. Test on full difficulty range (1-10) with larger sample size
2. Evaluate on real PDF documents
3. Implement true constrained generation with Outlines
4. Build production `LLMDetector` class
5. Integrate with visual and heuristic detectors for hybrid approach

## Code Artifacts

- Prompt templates: `pdf_splitter/detection/experiments/prompts/`
- Testing framework: `enhanced_synthetic_tests.py`, `test_optimal_prompts.py`
- Model formatting: `model_formatting.py`
- Results: `pdf_splitter/detection/experiments/results/`

## References

- Original research document: `docs/prompt_engineering_reference/prompt_engineering_research.txt`
- Implementation based on sections discussing Gemma/Phi formatting, Chain-of-Draft, and constrained generation